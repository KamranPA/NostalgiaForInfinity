"""
NostalgiaForInfinityX7ML.py

Thin subclass of the upstream NostalgiaForInfinityX7 strategy that adds
ML-snapshot logging on every order fill (entry, rebuy, exit), without
touching the upstream file itself. This means an `nfi-updater` job (or a
manual git pull from iterativv/NostalgiaForInfinity) can freely overwrite
NostalgiaForInfinityX7.py at any time without wiping out this project's
custom data-collection logic.

Point your config's "strategy" field at:
    "NostalgiaForInfinityX7ML"
instead of:
    "NostalgiaForInfinityX7"

Both this file and the upstream NostalgiaForInfinityX7.py must sit next
to each other (repo root), since this file imports the class directly
from it.
"""

import logging

import numpy as np
from freqtrade.persistence import Trade

from NostalgiaForInfinityX7 import NostalgiaForInfinityX7

log = logging.getLogger(__name__)


class NostalgiaForInfinityX7ML(NostalgiaForInfinityX7):
    """
    Identical strategy logic to NostalgiaForInfinityX7, with one addition:
    on every order fill, a snapshot of pair-level and BTC-context
    indicators is written to freqtrade's built-in trade_custom_data table
    via trade.set_custom_data().

    ML analysis then has the *conditions at the time of the signal*
    available, instead of only the final price/profit outcome.

    Data lives in freqtrade's built-in custom_data table (persisted to
    the same SQLite db as trades/orders), queryable later via:
        SELECT * FROM trade_custom_data WHERE ...
    """

    def order_filled(self, pair, trade, order, current_time, **kwargs) -> None:
        # Let the upstream strategy do whatever it normally does on a fill
        # first (currently a no-op in NFI, but future-proofing in case
        # that ever changes upstream).
        super().order_filled(pair, trade, order, current_time, **kwargs)

        try:
            is_entry_fill = order.ft_order_side == trade.entry_side
            fill_number = (
                trade.nr_of_successful_entries if is_entry_fill else None
            )

            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            candle = df.iloc[-1] if len(df) >= 1 else None

            def safe_get(row, col, default=None):
                if row is None or col not in row:
                    return default
                val = row[col]
                try:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return default
                except TypeError:
                    pass
                return float(val) if isinstance(val, (int, float)) else val

            snapshot = {
                "fill_time": current_time.isoformat(),
                "fill_type": "entry"
                if is_entry_fill
                else ("exit" if order.ft_order_side == trade.exit_side else "other"),
                "fill_number": fill_number,
                "order_side": order.ft_order_side,
                "order_tag": order.ft_order_tag,
                "fill_price": safe_get({"p": order.average or order.price}, "p"),
                "current_profit_pct": None,
                # Pair-level indicators, from the already-computed dataframe
                # (nothing extra calculated here, just read off the last candle).
                "rsi_14": safe_get(candle, "RSI_14"),
                "rsi_3": safe_get(candle, "RSI_3"),
                # ADX is only computed on the 4h informative timeframe in
                # this strategy (used for entry conditions #7/#505), not on
                # the base 5m dataframe -- so it carries the "_4h" suffix
                # that merge_informative_pair() adds.
                "adx_14": safe_get(candle, "ADX_14_4h"),
                "plus_di_14": safe_get(candle, "PLUS_DI_14_4h"),
                "minus_di_14": safe_get(candle, "MINUS_DI_14_4h"),
                "ema_20": safe_get(candle, "EMA_20"),
                "ema_50": safe_get(candle, "EMA_50"),
                "ema_200": safe_get(candle, "EMA_200"),
                "close": safe_get(candle, "close"),
                "volume": safe_get(candle, "volume"),
                # Broad market (BTC) context at the same moment -- this is what
                # lets a later analysis tell "independent weak signal" apart
                # from "correlated market-wide dip", instead of guessing from
                # open-timestamp clustering alone.
                # BTC informative indicators are only fetched on the 4h
                # timeframe (see btc_info_timeframes = ["4h"]) and merged
                # with the "_4h" suffix -- plain "BTC_RSI_14" etc. don't
                # exist in the dataframe and would silently resolve to None.
                "btc_rsi_14": safe_get(candle, "BTC_RSI_14_4h"),
                "btc_ema_20": safe_get(candle, "BTC_EMA_20_4h"),
                "btc_ema_200": safe_get(candle, "BTC_EMA_200_4h"),
                "btc_roc_3": safe_get(candle, "BTC_ROC_3_4h"),
                # Portfolio pressure at the moment of this fill -- were slots
                # scarce (bot forced to be selective) or plentiful?
                "open_trade_count_at_fill": Trade.get_open_trade_count(),
                "max_open_trades_config": self.config.get("max_open_trades"),
            }

            # current_profit isn't passed into order_filled; compute a cheap
            # approximation for rebuy/exit fills from the trade's own state.
            if not is_entry_fill and trade.open_rate:
                try:
                    fill_rate = order.average or order.price or trade.open_rate
                    snapshot["current_profit_pct"] = round(
                        ((fill_rate / trade.open_rate) - 1.0) * 100.0, 4
                    )
                except (TypeError, ZeroDivisionError):
                    pass

            trade.set_custom_data(
                key=f"ml_snapshot_fill_{trade.nr_of_successful_entries}_{order.ft_order_side}",
                value=snapshot,
            )

            # Also keep a single always-overwritten "entry_context" key holding
            # ONLY the very first entry's snapshot, so querying "what were the
            # conditions when this trade was opened" doesn't require knowing
            # how many fills happened later.
            if is_entry_fill and trade.nr_of_successful_entries == 1:
                trade.set_custom_data(key="entry_context", value=snapshot)
        except Exception:
            # Analytics logging must never be able to break live trading.
            # If anything above fails (missing column, None dataframe, etc.)
            # just skip logging for this fill and continue normally.
            log.warning(
                f"[{current_time}] ml_snapshot logging failed for {pair}",
                exc_info=True,
            )
