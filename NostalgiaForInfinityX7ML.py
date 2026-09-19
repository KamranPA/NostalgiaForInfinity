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

SNAPSHOT SCHEMA v2 (2026-09-18): added enter_tag/exit_reason (previously
only visible via Telegram, not stored in the snapshot itself), ATR_14,
1h/1d context indicators, and a strategy_version field so future ML
analysis can tell which snapshot shape a given row came from.

1h/1d COLUMN NAMES (verified 2026-09-18 directly against
NostalgiaForInfinityX7.py's informative_1h_indicators()/
informative_1d_indicators()): 1h only computes EMA_12/EMA_200 (there is
no EMA_50_1h), and neither the 1h nor 1d informative function computes
ADX at all (ADX_14 only exists on the 4h merge, as ADX_14_4h) -- so
those two guessed fields from the first draft were dropped/corrected
below. RANGE_PCT_14_1d (used upstream by the bad-trade/stale-range
check) is included instead as the 1d volatility figure.

ATR: not computed anywhere in the upstream strategy, so it isn't
available on any existing dataframe column. It's computed here instead,
directly from the base-timeframe OHLC via talib, wrapped in the same
try/except as everything else so a computation failure just drops this
one field rather than breaking the fill.
"""

import logging

import numpy as np
import talib.abstract as ta
from freqtrade.persistence import Trade

from NostalgiaForInfinityX7 import NostalgiaForInfinityX7

log = logging.getLogger(__name__)

ML_SNAPSHOT_SCHEMA_VERSION = 2

# Overrides the upstream default of ["4h"] (see NostalgiaForInfinityX7.py's
# commented-out alternatives at that line) so BTC_EMA_20/BTC_ROC_3 (only
# computed on 1h/15m/5m) and BTC_EMA_200 (only computed on 1d) actually
# get merged onto the base dataframe instead of silently staying None.
# populate_indicators() iterates self.btc_info_timeframes dynamically and
# merges each timeframe independently -- confirmed by reading that loop
# directly -- so this is purely additive: BTC_RSI_14_4h and every existing
# entry/exit condition that depends on it are completely unaffected.
# Placed here (subclass) rather than edited into the upstream file so an
# nfi-updater re-sync of NostalgiaForInfinityX7.py from iterativv can't
# silently revert it back to ["4h"] only.
BTC_INFO_TIMEFRAMES_OVERRIDE = ["4h", "1h", "1d"]


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

    btc_info_timeframes = BTC_INFO_TIMEFRAMES_OVERRIDE

    @staticmethod
    def _safe_atr_14(df, timeperiod: int = 14):
        """ATR isn't computed anywhere upstream, so it's derived here
        straight from the base-timeframe OHLC using talib. Returns None
        (rather than raising) if the dataframe is too short or missing
        the expected columns."""
        try:
            if df is None or len(df) < timeperiod + 1:
                return None
            atr_series = ta.ATR(df, timeperiod=timeperiod)
            val = atr_series.iloc[-1]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)
        except Exception:
            return None

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

            # Strategy version: prefer upstream's own version() if it defines
            # one (some NFI releases report their own version string there),
            # falling back to just this subclass's snapshot schema version.
            try:
                upstream_version = super().version()
            except Exception:
                upstream_version = None

            snapshot = {
                "schema_version": ML_SNAPSHOT_SCHEMA_VERSION,
                "strategy_version": upstream_version,
                "fill_time": current_time.isoformat(),
                "fill_type": "entry"
                if is_entry_fill
                else ("exit" if order.ft_order_side == trade.exit_side else "other"),
                "fill_number": fill_number,
                "order_side": order.ft_order_side,
                "order_tag": order.ft_order_tag,
                "fill_price": safe_get({"p": order.average or order.price}, "p"),
                "current_profit_pct": None,
                # Explicit enter_tag/exit_reason: Telegram already surfaces
                # these from trade.enter_tag/exit_reason, but they weren't
                # previously copied into the snapshot itself -- without them,
                # joining a snapshot row back to "which setup fired" required
                # a separate join against the trades table.
                "enter_tag": trade.enter_tag,
                "exit_reason": trade.exit_reason,
                # Pair-level indicators, from the already-computed dataframe
                # (nothing extra calculated here, just read off the last candle).
                "rsi_14": safe_get(candle, "RSI_14"),
                "rsi_3": safe_get(candle, "RSI_3"),
                "atr_14": self._safe_atr_14(df),
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
                # Higher-timeframe context (1h/1d) -- column names verified
                # directly against informative_1h_indicators()/
                # informative_1d_indicators() upstream (see module docstring).
                "rsi_14_1h": safe_get(candle, "RSI_14_1h"),
                "ema_12_1h": safe_get(candle, "EMA_12_1h"),
                "ema_200_1h": safe_get(candle, "EMA_200_1h"),
                "rsi_14_1d": safe_get(candle, "RSI_14_1d"),
                "ema_50_1d": safe_get(candle, "EMA_50_1d"),
                "ema_200_1d": safe_get(candle, "EMA_200_1d"),
                "range_pct_14_1d": safe_get(candle, "RANGE_PCT_14_1d"),
                # Broad market (BTC) context at the same moment -- this is what
                # lets a later analysis tell "independent weak signal" apart
                # from "correlated market-wide dip", instead of guessing from
                # open-timestamp clustering alone.
                # BTC informative columns come from different timeframes
                # upstream, confirmed directly against
                # btc_informative_4h/1h/1d_indicators(): BTC_RSI_14 is only
                # computed on 4h; BTC_EMA_20/BTC_ROC_3 only on 1h/15m/5m;
                # BTC_EMA_200 only on 1d. btc_info_timeframes is overridden
                # above (BTC_INFO_TIMEFRAMES_OVERRIDE) specifically so all
                # three of these merge onto the dataframe with real values
                # instead of resolving to None.
                "btc_rsi_14": safe_get(candle, "BTC_RSI_14_4h"),
                "btc_ema_20_1h": safe_get(candle, "BTC_EMA_20_1h"),
                "btc_ema_200_1d": safe_get(candle, "BTC_EMA_200_1d"),
                "btc_roc_3_1h": safe_get(candle, "BTC_ROC_3_1h"),
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

            # exit_reason is often only finalized on the trade right as/after
            # the exit order fills, so entry_context's exit_reason usually
            # stays None -- also refresh entry_context's exit_reason (without
            # touching anything else about it) once an exit fill lands, so
            # "what closed this trade" is queryable from the same row.
            elif not is_entry_fill:
                try:
                    existing = trade.get_custom_data("entry_context")
                    if isinstance(existing, dict) and trade.exit_reason:
                        existing["exit_reason"] = trade.exit_reason
                        trade.set_custom_data(key="entry_context", value=existing)
                except Exception:
                    pass
        except Exception:
            # Analytics logging must never be able to break live trading.
            # If anything above fails (missing column, None dataframe, etc.)
            # just skip logging for this fill and continue normally.
            log.warning(
                f"[{current_time}] ml_snapshot logging failed for {pair}",
                exc_info=True,
            )
