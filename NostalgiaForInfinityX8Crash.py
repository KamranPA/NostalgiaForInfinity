"""
NostalgiaForInfinityX8Crash  (EXPERIMENT, research only)

X8 + a "sharp drop without a bounce" gate on NEW LONG entries
(first entry only). Rebuys / grinds of open trades and shorts are untouched.

crash_scope decides WHOSE price is judged (1h candles, lookahead-safe merge):
  "btc"  (default) -> BTC/USDT:USDT, one market-wide gate
  "pair"           -> the traded pair itself, each coin judged on its own

Gate is ON when BOTH hold:
  1) price is >= crash_drop_pct % below its high of the last crash_high_hours
  2) price is <= crash_near_low_pct % above its low of the last crash_low_hours

Parameters come from config["guard_parameters"] (workflow input guard_parameters),
e.g. {"crash_scope": "pair", "crash_drop_pct": 40, "crash_near_low_pct": 5}.
Defaults are UNTESTED first guesses (a single coin drops much more than BTC,
so scope "pair" needs a larger crash_drop_pct than scope "btc").
crash_drop_pct = 0 turns the gate off (identical to plain X8).
"""
import logging

from pandas import DataFrame

from freqtrade.strategy import merge_informative_pair

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

log = logging.getLogger(__name__)


class NostalgiaForInfinityX8Crash(NostalgiaForInfinityX8):
    crash_scope = "btc"  # "btc" or "pair"
    crash_drop_pct = 20.0
    crash_high_hours = 720  # 30 days
    crash_near_low_pct = 5.0
    crash_low_hours = 168  # 7 days
    crash_btc_pair = "BTC/USDT:USDT"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        gp = config.get("guard_parameters", {}) or {}
        for key, cast in (
            ("crash_scope", str),
            ("crash_drop_pct", float),
            ("crash_high_hours", int),
            ("crash_near_low_pct", float),
            ("crash_low_hours", int),
        ):
            if key in gp:
                setattr(self, key, cast(gp[key]))
        if self.crash_scope not in ("btc", "pair"):
            raise ValueError(f"crash_scope must be 'btc' or 'pair', got {self.crash_scope!r}")
        log.info(
            "[Crash] params scope=%s drop_pct=%s high_hours=%s near_low_pct=%s low_hours=%s",
            self.crash_scope,
            self.crash_drop_pct,
            self.crash_high_hours,
            self.crash_near_low_pct,
            self.crash_low_hours,
        )

    def informative_pairs(self):
        pairs = list(super().informative_pairs())
        if self.crash_drop_pct > 0:
            wanted = [self.crash_btc_pair]
            if self.crash_scope == "pair" and self.dp:
                wanted += list(self.dp.current_whitelist())
            for p in wanted:
                item = (p, "1h")
                if item not in pairs:
                    pairs.append(item)
        return pairs

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df = super().populate_entry_trend(df, metadata)
        if self.crash_drop_pct <= 0 or not self.dp:
            return df

        judged = self.crash_btc_pair if self.crash_scope == "btc" else metadata["pair"]
        src = self.dp.get_pair_dataframe(pair=judged, timeframe="1h")
        if src is None or src.empty:
            log.warning("[Crash] %s no 1h data for %s, gate off (fail-open)", metadata.get("pair"), judged)
            return df

        hi = src["high"].rolling(self.crash_high_hours, min_periods=24).max()
        lo = src["low"].rolling(self.crash_low_hours, min_periods=24).min()
        info = DataFrame(
            {
                "date": src["date"],
                "crash_drop": (1.0 - src["close"] / hi) * 100.0,
                "crash_above_low": (src["close"] / lo - 1.0) * 100.0,
            }
        )
        df = merge_informative_pair(df, info, self.timeframe, "1h", ffill=True)

        regime = (df["crash_drop_1h"] >= self.crash_drop_pct) & (
            df["crash_above_low_1h"] <= self.crash_near_low_pct
        )
        signals = int((df["enter_long"] == 1).sum())
        blocked = int(((df["enter_long"] == 1) & regime).sum())
        df.loc[regime, "enter_long"] = 0
        log.info("[Crash] %s signals=%d blocked=%d", metadata.get("pair"), signals, blocked)
        return df
