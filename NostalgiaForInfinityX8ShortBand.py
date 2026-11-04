"""
NostalgiaForInfinityX8ShortBand - EXPERIMENT. Thin subclass of X8, shorts only.

Idea: do not open a SHORT when the price is already stretched toward the LOWER edge of a
volatility band (the move may already be over / a squeeze may come). Longs are untouched.

The band (per pair, from DAILY candles, no look-ahead):
  * daily log returns -> EWMA variance (RiskMetrics style) -> sigma for the next day
  * band for a horizon of h days: close_prev_day * exp(-z * sigma * sqrt(h))  ..  exp(+z * sigma * sqrt(h))
  * the band of day D-1 becomes usable only from the start of day D
  * position in band = (price - lower) / (upper - lower)   (0 = lower edge, 1 = upper edge)
A short entry signal is cancelled when position < band_near. If there is not enough
daily data, the filter does nothing (fail-open) and the signal stays.

Parameters (all optional) come from the backtest input "guard_parameters", e.g.
  {"band_z": 1.645, "band_horizon_days": 1, "band_near": 0.2}
  band_z              1.645 = ~90% band, 1.28 = ~80%          (default 1.645)
  band_horizon_days   horizon h in days                       (default 1)
  band_near           block shorts in the lowest X of the band (default 0.2; 0 = only below the band)
  band_ewma_lambda    EWMA decay                              (default 0.94)
  band_min_days       daily candles needed before it acts     (default 30)

Per pair it logs:  [ShortBand] PAIR signals=N blocked=M   (candles, not trades).
Only for backtests / research. Do not run it live without testing.
"""
import logging

import numpy as np
import pandas as pd

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)


def band_position(dates, close, daily, z=1.645, h_days=1, lam=0.94, min_days=30):
  """Position of each 5m close inside the daily volatility band (Series aligned with `close`).
  Returns None if there is not enough daily data."""
  if daily is None or len(daily) == 0:
    return None
  d = daily[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
  if len(d) < min_days + 2:
    return None
  r = np.log(d["close"] / d["close"].shift(1))
  var = (r * r).ewm(alpha=1.0 - lam, adjust=False, min_periods=min_days).mean()
  sd = np.sqrt(var) * np.sqrt(h_days)
  d["lo"] = d["close"] * np.exp(-z * sd)
  d["hi"] = d["close"] * np.exp(z * sd)
  d["avail"] = pd.to_datetime(d["date"], utc=True) + pd.Timedelta(days=1)  # usable after that day closed
  d = d.dropna(subset=["lo", "hi"])[["avail", "lo", "hi"]]
  d["avail"] = d["avail"].astype("datetime64[ns, UTC]")
  left = pd.DataFrame(
    {"date": pd.to_datetime(pd.Series(dates).values, utc=True).astype("datetime64[ns, UTC]"), "c": np.asarray(close, dtype=float)}
  )
  merged = pd.merge_asof(left, d, left_on="date", right_on="avail", direction="backward")
  width = (merged["hi"] - merged["lo"]).replace(0, np.nan)
  pos = (merged["c"] - merged["lo"]) / width
  return pd.Series(pos.values, index=close.index)


class NostalgiaForInfinityX8ShortBand(NostalgiaForInfinityX8):
  band_z = 1.645
  band_horizon_days = 1
  band_near = 0.2
  band_ewma_lambda = 0.94
  band_min_days = 30

  def _band_param(self, name, default):
    gp = self.config.get("guard_parameters") or {}
    try:
      return float(gp.get(name, default))
    except (TypeError, ValueError):
      return default

  def populate_entry_trend(self, df, metadata):
    df = super().populate_entry_trend(df, metadata)
    if "enter_short" not in df.columns:
      return df
    sig = df["enter_short"] == 1
    n_sig = int(sig.sum())
    pair = metadata.get("pair", "?")
    if n_sig == 0:
      logger.info("[ShortBand] %s signals=0 blocked=0", pair)
      return df
    try:
      daily = self.dp.get_pair_dataframe(pair=pair, timeframe="1d")
      pos = band_position(
        df["date"],
        df["close"],
        daily,
        z=self._band_param("band_z", self.band_z),
        h_days=self._band_param("band_horizon_days", self.band_horizon_days),
        lam=self._band_param("band_ewma_lambda", self.band_ewma_lambda),
        min_days=int(self._band_param("band_min_days", self.band_min_days)),
      )
    except Exception as e:  # never break the run because of the experiment
      logger.warning("[ShortBand] %s filter skipped (%r)", pair, e)
      pos = None
    if pos is None:
      logger.info("[ShortBand] %s signals=%d blocked=0 (no daily data, fail-open)", pair, n_sig)
      return df
    near = self._band_param("band_near", self.band_near)
    block = sig & (pos < near).fillna(False)
    n_block = int(block.sum())
    if n_block:
      df.loc[block, "enter_short"] = 0
    logger.info("[ShortBand] %s signals=%d blocked=%d", pair, n_sig, n_block)
    return df
