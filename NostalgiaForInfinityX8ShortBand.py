"""
NostalgiaForInfinityX8ShortBand - EXPERIMENT (v2). Thin subclass of X8, shorts only.

Idea: do not open a SHORT when the price is already stretched toward the LOWER edge of a
volatility band (the move may already be over / a squeeze may come). Longs are untouched.

v2: the band can now be built from 1h, 4h or 1d candles (v1 used only 1d, which did not match
5m entries). Per pair, no look-ahead:
  * candle log returns -> EWMA variance (RiskMetrics style) -> sigma per candle
  * band for a horizon of h candles: prev_close * exp(-z * sigma * sqrt(h)) .. exp(+z * sigma * sqrt(h))
  * the band of a candle is usable only after that candle has closed
  * position in band = (price - lower) / (upper - lower)   (0 = lower edge, 1 = upper edge)
A short entry signal is cancelled when position < band_near. If there is not enough candle
data, the filter does nothing (fail-open) and the signal stays.

Parameters (all optional) come from the backtest input "guard_parameters", e.g.
  {"band_timeframe": "1h", "band_horizon": 3, "band_near": 0.2}
  band_timeframe     "1h" | "4h" | "1d"                           (default "1h")
  band_horizon       horizon h in candles of that timeframe        (default 3; old name band_horizon_days also works)
  band_z             1.645 = ~90% band, 1.28 = ~80%                (default 1.645)
  band_near          block shorts in the lowest X of the band      (default 0.2; 0 = only below the band)
  band_ewma_lambda   EWMA decay                                    (default 0.94)
  band_min_bars      candles needed before it acts                 (default 100 for 1h/4h, 30 for 1d)

Per pair it logs:  [ShortBand] PAIR signals=N blocked=M   (candles, not trades).
Only for backtests / research. Do not run it live without testing.
"""
import logging

import numpy as np
import pandas as pd

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)

TF_DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


def band_position(dates, close, bars, tf="1h", z=1.645, h_bars=3, lam=0.94, min_bars=100):
  """Position of each 5m close inside the volatility band built from `bars` (candles of timeframe tf).
  Returns a Series aligned with `close`, or None if there is not enough candle data."""
  if bars is None or len(bars) == 0 or tf not in TF_DELTA:
    return None
  d = bars[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
  if len(d) < min_bars + 2:
    return None
  r = np.log(d["close"] / d["close"].shift(1))
  var = (r * r).ewm(alpha=1.0 - lam, adjust=False, min_periods=min_bars).mean()
  sd = np.sqrt(var) * np.sqrt(h_bars)
  d["lo"] = d["close"] * np.exp(-z * sd)
  d["hi"] = d["close"] * np.exp(z * sd)
  # candle 'date' is its OPEN time; the band is usable once that candle has closed
  d["avail"] = pd.to_datetime(d["date"], utc=True) + TF_DELTA[tf]
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
  band_timeframe = "1h"
  band_horizon = 3
  band_z = 1.645
  band_near = 0.2
  band_ewma_lambda = 0.94

  def _gp(self):
    return self.config.get("guard_parameters") or {}

  def _num(self, name, default, alias=None):
    gp = self._gp()
    val = gp.get(name, gp.get(alias, default) if alias else default)
    try:
      return float(val)
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
    tf = str(self._gp().get("band_timeframe", self.band_timeframe))
    try:
      bars = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
      default_min = 30 if tf == "1d" else 100
      pos = band_position(
        df["date"],
        df["close"],
        bars,
        tf=tf,
        z=self._num("band_z", self.band_z),
        h_bars=self._num("band_horizon", self.band_horizon, alias="band_horizon_days"),
        lam=self._num("band_ewma_lambda", self.band_ewma_lambda),
        min_bars=int(self._num("band_min_bars", default_min)),
      )
    except Exception as e:  # never break the run because of the experiment
      logger.warning("[ShortBand] %s filter skipped (%r)", pair, e)
      pos = None
    if pos is None:
      logger.info("[ShortBand] %s signals=%d blocked=0 (no %s data, fail-open)", pair, n_sig, tf)
      return df
    near = self._num("band_near", self.band_near)
    block = sig & (pos < near).fillna(False)
    n_block = int(block.sum())
    if n_block:
      df.loc[block, "enter_short"] = 0
    logger.info("[ShortBand] %s signals=%d blocked=%d", pair, n_sig, n_block)
    return df
