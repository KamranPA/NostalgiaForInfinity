# NostalgiaForInfinityX8BtcGate  (EXPERIMENT, research only)
#
# A thin, SEPARATE subclass of NostalgiaForInfinityX8 (pinned v18.0.155). It never edits or copies
# the upstream file. It adds ONE rule: a market-regime gate on NEW LONG entries.
#
# A new long entry is refused when EITHER condition holds (BTC/USDT:USDT, closed 4h candles):
#   slow : BTC close is below its EMA of ema_days days  (default 50 days; 300 candles of 4h)
#   fast : BTC fell by drop_pct percent or more over the last drop_days days (default 8 pct, 3 days)
# Shorts are never gated. Forced entries are never gated. Rebuys, grinds and every other
# position adjustment of an OPEN trade are never gated (they do not go through the first-entry check,
# and the code also skips any pair that already has an open trade).
#
# Why 4h and not 1d: X8 already loads BTC 4h (btc_info_timeframes = ["4h"]), so no extra data is
# needed and no X8 setting changes. EMA(300) on 4h candles is the same span as EMA(50) on 1d candles.
# Only candles that are already CLOSED at the entry time are used (no look-ahead).
#
# Thresholds are fixed IN ADVANCE on purpose (50 days, 3 days, 8 pct). Do not tune them on the same
# backtest windows you judge them on; that is the over-fitting seen before with the entry cap and stale.
#
# Optional override through "guard_parameters" in the config (workflow input guard_parameters):
#   btc_gate_ema_days (default 50), btc_gate_drop_days (default 3), btc_gate_drop_pct (default 8).
#   Set btc_gate_ema_days to 0 to switch the slow rule off, btc_gate_drop_pct to 0 for the fast rule.
#   With both off the class behaves exactly like X8.
#
# Log lines: "[BtcGate] rules ..." at start-up and "[BtcGate] PAIR blocked reason=..." per refusal.
# If BTC data is missing or too short, the gate FAILS OPEN (entry allowed) and logs it once.
import logging

import numpy as np
import pandas as pd

from freqtrade.persistence import Trade
from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)

CANDLES_PER_DAY = 6  # 4h candles


def btc_gate_state(dates, closes, ema, now, ema_candles, drop_candles, drop_pct):
  # dates: numpy datetime64[ns] array (UTC), closes/ema: float arrays of equal length, now: pandas
  # Timestamp (UTC). Returns (reasons, info) where reasons is a list of "slow"/"fast" and info is a dict.
  # Uses the last candle that was already closed at "now" (open time + 4h <= now).
  limit = np.datetime64(now.tz_convert("UTC").tz_localize(None) - pd.Timedelta(hours=4))
  idx = int(np.searchsorted(dates, limit, side="right")) - 1
  if idx < 0:
    return None, None
  reasons = []
  info = {"close": float(closes[idx]), "ema": None, "drop_pct": None}
  if ema_candles > 0 and idx >= ema_candles and not np.isnan(ema[idx]):
    info["ema"] = float(ema[idx])
    if closes[idx] < ema[idx]:
      reasons.append("slow")
  if drop_pct > 0 and drop_candles > 0 and idx - drop_candles >= 0:
    past = closes[idx - drop_candles]
    if past > 0:
      change = (closes[idx] / past - 1.0) * 100.0
      info["drop_pct"] = float(change)
      if change <= -drop_pct:
        reasons.append("fast")
  return reasons, info


class NostalgiaForInfinityX8BtcGate(NostalgiaForInfinityX8):
  btc_gate_ema_days = 50.0
  btc_gate_drop_days = 3.0
  btc_gate_drop_pct = 8.0

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self._btc_gate_cache = None
    self._btc_gate_warned = False
    logger.info(
      "[BtcGate] rules: slow=close<EMA(%s days) fast=drop>=%s pct in %s days (4h candles, longs only)",
      self._gate_param("btc_gate_ema_days", self.btc_gate_ema_days),
      self._gate_param("btc_gate_drop_pct", self.btc_gate_drop_pct),
      self._gate_param("btc_gate_drop_days", self.btc_gate_drop_days),
    )

  def version(self) -> str:
    return super().version() + "-btcgate"

  def _gate_param(self, name, default):
    gp = self.config.get("guard_parameters") or {}
    try:
      return float(gp.get(name, default))
    except (TypeError, ValueError):
      return float(default)

  def _btc_arrays(self):
    df = self.dp.get_pair_dataframe(pair=self.btc_informative_pair(), timeframe="4h")
    if df is None or len(df) == 0:
      return None
    key = (len(df), df["date"].iloc[-1])
    if self._btc_gate_cache is not None and self._btc_gate_cache[0] == key:
      return self._btc_gate_cache[1]
    ema_candles = int(round(self._gate_param("btc_gate_ema_days", self.btc_gate_ema_days) * CANDLES_PER_DAY))
    dates = df["date"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    closes = df["close"].to_numpy(dtype=float)
    if ema_candles > 0:
      ema = pd.Series(closes).ewm(span=ema_candles, adjust=False).mean().to_numpy()
    else:
      ema = np.full(len(closes), np.nan)
    arrays = (dates, closes, ema)
    self._btc_gate_cache = (key, arrays)
    return arrays

  def confirm_trade_entry(
    self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs
  ) -> bool:
    if side == "long" and entry_tag != "force_entry" and not Trade.get_trades_proxy(pair=pair, is_open=True):
      try:
        arrays = self._btc_arrays()
        if arrays is None:
          if not self._btc_gate_warned:
            logger.warning("[BtcGate] no BTC 4h data, gate is off (fail-open)")
            self._btc_gate_warned = True
        else:
          ema_candles = int(round(self._gate_param("btc_gate_ema_days", self.btc_gate_ema_days) * CANDLES_PER_DAY))
          drop_candles = int(round(self._gate_param("btc_gate_drop_days", self.btc_gate_drop_days) * CANDLES_PER_DAY))
          drop_pct = self._gate_param("btc_gate_drop_pct", self.btc_gate_drop_pct)
          dates, closes, ema = arrays
          reasons, info = btc_gate_state(
            dates, closes, ema, pd.Timestamp(current_time), ema_candles, drop_candles, drop_pct
          )
          if reasons:
            logger.info(
              "[BtcGate] %s blocked reason=%s btc_close=%.1f ema=%s change_pct=%s",
              pair,
              "+".join(reasons),
              info["close"],
              "n/a" if info["ema"] is None else "%.1f" % info["ema"],
              "n/a" if info["drop_pct"] is None else "%.1f" % info["drop_pct"],
            )
            return False
      except Exception as e:  # never break the bot because of the experiment
        if not self._btc_gate_warned:
          logger.warning("[BtcGate] gate error, entry allowed (%r)", e)
          self._btc_gate_warned = True
    return super().confirm_trade_entry(
      pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs
    )
