"""
NostalgiaForInfinityX8Cap.py  (SAVE AT THE REPO ROOT, next to NostalgiaForInfinityX8.py)

X8 with ONE extra, separate field:  guard_max_entries

WHAT IT DOES
------------
Caps the TOTAL number of entry orders (the first entry plus every rebuy /
grind re-entry) a single trade may have. Once a trade has reached the cap,
every further position INCREASE that X8 asks for is dropped. Exits and
partial exits (de-risk, grind sells, normal exits, stop-loss) are NOT
touched, so the trade can still shrink or close the way X8 decides.

WHY
---
In the backtests (two 90-day windows, 280 trades) 95% of winning trades
used 2 entries or fewer, while the big losers kept adding: BEAT 123 entries,
BASED 92, SOXL 37, AERO 32, KAT 21. Only 4 trades with 4+ entries won, and
together they made about 21 USDT; the 7 with 4+ entries that lost made about
846 USDT of losses. This is hindsight on 12 losing trades, so it is a
hypothesis to test on an independent window, not a proven fix.

THE FIELD
---------
    guard_max_entries = 4     # 0 (or negative) = OFF, behaves exactly like X8

Change it without editing the file, either way:
  * config:   "guard_parameters": {"guard_max_entries": 3}
  * config:   "entry_cap_parameters": {"guard_max_entries": 3}
(the second block wins if both are present). Only known fields are applied.

NOT COVERED / KNOWN LIMITS
--------------------------
* Capped trades stay open: this stops them GROWING, it does not close them.
  Their loss can still be large if the coin keeps falling.
* "Entries" = filled entry orders of the trade (trade.nr_of_successful_entries),
  so a grind that sells part and buys back counts every buy-back.
* Written against freqtrade's adjust_trade_position contract: the return
  value is None, a number (positive = add, negative = partial exit) or a
  (number, "tag") tuple. Anything else is passed through unchanged.
"""

import logging
from datetime import datetime
from typing import Optional

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

log = logging.getLogger(__name__)


class NostalgiaForInfinityX8Cap(NostalgiaForInfinityX8):
  # ---- the separate field (0 = off) ------------------------------------
  guard_max_entries = 4

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    self._cap_logged = set()
    cfg = config or {}
    # same override style as the Guard: known fields only
    for block in ("guard_parameters", "entry_cap_parameters"):
      params = cfg.get(block) or {}
      if not isinstance(params, dict):
        continue
      if "guard_max_entries" in params:
        try:
          self.guard_max_entries = int(params["guard_max_entries"])
        except (TypeError, ValueError):
          log.warning("guard_max_entries in %s is not an integer, ignored: %r", block, params["guard_max_entries"])

  def version(self) -> str:
    return super().version() + "+cap"

  def adjust_trade_position(
    self,
    trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    min_stake: Optional[float],
    max_stake: float,
    current_entry_rate: float,
    current_exit_rate: float,
    current_entry_profit: float,
    current_exit_profit: float,
    **kwargs,
  ):
    result = super().adjust_trade_position(
      trade,
      current_time,
      current_rate,
      current_profit,
      min_stake,
      max_stake,
      current_entry_rate,
      current_exit_rate,
      current_entry_profit,
      current_exit_profit,
      **kwargs,
    )

    cap = self.guard_max_entries
    if cap is None or cap <= 0 or result is None:
      return result

    amount = result[0] if isinstance(result, tuple) else result
    try:
      is_add = amount is not None and float(amount) > 0
    except (TypeError, ValueError):
      return result
    if not is_add:
      return result  # exits / partial exits are never blocked

    if trade.nr_of_successful_entries >= cap:
      if trade.id not in self._cap_logged:
        self._cap_logged.add(trade.id)
        log.info(
          "[entry cap] %s trade %s reached %s entries (cap %s): further adds are dropped",
          trade.pair, trade.id, trade.nr_of_successful_entries, cap,
        )
      return None
    return result
