"""
NostalgiaForInfinityX8Guard
===========================

A thin, SEPARATE subclass of NostalgiaForInfinityX8. It never edits or copies
the upstream file, so the daily upstream sync cannot overwrite it and X8
itself (and the X8 bots) are untouched. It only:

  1. Overrides a few NFI class attributes (values only, no logic copied):
       bad_trade_controller_stale_max_daily_range_pct = 15.0   (upstream 7.0)
     -> the "abandon" exit can fire on volatile alts too (AERO/BEAT case).

  2. Wraps adjust_trade_position() to cap how far a position can be grinded:
       guard_max_stake_multiple  total open stake <= N x first-entry stake
       guard_max_entries         max number of successful entries (0 = off)
     Only NEW ENTRIES (positive stake) are blocked. Partial exits / derisk /
     profit-taking returned by NFI always pass through unchanged.

Any attribute (NFI's or this class's) can be changed WITHOUT editing files via
an optional "guard_parameters" block in the config, e.g.:

    "guard_parameters": {
        "guard_max_stake_multiple": 4.0,
        "bad_trade_controller_derisk_hold_short_enable": true
    }

Unknown keys are ignored with a warning in the log.

Usage: strategy name = NostalgiaForInfinityX8Guard. Needs NostalgiaForInfinityX8.py
in the same strategies folder.
"""

import logging
from typing import Optional

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

log = logging.getLogger(__name__)


class NostalgiaForInfinityX8Guard(NostalgiaForInfinityX8):
    # ---- NFI attribute overrides (values only) ----
    bad_trade_controller_stale_max_daily_range_pct = 15.0

    # ---- Guard parameters ----
    guard_max_stake_multiple = 5.0  # 0 / None = off
    guard_max_entries = 0  # 0 / None = off

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._guard_logged = set()
        overrides = config.get("guard_parameters") or {}
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
                log.info("[guard] parameter %s set to %r", key, value)
            else:
                log.warning("[guard] unknown parameter ignored: %s", key)
        log.info(
            "[guard] active: stale_range=%s max_stake_multiple=%s max_entries=%s",
            self.bad_trade_controller_stale_max_daily_range_pct,
            self.guard_max_stake_multiple,
            self.guard_max_entries,
        )

    def version(self) -> str:
        return super().version() + "-guard"

    @staticmethod
    def _guard_initial_stake(trade) -> Optional[float]:
        """Margin (stake) of the first filled entry order, or None if unknown."""
        try:
            orders = trade.select_filled_orders(trade.entry_side)
            if not orders:
                return None
            first = orders[0]
            cost = first.safe_cost
            if not cost:
                cost = first.safe_filled * first.safe_price
            return float(cost) / float(trade.leverage or 1.0)
        except Exception:
            return None

    def adjust_trade_position(
        self,
        trade,
        current_time,
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
        if result is None:
            return None

        stake = result[0] if isinstance(result, tuple) else result
        # Partial exits / derisk (negative) always pass through.
        if stake is None or stake <= 0:
            return result

        reason = None
        if self.guard_max_entries and trade.nr_of_successful_entries >= self.guard_max_entries:
            reason = f"entries {trade.nr_of_successful_entries} >= {self.guard_max_entries}"
        elif self.guard_max_stake_multiple:
            base = self._guard_initial_stake(trade)
            if base and (trade.stake_amount + stake) > base * self.guard_max_stake_multiple:
                reason = (
                    f"stake {trade.stake_amount + stake:.1f} > "
                    f"{self.guard_max_stake_multiple}x first entry ({base:.1f})"
                )

        if reason is None:
            return result

        if trade.id not in self._guard_logged:
            self._guard_logged.add(trade.id)
            log.info("[guard] %s: further entries blocked (%s)", trade.pair, reason)
        return None
