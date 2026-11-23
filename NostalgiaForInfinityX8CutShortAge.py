# NostalgiaForInfinityX8CutShortAge (EXPERIMENT, research only)
#
# X8Cut (the entry-count cut rule for longs) plus ONE extra rule for SHORT trades:
# a short that has been open for at least short_age_days and is losing at least
# short_age_loss (fraction of margin, e.g. 0.15 = 15 pct) is closed.
# Reason: in the 180 day backtest two shorts (AERO 41 days, PUMP 13 days) stayed open
# at a loss and made most of the remaining loss after the long cut rule.
#
# guard_parameters (JSON object), all optional. Every X8Cut key still works:
#   short_age_days : 0 = off (default, then exactly X8Cut). e.g. 14
#   short_age_loss : default 0.15
# Example: {"cut_entries": 3, "cut_lock_days": 14, "cut_apply_to": "long",
#           "short_age_days": 14, "short_age_loss": 0.15}
# Exit reason of these trades: short_age_<days>d

import logging

from NostalgiaForInfinityX8Cut import NostalgiaForInfinityX8Cut

log = logging.getLogger(__name__)


class NostalgiaForInfinityX8CutShortAge(NostalgiaForInfinityX8Cut):
    short_age_days = 0
    short_age_loss = 0.15

    def __init__(self, config: dict) -> None:
        gp = dict(config.get("guard_parameters", {}) or {})
        mine = {k: gp.pop(k) for k in list(gp) if k.startswith("short_age_")}
        # hide our keys from the parent class, then put them back
        config["guard_parameters"] = gp
        super().__init__(config)
        config["guard_parameters"] = {**gp, **mine}
        if "short_age_days" in mine:
            self.short_age_days = float(mine["short_age_days"])
        if "short_age_loss" in mine:
            self.short_age_loss = float(mine["short_age_loss"])
        log.info(
            "[ShortAge] short_age_days=%s short_age_loss=%s",
            self.short_age_days,
            self.short_age_loss,
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        reason = super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)
        if reason:
            return reason
        if self.short_age_days > 0 and trade.is_short:
            age_days = (current_time - trade.open_date_utc).total_seconds() / 86400.0
            if age_days >= self.short_age_days and current_profit <= -self.short_age_loss:
                log.info(
                    "[ShortAge] %s age_days=%.1f profit=%.3f",
                    pair,
                    age_days,
                    current_profit,
                )
                return "short_age_%dd" % int(self.short_age_days)
        return None
