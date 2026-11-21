# NostalgiaForInfinityX8ShortOn (EXPERIMENT, research only)
#
# X8 + the option to switch ON more short entry conditions (and OFF some that are
# already on). X8 v18.0.155 runs only 4 of its 31 short conditions
# (501, 502, 542, 562); the rest are False or commented out in
# short_entry_signal_params. With guard_parameters empty this class behaves
# exactly like plain X8.
#
# guard_parameters (JSON object), all keys optional:
#   short_families : list of family names to switch on, from
#                    normal, quick, rebuy, high_profit, rapid, top_coins, scalp
#   short_enable   : list of single condition numbers to switch on, e.g. [543, 544]
#   short_disable  : list of condition numbers to switch off, e.g. [502]
# Example: {"short_families": ["quick"]}
#
# Condition numbers that do not exist in the strategy are rejected with an error.

import logging

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

log = logging.getLogger(__name__)


class NostalgiaForInfinityX8ShortOn(NostalgiaForInfinityX8):
    SHORT_FAMILIES = {
        "normal": [503, 504, 505, 506],
        "quick": [541, 543, 544, 545, 546, 548],
        "rebuy": [561, 563],
        "high_profit": [592, 593],
        "rapid": [605],
        "top_coins": [641, 642],
        "scalp": [661, 662, 663, 664, 665, 666, 667, 669, 670, 671],
    }
    # every short condition that has a code block in X8 v18.0.155
    SHORT_VALID = {501, 502, 542, 562} | {n for v in SHORT_FAMILIES.values() for n in v}

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        gp = config.get("guard_parameters", {}) or {}

        enable = []
        for fam in gp.get("short_families", []) or []:
            if fam not in self.SHORT_FAMILIES:
                raise ValueError("unknown short family %r, allowed: %s" % (fam, sorted(self.SHORT_FAMILIES)))
            enable += self.SHORT_FAMILIES[fam]
        for n in gp.get("short_enable", []) or []:
            enable.append(int(n))
        disable = [int(n) for n in (gp.get("short_disable", []) or [])]

        for n in enable + disable:
            if n not in self.SHORT_VALID:
                raise ValueError("short condition %s does not exist in this strategy" % n)

        params = dict(self.short_entry_signal_params)
        for n in enable:
            params["short_entry_condition_%d_enable" % n] = True
        for n in disable:
            params["short_entry_condition_%d_enable" % n] = False
        self.short_entry_signal_params = params

        on = sorted(
            int(k.rsplit("_", 2)[1]) for k, v in params.items() if v and k.startswith("short_entry_condition_")
        )
        log.info("[ShortOn] short conditions enabled (%d): %s", len(on), on)
