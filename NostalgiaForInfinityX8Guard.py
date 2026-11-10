# NostalgiaForInfinityX8Guard  (v2 - "cut rule")
#
# A thin, SEPARATE subclass of NostalgiaForInfinityX8. It never edits or copies the upstream
# file, so the daily upstream sync cannot overwrite it and X8 itself is untouched.
#
# v1 of this class raised the abandon threshold (stale_range 15) and capped stake. v2 replaces
# that with the "cut rule" tested in the X8Cut backtests: close the WHOLE position of a deep
# grinder, then lock the pair for a few days so the strategy does not buy the same falling coin
# again at once. Every other X8 behaviour (entries, grinds, derisk, normal exits) is unchanged,
# and stale_range is back to the X8 default.
#
# Rules (all OFF unless the config turns them on, see "guard_parameters" in the bot config):
#   cut_entries      close the trade when it has this many FILLED entry orders (first entry = 1).
#   cut_age_hours    close the trade when it is at least this old (hours since first entry) AND
#                    the price is cut_adverse_pct or more against the FIRST entry price.
#   cut_adverse_pct  adverse move for the age trigger, percent (default 12).
#   cut_lock_days    after a cut, block new entries on that pair for this many days (default 3).
#   cut_apply_to     "both" (default) or "long". "long" = shorts are never cut.
#   cut_start_date   UTC time like "2026-11-09 00:00". Trades OPENED BEFORE it are never cut and
#                    keep plain X8 behaviour. This keeps trades inherited from before the switch
#                    identical on both bots, so the comparison with the X8 bot is fair.
# A wrong value in cut_apply_to or cut_start_date stops the bot at start-up (on purpose).
#
# Exit reasons written on a cut trade: cut_entries_<N>  or  cut_age_<H>h_<D>pct
# Logs: "[guard] cut rule ..." at start-up and "[Cut] PAIR reason=... entries=N age_h=H adverse_pct=D".
#
# Same logic as NostalgiaForInfinityX8Cut.py (used for backtests); this file is self-contained so
# the bot workflow needs no change.
import logging
from datetime import datetime, timedelta, timezone

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)


class NostalgiaForInfinityX8Guard(NostalgiaForInfinityX8):
  cut_entries = 0
  cut_age_hours = 0.0
  cut_adverse_pct = 12.0
  cut_lock_days = 3.0

  def __init__(self, config: dict) -> None:
    super().__init__(config)
    # Validate once at start-up so a typo in the config stops the bot immediately.
    apply_to = self._cut_apply_to()
    start = self._cut_start()
    logger.info(
      "[guard] cut rule: entries=%s age_h=%s adverse_pct=%s lock_days=%s apply_to=%s start=%s",
      int(self._cut_param("cut_entries", self.cut_entries)),
      self._cut_param("cut_age_hours", self.cut_age_hours),
      self._cut_param("cut_adverse_pct", self.cut_adverse_pct),
      self._cut_param("cut_lock_days", self.cut_lock_days),
      apply_to,
      start.isoformat() if start else "none (all trades)",
    )

  def version(self) -> str:
    return super().version() + "-guard"

  def _cut_param(self, name, default):
    gp = self.config.get("guard_parameters") or {}
    try:
      return float(gp.get(name, default))
    except (TypeError, ValueError):
      return float(default)

  def _cut_apply_to(self):
    gp = self.config.get("guard_parameters") or {}
    value = str(gp.get("cut_apply_to", "both")).strip().lower()
    if value not in ("both", "long"):
      raise ValueError("cut_apply_to must be 'both' or 'long', got %r" % (value,))
    return value

  def _cut_start(self):
    gp = self.config.get("guard_parameters") or {}
    raw = gp.get("cut_start_date")
    if not raw:
      return None
    try:
      d = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
      raise ValueError("cut_start_date must look like '2026-11-09 00:00' (UTC), got %r" % (raw,))
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)

  def _entry_count(self, trade):
    # Number of FILLED entry orders (the first entry counts as 1). Uses the order list and falls
    # back to nr_of_successful_entries (the attribute X8 itself uses). It never defaults to 1
    # silently: if neither can be read it raises, so the problem shows up in the log.
    try:
      return len(trade.select_filled_orders(trade.entry_side))
    except Exception:
      return int(trade.nr_of_successful_entries)

  def _first_entry_price(self, trade):
    try:
      orders = trade.select_filled_orders(trade.entry_side)
    except Exception:
      return None
    if not orders:
      return None
    first = orders[0]
    price = getattr(first, "safe_price", None) or getattr(first, "price", None)
    return float(price) if price else None

  def _cut_reason(self, trade, current_time, current_rate):
    # Returns (reason, info) when the trade should be cut now, else (None, None).
    max_entries = int(self._cut_param("cut_entries", self.cut_entries))
    age_limit = self._cut_param("cut_age_hours", self.cut_age_hours)
    adverse_limit = self._cut_param("cut_adverse_pct", self.cut_adverse_pct)
    if max_entries <= 0 and age_limit <= 0:
      return None, None
    apply_to = self._cut_apply_to()
    if trade.is_short and apply_to == "long":
      return None, None
    start = self._cut_start()
    if start is not None and trade.open_date_utc < start:
      return None, None

    n_entries = self._entry_count(trade)
    age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
    p0 = self._first_entry_price(trade)
    adverse = None
    if p0 and current_rate:
      adverse = (current_rate - p0) / p0 * 100.0 if trade.is_short else (p0 - current_rate) / p0 * 100.0

    if max_entries > 0 and n_entries >= max_entries:
      return f"cut_entries_{max_entries}", (n_entries, age_h, adverse)
    if age_limit > 0 and adverse is not None and age_h >= age_limit and adverse >= adverse_limit:
      return f"cut_age_{age_limit:g}h_{adverse_limit:g}pct", (n_entries, age_h, adverse)
    return None, None

  def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
    reason, info = self._cut_reason(trade, current_time, current_rate)
    if reason:
      n_entries, age_h, adverse = info
      logger.info(
        "[Cut] %s reason=%s entries=%d age_h=%.1f adverse_pct=%s",
        pair, reason, n_entries, age_h, "n/a" if adverse is None else f"{adverse:.1f}",
      )
      return reason
    return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)

  def confirm_trade_exit(
    self, pair, trade, order_type, amount, rate, time_in_force, exit_reason, current_time, **kwargs
  ) -> bool:
    # X8's own confirm_trade_exit can veto exits (hold logic, exit_profit_only); a cut must not be
    # vetoed, otherwise the trade would just keep grinding. Then lock the pair.
    if isinstance(exit_reason, str) and exit_reason.startswith("cut_"):
      lock_days = self._cut_param("cut_lock_days", self.cut_lock_days)
      if lock_days > 0:
        try:
          self.lock_pair(pair, current_time + timedelta(days=lock_days), reason="cut", side="long")
        except Exception as e:  # never break the bot because of the experiment
          logger.warning("[Cut] %s could not lock pair (%r)", pair, e)
      return True
    return super().confirm_trade_exit(
      pair, trade, order_type, amount, rate, time_in_force, exit_reason, current_time, **kwargs
    )
