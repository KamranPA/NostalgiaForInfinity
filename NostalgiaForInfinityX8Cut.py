"""
NostalgiaForInfinityX8Cut - EXPERIMENT. Thin subclass of X8 (long and short are both handled).
"""
NostalgiaForInfinityX8Cut - EXPERIMENT. Thin subclass of X8 (long and short are both handled).

Problem it targets: a handful of trades keep rebuying (grind) while the coin keeps falling, stay
open for days and make almost all of the losses. In our backtests a trade that reached 5 or more
entries lost money in 8 of 11 cases, and the few that ended in profit made only a few USDT.

Idea: instead of waiting for the strategy's own derisk / abandon logic, CLOSE THE WHOLE POSITION
when a trade looks like such a grinder, then LOCK THE PAIR for a few days so the strategy does not
buy the same falling coin again at once. Everything else of X8 is untouched; the normal X8 exits
keep working for every trade that does not trigger a cut.

Two independent triggers, both OFF by default (all zero = this class behaves exactly like X8).
Parameters come from the backtest input "guard_parameters", e.g.
  {"cut_entries": 3}
  {"cut_age_hours": 72, "cut_adverse_pct": 12}
  {"cut_entries": 5, "cut_age_hours": 72, "cut_adverse_pct": 12, "cut_lock_days": 3}

  cut_entries      close the trade when it has this many ENTRY orders filled (the first entry
                   counts as 1). 0 = off.
  cut_age_hours    close the trade when it is at least this old (hours since the first entry)
                   AND the price is cut_adverse_pct or more against the FIRST entry price
                   (below it for a long, above it for a short). 0 = off.
  cut_adverse_pct  adverse move for the age trigger, in percent (default 12).
  cut_lock_days    after a cut, block new entries on that pair for this many days (default 3).

Exit reasons written to the trade: cut_entries_<N>  and  cut_age_<H>h_<D>pct
Per cut it logs:  [Cut] PAIR reason=... entries=N age_h=H adverse_pct=D

v2 (2026-11-07): fixed cut_entries, which never fired in v1 (it read a freqtrade attribute that does not
exist, nr_of_entries, and silently defaulted to 1). The entry count now comes from the filled orders.

Only for backtests / research. Do not run it live without testing.
"""
import logging
from datetime import timedelta

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)


class NostalgiaForInfinityX8Cut(NostalgiaForInfinityX8):
  cut_entries = 0
  cut_age_hours = 0.0
  cut_adverse_pct = 12.0
  cut_lock_days = 3.0

  def _cut_param(self, name, default):
    gp = self.config.get("guard_parameters") or {}
    try:
      return float(gp.get(name, default))
    except (TypeError, ValueError):
      return float(default)

  def _entry_count(self, trade):
    """Number of FILLED entry orders (the first entry counts as 1). Uses the order list; falls back to
    freqtrade's nr_of_successful_entries (the attribute X8 itself uses). Never defaults to 1 silently:
    if neither can be read it raises, so the problem is visible in the log instead of the rule being off."""
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
    """Returns (reason, info) when the trade should be cut now, else (None, None)."""
    max_entries = int(self._cut_param("cut_entries", self.cut_entries))
    age_limit = self._cut_param("cut_age_hours", self.cut_age_hours)
    adverse_limit = self._cut_param("cut_adverse_pct", self.cut_adverse_pct)
    if max_entries <= 0 and age_limit <= 0:
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
          self.lock_pair(pair, current_time + timedelta(days=lock_days), reason="cut")
        except Exception as e:  # never break the run because of the experiment
          logger.warning("[Cut] %s could not lock pair (%r)", pair, e)
      return True
    return super().confirm_trade_exit(
      pair, trade, order_type, amount, rate, time_in_force, exit_reason, current_time, **kwargs
    )
Problem it targets: a handful of trades keep rebuying (grind) while the coin keeps falling, stay
open for days and make almost all of the losses. In our backtests a trade that reached 5 or more
entries lost money in 8 of 11 cases, and the few that ended in profit made only a few USDT.

Idea: instead of waiting for the strategy's own derisk / abandon logic, CLOSE THE WHOLE POSITION
when a trade looks like such a grinder, then LOCK THE PAIR for a few days so the strategy does not
buy the same falling coin again at once. Everything else of X8 is untouched; the normal X8 exits
keep working for every trade that does not trigger a cut.

Two independent triggers, both OFF by default (all zero = this class behaves exactly like X8).
Parameters come from the backtest input "guard_parameters", e.g.
  {"cut_entries": 3}
  {"cut_age_hours": 72, "cut_adverse_pct": 12}
  {"cut_entries": 5, "cut_age_hours": 72, "cut_adverse_pct": 12, "cut_lock_days": 3}

  cut_entries      close the trade when it has this many ENTRY orders filled (the first entry
                   counts as 1). 0 = off.
  cut_age_hours    close the trade when it is at least this old (hours since the first entry)
                   AND the price is cut_adverse_pct or more against the FIRST entry price
                   (below it for a long, above it for a short). 0 = off.
  cut_adverse_pct  adverse move for the age trigger, in percent (default 12).
  cut_lock_days    after a cut, block new entries on that pair for this many days (default 3).

Exit reasons written to the trade: cut_entries_<N>  and  cut_age_<H>h_<D>pct
Per cut it logs:  [Cut] PAIR reason=... entries=N age_h=H adverse_pct=D

Only for backtests / research. Do not run it live without testing.
"""
import logging
from datetime import timedelta

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

logger = logging.getLogger(__name__)


class NostalgiaForInfinityX8Cut(NostalgiaForInfinityX8):
  cut_entries = 0
  cut_age_hours = 0.0
  cut_adverse_pct = 12.0
  cut_lock_days = 3.0

  def _cut_param(self, name, default):
    gp = self.config.get("guard_parameters") or {}
    try:
      return float(gp.get(name, default))
    except (TypeError, ValueError):
      return float(default)

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
    """Returns (reason, info) when the trade should be cut now, else (None, None)."""
    max_entries = int(self._cut_param("cut_entries", self.cut_entries))
    age_limit = self._cut_param("cut_age_hours", self.cut_age_hours)
    adverse_limit = self._cut_param("cut_adverse_pct", self.cut_adverse_pct)
    if max_entries <= 0 and age_limit <= 0:
      return None, None

    n_entries = int(getattr(trade, "nr_of_entries", 1) or 1)
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
          self.lock_pair(pair, current_time + timedelta(days=lock_days), reason="cut")
        except Exception as e:  # never break the run because of the experiment
          logger.warning("[Cut] %s could not lock pair (%r)", pair, e)
      return True
    return super().confirm_trade_exit(
      pair, trade, order_type, amount, rate, time_in_force, exit_reason, current_time, **kwargs
    )
