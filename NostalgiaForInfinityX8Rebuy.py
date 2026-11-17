# NostalgiaForInfinityX8Rebuy
#
# A thin, SEPARATE subclass of NostalgiaForInfinityX8 (pinned v18.0.155). It never edits or copies
# the upstream file. Its only change: in the two SHORT rebuy functions
#   short_rebuy_adjust_trade_position_v3
#   short_rebuy_adjust_trade_position_v4
# the filter  close < EMA_26 * 1.012  is replaced by  close > EMA_26 * 1.012.
#
# Why: every other short condition of this kind (short_rebuy_adjust_trade_position at line 61353,
# and the three short_rebuy_entry_* functions) uses ">", and the long mirror is close < EMA_26 * 0.988.
# With "<" the short filter is true almost always, so it does nothing. Likely a copy-paste slip upstream.
#
# How: at import time the parent's source of each function is read, the ONE expected line is swapped,
# and the function is recompiled inside the parent's module. If the parent changes and the line is
# no longer found exactly once, the import fails on purpose (so the test can never run unpatched).
#
# Use for BACKTESTS first, with X8 (pinned) as the baseline. Do not touch X8 or X8Guard.
import inspect
import sys
import textwrap

from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

_BAD = 'last_candle["close"] < (last_candle["EMA_26"] * 1.012)'
_GOOD = 'last_candle["close"] > (last_candle["EMA_26"] * 1.012)'


def _patched(name):
  func = getattr(NostalgiaForInfinityX8, name)
  src = textwrap.dedent(inspect.getsource(func))
  found = src.count(_BAD)
  if found != 1:
    raise RuntimeError("[rebuy-fix] %s: expected the line exactly once, found %d" % (name, found))
  module = sys.modules[func.__module__]
  ns = {}
  exec(compile(src.replace(_BAD, _GOOD), "<%s patched>" % name, "exec"), module.__dict__, ns)
  return ns[name]


class NostalgiaForInfinityX8Rebuy(NostalgiaForInfinityX8):
  short_rebuy_adjust_trade_position_v3 = _patched("short_rebuy_adjust_trade_position_v3")
  short_rebuy_adjust_trade_position_v4 = _patched("short_rebuy_adjust_trade_position_v4")

  def version(self) -> str:
    return super().version() + "-rebuyfix"
