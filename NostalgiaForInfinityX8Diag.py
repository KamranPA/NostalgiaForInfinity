"""
NostalgiaForInfinityX8Diag - READ-ONLY DIAGNOSTIC subclass of X8. Save in the repo root.

It behaves exactly like X8 (same signals, same trades). The only addition: while X8
builds its long entry conditions, it records, for every long tag, the list of
sub-conditions, and counts per candle how many of them FAIL. From that it dumps
(one small JSON per pair, folder DIAG_OUT) these numbers for two windows:

  block window : DIAG_BLOCK_START .. DIAG_END      (the days with no entries)
  reference    : DIAG_REF_START   .. DIAG_BLOCK_START (days with entries)

  * nm  : "near misses" = candles where exactly ONE sub-condition of a tag failed,
          counted per (tag, sub-condition). The sub-condition that is a near miss
          far more often in the block window is the one that blocked the entries.
  * dip : on candles with RSI_3 < 10, how many sub-conditions the CLOSEST tag misses.
  * any_pass: candles where some tag passed completely (= an entry signal).

Used only with filter-diagnose.yml. Never used by the live bots.
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import NostalgiaForInfinityX8 as _m
from NostalgiaForInfinityX8 import NostalgiaForInfinityX8

_B0 = pd.Timestamp(os.environ.get("DIAG_BLOCK_START", "2026-10-15 03:00"), tz="UTC")
_E = pd.Timestamp(os.environ.get("DIAG_END", "2026-10-20 00:00"), tz="UTC")
_R0 = pd.Timestamp(os.environ.get("DIAG_REF_START", "2026-10-06 00:00"), tz="UTC")
_OUT = os.environ.get("DIAG_OUT", "/tmp/diag")

_STATE = {"active": False, "last": None, "records": []}


def _install_hooks():
  if getattr(_m, "_diag_hooks_installed", False):
    return
  orig_and = _m._and_entry_conditions
  orig_tag = _m._append_entry_tag

  def and_wrap(conditions):
    if _STATE["active"]:
      _STATE["last"] = list(conditions)
    return orig_and(conditions)

  def tag_wrap(entry_tags, mask, tag):
    if _STATE["active"]:
      lst = _STATE["last"]
      _STATE["last"] = None
      try:
        idx = int(str(tag).strip())
      except ValueError:
        idx = None
      if lst is not None and idx is not None and idx < 500:  # long tags only
        _STATE["records"].append((idx, lst))
    return orig_tag(entry_tags, mask, tag)

  _m._and_entry_conditions = and_wrap
  _m._append_entry_tag = tag_wrap
  _m._diag_hooks_installed = True


def _analyze(df, pair, records):
  n = len(df)
  dates = pd.to_datetime(df["date"], utc=True)
  wb = ((dates >= _B0) & (dates < _E)).to_numpy()
  wr = ((dates >= _R0) & (dates < _B0)).to_numpy()
  wins = [wb, wr]
  dip = (df["RSI_3"] < 10).to_numpy() if "RSI_3" in df.columns else np.zeros(n, dtype=bool)

  minfail = np.full(n, 99, dtype=int)
  nm = defaultdict(lambda: [0, 0])
  pas = defaultdict(lambda: [0, 0])
  nel = {}
  for idx, conds in records:
    arrs = []
    ok = True
    for c in conds:
      if _m._is_entry_bool_scalar(c):
        arrs.append(np.full(n, bool(c)))
      else:
        a = np.asarray(_m._entry_bool_array(c), dtype=bool)
        if a.shape != (n,):
          ok = False
          break
        arrs.append(a)
    if not ok or not arrs:
      continue
    fail = ~np.vstack(arrs)  # (k, n)
    nfail = fail.sum(axis=0)
    minfail = np.minimum(minfail, nfail)
    nel[str(idx)] = int(fail.shape[0])
    for wi, w in enumerate(wins):
      pas[str(idx)][wi] += int(((nfail == 0) & w).sum())
      one = (nfail == 1) & w
      if one.any():
        js = fail[:, one].argmax(axis=0)
        vals, cnts = np.unique(js, return_counts=True)
        for j, cnt in zip(vals, cnts):
          nm[f"{idx}:{int(j)}"][wi] += int(cnt)

  out = {
    "pair": pair,
    "n_tags": len(nel),
    "n_elems": nel,
    "nm": dict(nm),
    "pass": dict(pas),
    "any_pass": [int(((minfail == 0) & w).sum()) for w in wins],
    "any_pass_dates": [str(d) for d in dates[(minfail == 0) & (wb | wr)]][:200],
    "dip_total": [int((dip & w).sum()) for w in wins],
    "dip_hist": [
      np.bincount(np.minimum(minfail[dip & w], 6), minlength=7).tolist() for w in wins
    ],
  }
  os.makedirs(_OUT, exist_ok=True)
  with open(os.path.join(_OUT, pair.replace("/", "_").replace(":", "_") + ".json"), "w") as f:
    json.dump(out, f)


class NostalgiaForInfinityX8Diag(NostalgiaForInfinityX8):
  def populate_entry_trend(self, df, metadata):
    _install_hooks()
    _STATE["records"] = []
    _STATE["last"] = None
    _STATE["active"] = True
    try:
      df = super().populate_entry_trend(df, metadata)
    finally:
      _STATE["active"] = False
    try:
      _analyze(df, metadata.get("pair", "?"), _STATE["records"])
    except Exception as e:  # never break the backtest because of the diagnostic
      print(f"[diag] analysis failed for {metadata.get('pair')}: {e!r}")
    _STATE["records"] = []
    return df
