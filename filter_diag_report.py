#!/usr/bin/env python3
"""
filter_diag_report.py - reads the JSON files written by NostalgiaForInfinityX8Diag
and prints which long-entry sub-condition blocked the entries. Save in the repo root.

Usage: python filter_diag_report.py --diag-dir /tmp/diag --strategy user_data/strategies/NostalgiaForInfinityX8.py \
         --block-start "2026-10-15 03:00" --end "2026-10-20 00:00" --ref-start "2026-10-06 00:00"
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd

IF_RE = re.compile(r"^\s*if long_entry_condition_index == (\d+):\s*$")
APPEND = "long_entry_logic.append("
VOL = 'long_entry_logic.append(df["volume"] > 0)'


def parse_labels(path):
    """{tag: [label of element 1, label of element 2, ...]} (element 0 is the constant True)."""
    labels = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return labels
    starts = [(i, int(IF_RE.match(l).group(1))) for i, l in enumerate(lines) if IF_RE.match(l)]
    vol_lines = [i for i, l in enumerate(lines) if VOL in l]
    for k, (i, tag) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        later_vol = [v for v in vol_lines if v > i]
        if later_vol:
            end = min(end, later_vol[0])
        block, labs = lines[i:end], []
        for j, l in enumerate(block):
            if APPEND in l and not l.strip().startswith("#"):
                # collect the statement until its parentheses balance
                depth, stmt = 0, []
                for t in block[j:j + 60]:
                    code = t.split("#", 1)[0]
                    depth += code.count("(") - code.count(")")
                    stmt.append(t.strip())
                    if depth <= 0:
                        break
                first = stmt[0].replace(APPEND, "").strip()
                comments = [x.lstrip("# ").strip() for x in stmt[1:] if x.startswith("#")]
                codes = [x for x in stmt[1:] if x and not x.startswith("#")]
                if len(stmt) == 1:
                    label = first.rstrip(")")
                else:
                    label = (comments[0] if comments else (first or (codes[0] if codes else "")))
                    if codes:
                        label += " :: " + codes[0]
                labs.append(label[:120])
        labels[tag] = labs
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--block-start", default="2026-10-15 03:00")
    ap.add_argument("--end", default="2026-10-20 00:00")
    ap.add_argument("--ref-start", default="2026-10-06 00:00")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ts = lambda s: pd.Timestamp(s, tz="UTC")
    d_block = (ts(a.end) - ts(a.block_start)).total_seconds() / 86400
    d_ref = (ts(a.block_start) - ts(a.ref_start)).total_seconds() / 86400
    days = [d_block, d_ref]

    files = sorted(glob.glob(os.path.join(a.diag_dir, "*.json")))
    if not files:
        print("No diag files found - the diagnostic strategy did not run or wrote nothing.")
        return

    nm = defaultdict(lambda: [0, 0])
    nel = {}
    any_pass = [0, 0]
    dip_total = [0, 0]
    hist = [[0] * 7, [0] * 7]
    pass_dates = []
    for f in files:
        d = json.load(open(f))
        for k, v in d["nm"].items():
            nm[k][0] += v[0]
            nm[k][1] += v[1]
        nel.update(d["n_elems"])
        for w in (0, 1):
            any_pass[w] += d["any_pass"][w]
            dip_total[w] += d["dip_total"][w]
            for i in range(7):
                hist[w][i] += d["dip_hist"][w][i]
        pass_dates += [(x, d["pair"]) for x in d["any_pass_dates"]]
    labels = parse_labels(a.strategy)

    L = []
    w = L.append
    w(f"pairs analysed: {len(files)}")
    w(f"block window {a.block_start} .. {a.end} = {d_block:.1f} days; reference {a.ref_start} .. {a.block_start} = {d_ref:.1f} days")
    w("")
    w("1) Candles where some long tag passed completely (= entry signal before slots/stake limits)")
    w(f"   block window: {any_pass[0]}   reference: {any_pass[1]}   (backtest had 0 signals in the block window)")
    if any_pass[0]:
        in_block = sorted((x, p) for x, p in pass_dates if pd.Timestamp(x) >= ts(a.block_start))
        w("   block-window signals (date, pair): " + "; ".join(f"{x[:16]} {p}" for x, p in in_block[:10]))
    w("")
    w("2) Dip candles (RSI_3 < 10): how many sub-conditions the CLOSEST long tag misses (per day)")
    w("   misses:        0      1      2      3      4      5     6+")
    for wi, name in ((0, "block"), (1, "reference")):
        row = "  ".join(f"{hist[wi][i] / days[wi]:6.1f}" for i in range(7))
        w(f"   {name:10s} {row}   (dip candles/day: {dip_total[wi] / days[wi]:.0f})")
    w("")
    w(f"3) Top {a.top} sub-conditions that were the ONLY failing one (near misses per day), ranked by block window")
    w("   tag:elem | block/day | ref/day | text of the sub-condition")
    ranked = sorted(nm.items(), key=lambda kv: kv[1][0], reverse=True)[: a.top]
    for key, (b, r) in ranked:
        tag, elem = key.split(":")
        elem = int(elem)
        labs = labels.get(int(tag), [])
        k = nel.get(tag)
        if elem == 0:
            text = "(constant True)"
        elif k is not None and elem == k - 1:
            text = 'df["volume"] > 0'
        elif k is not None and len(labs) + 2 == k and elem - 1 < len(labs):
            text = labs[elem - 1]
        else:
            text = "(text not matched - open the file at this tag)"
        w(f"   {tag:>4s}:{elem:<2d}  | {b / days[0]:8.1f}  | {r / days[1]:7.1f} | {text}")
    text = "\n".join(L)
    print(text)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "filter_diag_report.txt"), "w").write(text + "\n")
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
