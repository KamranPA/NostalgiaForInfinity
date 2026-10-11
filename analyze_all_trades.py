#!/usr/bin/env python3
"""
analyze_all_trades.py -- pattern search over EVERY backtest trade (no "open > N days" filter).

Question 1: which trades walk into a deep loss?   (what is known AT ENTRY)
Question 2: of the trades that did walk into a deep loss, which came back?
            (entry-time AND path features; path features are descriptive only)

Needs analyze_long_trades.py next to it (reuses its loader).
Input: freqtrade backtest results (.zip/.json). Output: Markdown on stdout + CSV (--csv).

  python analyze_all_trades.py [--deep 0.5] [--only-strategy NostalgiaForInfinityX8]
                               [--csv all.csv] FILE [FILE ...]

Definitions
  deep      = worst price move against the first entry x leverage <= -DEEP of margin
  loss      = closed with profit_abs < 0
  recovered = deep AND closed with profit_abs > 0
Statistics: Fisher exact test per category vs the rest; AUC for numeric features.
With dozens of tests some "significant" results appear by chance -> the report prints
the number of tests and the Bonferroni threshold. Treat results as leads, not laws.
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import analyze_long_trades as al


# ------------------------------------------------------------------ stats
def fisher_two_sided(a, b, c, d):
    """2x2 table [[a,b],[c,d]] -> two-sided Fisher exact p."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def pmf(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p_obs * (1 + 1e-9)))


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else (0.5 if p == q else 0.0)
    return s / (len(pos) * len(neg))


def bucket(x, edges, labels):
    if x is None:
        return "n/a"
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


# ------------------------------------------------------------------ features
def add_features(trades, args):
    by_win = defaultdict(list)
    for t in trades:
        by_win[t["window"]].append(t)
    for w, ts in by_win.items():
        ts.sort(key=lambda t: t["open_dt"])
        by_pair = defaultdict(list)
        for t in ts:
            by_pair[t["pair"]].append(t)
        prev_open = None
        for i, t in enumerate(ts):
            o = t["open_dt"]
            t["n_open"] = sum(1 for u in ts if u is not t and u["open_dt"] <= o and
                              (u["close_dt"] is None or u["close_dt"] > o))
            t["burst"] = sum(1 for u in ts if u is not t and abs((u["open_dt"] - o).total_seconds()) <= 300)
            t["gap_prev_h"] = ((o - prev_open).total_seconds() / 3600.0) if prev_open else None
            prev_open = o
            earlier = [u for u in by_pair[t["pair"]] if u is not t and u["open_dt"] < o]
            if not earlier:
                t["prev_pair"], t["prev_pair_h"] = "no earlier trade", None
            else:
                u = earlier[-1]
                t["prev_pair_h"] = (o - u["open_dt"]).total_seconds() / 3600.0
                if u["close_dt"] is None or u["close_dt"] > o:
                    t["prev_pair"] = "earlier trade still open"
                else:
                    t["prev_pair"] = "earlier won" if (u["profit_abs"] or 0) > 0 else "earlier lost"
    for t in trades:
        o = t["open_dt"]
        t["hour"] = o.hour
        t["hour_b"] = bucket(o.hour, [6, 12, 18, 25], ["00-06", "06-12", "12-18", "18-24"])
        t["weekday"] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][o.weekday()]
        t["wkend"] = "weekend" if o.weekday() >= 5 else "weekday"
        fp = t["first_price"] or t["open_rate"]
        t["log_price"] = math.log10(fp) if fp and fp > 0 else None
        t["price_b"] = bucket(fp, [0.01, 0.1, 1, 10, 1e18], ["<0.01", "0.01-0.1", "0.1-1", "1-10", ">=10"])
        t["direction"] = "SHORT" if t["is_short"] else "LONG"
        t["tag0"] = str(t["tag"] or "").split()[0] if str(t["tag"] or "").split() else "?"
        t["n_tags"] = len(str(t["tag"] or "").split())
        t["n_tags_b"] = "1 tag" if t["n_tags"] <= 1 else ("2 tags" if t["n_tags"] == 2 else "3+ tags")
        t["n_open_b"] = bucket(t["n_open"], [2, 4, 6, 8, 99], ["0-1", "2-3", "4-5", "6-7", "8+"])
        t["burst_b"] = bucket(t["burst"], [1, 2, 4, 999], ["alone", "1 other", "2-3 others", "4+ others"])
        t["gap_b"] = bucket(t["gap_prev_h"], [0.1, 1, 6, 1e9], ["<6min", "6-60min", "1-6h", ">6h"])
        t["prev_pair_h_b"] = bucket(t["prev_pair_h"], [6, 24, 72, 1e9], ["<6h", "6-24h", "1-3d", ">3d"])
        # path features (descriptive only -- NOT known at entry)
        t["entries_b"] = bucket(t["n_entries"], [2, 3, 5, 10, 1e9], ["1", "2", "3-4", "5-9", "10+"])
        dl = 0
        for od in t["orders"]:
            tg = str(od.get("tag") or "")
            if od["side"] == "exit" and tg.startswith("derisk_level_"):
                try:
                    dl = max(dl, int(tg.split("derisk_level_")[1][0]))
                except (ValueError, IndexError):
                    pass
        t["derisk"] = dl
        t["derisk_b"] = f"level {dl}" if dl else "none"
        t["dur_b"] = bucket(t["dur"], [0.25, 1, 2, 5, 14, 1e9], ["<6h", "6-24h", "1-2d", "2-5d", "5-14d", "14d+"])
        t["dca1_b"] = ("no DCA" if t["time_to_first_dca"] is None else
                       bucket(t["time_to_first_dca"] * 24, [1, 6, 24, 1e9], ["DCA <1h", "DCA 1-6h", "DCA 6-24h", "DCA >1d"]))
        t["mfe_b"] = bucket(t["mfe_margin"], [0.05, 0.15, 0.4, 9e9], ["<5%", "5-15%", "15-40%", "40%+"])
        # targets
        t["closed"] = not t["is_open"]
        t["deep"] = t["mae_margin"] is not None and t["mae_margin"] <= -args.deep
        t["loss"] = t["closed"] and (t["profit_abs"] or 0) < 0
        t["recovered"] = t["deep"] and t["closed"] and (t["profit_abs"] or 0) > 0


ENTRY_CAT = ["family", "direction", "tag0", "n_tags_b", "hour_b", "wkend", "weekday", "price_b",
             "n_open_b", "burst_b", "gap_b", "prev_pair", "prev_pair_h_b"]
ENTRY_NUM = ["log_price", "n_open", "burst", "gap_prev_h", "hour", "stake0", "n_tags"]
PATH_CAT = ["entries_b", "derisk_b", "dca1_b", "dur_b", "mfe_b"]
PATH_NUM = ["n_entries", "n_partial_exits", "dur", "mfe_margin", "time_to_first_dca"]


# ------------------------------------------------------------------ tables
def cat_findings(universe, target, feats, wins, min_n=8):
    N = len(universe)
    K = sum(1 for t in universe if t[target])
    base = K / N if N else 0
    found = []
    for f in feats:
        g = defaultdict(list)
        for t in universe:
            g[t[f]].append(t)
        for val, ts in g.items():
            n = len(ts)
            if n < min_n or val == "n/a":
                continue
            k = sum(1 for t in ts if t[target])
            p = fisher_two_sided(k, n - k, K - k, N - n - (K - k))
            per = []
            for w in wins:
                tw = [t for t in ts if t["window"] == w]
                per.append(f"{sum(1 for t in tw if t[target])}/{len(tw)}")
            found.append(dict(feat=f, val=val, n=n, k=k, rate=k / n, lift=(k / n / base) if base else None,
                              p=p, per=per))
    return N, K, base, found


def print_findings(title, N, K, base, found, wins, n_tests, top=14):
    print(f"### {title}\n")
    print(f"Universe {N} trades, target {K} ({base * 100:.1f}%). Tests run: {n_tests} "
          f"(Bonferroni p < {0.05 / max(n_tests, 1):.4f}). Per-window columns: " + ", ".join(wins) + " as target/total.\n")
    found = sorted(found, key=lambda x: x["p"])[:top]
    rows = [(f["feat"], f["val"], f["n"], f"{f['rate'] * 100:.0f}%", f"x{f['lift']:.1f}" if f["lift"] else "—",
             f"{f['p']:.4f}" + (" *" if f["p"] < 0.05 / max(n_tests, 1) else ""), " · ".join(f["per"]))
            for f in found]
    print(al.table(["feature", "value", "n", "rate", "lift", "p", "per window"], rows))


def num_table(universe, target, feats, title):
    pos = [t for t in universe if t[target]]
    neg = [t for t in universe if not t[target]]
    rows = []
    for f in feats:
        a = [t[f] for t in pos if t[f] is not None]
        b = [t[f] for t in neg if t[f] is not None]
        v = auc(a, b)
        if v is None:
            continue
        med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
        rows.append((f, f"{v:.2f}", "higher → more" if v > 0.5 else "lower → more",
                     f"{med(a):.3g}", f"{med(b):.3g}", len(a), len(b)))
    rows.sort(key=lambda r: -abs(float(r[1]) - 0.5))
    print(f"### {title}: numeric features (AUC; 0.5 = no signal)\n")
    print(al.table(["feature", "AUC", "direction", "median (target)", "median (rest)", "n target", "n rest"], rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", type=float, default=0.5, help="deep = adverse move x leverage >= this share of margin")
    ap.add_argument("--only-strategy", default="NostalgiaForInfinityX8")
    ap.add_argument("--csv")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    trades, seen = [], set()
    for p in args.files:
        if p.lower().endswith((".sqlite", ".db")):
            continue
        label = os.path.splitext(os.path.basename(p))[0]
        try:
            ts = al.load_backtest(p, label)
        except Exception as e:  # noqa: BLE001
            print(f"_could not read {p}: {e}_\n")
            continue
        for t in ts:
            strat = t["source"].split(":", 1)[-1]
            if args.only_strategy and strat != args.only_strategy:
                continue
            if not t["open_dt"]:
                continue
            key = (t["pair"], t["open_dt"], t["is_short"])
            if key in seen:
                continue
            seen.add(key)
            t["window"] = t["source"]
            trades.append(al.enrich(t))
    if not trades:
        print("_No trades found (check --only-strategy and the input files)._")
        return

    wins_sorted = sorted({t["window"] for t in trades}, key=lambda w: min(t["open_dt"] for t in trades if t["window"] == w))
    short = {w: f"W{i + 1}" for i, w in enumerate(wins_sorted)}
    for t in trades:
        t["window_id"] = short[t["window"]]
    wins = [short[w] for w in wins_sorted]
    for t in trades:
        t["window"] = t["window_id"]
    add_features(trades, args)

    N = len(trades)
    print(f"# Pattern search over ALL trades ({args.only_strategy})\n")
    rows = []
    for w, src in zip(wins, wins_sorted):
        ts = [t for t in trades if t["window"] == w]
        rows.append((w, src[:60], len(ts), f"{min(t['open_dt'] for t in ts):%Y-%m-%d}→{max(t['open_dt'] for t in ts):%Y-%m-%d}",
                     sum(t["deep"] for t in ts), sum(t["loss"] for t in ts), sum(t["recovered"] for t in ts),
                     al.usd(sum((t["profit_abs"] or 0) for t in ts if t["closed"]))))
    print(al.table(["id", "source", "trades", "span", f"deep(>={args.deep:.0%} margin)", "closed at loss", "recovered", "P&L"], rows))

    # ---- A: entry-time features -> deep / loss
    n_tests = 0
    for target, title in (("deep", f"A1. What leads into a DEEP loss (>= {args.deep:.0%} of margin)"),
                          ("loss", "A2. What ends in a LOSS (closed, profit < 0)")):
        Nn, K, base, found = cat_findings(trades, target, ENTRY_CAT, wins)
        n_tests = len(found)
        print(f"\n## {title} — known at entry\n")
        if K == 0 or K == Nn:
            print("_Target is empty or universal — nothing to separate._\n")
            continue
        print_findings("Categorical patterns", Nn, K, base, found, wins, n_tests)
        num_table(trades, target, ENTRY_NUM, "Entry-time")

    # ---- B: among deep & closed -> recovered
    deep_closed = [t for t in trades if t["deep"] and t["closed"]]
    print("\n## B. Of the DEEP trades that closed: which came back?\n")
    n_rec = sum(t["recovered"] for t in deep_closed)
    print(f"{len(deep_closed)} deep & closed trades, {n_rec} recovered (closed in profit), "
          f"{len(deep_closed) - n_rec} did not. Path features are only visible AFTER the fact — "
          "they describe, they do not predict.\n")
    if 0 < n_rec < len(deep_closed):
        Nn, K, base, found = cat_findings(deep_closed, "recovered", ENTRY_CAT + PATH_CAT, wins, min_n=5)
        print_findings("Categorical patterns (recovered vs not)", Nn, K, base, found, wins, len(found))
        num_table(deep_closed, "recovered", ENTRY_NUM + PATH_NUM, "Recovered vs not")
        rows = []
        for t in sorted(deep_closed, key=lambda x: -(x["profit_abs"] or 0))[:40]:
            rows.append((t["window"], t["pair"].split("/")[0], t["direction"], t["tag0"], t["family"],
                         f"{t['mae_margin'] * 100:.0f}%", t["n_entries"], t["derisk_b"], f"{t['dur']:.1f}d",
                         t["exit_reason"] or "", al.usd(t["profit_abs"]), "RECOVERED" if t["recovered"] else "—"))
        print("### Deep trades (best → worst, first 40)\n")
        print(al.table(["win", "pair", "dir", "tag", "family", "worst (margin)", "entries", "derisk", "days",
                        "exit", "P&L", ""], rows))
    else:
        print("_All deep trades ended the same way — nothing to compare._\n")

    print("\n## Caveats\n")
    print("- Few deep trades and many tests: expect some false positives. A pattern deserves attention "
          "only if it holds in BOTH windows (see the per-window column) and has a sensible mechanism.\n"
          "- Entry-time features here are limited to tag, direction, time, price level, crowding and the "
          "pair's previous trade. Indicator values at entry (RSI, ATR, trend...) are NOT in the result "
          "file; export them with `freqtrade backtesting --export signals` to go further.\n")

    if args.csv:
        cols = ["window", "pair", "direction", "tag", "tag0", "family", "n_tags", "open_dt", "close_dt", "dur",
                "hour", "weekday", "log_price", "stake0", "n_open", "burst", "gap_prev_h", "prev_pair", "prev_pair_h",
                "n_entries", "n_partial_exits", "derisk", "time_to_first_dca", "mae_margin", "mfe_margin",
                "exit_reason", "profit_abs", "profit_ratio", "deep", "loss", "recovered", "closed"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for t in trades:
                w.writerow([t.get(c) for c in cols])


if __name__ == "__main__":
    main()
