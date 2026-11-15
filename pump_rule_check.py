# pump_rule_check.py
#
# Question: do LONG entries made right after a sharp pump, inside a long downtrend,
# turn into troubled trades more often than other long entries -- and would a filter
# for that pattern earn money, or only remove winners too?
#
# Input : trades_with_features.csv  (written by pair_features.py analyze; one row per trade,
#         features come from the last closed DAILY candle before the entry day)
# Output: report text on screen and in the --out file.
#
# Everything is decided BEFORE looking at the data:
#   pattern  = PUMP and DOWNTREND
#   PUMP     = ret_7d >= a      or  best_day_30d >= b     (a, b in 0.15 / 0.25 / 0.40)
#   DOWNTREND= dist_high_90d <= -d                          (d in 0.40 / 0.50 / 0.70)
#   18 combinations in total. The best of them is judged with a permutation test that
#   shuffles the outcome labels inside each run and takes the best of all 18 combinations
#   every time, so choosing the best combination does not count as a discovery.
#
# Usage: python pump_rule_check.py --csv trades_with_features.csv --out pump_report.md
#
# Only needs pandas and numpy.

import argparse
import itertools

import numpy as np
import pandas as pd

PUMP_A = [0.15, 0.25, 0.40]   # ret_7d at least
PUMP_B = [0.15, 0.25, 0.40]   # best single day in the last 30 days at least
DOWN_D = [0.40, 0.50, 0.70]   # price at least this far below the 90 day high
MIN_TOTAL = 12                # fewest matching trades for a verdict
MIN_PER_RUN = 4
N_PERM = 2000
ALPHA = 0.10                  # lenient on purpose: there are very few troubled trades


def truthy(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def load(path):
    df = pd.read_csv(path)
    need = ["run", "pair", "is_short", "profit_abs", "days", "adverse_move",
            "ret_7d", "best_day_30d", "dist_high_90d", "ret_90d"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit("missing columns: " + ", ".join(miss))
    df["is_short"] = truthy(df["is_short"])
    df = df[~df["is_short"]].copy()
    df["lose"] = df["profit_abs"] < 0
    df["deep20"] = df["adverse_move"] >= 0.20
    df["trouble"] = df["lose"] | df["deep20"]
    df["stuck10"] = df["days"] > 10
    n0 = len(df)
    df = df.dropna(subset=["ret_7d", "best_day_30d", "dist_high_90d"]).reset_index(drop=True)
    return df, n0


def combos():
    out = []
    for kind, vals in (("ret_7d", PUMP_A), ("best_day_30d", PUMP_B)):
        for v in vals:
            for d in DOWN_D:
                out.append((kind, v, d))
    return out


def mask_for(df, kind, v, d):
    pump = df[kind] >= v
    down = df["dist_high_90d"] <= -d
    return (pump & down).to_numpy()


def stratified_z(M, y, runs):
    # M: n x k boolean matrix (pattern), y: n boolean outcome, runs: n labels
    # z-score of the number of troubled trades inside the pattern, summed over runs
    S = np.zeros(M.shape[1])
    V = np.zeros(M.shape[1])
    for r in np.unique(runs):
        idx = runs == r
        yy = y[idx].astype(float)
        N = idx.sum()
        K = yy.sum()
        if N < 3:
            continue
        m = M[idx].astype(float)
        nm = m.sum(axis=0)
        k = yy @ m
        p = K / N
        E = nm * p
        var = nm * p * (1 - p) * (N - nm) / max(N - 1, 1)
        S += k - E
        V += var
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(V > 0, S / np.sqrt(V), 0.0)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="pump_report.md")
    args = ap.parse_args()

    df, n0 = load(args.csv)
    runs = df["run"].astype(str).to_numpy()
    run_ids = list(pd.unique(runs))
    cs = combos()
    M = np.column_stack([mask_for(df, *c) for c in cs])
    counts = M.sum(axis=0)

    lines = []
    w = lines.append
    w("# Pump inside a downtrend: check of LONG entries")
    w("")
    w("Long trades: %d (with features: %d). Runs: %s." % (n0, len(df), ", ".join(run_ids)))
    base = df.groupby("run").agg(n=("pair", "count"), trouble=("trouble", "sum"),
                                 lose=("lose", "sum"), stuck10=("stuck10", "sum"),
                                 pnl=("profit_abs", "sum")).round(1)
    w("")
    w("Baseline per run:")
    w(base.to_string())
    w("")
    ntrouble = int(df["trouble"].sum())
    if ntrouble < 8:
        w("Only %d troubled long trades in total: too few for any rule. Stop here." % ntrouble)
        text = "\n".join(lines) + "\n"
        open(args.out, "w").write(text)
        print(text)
        return

    y = df["trouble"].to_numpy()
    z_obs = stratified_z(M, y, runs)
    ok = counts >= MIN_TOTAL
    rng = np.random.default_rng(1)
    maxz = np.zeros(N_PERM)
    for i in range(N_PERM):
        yp = y.copy()
        for r in run_ids:
            idx = np.where(runs == r)[0]
            yp[idx] = rng.permutation(y[idx])
        zz = stratified_z(M, yp, runs)
        zz = np.where(ok, zz, -9.0)
        maxz[i] = zz.max()

    rows = []
    for j, (kind, v, d) in enumerate(cs):
        m = M[:, j]
        sub = df[m]
        rest = df[~m]
        row = dict(pump="%s>=%.2f" % (kind, v), down="dist_high_90d<=-%.2f" % d,
                   n=int(m.sum()), trouble_rate=(sub["trouble"].mean() if len(sub) else np.nan),
                   rest_rate=rest["trouble"].mean(), pnl=sub["profit_abs"].sum(),
                   z=z_obs[j])
        for r in run_ids:
            s = sub[sub["run"].astype(str) == r]
            row["n_" + r[-4:]] = len(s)
        rows.append(row)
    t = pd.DataFrame(rows)
    t["p_adj"] = [float((maxz >= z).mean()) if ok[j] else np.nan for j, z in enumerate(t["z"])]
    tt = t[t["n"] >= MIN_TOTAL].sort_values("z", ascending=False)

    w("## All %d combinations (only those with at least %d matching trades are ranked)" % (len(cs), MIN_TOTAL))
    w("")
    show = tt.copy() if len(tt) else t.copy()
    show["trouble_rate"] = show["trouble_rate"].round(2)
    show["rest_rate"] = show["rest_rate"].round(2)
    show["pnl"] = show["pnl"].round(0)
    show["z"] = show["z"].round(2)
    show["p_adj"] = show["p_adj"].round(3)
    w(show.to_string(index=False))
    w("")

    if not len(tt):
        w("VERDICT: no combination matches %d or more trades. The pattern is too rare in this data." % MIN_TOTAL)
    else:
        best = tt.iloc[0]
        j = int(t.index[(t["pump"] == best["pump"]) & (t["down"] == best["down"])][0])
        m = M[:, j]
        sub = df[m]
        w("## Best combination: %s and %s" % (best["pump"], best["down"]))
        w("")
        crit = []
        per_run_ok = True
        per_run_dir = True
        per_run_pnl = True
        for r in run_ids:
            a = df[(df["run"].astype(str) == r)]
            s = a[m[a.index]]
            if len(s) < MIN_PER_RUN:
                per_run_ok = False
            rr = a[~m[a.index]]
            if len(s) and not (s["trouble"].mean() > rr["trouble"].mean()):
                per_run_dir = False
            if len(s) and not (s["profit_abs"].sum() < 0):
                per_run_pnl = False
            w("run %s: matching %d, troubled rate %.2f vs %.2f, net profit of matching trades %.1f" % (
                r, len(s), s["trouble"].mean() if len(s) else float("nan"),
                rr["trouble"].mean(), s["profit_abs"].sum()))
        winners = sub[sub["profit_abs"] > 0]
        losers = sub[sub["profit_abs"] < 0]
        w("")
        w("Matching trades: %d; winners %d (profit %.1f), losers %d (loss %.1f), net %.1f." % (
            len(sub), len(winners), winners["profit_abs"].sum(), len(losers),
            losers["profit_abs"].sum(), sub["profit_abs"].sum()))
        w("Troubled cases inside the pattern: %d of %d troubled long trades overall." % (
            int(sub["trouble"].sum()), ntrouble))
        w("")
        crit.append(("enough trades (total >= %d, each run >= %d)" % (MIN_TOTAL, MIN_PER_RUN),
                     len(sub) >= MIN_TOTAL and per_run_ok))
        crit.append(("troubled rate higher inside the pattern in EVERY run", per_run_dir))
        crit.append(("adjusted p-value <= %.2f (best of %d combinations)" % (ALPHA, len(cs)),
                     float(best["p_adj"]) <= ALPHA))
        crit.append(("net profit of the matching trades is negative in EVERY run "
                     "(a filter would add money, winners included)", per_run_pnl))
        for name, passed in crit:
            w("- %s: %s" % (name, "PASS" if passed else "FAIL"))
        w("")
        if all(p for _, p in crit):
            w("VERDICT: the pattern passes all four checks. Next step: test it as a real entry filter "
              "in a backtest on both windows and read the cost on winners.")
        else:
            w("VERDICT: the pattern does not pass all four checks. Not enough evidence for a filter.")
        w("")
        w("## Trades inside the best pattern (worst first)")
        cols = [c for c in ["run", "pair", "enter_tag", "open_date", "days", "n_entries", "profit_abs",
                            "adverse_move", "ret_7d", "best_day_30d", "dist_high_90d", "exit_reason"] if c in sub.columns]
        s2 = sub.sort_values("profit_abs")[cols].copy()
        s2["pair"] = s2["pair"].str.replace("/USDT:USDT", "", regex=False)
        for c in ["days", "profit_abs"]:
            s2[c] = s2[c].round(1)
        for c in ["adverse_move", "ret_7d", "best_day_30d", "dist_high_90d"]:
            if c in s2.columns:
                s2[c] = s2[c].round(2)
        w(s2.head(25).to_string(index=False))

    # the two single conditions for comparison
    w("")
    w("## Single conditions for comparison (not the pattern)")
    for name, mm in (("DOWNTREND only: dist_high_90d <= -0.50", (df["dist_high_90d"] <= -0.5).to_numpy()),
                     ("PUMP only: best_day_30d >= 0.25", (df["best_day_30d"] >= 0.25).to_numpy()),
                     ("PUMP only: ret_7d >= 0.25", (df["ret_7d"] >= 0.25).to_numpy())):
        s = df[mm]
        r = df[~mm]
        w("%s: %d trades, troubled rate %.2f vs %.2f, net profit %.1f" % (
            name, len(s), s["trouble"].mean() if len(s) else float("nan"), r["trouble"].mean(),
            s["profit_abs"].sum()))
    w("")
    w("Note: the daily features end one day before the entry. Pairs listed on OKX less than 90 days "
      "before the entry have a shorter history and are treated as they are.")
    text = "\n".join(lines) + "\n"
    open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
