#!/usr/bin/env python3
"""
pair_features.py -- do the pairs that produced the big losses look different BEFORE the entry?

Two sub-commands (used by .github/workflows/pair-features.yml):

  python pair_features.py pairs   --artifacts DIR --runs ID1,ID2 --out pairs.json
      Union of the pairlist snapshots of the given backtest runs (+ BTC), used to decide
      which daily candles must be downloaded.

  python pair_features.py analyze --data DIR --artifacts DIR --runs ID1,ID2 --out OUTDIR
                                  [--history-start YYYYMMDD]
      1. builds features per pair and day from the DAILY candles (only information that
         existed before the entry: the last fully closed daily candle before the trade opened),
      2. joins them with the trades of the backtest runs,
      3. compares troubled trades with normal ones (AUC + permutation test that corrects for
         testing many features), and checks the first run's findings on the second run.

Outputs in OUTDIR: report.md, trades_with_features.csv, pair_features_daily.csv.gz

Run IDs: FIRST = discovery period, SECOND = out-of-sample check.
Nothing here changes the backtest data; it only reads it.
Needs: pandas, numpy, pyarrow (feather files written by `freqtrade download-data`).
"""
import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

BTC = "BTC/USDT:USDT"
# Bases that are stocks/ETFs (not crypto). Heuristic list -- edit via env STOCK_BASES.
STOCK_BASES = set(
    os.environ.get(
        "STOCK_BASES",
        "SOXL,TQQQ,SQQQ,SPY,QQQ,AAPL,NVDA,TSLA,MSFT,AMZN,GOOGL,META,COIN,MSTR,HOOD,CRCL,AMD,PLTR",
    ).split(",")
)
FEATURES = [
    "ret_7d", "ret_30d", "ret_90d", "vol_30d", "worst_day_30d", "best_day_30d",
    "big_move_days_30d", "range_14d", "dist_high_90d", "dist_low_90d",
    "dollar_vol_30d", "vol_spike_3v90", "age_days", "btc_ret_7d", "btc_ret_30d",
    "rel_ret_30d", "corr_btc_60d",
]


# ----------------------------------------------------------------------------- helpers
def file_key(pair):
    return pair.replace("/", "_").replace(":", "_")


def read_pairlist(artifacts, run_id):
    paths = glob.glob(os.path.join(artifacts, f"pairlist-{run_id}", "**", "pairlist_snapshot.json"), recursive=True)
    if not paths:
        raise SystemExit(f"pairlist-{run_id}: pairlist_snapshot.json not found under {artifacts}")
    with open(paths[0]) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"pairlist-{run_id}: unexpected content")
    return data


def load_trades(artifacts, run_id):
    base = os.path.join(artifacts, f"backtest-{run_id}")
    data = None
    zips = [z for z in glob.glob(os.path.join(base, "**", "backtest-result-*.zip"), recursive=True)
            if "meta" not in os.path.basename(z)]
    jsons = [j for j in glob.glob(os.path.join(base, "**", "backtest-result-*.json"), recursive=True)
             if not j.endswith(".meta.json") and "_config" not in j]
    if zips:
        with zipfile.ZipFile(sorted(zips)[-1]) as zf:
            names = [n for n in zf.namelist()
                     if n.endswith(".json") and "_config" not in n and "_meta" not in n]
            if names:
                with zf.open(names[0]) as fh:
                    data = json.load(fh)
    elif jsons:
        with open(sorted(jsons)[-1]) as fh:
            data = json.load(fh)
    if data is None:
        raise SystemExit(f"backtest-{run_id}: no result json found under {base}")
    key = next(iter(data["strategy"]))
    rows = []
    for t in data["strategy"][key]["trades"]:
        orders = sorted(t.get("orders") or [], key=lambda o: o.get("order_filled_timestamp", 0))
        entries = [o for o in orders if o.get("ft_is_entry")]
        p0 = entries[0]["safe_price"] if entries else t["open_rate"]
        short = bool(t.get("is_short"))
        if short:
            adverse = (t["max_rate"] / p0 - 1.0) if p0 else np.nan
        else:
            adverse = (1.0 - t["min_rate"] / p0) if p0 else np.nan
        rows.append(dict(
            run=str(run_id), pair=t["pair"], is_short=short, enter_tag=str(t.get("enter_tag")),
            open_date=pd.Timestamp(t["open_date"]), close_date=pd.Timestamp(t["close_date"]),
            days=t["trade_duration"] / 1440.0, profit_abs=t["profit_abs"],
            exit_reason=t.get("exit_reason"), n_entries=len(entries), first_entry_price=p0,
            adverse_move=adverse,
        ))
    df = pd.DataFrame(rows)
    df["open_date"] = pd.to_datetime(df["open_date"], utc=True)
    df["close_date"] = pd.to_datetime(df["close_date"], utc=True)
    df["lose"] = df["profit_abs"] < 0
    df["stuck10"] = df["days"] > 10
    df["deep20"] = df["adverse_move"] >= 0.20
    df["trouble"] = df["lose"] | df["deep20"]
    return df


def read_daily(data_root, pair):
    key = file_key(pair)
    hits = glob.glob(os.path.join(data_root, "**", f"{key}-1d-futures.feather"), recursive=True)
    if not hits:
        return None
    df = pd.read_feather(hits[0])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    return df.drop_duplicates("date").sort_values("date").set_index("date")


def build_features(d, btc_close, history_start):
    c = d["close"].astype(float)
    r = c.pct_change()
    f = pd.DataFrame(index=d.index)
    f["ret_7d"] = c / c.shift(7) - 1
    f["ret_30d"] = c / c.shift(30) - 1
    f["ret_90d"] = c / c.shift(90) - 1
    f["vol_30d"] = np.log(c).diff().rolling(30, min_periods=20).std()
    f["worst_day_30d"] = r.rolling(30, min_periods=20).min()
    f["best_day_30d"] = r.rolling(30, min_periods=20).max()
    f["big_move_days_30d"] = (r.abs() > 0.20).astype(float).rolling(30, min_periods=20).sum()
    f["range_14d"] = ((d["high"] - d["low"]) / c).rolling(14, min_periods=10).mean()
    f["dist_high_90d"] = c / d["high"].rolling(90, min_periods=30).max() - 1
    f["dist_low_90d"] = c / d["low"].rolling(90, min_periods=30).min() - 1
    f["dollar_vol_30d"] = (c * d["volume"]).rolling(30, min_periods=20).mean()   # relative size only
    f["vol_spike_3v90"] = d["volume"].rolling(3).mean() / d["volume"].rolling(90, min_periods=30).mean()
    first = d.index[0]
    f["age_days"] = (d.index - first).days
    f["age_truncated"] = first <= (history_start + pd.Timedelta(days=3))   # listed before our data starts
    b = btc_close.reindex(d.index)
    f["btc_ret_7d"] = b / b.shift(7) - 1
    f["btc_ret_30d"] = b / b.shift(30) - 1
    f["rel_ret_30d"] = f["ret_30d"] - f["btc_ret_30d"]
    f["corr_btc_60d"] = r.rolling(60, min_periods=30).corr(b.pct_change())
    return f


def auc_from(ranks, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 3 or n0 < 3:
        return np.nan
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def auc_table(df, target, features, n_perm=400, seed=0):
    """AUC per feature (P(troubled trade has a higher value)) + permutation max-null threshold."""
    y_all = df[target].to_numpy(bool)
    masks, ranks = {}, {}
    for f in features:
        m = df[f].notna().to_numpy()
        masks[f] = m
        ranks[f] = df.loc[m, f].rank(method="average").to_numpy()
    obs = {f: auc_from(ranks[f], y_all[masks[f]]) for f in features}
    rng = np.random.default_rng(seed)
    mx = []
    for _ in range(n_perm):
        yp = rng.permutation(y_all)
        vals = [abs(auc_from(ranks[f], yp[masks[f]]) - 0.5) for f in features]
        vals = [v for v in vals if not np.isnan(v)]
        mx.append(max(vals) if vals else 0.0)
    thr = float(np.quantile(mx, 0.95)) if mx else np.nan
    rows = []
    for f in features:
        a = obs[f]
        if np.isnan(a):
            continue
        m = masks[f]
        rows.append(dict(feature=f, auc=a, dev=abs(a - 0.5),
                         med_bad=float(df.loc[m & y_all, f].median()),
                         med_ok=float(df.loc[m & ~y_all, f].median())))
    t = pd.DataFrame(rows).sort_values("dev", ascending=False).reset_index(drop=True)
    return t, thr


def md_table(df, floatfmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- commands
def cmd_pairs(args):
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    pairs = set([BTC])
    for r in runs:
        pairs.update(read_pairlist(args.artifacts, r))
    out = sorted(pairs)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out)} pairs (incl. BTC) -> {args.out}")


def cmd_analyze(args):
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    os.makedirs(args.out, exist_ok=True)
    history_start = pd.Timestamp(args.history_start, tz="UTC")

    trades = pd.concat([load_trades(args.artifacts, r) for r in runs], ignore_index=True)
    print(f"trades: {len(trades)} from runs {runs}")

    btc = read_daily(args.data, BTC)
    if btc is None:
        raise SystemExit("BTC daily candles missing -- the download step failed?")
    btc_close = btc["close"].astype(float)

    feats, missing = {}, []
    for pair in sorted(trades["pair"].unique()):
        d = read_daily(args.data, pair)
        if d is None or len(d) < 40:
            missing.append(pair)
            continue
        feats[pair] = build_features(d, btc_close, history_start)

    # all pairs of the pairlists go into the daily file, traded or not
    daily = []
    for r in runs:
        for pair in read_pairlist(args.artifacts, r):
            if pair in feats or pair in missing:
                continue
            d = read_daily(args.data, pair)
            if d is None or len(d) < 40:
                missing.append(pair)
                continue
            feats[pair] = build_features(d, btc_close, history_start)
    for pair, f in feats.items():
        x = f.copy()
        x.insert(0, "pair", pair)
        daily.append(x.reset_index())
    dd = pd.concat(daily, ignore_index=True)
    keep_from = trades["open_date"].min().normalize() - pd.Timedelta(days=3)   # keep the file small
    dd = dd[dd["date"] >= keep_from]
    dd.to_csv(os.path.join(args.out, "pair_features_daily.csv.gz"), index=False,
              compression="gzip", float_format="%.5g")

    # join: features of the last fully closed daily candle BEFORE the entry day
    trades["feat_date"] = trades["open_date"].dt.normalize() - pd.Timedelta(days=1)
    rows = []
    for _, t in trades.iterrows():
        f = feats.get(t["pair"])
        if f is None or t["feat_date"] not in f.index:
            rows.append({k: np.nan for k in FEATURES + ["age_truncated"]})
        else:
            rows.append(f.loc[t["feat_date"], FEATURES + ["age_truncated"]].to_dict())
    tf = pd.concat([trades.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    tf["base"] = tf["pair"].str.split("/").str[0]
    tf["asset_type"] = np.where(tf["base"].isin(STOCK_BASES), "stock_etf", "crypto")
    tf.to_csv(os.path.join(args.out, "trades_with_features.csv"), index=False)

    ok = tf.dropna(subset=["ret_30d", "vol_30d"]).copy()
    lines = []
    w = lines.append
    w("# Pair features vs trade outcomes")
    w("")
    w(f"Runs: {', '.join(runs)} (first = discovery, second = out-of-sample).")
    w(f"Trades: {len(tf)}; with usable features: {len(ok)}. Pairs without daily data: "
      f"{', '.join(missing) if missing else 'none'}.")
    w("Features use only the last fully closed daily candle before the entry day.")
    w("")
    w("Targets: `lose` (closed with a loss), `stuck10` (open more than 10 days), `deep20` "
      "(price moved 20%+ against the FIRST entry), `trouble` (lose or deep20).")
    w("")
    w("## How many troubled trades")
    cnt = ok.groupby("run").agg(trades=("pair", "count"), lose=("lose", "sum"), stuck10=("stuck10", "sum"),
                                deep20=("deep20", "sum"), trouble=("trouble", "sum")).reset_index()
    w(md_table(cnt))
    w("")
    w("## Asset type")
    at = ok.groupby("asset_type").agg(trades=("pair", "count"), trouble=("trouble", "sum"),
                                      pnl=("profit_abs", "sum")).reset_index()
    w(md_table(at))
    w("")

    for target in ["trouble", "lose", "stuck10"]:
        n_bad = int(ok[target].sum())
        w(f"## Target `{target}` -- all runs together ({n_bad} of {len(ok)})")
        if n_bad < 6:
            w("Too few cases for a comparison.")
            w("")
            continue
        t, thr = auc_table(ok, target, FEATURES)
        w(f"AUC 0.5 = no difference. A feature is only convincing if |AUC-0.5| is above "
          f"**{thr:.3f}** (the 95% level of the best chance result among {len(FEATURES)} features).")
        w("")
        w(md_table(t.head(8)))
        w("")
        # discovery -> out-of-sample
        if len(runs) >= 2:
            a = ok[ok["run"] == runs[0]]
            b = ok[ok["run"] == runs[1]]
            if int(a[target].sum()) >= 5 and int(b[target].sum()) >= 3:
                ta, _ = auc_table(a, target, FEATURES, n_perm=0)
                tb, _ = auc_table(b, target, FEATURES, n_perm=0)
                m = ta.head(6).merge(tb[["feature", "auc"]], on="feature", how="left", suffixes=("_first", "_second"))
                m["same_side"] = np.where((m["auc_first"] - 0.5) * (m["auc_second"] - 0.5) > 0, "yes", "no")
                w(f"Top features of run {runs[0]} checked on run {runs[1]}:")
                w("")
                w(md_table(m[["feature", "auc_first", "auc_second", "same_side"]]))
                w("")
            else:
                w(f"Run-to-run check skipped (too few cases: {int(a[target].sum())} / {int(b[target].sum())}).")
                w("")

    w("## Losing trades: where each feature stands (percentile among ALL trades, 0-100)")
    pc = ok.copy()
    for f in ["vol_30d", "worst_day_30d", "ret_30d", "dist_high_90d", "vol_spike_3v90", "age_days", "dollar_vol_30d"]:
        pc[f] = (pc[f].rank(pct=True) * 100).round(0)
    L = pc[pc["lose"]].sort_values("profit_abs")
    cols = ["run", "pair", "is_short", "profit_abs", "days", "vol_30d", "worst_day_30d", "ret_30d",
            "dist_high_90d", "vol_spike_3v90", "age_days", "dollar_vol_30d"]
    L = L[cols].copy()
    L["pair"] = L["pair"].str.replace("/USDT:USDT", "", regex=False)
    L["profit_abs"] = L["profit_abs"].round(0)
    L["days"] = L["days"].round(0)
    w(md_table(L, "{:.0f}"))
    w("")
    w("age_days is capped by the start of the downloaded data; pairs listed before it all look equally old.")
    w("")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--artifacts", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--data", required=True)
    a.add_argument("--artifacts", required=True)
    a.add_argument("--runs", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--history-start", default="20240101")
    args = ap.parse_args()
    {"pairs": cmd_pairs, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
