#!/usr/bin/env python3
"""
dip_scan.py - read-only diagnostic. Save in the repo root.

Question it answers: on each day, were there sharp 5-minute dips (RSI(3) < 10 / < 20)
on the futures pairs the bots trade? If dips happened but the bot did not enter,
the entry filters are the reason. If no dips happened, it is just the market.

It downloads 5m candles from OKX (ccxt, public data, no keys), computes RSI(3),
and prints one row per UTC day.

Usage: python dip_scan.py --pairs-dir /tmp/dip/pl --start 20261006 --out /tmp/dip/out
"""
import argparse
import datetime as dt
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd

PAIR_RE = re.compile(r"[A-Z0-9]{1,20}/USDT:USDT")
BTC = "BTC/USDT:USDT"
STEP_MS = 5 * 60 * 1000


def load_pairs(path):
    pairs = []
    if path and os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
            try:
                with open(f, errors="ignore") as fh:
                    pairs += PAIR_RE.findall(fh.read())
            except Exception:
                continue
    return sorted(set(pairs))


def top_pairs(ex, n=100):
    ex.load_markets()
    tk = ex.fetch_tickers()
    rows = []
    for sym, t in tk.items():
        if sym.endswith("/USDT:USDT") and t.get("quoteVolume"):
            rows.append((t["quoteVolume"], sym))
    rows.sort(reverse=True)
    return [s for _, s in rows[:n]]


def wilder_rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(ad == 0) & (au > 0)] = 100.0
    return out


def fetch_5m(ex, sym, start_ms, end_ms):
    rows, cur, guard = [], start_ms, 0
    while cur < end_ms and guard < 500:
        guard += 1
        batch = None
        for attempt in range(3):
            try:
                batch = ex.fetch_ohlcv(sym, "5m", since=cur, limit=100)
                break
            except Exception as e:  # noqa
                print(f"  retry {sym} ({e.__class__.__name__})", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        if not batch:
            break
        rows += batch
        last = batch[-1][0]
        if last < cur:
            break
        cur = last + STEP_MS
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df = df.drop_duplicates("t").sort_values("t")
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df


def scan(frames, report_start):
    """frames: {pair: DataFrame(o,h,l,c,v) with UTC index}. Returns daily table."""
    series = {}
    for sym, df in frames.items():
        r = wilder_rsi(df["c"], 3)
        series[sym] = r
    wide = pd.concat(series, axis=1)
    wide = wide[wide.index >= report_start]
    lt10 = wide < 10
    lt20 = wide < 20
    day = wide.index.floor("D")
    res = pd.DataFrame(
        {
            "candles_lt10": lt10.sum(axis=1).groupby(day).sum(),
            "pairs_lt10": lt10.groupby(day).any().sum(axis=1),
            "max_pairs_same_candle_lt10": lt10.sum(axis=1).groupby(day).max(),
            "candles_lt20": lt20.sum(axis=1).groupby(day).sum(),
            "pairs_lt20": lt20.groupby(day).any().sum(axis=1),
        }
    )
    if BTC in frames:
        b = frames[BTC]
        b = b[b.index >= report_start]
        g = b.groupby(b.index.floor("D"))
        res["btc_low"] = g["l"].min()
        res["btc_high"] = g["h"].max()
        res["btc_range_pct"] = ((res["btc_high"] / res["btc_low"] - 1) * 100).round(2)
    res.index = res.index.strftime("%Y-%m-%d")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", default="")
    ap.add_argument("--start", default="20261006", help="YYYYMMDD (UTC)")
    ap.add_argument("--out", default="/tmp/dip/out")
    a = ap.parse_args()

    import ccxt

    ex = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    pairs = load_pairs(a.pairs_dir)
    if pairs:
        print(f"pairs from pairlist artifact: {len(pairs)}")
    else:
        print("no pairlist found -> using top 100 USDT swaps by volume today")
        pairs = top_pairs(ex, 100)
    if BTC not in pairs:
        pairs.append(BTC)

    report_start = pd.Timestamp(dt.datetime.strptime(a.start, "%Y%m%d"), tz="UTC")
    fetch_start = report_start - pd.Timedelta(days=1)  # warm-up for RSI
    end = pd.Timestamp.now(tz="UTC")
    s_ms, e_ms = int(fetch_start.timestamp() * 1000), int(end.timestamp() * 1000)
    expected = (e_ms - s_ms) // STEP_MS

    frames, thin = {}, []
    for i, sym in enumerate(pairs, 1):
        df = fetch_5m(ex, sym, s_ms, e_ms)
        if df is None or len(df) < 50:
            thin.append((sym, 0))
            continue
        frames[sym] = df
        if len(df) < 0.9 * expected:
            thin.append((sym, len(df)))
        if i % 10 == 0:
            print(f"  {i}/{len(pairs)} pairs done")

    os.makedirs(a.out, exist_ok=True)
    res = scan(frames, report_start)
    res.to_csv(os.path.join(a.out, "dip_scan_daily.csv"))

    lines = [
        f"pairs used: {len(frames)} of {len(pairs)}; expected candles per pair: {expected}",
        f"pairs with incomplete data (new listings or gaps): {len(thin)}",
        "",
        res.to_string(),
    ]
    text = "\n".join(lines)
    print(text)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a") as fh:
            fh.write("```\n" + text + "\n```\n")


if __name__ == "__main__":
    main()
