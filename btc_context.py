#!/usr/bin/env python3
"""
btc_context.py  (hourly version)

Question: did the BAD trades (deep grinders / big losers) open while Bitcoin
was already falling over the HOURS before entry? The strategy runs on 5-minute
candles and a typical winner lasts about 3 hours, so daily BTC measures (like a
daily EMA200) are far too slow. This version uses BTC 1-hour candles:

  - BTC return over the previous 1 / 4 / 12 / 24 hours
  - BTC distance below its 24-hour high
  - BTC below its 1h EMA50 (about 2 days) and 1h EMA200 (about 8 days)

Only hourly candles that were COMPLETED before the entry are used, so there is
no look-ahead. The last completed hour can be up to 1 hour old for an entry
that happens late in the hour (the hourly resolution is the limit here).

INPUT
  <root>/<run_id>/...   = what `gh run download <run_id> --name backtest-<run_id>`
                          produces (it contains backtest_results/*.zip).
  BTC 1h closes          = downloaded from OKX with ccxt (BTC/USDT:USDT, 1h),
                          or a CSV passed with --btc-csv (timestamp + close).

A trade is BAD when it lost more than 5 USDT or needed 5+ entries (grind/rebuy
fills); GOOD when it made money with fewer than 5 entries. Samples are small
(roughly a dozen bad trades per window): treat results as a hint, not proof.

USAGE
  python btc_context.py bt --out report.txt
  python btc_context.py bt --btc-csv btc_1h.csv --out report.txt
"""

import argparse
import csv
import glob
import json
import os
import statistics as st
import sys
import time
import zipfile
from datetime import datetime, timezone

HOUR = 3600
BAD_LOSS_USDT = -5.0
BAD_MIN_ENTRIES = 5

# Trades that are open right now on the live bots (UTC), for context only.
LIVE_TRADES = [
    ("HUMA", "2026-11-04 14:00"),
    ("SNXX", "2026-11-04 15:45"),
    ("SOXL", "2026-10-29 23:00"),
]


# ----------------------------------------------------------------------
# Loading backtest trades
# ----------------------------------------------------------------------

def load_trades_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist()
                 if n.endswith(".json") and not n.endswith("_config.json")
                 and not n.endswith(".meta.json")]
        if not names:
            return []
        data = json.loads(z.read(names[0]))
    strat = data.get("strategy", {})
    if not strat:
        return []
    return strat[list(strat)[0]].get("trades", [])


def load_run(run_dir):
    zips = sorted(glob.glob(os.path.join(run_dir, "**", "backtest-result-*.zip"), recursive=True))
    trades = []
    for zp in zips:
        trades.extend(load_trades_from_zip(zp))
    out = []
    for x in trades:
        orders = x.get("orders") or []
        n_entries = sum(1 for o in orders if o.get("ft_is_entry"))
        out.append({
            "pair": x["pair"],
            "open_ts": int(x["open_timestamp"] / 1000),
            "profit": float(x["profit_abs"]),
            "short": bool(x.get("is_short")),
            "tag": str(x.get("enter_tag", "")).strip(),
            "n_entries": max(n_entries, 1),
            "days": float(x.get("trade_duration", 0)) / 1440.0,
        })
    return out


# ----------------------------------------------------------------------
# BTC hourly candles
# ----------------------------------------------------------------------

def hour_floor(ts):
    return ts - (ts % HOUR)


def read_btc_csv(path):
    """CSV with a timestamp/date column first and a 'close' column. Accepts
    'YYYY-MM-DD HH:MM[:SS]' or epoch seconds/milliseconds."""
    series = {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    ci = header.index("close") if "close" in header else 1
    for r in rows[1:]:
        if len(r) <= ci or not r[0].strip():
            continue
        raw = r[0].strip()
        try:
            if raw.replace(".", "", 1).isdigit():
                v = float(raw)
                ts = int(v / 1000) if v > 1e11 else int(v)
            else:
                ts = int(datetime.fromisoformat(raw.replace("T", " ")[:19]).replace(tzinfo=timezone.utc).timestamp())
            series[hour_floor(ts)] = float(r[ci].replace(",", ""))
        except ValueError:
            continue
    return series


def fetch_hourly(exchange, symbol, since_ts, until_ts, max_calls=400):
    """{hour_start_ts: close}. Paginates forward, 100 candles per call."""
    series = {}
    since = since_ts * 1000
    for _ in range(max_calls):
        try:
            candles = exchange.fetch_ohlcv(symbol, "1h", since=since, limit=100)
        except Exception as e:
            print(f"  (fetch_ohlcv failed for {symbol}: {e})", file=sys.stderr)
            break
        if not candles:
            break
        new = 0
        for ts, _o, _h, _l, c, _v in candles:
            h = int(ts / 1000)
            if h not in series:
                series[h] = float(c)
                new += 1
        last = int(candles[-1][0] / 1000)
        if new == 0 or last >= until_ts:
            break
        since = (last + HOUR) * 1000
    return series


def ema(values, span):
    alpha = 2.0 / (span + 1)
    out, e = [], None
    for v in values:
        e = v if e is None else alpha * v + (1 - alpha) * e
        out.append(e)
    return out


class Btc:
    def __init__(self, series):
        self.hours = sorted(series)
        self.closes = [series[h] for h in self.hours]
        self.index = {h: i for i, h in enumerate(self.hours)}
        self.ema50 = ema(self.closes, 50)
        self.ema200 = ema(self.closes, 200)

    def last_completed(self, ts):
        """Index of the last hourly candle that CLOSED at or before `ts`."""
        target = hour_floor(ts) - HOUR
        for back in range(0, 6):
            i = self.index.get(target - back * HOUR)
            if i is not None:
                return i
        return None

    def context(self, ts):
        i = self.last_completed(ts)
        if i is None or i < 24:
            return None
        c = self.closes
        hi24 = max(c[i - 23:i + 1])
        return {
            "r1": c[i] / c[i - 1] - 1,
            "r4": c[i] / c[i - 4] - 1,
            "r12": c[i] / c[i - 12] - 1,
            "r24": c[i] / c[i - 24] - 1,
            "dd24": c[i] / hi24 - 1,
            "below50": (c[i] < self.ema50[i]) if i >= 50 else None,
            "below200": (c[i] < self.ema200[i]) if i >= 200 else None,
        }


# ----------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def med(values):
    v = [x for x in values if x is not None]
    return st.median(v) if v else None


def classify(t):
    if t["profit"] < BAD_LOSS_USDT or t["n_entries"] >= BAD_MIN_ENTRIES:
        return "bad"
    if t["profit"] > 0:
        return "good"
    return "other"


def bucket_rows(trades, buckets):
    rows = []
    for label, pred in buckets:
        sel = [t for t in trades if t["ctx"] and pred(t["ctx"])]
        bad = sum(1 for t in sel if t["cls"] == "bad")
        prof = sum(t["profit"] for t in sel)
        rate = f"{bad / len(sel) * 100:.0f}%" if sel else "-"
        rows.append(f"  {label:<26} trades {len(sel):>4}   bad {bad:>3} ({rate:>4})   profit {prof:+9.1f}")
    return rows


def flag(v):
    return "n/a" if v is None else ("below" if v else "above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with one sub-folder per downloaded run id")
    ap.add_argument("--out", default="btc_context_report.txt")
    ap.add_argument("--btc-csv", help="BTC hourly CSV instead of downloading from OKX")
    args = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if os.path.isdir(d):
            tr = load_run(d)
            if tr:
                runs[os.path.basename(d)] = tr
    if not runs:
        print("No trades found under", args.root)
        sys.exit(1)

    all_ts = [t["open_ts"] for tr in runs.values() for t in tr]
    live_ts = [int(datetime.strptime(w, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())
               for _, w in LIVE_TRADES]
    start = hour_floor(min(all_ts + live_ts)) - 15 * 24 * HOUR   # warm-up for EMA200 (1h)
    now = int(time.time())

    if args.btc_csv:
        series = read_btc_csv(args.btc_csv)
    else:
        import ccxt
        exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        series = fetch_hourly(exchange, "BTC/USDT:USDT", start, now)
    if len(series) < 500:
        print(f"Only {len(series)} BTC hourly candles - not enough.")
        sys.exit(1)
    btc = Btc(series)

    first_h, last_h = btc.hours[0], btc.hours[-1]
    expected = (last_h - first_h) // HOUR + 1
    lines = [f"BTC 1h candles: {len(btc.hours)} of ~{expected} expected  "
             f"({datetime.fromtimestamp(first_h, timezone.utc):%Y-%m-%d %H:%M} -> "
             f"{datetime.fromtimestamp(last_h, timezone.utc):%Y-%m-%d %H:%M} UTC)", ""]

    r4_buckets = [
        ("BTC 4h below -1.5%", lambda c: c["r4"] < -0.015),
        ("BTC 4h -1.5% .. -0.5%", lambda c: -0.015 <= c["r4"] < -0.005),
        ("BTC 4h -0.5% .. +0.5%", lambda c: -0.005 <= c["r4"] <= 0.005),
        ("BTC 4h above +0.5%", lambda c: c["r4"] > 0.005),
    ]
    r24_buckets = [
        ("BTC 24h below -3%", lambda c: c["r24"] < -0.03),
        ("BTC 24h -3% .. -1%", lambda c: -0.03 <= c["r24"] < -0.01),
        ("BTC 24h -1% .. +1%", lambda c: -0.01 <= c["r24"] <= 0.01),
        ("BTC 24h above +1%", lambda c: c["r24"] > 0.01),
    ]
    ema_buckets = [
        ("BTC below 1h EMA50", lambda c: c["below50"] is True),
        ("BTC above 1h EMA50", lambda c: c["below50"] is False),
        ("BTC below 1h EMA200", lambda c: c["below200"] is True),
        ("BTC above 1h EMA200", lambda c: c["below200"] is False),
    ]
    r1_buckets = [
        ("BTC fell >0.5% last hour", lambda c: c["r1"] < -0.005),
        ("BTC otherwise", lambda c: c["r1"] >= -0.005),
    ]

    for run_id, trades in runs.items():
        longs = [t for t in trades if not t["short"]]
        first = min(t["open_ts"] for t in trades)
        last = max(t["open_ts"] for t in trades)
        lines.append("=" * 78)
        lines.append(f"RUN {run_id}: {len(trades)} trades ({len(longs)} long), "
                     f"{datetime.fromtimestamp(first, timezone.utc):%Y-%m-%d} -> "
                     f"{datetime.fromtimestamp(last, timezone.utc):%Y-%m-%d}")
        i0, i1 = btc.last_completed(first), btc.last_completed(last)
        if i0 is not None and i1 is not None:
            lines.append(f"BTC over the window: {pct(btc.closes[i1] / btc.closes[i0] - 1)}")
        lines.append("")

        for t in longs:
            t["ctx"] = btc.context(t["open_ts"])
            t["cls"] = classify(t)
        usable = [t for t in longs if t["ctx"]]
        bad = [t for t in usable if t["cls"] == "bad"]
        good = [t for t in usable if t["cls"] == "good"]
        lines.append(f"Long trades with BTC context: {len(usable)}  |  bad {len(bad)} "
                     f"(loss > {-BAD_LOSS_USDT:.0f} USDT or {BAD_MIN_ENTRIES}+ entries)  |  good {len(good)}")
        for name, grp in (("bad", bad), ("good", good)):
            c = [t["ctx"] for t in grp]
            below50 = [x["below50"] for x in c if x["below50"] is not None]
            lines.append(f"  {name:<5} median BTC 1h {pct(med([x['r1'] for x in c]))}  "
                         f"4h {pct(med([x['r4'] for x in c]))}  12h {pct(med([x['r12'] for x in c]))}  "
                         f"24h {pct(med([x['r24'] for x in c]))}  below24hHigh {pct(med([x['dd24'] for x in c]))}  "
                         f"below EMA50(1h): {sum(1 for v in below50 if v)}/{len(below50)}")
        lines.append("")
        lines.append("By BTC 4-hour return before entry:")
        lines += bucket_rows(usable, r4_buckets)
        lines.append("By BTC 24-hour return before entry:")
        lines += bucket_rows(usable, r24_buckets)
        lines.append("By BTC last-hour move:")
        lines += bucket_rows(usable, r1_buckets)
        lines.append("By BTC vs hourly EMAs:")
        lines += bucket_rows(usable, ema_buckets)
        lines.append("")
        lines.append("Worst 8 long trades:")
        for t in sorted(usable, key=lambda t: t["profit"])[:8]:
            c = t["ctx"]
            lines.append(f"  {t['pair'].split('/')[0]:<8} tag {t['tag']:<5} {t['profit']:+8.1f} USDT  "
                         f"{t['n_entries']:>3} entries {t['days']:5.1f}d  "
                         f"BTC 1h {pct(c['r1'])} 4h {pct(c['r4'])} 24h {pct(c['r24'])} "
                         f"EMA50 {flag(c['below50'])} EMA200 {flag(c['below200'])}")
        lines.append("")

    lines.append("=" * 78)
    lines.append("TRADES OPEN ON THE LIVE BOTS (BTC context at entry)")
    for name, when in LIVE_TRADES:
        ts = int(datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())
        c = btc.context(ts)
        if not c:
            lines.append(f"  {name} {when}: no BTC data for that time")
            continue
        lines.append(f"  {name:<6} {when}  BTC 1h {pct(c['r1'])} 4h {pct(c['r4'])} 12h {pct(c['r12'])} "
                     f"24h {pct(c['r24'])}  below24hHigh {pct(c['dd24'])}  "
                     f"EMA50 {flag(c['below50'])} EMA200 {flag(c['below200'])}")
    lines.append("")
    lines.append("Reminder: small samples ('bad' is a handful of trades per window). A pattern here "
                 "is a hint to test with a filter backtest, not a proven rule. Hourly candles are the "
                 "finest resolution used; very fast 5-minute shocks inside an hour are not visible.")

    text = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
