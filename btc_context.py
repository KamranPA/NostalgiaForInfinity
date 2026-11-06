#!/usr/bin/env python3
"""
btc_context.py

Question it answers: did the BAD trades (deep grinders / big losers) tend to
open while Bitcoin was falling or below its 200-day average, and how strongly
did the coin move with Bitcoin before entry?

INPUT
  <root>/<run_id>/...   = what `gh run download <run_id> --name backtest-<run_id>`
                          produces (it contains backtest_results/*.zip).
  Bitcoin daily closes   = downloaded from OKX with ccxt (BTC/USDT:USDT, 1d),
                          or a CSV passed with --btc-csv (Date + Close columns,
                          e.g. Yahoo Finance "BTC-USD" history).

WHAT IT MEASURES (per LONG trade, using only candles COMPLETED before the entry
day, so there is no look-ahead):
  - BTC return over the previous 3 / 7 days
  - BTC above / below its 200-day EMA
  - correlation + beta of the coin's daily returns vs BTC over the previous
    60 days (optional, needs the coin's daily candles; skipped with --no-coin-corr)

A trade is called BAD when it lost more than 5 USDT or needed 5+ entries
(grind/rebuy fills); GOOD when it made money with fewer than 5 entries.
Samples are small (about a dozen bad trades per window): read the output as
a hint, not as proof.

USAGE
  python btc_context.py bt --out report.txt
  python btc_context.py bt --btc-csv btc.csv --no-coin-corr --out report.txt
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys
import time
import zipfile
from datetime import datetime, timezone, timedelta

DAY = 86400
BAD_LOSS_USDT = -5.0
BAD_MIN_ENTRIES = 5

# Trades that are open right now on the live bots (UTC), for context only.
LIVE_TRADES = [
    ("HUMA/USDT:USDT", "2026-11-04 14:00"),
    ("SNXX/USDT:USDT", "2026-11-04 15:45"),
    ("SOXL/USDT:USDT", "2026-10-29 23:00"),
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
# Daily candles
# ----------------------------------------------------------------------

def day_floor(ts):
    return ts - (ts % DAY)


def read_btc_csv(path):
    series = {}
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    ci = header.index("close") if "close" in header else 1
    for r in rows[1:]:
        if len(r) <= ci or not r[0].strip():
            continue
        try:
            d = datetime.fromisoformat(r[0].strip()[:10]).replace(tzinfo=timezone.utc)
            series[int(d.timestamp())] = float(r[ci].replace(",", ""))
        except ValueError:
            continue
    return series


def fetch_daily(exchange, symbol, since_ts, until_ts):
    """{day_start_ts: close}. Paginates forward; stops when nothing new."""
    series = {}
    since = since_ts * 1000
    for _ in range(40):
        try:
            candles = exchange.fetch_ohlcv(symbol, "1d", since=since, limit=100)
        except Exception as e:
            print(f"  (fetch_ohlcv failed for {symbol}: {e})", file=sys.stderr)
            break
        if not candles:
            break
        new = 0
        for ts, _o, _h, _l, c, _v in candles:
            d = int(ts / 1000)
            if d not in series:
                series[d] = float(c)
                new += 1
        last = int(candles[-1][0] / 1000)
        if new == 0 or last >= until_ts:
            break
        since = (last + DAY) * 1000
    return series


def sorted_series(series):
    days = sorted(series)
    return days, [series[d] for d in days]


def ema(values, span):
    alpha = 2.0 / (span + 1)
    out, e = [], None
    for v in values:
        e = v if e is None else alpha * v + (1 - alpha) * e
        out.append(e)
    return out


class Btc:
    def __init__(self, series):
        self.days, self.closes = sorted_series(series)
        self.index = {d: i for i, d in enumerate(self.days)}
        self.ema200 = ema(self.closes, 200)

    def last_completed(self, ts):
        """Index of the last daily candle that closed before `ts`."""
        target = day_floor(ts) - DAY
        # walk back to the nearest available day (gaps / weekends safe)
        for back in range(0, 6):
            i = self.index.get(target - back * DAY)
            if i is not None:
                return i
        return None

    def context(self, ts):
        i = self.last_completed(ts)
        if i is None:
            return None
        c = self.closes
        r3 = c[i] / c[i - 3] - 1 if i >= 3 else None
        r7 = c[i] / c[i - 7] - 1 if i >= 7 else None
        below = (c[i] < self.ema200[i]) if i >= 199 else None
        return {"r3": r3, "r7": r7, "below_ema": below, "idx": i}


def corr_beta(coin, btc, i_btc, window=60):
    """Correlation and beta of the coin's daily returns vs BTC's over the
    `window` days up to BTC index i_btc (matched on the same day)."""
    if not coin or i_btc is None or i_btc < window:
        return None, None
    rc, rb = [], []
    for k in range(i_btc - window + 1, i_btc + 1):
        d0, d1 = btc.days[k - 1], btc.days[k]
        if d0 in coin and d1 in coin and coin[d0] > 0:
            rc.append(coin[d1] / coin[d0] - 1)
            rb.append(btc.closes[k] / btc.closes[k - 1] - 1)
    if len(rc) < 30:
        return None, None
    mc, mb = st.fmean(rc), st.fmean(rb)
    cov = sum((a - mc) * (b - mb) for a, b in zip(rc, rb)) / (len(rc) - 1)
    vc = st.pvariance(rc) * len(rc) / (len(rc) - 1)
    vb = st.pvariance(rb) * len(rb) / (len(rb) - 1)
    if vc <= 0 or vb <= 0:
        return None, None
    return cov / math.sqrt(vc * vb), cov / vb


# ----------------------------------------------------------------------
# Report helpers
# ----------------------------------------------------------------------

def pct(x):
    return "n/a" if x is None else f"{x * 100:+.1f}%"


def med(values):
    v = [x for x in values if x is not None]
    return st.median(v) if v else None


def classify(t):
    if t["profit"] < BAD_LOSS_USDT or t["n_entries"] >= BAD_MIN_ENTRIES:
        return "bad"
    if t["profit"] > 0:
        return "good"
    return "other"


def bucket_table(trades, key_fn, labels):
    rows = []
    for label, pred in labels:
        sel = [t for t in trades if t.get("ctx") and pred(t)]
        bad = sum(1 for t in sel if t["cls"] == "bad")
        prof = sum(t["profit"] for t in sel)
        rate = f"{bad / len(sel) * 100:.0f}%" if sel else "-"
        rows.append(f"  {label:<26} trades {len(sel):>4}   bad {bad:>3} ({rate:>4})   profit {prof:+9.1f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with one sub-folder per downloaded run id")
    ap.add_argument("--out", default="btc_context_report.txt")
    ap.add_argument("--btc-csv", help="BTC daily CSV (Date,...,Close) instead of downloading")
    ap.add_argument("--no-coin-corr", action="store_true")
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
    start = day_floor(min(all_ts)) - 320 * DAY      # enough history for EMA200
    now = int(time.time())

    exchange = None
    if not args.btc_csv or not args.no_coin_corr:
        import ccxt
        exchange = ccxt.okx({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    if args.btc_csv:
        btc_series = read_btc_csv(args.btc_csv)
    else:
        btc_series = fetch_daily(exchange, "BTC/USDT:USDT", start, now)
    if len(btc_series) < 250:
        print(f"Only {len(btc_series)} BTC daily candles - not enough (need ~250+).")
        sys.exit(1)
    btc = Btc(btc_series)
    lines = [f"BTC daily candles: {len(btc.days)}  "
             f"({datetime.fromtimestamp(btc.days[0], timezone.utc):%Y-%m-%d} -> "
             f"{datetime.fromtimestamp(btc.days[-1], timezone.utc):%Y-%m-%d})", ""]

    coin_cache = {}

    def coin_series(pair):
        if args.no_coin_corr or exchange is None:
            return None
        if pair not in coin_cache:
            coin_cache[pair] = fetch_daily(exchange, pair, start, now)
        return coin_cache[pair]

    for run_id, trades in runs.items():
        longs = [t for t in trades if not t["short"]]
        first = min(t["open_ts"] for t in trades)
        last = max(t["open_ts"] for t in trades)
        i0, i1 = btc.last_completed(first), btc.last_completed(last)
        lines.append("=" * 78)
        lines.append(f"RUN {run_id}: {len(trades)} trades ({len(longs)} long), "
                     f"{datetime.fromtimestamp(first, timezone.utc):%Y-%m-%d} -> "
                     f"{datetime.fromtimestamp(last, timezone.utc):%Y-%m-%d}")
        if i0 is not None and i1 is not None:
            lines.append(f"BTC over the window: {pct(btc.closes[i1] / btc.closes[i0] - 1)}")
        lines.append("")

        for t in longs:
            t["ctx"] = btc.context(t["open_ts"])
            t["cls"] = classify(t)
            if t["ctx"]:
                cs = coin_series(t["pair"])
                t["corr"], t["beta"] = corr_beta(cs, btc, t["ctx"]["idx"])
            else:
                t["corr"] = t["beta"] = None

        bad = [t for t in longs if t["cls"] == "bad"]
        good = [t for t in longs if t["cls"] == "good"]
        lines.append(f"Long trades: bad {len(bad)} (loss > {-BAD_LOSS_USDT:.0f} USDT or "
                     f"{BAD_MIN_ENTRIES}+ entries), good {len(good)}")
        for name, grp in (("bad", bad), ("good", good)):
            c = [t["ctx"] for t in grp if t["ctx"]]
            lines.append(f"  {name:<5} median BTC 3d {pct(med([x['r3'] for x in c]))}   "
                         f"7d {pct(med([x['r7'] for x in c]))}   "
                         f"below EMA200: {sum(1 for x in c if x['below_ema'])}/"
                         f"{sum(1 for x in c if x['below_ema'] is not None)}"
                         + ("" if args.no_coin_corr else
                            f"   corr60 {med([t['corr'] for t in grp]) if med([t['corr'] for t in grp]) is None else round(med([t['corr'] for t in grp]), 2)}"
                            f"   beta {med([t['beta'] for t in grp]) if med([t['beta'] for t in grp]) is None else round(med([t['beta'] for t in grp]), 2)}"))
        lines.append("")
        lines.append("By BTC 7-day return before entry:")
        lines += bucket_table(longs, None, [
            ("BTC 7d below -10%", lambda t: t["ctx"]["r7"] is not None and t["ctx"]["r7"] < -0.10),
            ("BTC 7d -10% .. -3%", lambda t: t["ctx"]["r7"] is not None and -0.10 <= t["ctx"]["r7"] < -0.03),
            ("BTC 7d -3% .. +3%", lambda t: t["ctx"]["r7"] is not None and -0.03 <= t["ctx"]["r7"] <= 0.03),
            ("BTC 7d above +3%", lambda t: t["ctx"]["r7"] is not None and t["ctx"]["r7"] > 0.03),
        ])
        lines.append("By BTC vs its 200-day EMA:")
        lines += bucket_table(longs, None, [
            ("BTC below EMA200", lambda t: t["ctx"]["below_ema"] is True),
            ("BTC above EMA200", lambda t: t["ctx"]["below_ema"] is False),
        ])
        lines.append("")
        lines.append("Worst 8 long trades:")
        for t in sorted(longs, key=lambda t: t["profit"])[:8]:
            c = t["ctx"] or {}
            lines.append(f"  {t['pair'].split('/')[0]:<8} tag {t['tag']:<5} {t['profit']:+8.1f} USDT  "
                         f"{t['n_entries']:>3} entries {t['days']:5.1f}d  "
                         f"BTC3d {pct(c.get('r3'))} 7d {pct(c.get('r7'))} "
                         f"belowEMA200={c.get('below_ema')}"
                         + ("" if args.no_coin_corr else
                            f"  corr60={None if t['corr'] is None else round(t['corr'], 2)}"))
        lines.append("")

    lines.append("=" * 78)
    lines.append("TRADES OPEN ON THE LIVE BOTS (context at entry)")
    for pair, when in LIVE_TRADES:
        ts = int(datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp())
        c = btc.context(ts)
        if not c:
            lines.append(f"  {pair} {when}: no BTC data for that date (CSV/ccxt ends earlier)")
            continue
        extra = ""
        if not args.no_coin_corr:
            corr, beta = corr_beta(coin_series(pair), btc, c["idx"])
            extra = f"  corr60={None if corr is None else round(corr, 2)} beta={None if beta is None else round(beta, 2)}"
        lines.append(f"  {pair.split('/')[0]:<6} {when}  BTC3d {pct(c['r3'])} 7d {pct(c['r7'])} "
                     f"belowEMA200={c['below_ema']}{extra}")
    lines.append("")
    lines.append("Reminder: small samples. 'bad' is about a dozen trades per window, so a pattern here "
                 "is a hint to test with a filter backtest, not a proven rule.")

    text = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
