#!/usr/bin/env python3
"""
analyze_long_trades.py -- read-only deep dive on every trade that stayed open
longer than N days (default 2).

Inputs (any mix, any number):
  *.zip / *.json   freqtrade backtest results (the file in user_data/backtest_results)
  *.sqlite         dry-run state files (tradesv3.*.sqlite)

Output: Markdown report on stdout + optional flat CSV (--csv).

  python analyze_long_trades.py [--min-days 2] [--csv out.csv] FILE [FILE ...]

What it shows (only what the data really contains):
  * per-trade: every entry/DCA/partial-exit (time, price, price change vs first
    entry, cumulative margin), worst & best price excursion (min_rate/max_rate,
    also x leverage), funding fees, exit reason, outcome class
  * aggregates: share of total P&L, outcome classes, "survival" table
    (win rate / P&L of trades that reached 2,3,5,7,10,14,21,30 days),
    by tag family, direction, DCA depth, duration bucket, pair
NOT available from these files: the full price path between entry and exit
(only min/max). Say so if you need a candle-by-candle phase 2.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import statistics as st
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone

NOW = datetime.now(timezone.utc)

TAG_FAMILIES = [
    (1, 13, "Normal"), (21, 26, "Pump"), (41, 53, "Quick"), (61, 68, "Rebuy"),
    (81, 82, "High Profit"), (101, 110, "Rapid"), (120, 120, "Grind"), (121, 121, "BTC"),
    (141, 145, "Top Coins"), (161, 173, "Scalp"),
    (501, 513, "Short Normal"), (521, 526, "Short Pump"), (541, 553, "Short Quick"),
    (561, 568, "Short Rebuy"), (581, 582, "Short High Profit"), (601, 610, "Short Rapid"),
    (620, 620, "Short Grind"), (621, 621, "Short BTC"), (641, 645, "Short Top Coins"),
    (661, 673, "Short Scalp"),
]
BUCKETS = [(2, 3, "2-3d"), (3, 7, "3-7d"), (7, 14, "7-14d"), (14, 21, "14-21d"), (21, 9999, "21d+")]
SURVIVAL = [2, 3, 5, 7, 10, 14, 21, 30]


# ------------------------------------------------------------------ helpers
def parse_dt(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, tz=timezone.utc)
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v).strip()
    try:
        d = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                d = datetime.strptime(s, fmt)
                break
            except ValueError:
                d = None
        if d is None:
            return None
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def fnum(x, default=None):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


def family(tag):
    first = str(tag or "").split()[0] if str(tag or "").split() else ""
    try:
        n = int(first)
    except ValueError:
        return "Other"
    for lo, hi, name in TAG_FAMILIES:
        if lo <= n <= hi:
            return name
    return "Other"


def fd(dt):
    return "—" if dt is None else dt.strftime("%Y-%m-%d %H:%M")


def pct(x, d=2):
    return "—" if x is None else f"{x * 100:+.{d}f}%"


def usd(x):
    return "—" if x is None else f"{x:+.2f}"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ loaders
def norm_bt_trade(t, source):
    lev = fnum(t.get("leverage"), 1.0) or 1.0
    is_short = bool(t.get("is_short"))
    orders = []
    for o in t.get("orders") or []:
        dt = parse_dt(o.get("order_filled_timestamp") or o.get("order_timestamp"))
        if dt is None:
            continue
        is_entry = o.get("ft_is_entry")
        if is_entry is None:
            is_entry = (o.get("ft_order_side") or o.get("side")) == ("sell" if is_short else "buy")
        price = fnum(o.get("safe_price"), fnum(o.get("price")))
        cost = fnum(o.get("cost"))
        if cost is None and price is not None:
            cost = price * (fnum(o.get("amount"), 0.0) or 0.0)
        orders.append({"dt": dt, "side": "entry" if is_entry else "exit", "price": price,
                       "amount": fnum(o.get("amount")), "cost": cost, "tag": o.get("ft_order_tag")})
    return {
        "source": source, "id": t.get("id"), "pair": t.get("pair"), "is_short": is_short,
        "tag": t.get("enter_tag"), "open_dt": parse_dt(t.get("open_date") or t.get("open_timestamp")),
        "close_dt": None if t.get("is_open") else parse_dt(t.get("close_date") or t.get("close_timestamp")),
        "is_open": bool(t.get("is_open")), "open_rate": fnum(t.get("open_rate")),
        "close_rate": fnum(t.get("close_rate")), "min_rate": fnum(t.get("min_rate")),
        "max_rate": fnum(t.get("max_rate")), "leverage": lev,
        "profit_ratio": fnum(t.get("profit_ratio")), "profit_abs": fnum(t.get("profit_abs")),
        "exit_reason": t.get("exit_reason"), "funding": fnum(t.get("funding_fees"), 0.0),
        "stake": fnum(t.get("stake_amount")), "max_stake": fnum(t.get("max_stake_amount")),
        "orders": sorted(orders, key=lambda o: o["dt"]),
    }


def load_backtest(path, label):
    docs = []
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                b = os.path.basename(n).lower()
                if n.endswith(".json") and "config" not in b and "meta" not in b:
                    try:
                        docs.append(json.loads(z.read(n)))
                    except ValueError:
                        pass
    else:
        with open(path) as f:
            docs.append(json.load(f))
    out = []
    for d in docs:
        for strat, body in (d.get("strategy") or {}).items():
            for t in body.get("trades", []):
                out.append(norm_bt_trade(t, f"{label}:{strat}"))
    return out


def load_sqlite(path, label):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out = []
    try:
        trades = [dict(r) for r in conn.execute("SELECT * FROM trades")]
        ocols = {r[1] for r in conn.execute("PRAGMA table_info(orders)")}
        for t in trades:
            is_short = bool(t.get("is_short"))
            entry_side = "sell" if is_short else "buy"
            orders = []
            for o in conn.execute("SELECT * FROM orders WHERE ft_trade_id=? AND order_filled_date IS NOT NULL",
                                  (t["id"],)):
                o = dict(o)
                dt = parse_dt(o.get("order_filled_date"))
                price = fnum(o.get("average"), fnum(o.get("price")))
                cost = fnum(o.get("cost")) if "cost" in ocols else None
                if not cost and price is not None:
                    cost = price * (fnum(o.get("filled"), 0.0) or 0.0)
                orders.append({"dt": dt, "side": "entry" if o.get("ft_order_side") == entry_side else "exit",
                               "price": price, "amount": fnum(o.get("filled")), "cost": cost,
                               "tag": o.get("ft_order_tag")})
            is_open = bool(t.get("is_open"))
            out.append({
                "source": label, "id": t["id"], "pair": t["pair"], "is_short": is_short,
                "tag": t.get("enter_tag"), "open_dt": parse_dt(t.get("open_date")),
                "close_dt": None if is_open else parse_dt(t.get("close_date")), "is_open": is_open,
                "open_rate": fnum(t.get("open_rate")), "close_rate": fnum(t.get("close_rate")),
                "min_rate": fnum(t.get("min_rate")), "max_rate": fnum(t.get("max_rate")),
                "leverage": fnum(t.get("leverage"), 1.0) or 1.0,
                "profit_ratio": None if is_open else fnum(t.get("close_profit")),
                "profit_abs": None if is_open else fnum(t.get("close_profit_abs")),
                "exit_reason": t.get("exit_reason"), "funding": fnum(t.get("funding_fees"), 0.0),
                "stake": fnum(t.get("stake_amount")), "max_stake": None,
                "orders": sorted([o for o in orders if o["dt"]], key=lambda o: o["dt"]),
            })
    finally:
        conn.close()
    return out


# ------------------------------------------------------------------ enrich
def enrich(t):
    end = t["close_dt"] or NOW
    t["dur"] = max((end - t["open_dt"]).total_seconds() / 86400.0, 0.0) if t["open_dt"] else 0.0
    entries = [o for o in t["orders"] if o["side"] == "entry"]
    exits = [o for o in t["orders"] if o["side"] == "exit"]
    t["n_entries"] = max(len(entries), 1)
    t["n_partial_exits"] = max(len(exits) - (0 if t["is_open"] else 1), 0)
    first = (entries[0]["price"] if entries and entries[0]["price"] else t["open_rate"])
    t["first_price"] = first
    lev = t["leverage"]
    mae = mfe = None
    if first and t["min_rate"] and t["max_rate"]:
        if t["is_short"]:
            mae, mfe = 1 - t["max_rate"] / first, 1 - t["min_rate"] / first
        else:
            mae, mfe = t["min_rate"] / first - 1, t["max_rate"] / first - 1
    t["mae_price"], t["mfe_price"] = mae, mfe
    t["mae_margin"] = mae * lev if mae is not None else None
    t["mfe_margin"] = mfe * lev if mfe is not None else None
    t["time_to_first_dca"] = ((entries[1]["dt"] - entries[0]["dt"]).total_seconds() / 86400.0
                              if len(entries) > 1 else None)
    stake0 = (entries[0]["cost"] / lev) if entries and entries[0]["cost"] else t["stake"]
    t["stake0"] = stake0
    cum, mx = 0.0, 0.0
    for o in t["orders"]:
        c = (o["cost"] or 0.0) / lev
        cum += c if o["side"] == "entry" else -c
        mx = max(mx, cum)
        o["cum"] = max(cum, 0.0)
    t["max_margin"] = t["max_stake"] or mx or t["stake"]
    t["stake_growth"] = (t["max_margin"] / stake0) if stake0 else None
    r = (t["exit_reason"] or "")
    if t["is_open"]:
        t["outcome"] = "STILL OPEN"
    elif r.startswith("exit_bad_trade_abandon"):
        t["outcome"] = "ABANDON"
    elif "stop_loss" in r:
        t["outcome"] = "STOP LOSS"
    elif r == "force_exit":
        t["outcome"] = "FORCE EXIT"
    elif (t["profit_abs"] or 0) > 0:
        t["outcome"] = "PROFIT EXIT"
    else:
        t["outcome"] = "LOSS EXIT (other)"
    t["family"] = family(t["tag"])
    return t


# ------------------------------------------------------------------ report
def agg_rows(trades, keyfn, order=None):
    g = defaultdict(list)
    for t in trades:
        g[keyfn(t)].append(t)
    rows = []
    for k in (order or sorted(g, key=lambda x: -len(g[x]))):
        ts = g.get(k)
        if not ts:
            continue
        closed = [t for t in ts if not t["is_open"]]
        wins = sum(1 for t in closed if (t["profit_abs"] or 0) > 0)
        pr = [t["profit_ratio"] for t in closed if t["profit_ratio"] is not None]
        rows.append((k, len(ts), len(ts) - len(closed),
                     f"{wins / len(closed) * 100:.0f}%" if closed else "—",
                     pct(st.mean(pr)) if pr else "—", pct(st.median(pr)) if pr else "—",
                     usd(sum((t["profit_abs"] or 0) for t in closed)),
                     f"{st.mean(t['dur'] for t in ts):.1f}d"))
    return rows


AGG_H = ["group", "trades", "open", "win%", "avg %", "median %", "sum USDT", "avg days"]


def trade_block(i, t):
    lev = t["leverage"]
    d = "SHORT" if t["is_short"] else "LONG"
    L = [f"### {i}. {t['source']} · {t['pair']} · {d} · tag {t['tag']} ({t['family']}) — {t['outcome']}\n"]
    L.append(table(["field", "value"], [
        ("opened → closed", f"{fd(t['open_dt'])} → {fd(t['close_dt']) if t['close_dt'] else 'STILL OPEN'}"),
        ("duration", f"{t['dur']:.2f} days"),
        ("exit reason", t["exit_reason"] or "—"),
        ("result", f"{pct(t['profit_ratio'])} / {usd(t['profit_abs'])} USDT"),
        ("leverage", f"{lev:g}x"),
        ("first entry price", t["first_price"]),
        ("worst excursion (price / margin)", f"{pct(t['mae_price'])} / {pct(t['mae_margin'])}"),
        ("best excursion (price / margin)", f"{pct(t['mfe_price'])} / {pct(t['mfe_margin'])}"),
        ("entries (DCA) / partial exits", f"{t['n_entries']} / {t['n_partial_exits']}"),
        ("time to first DCA", f"{t['time_to_first_dca']:.2f} d" if t["time_to_first_dca"] is not None else "—"),
        ("margin first → max", f"{t['stake0'] or 0:.1f} → {t['max_margin'] or 0:.1f} USDT"
                              + (f" (x{t['stake_growth']:.1f})" if t["stake_growth"] else "")),
        ("funding fees", usd(t["funding"])),
    ]))
    rows = []
    for o in t["orders"]:
        chg = (o["price"] / t["first_price"] - 1) if o["price"] and t["first_price"] else None
        rows.append((fd(o["dt"]), f"{(o['dt'] - t['open_dt']).total_seconds() / 86400:.2f}d",
                     o["side"].upper() + (f" [{o['tag']}]" if o.get("tag") else ""),
                     o["price"], pct(chg), f"{o['cum']:.1f}"))
    if rows:
        L.append("\n" + table(["time", "since open", "order", "price", "vs first entry", "margin after"], rows))
    flags = []
    if t["mfe_margin"] is not None and t["mfe_margin"] > 0.03 and (t["profit_abs"] or 0) <= 0 and not t["is_open"]:
        flags.append(f"was up {pct(t['mfe_margin'])} (margin) before closing at a loss — gave it back")
    if t["mae_margin"] is not None and t["mae_margin"] < -0.5:
        flags.append(f"worst point {pct(t['mae_margin'])} of margin — deep drawdown")
    ent = [o for o in t["orders"] if o["side"] == "entry" and o["price"]]
    if len(ent) >= 3:
        falling = all((ent[k]["price"] < ent[k - 1]["price"]) != t["is_short"] for k in range(1, len(ent)))
        if falling:
            flags.append("every DCA was bought lower (long) / sold higher (short): pure averaging into the move")
    if t["stake_growth"] and t["stake_growth"] >= 3:
        flags.append(f"position grew x{t['stake_growth']:.1f} via DCA")
    if flags:
        L.append("\n**Flags:** " + "; ".join(flags) + "\n")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-days", type=float, default=2.0)
    ap.add_argument("--csv")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()

    all_t = []
    for p in a.files:
        label = os.path.splitext(os.path.basename(p))[0]
        try:
            ts = load_sqlite(p, label) if p.lower().endswith((".sqlite", ".db")) else load_backtest(p, label)
        except Exception as e:  # noqa: BLE001
            print(f"_could not read {p}: {e}_\n")
            continue
        all_t += [enrich(t) for t in ts if t["open_dt"]]

    long_t = [t for t in all_t if t["dur"] > a.min_days]
    print(f"# Trades open longer than {a.min_days:g} days — deep dive\n")

    # 0. inputs
    by_src = defaultdict(list)
    for t in all_t:
        by_src[t["source"]].append(t)
    rows = []
    for s, ts in by_src.items():
        lg = [t for t in ts if t["dur"] > a.min_days]
        tot = sum((t["profit_abs"] or 0) for t in ts if not t["is_open"])
        lgp = sum((t["profit_abs"] or 0) for t in lg if not t["is_open"])
        rows.append((s, len(ts), len(lg), f"{len(lg) / len(ts) * 100:.0f}%" if ts else "—",
                     usd(tot), usd(lgp), usd(tot - lgp)))
    print("## 0. Inputs\n")
    print(table(["source", "all trades", f">{a.min_days:g}d trades", "share", "P&L all (closed)",
                 f"P&L of >{a.min_days:g}d", "P&L of the rest"], rows))
    if not long_t:
        print("_No trade stayed open that long._")
        return

    print("## 1. Outcome classes\n")
    print(table(AGG_H, agg_rows(long_t, lambda t: t["outcome"])))

    print("## 2. Survival: what happens to trades that reached X days\n")
    sr = []
    for x in SURVIVAL:
        ts = [t for t in all_t if t["dur"] >= x]
        if not ts:
            continue
        cl = [t for t in ts if not t["is_open"]]
        w = sum(1 for t in cl if (t["profit_abs"] or 0) > 0)
        sr.append((f">= {x}d", len(ts), len(ts) - len(cl), f"{w / len(cl) * 100:.0f}%" if cl else "—",
                   usd(sum((t["profit_abs"] or 0) for t in cl)),
                   usd(st.mean([(t["profit_abs"] or 0) for t in cl])) if cl else "—"))
    print(table(["reached", "trades", "still open", "win% (closed)", "sum USDT", "avg USDT"], sr))

    print("## 3. Breakdowns (trades > threshold)\n")
    print("**By direction**\n")
    print(table(AGG_H, agg_rows(long_t, lambda t: "SHORT" if t["is_short"] else "LONG")))
    print("**By tag family**\n")
    print(table(AGG_H, agg_rows(long_t, lambda t: t["family"])))
    print("**By DCA depth (entries)**\n")
    print(table(AGG_H, agg_rows(long_t, lambda t: f"{min(t['n_entries'], 6)}{'+' if t['n_entries'] >= 6 else ''} entries",
                                order=[f"{i}{'+' if i == 6 else ''} entries" for i in range(1, 7)])))
    print("**By duration bucket**\n")
    print(table(AGG_H, agg_rows(long_t, lambda t: next((n for lo, hi, n in BUCKETS if lo <= t["dur"] < hi), "21d+"),
                                order=[n for _, _, n in BUCKETS])))
    print("**By pair (worst 15 by sum)**\n")
    pr = agg_rows(long_t, lambda t: t["pair"])
    pr.sort(key=lambda r: float(r[6]))
    print(table(AGG_H, pr[:15]))

    print(f"## 4. Every trade > {a.min_days:g} days (worst first)\n")
    key = lambda t: (t["profit_abs"] if t["profit_abs"] is not None else 1e9)  # noqa: E731
    for i, t in enumerate(sorted(long_t, key=key), 1):
        print(trade_block(i, t))

    if a.csv:
        cols = ["source", "id", "pair", "is_short", "tag", "family", "open_dt", "close_dt", "dur", "outcome",
                "exit_reason", "profit_ratio", "profit_abs", "leverage", "mae_price", "mae_margin", "mfe_price",
                "mfe_margin", "n_entries", "n_partial_exits", "time_to_first_dca", "stake0", "max_margin",
                "stake_growth", "funding"]
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for t in sorted(long_t, key=key):
                w.writerow([t.get(c) for c in cols])


if __name__ == "__main__":
    main()
