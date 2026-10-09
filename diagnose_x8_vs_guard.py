#!/usr/bin/env python3
"""
diagnose_x8_vs_guard.py  -- ONE-SHOT diagnostic (read-only).

Question: X8 futures and X8 Guard futures run the same entry logic, so why
do some entries appear on only one of them?

Reads (never writes):
  - the two SQLite state files
  - config_dryrun_futures_x8.json / config_dryrun_futures_x8guard.json
  - NostalgiaForInfinityX8Guard.py
and prints a Markdown report to stdout:

  1. Config differences (secrets masked)
  2. Guard strategy file (what it overrides)
  3. Entry matching (same pair + direction within 15 min) since Guard start
  4. For EVERY unmatched entry: at that moment, were the other bot's slots
     full? was the same pair already open there? -> verdict + log window
  5. Orders-table status counts (cancelled/unfilled entries leave no
     trade row, so they are invisible here -> that case is a log check)

Usage: python diagnose_x8_vs_guard.py <futures_x8.sqlite> <futures_x8guard.sqlite>
"""

import os
import re
import sys
import json
import sqlite3
from datetime import datetime, timedelta, timezone

CFG_X8 = "config_dryrun_futures_x8.json"
CFG_GUARD = "config_dryrun_futures_x8guard.json"
GUARD_FILE = "NostalgiaForInfinityX8Guard.py"
MATCH_MIN = 15
SECRET = re.compile(r"token|chat_id|password|secret|jwt|api_?key|username|\.key$|\.uid$", re.I)


# ---------------------------------------------------------------- helpers
def parse_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        d = datetime.fromisoformat(v)
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except ValueError:
        return None


def fmt(dt):
    return "—" if dt is None else dt.strftime("%m-%d %H:%M:%S")


def norm(pair):
    return pair.split(":")[0] if pair else pair


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def connect_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def load_trades(path):
    conn = connect_ro(path)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, pair, is_open, is_short, enter_tag, exit_reason, open_date, close_date, "
            "stake_amount, close_profit FROM trades ORDER BY open_date ASC")]
    finally:
        conn.close()
    for r in rows:
        r["is_open"] = bool(r["is_open"])
        r["is_short"] = bool(r["is_short"] or False)
        r["open_date"] = parse_dt(r["open_date"])
        r["close_date"] = parse_dt(r["close_date"])
    return rows


def bot_start(path):
    try:
        conn = connect_ro(path)
        row = conn.execute("SELECT datetime_value FROM KeyValueStore WHERE key='bot_start_time' "
                           "ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return parse_dt(row[0]) if row and row[0] else None
    except sqlite3.Error:
        return None


def order_status_counts(path):
    try:
        conn = connect_ro(path)
        rows = conn.execute("SELECT ft_order_side, status, COUNT(*) FROM orders "
                            "GROUP BY 1,2 ORDER BY 1,2").fetchall()
        conn.close()
        return rows
    except sqlite3.Error as e:
        return [("(orders table unreadable)", str(e), 0)]


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"_could not read {path}: {e}_")
        return {}


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d, sort_keys=True)
    else:
        out[prefix] = d
    return out


def open_at(trades, t, exclude_id=None):
    n = 0
    for x in trades:
        if x["id"] == exclude_id or x["open_date"] is None or x["open_date"] > t:
            continue
        if x["is_open"] or x["close_date"] is None or x["close_date"] > t:
            n += 1
    return n


def pair_state(trades, pair, t):
    """Other bot's trades on the same pair (any direction) around time t."""
    same = [x for x in trades if norm(x["pair"]) == norm(pair) and x["open_date"]]
    open_now = [x for x in same if x["open_date"] <= t and
                (x["is_open"] or x["close_date"] is None or x["close_date"] > t)]
    if open_now:
        x = open_now[0]
        return "OPEN", f"open since {fmt(x['open_date'])} (tag {x['enter_tag']})"
    if same:
        x = min(same, key=lambda y: abs((y["open_date"] - t).total_seconds()))
        return "NEAR", (f"nearest: opened {fmt(x['open_date'])}, "
                        f"closed {fmt(x['close_date']) if x['close_date'] else 'still open'}")
    return "NONE", "never traded this pair"


# ---------------------------------------------------------------- main
def main(x8_path, guard_path):
    cfg8, cfgg = load_json(CFG_X8), load_json(CFG_GUARD)
    max8 = int(cfg8.get("max_open_trades", 8) or 8)
    maxg = int(cfgg.get("max_open_trades", 8) or 8)

    print("# X8 vs X8 Guard — why do entries differ?\n")

    # 1. config diff
    print("## 1. Config differences (secrets masked)\n")
    f8, fg = flatten(cfg8), flatten(cfgg)
    rows = []
    for k in sorted(set(f8) | set(fg)):
        a, b = f8.get(k, "<absent>"), fg.get(k, "<absent>")
        if a != b:
            if SECRET.search(k):
                a, b = "***", "***"
            rows.append((k, a, b))
    print(table(["key", "X8", "Guard"], rows))
    print(f"`max_open_trades`: X8={max8}, Guard={maxg}\n")

    # 2. guard file
    print("## 2. Guard strategy file\n")
    if os.path.exists(GUARD_FILE):
        src = open(GUARD_FILE, encoding="utf-8", errors="replace").read()
        print(f"{len(src.splitlines())} lines. Methods/attrs it defines:\n")
        keys = [l.strip() for l in src.splitlines()
                if re.match(r"\s*(class |def |[A-Za-z_][A-Za-z0-9_]*\s*[:=])", l) and not l.strip().startswith("#")]
        print("```\n" + "\n".join(keys[:80]) + "\n```\n")
        print("<details><summary>full file</summary>\n\n```python\n" + src + "\n```\n</details>\n")
    else:
        print(f"_{GUARD_FILE} not found in repo root_\n")

    # 3. trades / matching
    t8, tg = load_trades(x8_path), load_trades(guard_path)
    start = bot_start(guard_path) or (min((x["open_date"] for x in tg if x["open_date"]), default=None))
    print("## 3. Entry matching\n")
    if start is None:
        print("_Guard has no start time or trades yet._")
        return
    w8 = [x for x in t8 if x["open_date"] and x["open_date"] >= start]
    wg = [x for x in tg if x["open_date"] and x["open_date"] >= start]
    print(f"Window start (Guard bot_start_time): **{fmt(start)} UTC**. "
          f"Entries in window: X8={len(w8)}, Guard={len(wg)}.\n")

    cands = []
    for i, a in enumerate(w8):
        for j, b in enumerate(wg):
            if norm(a["pair"]) == norm(b["pair"]) and a["is_short"] == b["is_short"]:
                gap = abs((a["open_date"] - b["open_date"]).total_seconds())
                if gap <= MATCH_MIN * 60:
                    cands.append((gap, i, j))
    cands.sort()
    ua, ub, matched = set(), set(), []
    for gap, i, j in cands:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); matched.append((i, j, gap))
    print(f"Matched (same pair+direction within {MATCH_MIN} min): **{len(matched)}**; "
          f"X8-only: **{len(w8) - len(matched)}**; Guard-only: **{len(wg) - len(matched)}**.\n")

    # 4. unmatched analysis
    print("## 4. Every unmatched entry — what was the OTHER bot doing at that moment?\n")
    rows, verdicts = [], {}
    for side, own, own_idx, other, other_max, other_name in (
            ("X8-only", w8, ua, tg, maxg, "Guard"),
            ("Guard-only", wg, ub, t8, max8, "X8")):
        for k, x in enumerate(own):
            if k in own_idx:
                continue
            t = x["open_date"]
            other_open = open_at(other, t)
            ps, psdesc = pair_state(other, x["pair"], t)
            if other_open >= other_max:
                verdict = f"{other_name} SLOTS FULL ({other_open}/{other_max})"
            elif ps == "OPEN":
                verdict = f"{other_name} already had this pair open"
            else:
                verdict = "no slot/pair reason in DB → check logs (unfilled order? pairlist?)"
            verdicts[verdict.split(" (")[0]] = verdicts.get(verdict.split(" (")[0], 0) + 1
            lo, hi = t - timedelta(minutes=3), t + timedelta(minutes=3)
            rows.append((side, x["pair"] + (" (short)" if x["is_short"] else ""), x["enter_tag"], fmt(t),
                         f"{other_name} open {other_open}/{other_max}", psdesc, verdict,
                         f"{lo.strftime('%H:%M')}–{hi.strftime('%H:%M')} UTC"))
    print(table(["side", "pair", "tag", "opened (UTC)", "other bot slots", "other bot on this pair",
                 "verdict", "log window"], rows))
    print("**Verdict counts:** " + (", ".join(f"{k}: {v}" for k, v in verdicts.items()) or "none") + "\n")

    # 5. orders
    print("## 5. Orders-table status counts\n")
    for name, p in (("X8", x8_path), ("Guard", guard_path)):
        print(f"**{name}**\n")
        print(table(["side", "status", "count"], order_status_counts(p)))
    print("Entry orders that were cancelled/unfilled before any fill normally leave NO row in "
          "`trades`, so they cannot appear above. If a verdict says *check logs*, search that "
          "bot's workflow log in the given window for the pair name and these phrases: "
          "`Cancelling entry order`, `Timeout`, `unfilled`, `Whitelist with`, `Not enough`, "
          "`Max open trades`.\n")

    # 6. timeline
    print("## 6. All entries in window (for eyeballing)\n")
    allrows = ([("X8", x["pair"], "S" if x["is_short"] else "L", x["enter_tag"], fmt(x["open_date"]),
                 fmt(x["close_date"]) if x["close_date"] else "open", x["exit_reason"] or "") for x in w8] +
               [("Guard", x["pair"], "S" if x["is_short"] else "L", x["enter_tag"], fmt(x["open_date"]),
                 fmt(x["close_date"]) if x["close_date"] else "open", x["exit_reason"] or "") for x in wg])
    allrows.sort(key=lambda r: (r[1], r[4]))
    print(table(["bot", "pair", "L/S", "tag", "opened", "closed", "exit"], allrows))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
