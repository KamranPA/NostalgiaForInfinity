"""
generate_dashboard_compare.py

Builds docs/compare.html — a SECOND dashboard page that compares the two
strategies head to head, per market:

    SPOT:     NostalgiaForInfinityX7ML   vs   NostalgiaForInfinityX8
    FUTURES:  NostalgiaForInfinityX7ML   vs   NostalgiaForInfinityX8

plus the full per-mode detail sections for the two X8 bots (the X7ML
ones stay on the main page, index.html).

It IMPORTS generate_dashboard.py and reuses its data-fetching and section
builders, so the main dashboard script is left untouched (zero regression
risk for index.html). It reads the four bots' SQLite state files
READ-ONLY, exactly like the main script does.

FAIRNESS RULE (the important design decision)
---------------------------------------------
The X7ML bots have been running longer (spot was also deliberately reset
on 2026-09-14), the X8 bots started later. Comparing all-time numbers
would hand X7ML a head start. So every head-to-head number on this page is
computed ONLY from trades OPENED at/after the X8 bot's start time, for
BOTH strategies, per market:

  window start = freqtrade's own `bot_start_time` (stored in the X8 bot's
  SQLite KeyValueStore table on its very first startup and carried across
  restarts with the DB), falling back to the X8 bot's first trade if that
  key isn't available.

Known residual unfairness (shown on the page too): X7ML enters the window
with trades already open from before it, occupying some of its
max_open_trades slots, while X8 starts with an empty wallet.

UNREALIZED P/L NOTE: this page values open trades as
    direction * (live_price - open_rate) * amount
(direction = -1 for shorts). This is leverage-correct in futures mode.
The main dashboard values open trades as stake_amount * price change %,
which in futures mode (stake_amount = margin, not position size)
understates unrealized P/L by the leverage factor and ignores the sign of
shorts — so futures "True Total" here can differ from index.html.

Usage:
    python generate_dashboard_compare.py \\
        <spot_x7ml.sqlite> <futures_x7ml.sqlite> \\
        <spot_x8.sqlite> <futures_x8.sqlite> \\
        <history.sqlite> <output_html_path>
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

import generate_dashboard as gd


OVERLAP_WINDOW_MINUTES = 15
OPEN_TOO_LONG_DAYS = 3          # "open longer than this" flag in the table
SMALL_SAMPLE_CLOSED = 30        # below this many closed trades, warn
SWAP_OPTIONS = {"defaultType": "swap"}

EXTRA_CSS = """
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .edge { background: rgba(63,185,80,0.12); }
  .warn-note { color: #d29922; font-size: 0.85rem; margin: 8px 0 0 0; }
  .nav { margin-bottom: 20px; font-size: 0.9rem; }
  .nav a { margin-right: 14px; }
  .legend-x7 { color: #d29922; font-weight: 600; }
  .legend-x8 { color: #58a6ff; font-weight: 600; }
"""


# ----------------------------------------------------------------------
# Runtime patches of the imported module (no edits to generate_dashboard.py)
# ----------------------------------------------------------------------

_price_cache = {}
_orig_fetch_live_prices = gd.fetch_live_prices
_orig_build_ml_html = gd.build_ml_readiness_html


def _cached_fetch_live_prices(pairs, exchange_id="kucoin", ccxt_options=None):
    """Same contract as gd.fetch_live_prices, but remembers prices so the
    four bots' overlapping pairs are only fetched once per page build."""
    ex_key = (exchange_id, json.dumps(ccxt_options, sort_keys=True))
    missing = [p for p in pairs if (ex_key, p) not in _price_cache]
    if missing:
        for p, v in _orig_fetch_live_prices(missing, exchange_id, ccxt_options).items():
            _price_cache[(ex_key, p)] = v
    return {p: _price_cache[(ex_key, p)] for p in pairs if (ex_key, p) in _price_cache}


def install_patches():
    gd.fetch_live_prices = _cached_fetch_live_prices
    # Tag 68 is a valid rebuy tag in upstream (X7 and X8) but missing from
    # the main script's family table — label it here.
    if gd.tag_family_name("68") is None:
        gd.TAG_FAMILIES.append((68, 68, "Rebuy"))


def build_x8_mode(*args, **kwargs):
    """gd.build_one_mode for an X8 bot: identical, except the 'ML Data
    Readiness' card is suppressed (X8 has no ML snapshot logging, so the
    card would only ever say 'no data')."""
    gd.build_ml_readiness_html = lambda readiness: ""
    try:
        return gd.build_one_mode(*args, **kwargs)
    finally:
        gd.build_ml_readiness_html = _orig_build_ml_html


# ----------------------------------------------------------------------
# Bot start time / comparison window
# ----------------------------------------------------------------------

def fetch_bot_start_time(db_path):
    """freqtrade stores `bot_start_time` in the KeyValueStore table at the
    very first startup of a database (= 'now' on a fresh DB) and keeps it
    across restarts. Returns an aware UTC datetime, or None if the file /
    table / key isn't there (older freqtrade, empty file, ...)."""
    if not db_path or not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT datetime_value FROM KeyValueStore "
            "WHERE key = 'bot_start_time' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if not row or row[0] is None:
        return None
    return gd.to_aware_utc(gd.parse_sqlite_datetime(row[0]))


def determine_window_start(raw_start, x8_trades):
    if raw_start is not None:
        return raw_start, "freqtrade bot_start_time"
    dated = [gd.to_aware_utc(t["open_date"]) for t in x8_trades if t["open_date"]]
    if dated:
        return min(dated), "first X8 trade (bot_start_time unavailable)"
    return None, None


def in_window(t, start):
    return t["open_date"] is not None and gd.to_aware_utc(t["open_date"]) >= start


def count_carry_over(trades, start):
    """Trades opened BEFORE the window that were still occupying a slot when
    it began."""
    n = 0
    for t in trades:
        od = t["open_date"]
        if od is None or gd.to_aware_utc(od) >= start:
            continue
        if t["is_open"]:
            n += 1
        elif t["close_date"] is not None and gd.to_aware_utc(t["close_date"]) >= start:
            n += 1
    return n


# ----------------------------------------------------------------------
# Window metrics
# ----------------------------------------------------------------------

def compute_window_metrics(trades, start, live_prices, entry_fills, wallet, now):
    sel = [t for t in trades if in_window(t, start)]
    closed = [t for t in sel if not t["is_open"] and t["close_profit"] is not None]
    open_ = [t for t in sel if t["is_open"]]

    profits = [float(t["close_profit"]) for t in closed]
    n_closed = len(closed)
    wins = sum(1 for p in profits if p > 0)
    realized = sum(gd.to_float(t["close_profit_abs"]) for t in closed)

    unrealized = 0.0
    unpriced = 0
    open_over = 0
    oldest_open = 0.0
    for t in open_:
        age = gd.duration_days(t["open_date"], now)
        oldest_open = max(oldest_open, age)
        if age > OPEN_TOO_LONG_DAYS:
            open_over += 1
        live = live_prices.get(t["pair"])
        if live and t["open_rate"]:
            direction = -1.0 if t["is_short"] else 1.0
            unrealized += direction * (live - t["open_rate"]) * gd.to_float(t["amount"])
        else:
            unpriced += 1

    capital_days = 0.0
    for t in closed:
        capital_days += gd.capital_days_for_trade(
            entry_fills.get(t["id"]), t["open_date"], t["close_date"],
            fallback_stake=gd.to_float(t["stake_amount"]),
            fallback_days=gd.duration_days(t["open_date"], t["close_date"]),
        )
    for t in open_:
        age = gd.duration_days(t["open_date"], now)
        capital_days += gd.capital_days_for_trade(
            entry_fills.get(t["id"]), gd.to_aware_utc(t["open_date"]), now,
            fallback_stake=gd.to_float(t["stake_amount"]), fallback_days=age,
        )

    true_total = realized + unrealized
    rate = (true_total / capital_days) if capital_days > 0 else None

    # Max drawdown of the REALIZED equity curve (USDT), from a 0 baseline.
    cum = peak = max_dd = 0.0
    for t in sorted(closed, key=lambda x: gd.to_aware_utc(x["close_date"])
                    if x["close_date"] else now):
        cum += gd.to_float(t["close_profit_abs"])
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    durations = [gd.duration_days(t["open_date"], t["close_date"]) for t in closed]
    worst = min(closed, key=lambda t: float(t["close_profit"])) if closed else None
    best = max(closed, key=lambda t: float(t["close_profit"])) if closed else None
    fill_counts = [len(entry_fills.get(t["id"], [])) for t in sel if entry_fills.get(t["id"])]

    return {
        "n_opened": len(sel),
        "n_closed": n_closed,
        "n_open": len(open_),
        "win_rate": (wins / n_closed) if n_closed else None,
        "realized": realized,
        "unrealized": unrealized,
        "unpriced": unpriced,
        "true_total": true_total,
        "true_total_pct": (true_total / wallet) if wallet else None,
        "rate": rate,
        "avg_profit": (float(np.mean(profits)) if profits else None),
        "avg_dur_days": (float(np.mean(durations)) if durations else None),
        "worst": (float(worst["close_profit"]), worst["pair"]) if worst else None,
        "best": (float(best["close_profit"]), best["pair"]) if best else None,
        "open_over": open_over,
        "oldest_open": oldest_open if open_ else None,
        "max_dd": max_dd,
        "avg_dca": (float(np.mean([max(c - 1, 0) for c in fill_counts])) if fill_counts else None),
    }


def equity_series(trades, start, now):
    """[{x: hours since window start, y: cumulative realized USDT}, ...]"""
    closed = sorted(
        [t for t in trades if in_window(t, start) and not t["is_open"]
         and t["close_date"] is not None and t["close_profit_abs"] is not None],
        key=lambda t: gd.to_aware_utc(t["close_date"]),
    )
    pts = [{"x": 0.0, "y": 0.0}]
    cum = 0.0
    for t in closed:
        cum += gd.to_float(t["close_profit_abs"])
        hrs = max((gd.to_aware_utc(t["close_date"]) - start).total_seconds() / 3600.0, 0.0)
        pts.append({"x": round(hrs, 2), "y": round(cum, 2)})
    end_h = max((now - start).total_seconds() / 3600.0, 0.0)
    pts.append({"x": round(end_h, 2), "y": round(cum, 2)})
    return pts


# ----------------------------------------------------------------------
# Signal overlap between two bots (generic)
# ----------------------------------------------------------------------

def compute_entry_overlap(a_trades, b_trades, start, window_minutes=OVERLAP_WINDOW_MINUTES,
                          longs_only=False):
    """One-to-one matching of entries: same pair (':SETTLE' stripped) and
    same direction, opened within `window_minutes` of each other. Closest
    gaps are matched first so a trade can never be counted twice."""
    def pick(trades):
        out = [t for t in trades if in_window(t, start)]
        if longs_only:
            out = [t for t in out if not t["is_short"]]
        return out

    a, b = pick(a_trades), pick(b_trades)
    window = window_minutes * 60
    b_by_key = defaultdict(list)
    for j, t in enumerate(b):
        b_by_key[(gd.normalize_pair(t["pair"]), bool(t["is_short"]))].append(j)

    cands = []
    for i, ta in enumerate(a):
        for j in b_by_key.get((gd.normalize_pair(ta["pair"]), bool(ta["is_short"])), []):
            gap = abs((gd.to_aware_utc(ta["open_date"]) - gd.to_aware_utc(b[j]["open_date"])).total_seconds())
            if gap <= window:
                cands.append((gap, i, j))
    cands.sort()

    used_a, used_b, matches = set(), set(), []
    for gap, i, j in cands:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((a[i], b[j], gap))

    same_tag = sum(1 for ta, tb, _ in matches if str(ta["enter_tag"]) == str(tb["enter_tag"]))
    both_closed = [(ta, tb) for ta, tb, _ in matches
                   if not ta["is_open"] and not tb["is_open"]
                   and ta["close_profit"] is not None and tb["close_profit"] is not None]
    return {
        "n_a": len(a), "n_b": len(b),
        "matched": len(matches), "same_tag": same_tag,
        "a_only": len(a) - len(matches), "b_only": len(b) - len(matches),
        "matches": sorted(matches, key=lambda m: gd.to_aware_utc(m[0]["open_date"]), reverse=True),
        "both_closed_n": len(both_closed),
        "both_closed_avg_a": (float(np.mean([float(x["close_profit"]) for x, _ in both_closed])) if both_closed else None),
        "both_closed_avg_b": (float(np.mean([float(y["close_profit"]) for _, y in both_closed])) if both_closed else None),
        "window_minutes": window_minutes,
    }


# ----------------------------------------------------------------------
# HTML helpers
# ----------------------------------------------------------------------

def money(v):
    return "—" if v is None else f"{v:+.2f} USDT"


def pct_s(v):
    return "—" if v is None else gd.fmt_pct(v)


def days_s(v):
    if v is None:
        return "—"
    return f"{v * 24:.1f}h" if v < 1 else f"{v:.1f}d"


def trade_ref(pair_pct):
    if pair_pct is None:
        return "—"
    pct, pair = pair_pct
    return f"{gd.fmt_pct(pct)} <span class=\"muted\">({gd.esc(pair)})</span>"


def cmp_row(label, a_txt, b_txt, a_val=None, b_val=None, higher_is_better=None):
    a_cls = b_cls = ""
    if (higher_is_better is not None and a_val is not None and b_val is not None
            and a_val != b_val):
        a_better = (a_val > b_val) if higher_is_better else (a_val < b_val)
        if a_better:
            a_cls = ' class="edge"'
        else:
            b_cls = ' class="edge"'
    return f"<tr><td>{label}</td><td{a_cls}>{a_txt}</td><td{b_cls}>{b_txt}</td></tr>"


def metrics_table_html(m7, m8):
    rows = [
        cmp_row("Trades opened (in window)", str(m7["n_opened"]), str(m8["n_opened"])),
        cmp_row("Closed / open now",
                f"{m7['n_closed']} / {m7['n_open']}", f"{m8['n_closed']} / {m8['n_open']}"),
        cmp_row("Win rate (closed)", pct_s(m7["win_rate"]).lstrip("+") if m7["win_rate"] is not None else "—",
                pct_s(m8["win_rate"]).lstrip("+") if m8["win_rate"] is not None else "—",
                m7["win_rate"], m8["win_rate"], True),
        cmp_row("Realized profit", money(m7["realized"]), money(m8["realized"]),
                m7["realized"], m8["realized"], True),
        cmp_row("Unrealized P/L (open)", money(m7["unrealized"]), money(m8["unrealized"]),
                m7["unrealized"], m8["unrealized"], True),
        cmp_row("<b>True Total</b> (realized + unrealized)",
                f"<b>{money(m7['true_total'])}</b>", f"<b>{money(m8['true_total'])}</b>",
                m7["true_total"], m8["true_total"], True),
        cmp_row("True Total (% of dry-run wallet)", pct_s(m7["true_total_pct"]), pct_s(m8["true_total_pct"]),
                m7["true_total_pct"], m8["true_total_pct"], True),
        cmp_row("Return per capital-day (blended)",
                f"{m7['rate'] * 100:.4f}%/day" if m7["rate"] is not None else "—",
                f"{m8['rate'] * 100:.4f}%/day" if m8["rate"] is not None else "—",
                m7["rate"], m8["rate"], True),
        cmp_row("Avg profit / closed trade", pct_s(m7["avg_profit"]), pct_s(m8["avg_profit"]),
                m7["avg_profit"], m8["avg_profit"], True),
        cmp_row("Avg closed duration", days_s(m7["avg_dur_days"]), days_s(m8["avg_dur_days"])),
        cmp_row("Worst closed trade", trade_ref(m7["worst"]), trade_ref(m8["worst"]),
                m7["worst"][0] if m7["worst"] else None, m8["worst"][0] if m8["worst"] else None, True),
        cmp_row("Best closed trade", trade_ref(m7["best"]), trade_ref(m8["best"])),
        cmp_row(f"Open longer than {OPEN_TOO_LONG_DAYS}d", str(m7["open_over"]), str(m8["open_over"]),
                m7["open_over"], m8["open_over"], False),
        cmp_row("Oldest open trade", days_s(m7["oldest_open"]), days_s(m8["oldest_open"]),
                m7["oldest_open"], m8["oldest_open"], False),
        cmp_row("Max drawdown (realized equity)", f"{m7['max_dd']:.2f} USDT", f"{m8['max_dd']:.2f} USDT",
                m7["max_dd"], m8["max_dd"], False),
        cmp_row("Avg DCA/rebuy fills per trade",
                f"{m7['avg_dca']:.2f}" if m7["avg_dca"] is not None else "—",
                f"{m8['avg_dca']:.2f}" if m8["avg_dca"] is not None else "—"),
    ]
    return f"""
  <div class="table-scroll">
  <table class="compare-table">
    <tr><th>Metric</th><th class="legend-x7">X7ML</th><th class="legend-x8">X8</th></tr>
    {''.join(rows)}
  </table>
  </div>
  <p class="muted" style="font-size:0.8rem;margin:10px 0 0 0;">
    Highlighted cell = the better value where "better" is unambiguous. Trades-opened and
    duration rows are not highlighted: more entries or shorter holds aren't better by themselves.
    Unrealized P/L uses live OKX prices; open trades without a price are counted as 0
    ({m7['unpriced']} for X7ML, {m8['unpriced']} for X8).
  </p>"""


def family_of(tag):
    first = str(tag).split()[0] if tag not in (None, "") else None
    return gd.tag_family_name(first) or "Other"


def family_table_html(trades7, trades8, start):
    agg = {}
    for side, trades in (("a", trades7), ("b", trades8)):
        for t in trades:
            if not in_window(t, start):
                continue
            d = agg.setdefault(family_of(t["enter_tag"]), {"a": [0, []], "b": [0, []]})
            d[side][0] += 1
            if not t["is_open"] and t["close_profit"] is not None:
                d[side][1].append(float(t["close_profit"]))
    if not agg:
        return '<p class="muted">No trades in the window yet.</p>'

    def cell(entry):
        n, ps = entry
        avg = gd.fmt_pct(float(np.mean(ps))) if ps else "—"
        return f"<td>{n}</td><td>{avg}</td>"

    rows = "".join(
        f"<tr><td>{gd.esc(fam)}</td>{cell(d['a'])}{cell(d['b'])}</tr>"
        for fam, d in sorted(agg.items(), key=lambda kv: -(kv[1]["a"][0] + kv[1]["b"][0]))
    )
    return f"""
  <div class="table-scroll"><table class="compare-table">
    <tr><th>Tag family</th><th class="legend-x7">X7ML opened</th><th class="legend-x7">avg closed</th>
        <th class="legend-x8">X8 opened</th><th class="legend-x8">avg closed</th></tr>
    {rows}
  </table></div>
  <p class="muted" style="font-size:0.75rem;margin:8px 0 0 0;">Multi-condition tags (e.g. "41 61") are filed under their first tag.</p>"""


def exit_reason_table_html(trades7, trades8, start):
    counts = defaultdict(lambda: [0, 0])
    for idx, trades in enumerate((trades7, trades8)):
        for t in trades:
            if in_window(t, start) and not t["is_open"]:
                counts[t["exit_reason"] or "unknown"][idx] += 1
    if not counts:
        return '<p class="muted">No closed trades in the window yet.</p>'
    rows = "".join(
        f"<tr><td>{gd.esc(r)}</td><td>{c[0]}</td><td>{c[1]}</td></tr>"
        for r, c in sorted(counts.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:12]
    )
    return f"""
  <div class="table-scroll"><table class="compare-table">
    <tr><th>Exit reason</th><th class="legend-x7">X7ML</th><th class="legend-x8">X8</th></tr>
    {rows}
  </table></div>"""


def result_cell(t):
    if t["is_open"] or t["close_profit"] is None:
        return '<span class="muted">open</span>'
    p = float(t["close_profit"])
    cls = "profit-pos" if p > 0 else "profit-neg"
    return f'<span class="{cls}">{gd.fmt_pct(p)}</span>'


def overlap_html(ov, name_a, name_b, title, show_results=True, note=""):
    if ov["n_a"] == 0 and ov["n_b"] == 0:
        return f'<div class="card"><h3>{title}</h3><p class="muted">No entries in the window yet.</p></div>'

    pa = (ov["matched"] / ov["n_a"] * 100) if ov["n_a"] else 0
    pb = (ov["matched"] / ov["n_b"] * 100) if ov["n_b"] else 0
    tag_pct = (ov["same_tag"] / ov["matched"] * 100) if ov["matched"] else 0

    like = ""
    if ov["both_closed_n"]:
        like = f"""
    <p style="margin:12px 0 0 0;">On the <b>{ov['both_closed_n']}</b> matched signals that both bots have already closed:
      {gd.esc(name_a)} averaged <b>{gd.fmt_pct(ov['both_closed_avg_a'])}</b>,
      {gd.esc(name_b)} averaged <b>{gd.fmt_pct(ov['both_closed_avg_b'])}</b>
      — same entry moment, different trade management.</p>"""

    rows = ""
    if show_results:
        for ta, tb, gap in ov["matches"][:12]:
            rows += f"""<tr>
              <td>{gd.esc(ta['pair'])}{' <span class="muted">(short)</span>' if ta['is_short'] else ''}</td>
              <td>{gd.fmt_tag(ta['enter_tag'])}</td><td>{gd.fmt_tag(tb['enter_tag'])}</td>
              <td>{gd.fmt_dt(ta['open_date'])}</td><td>{gd.fmt_dt(tb['open_date'])}</td>
              <td>{gap / 60:.1f}m</td><td>{result_cell(ta)}</td><td>{result_cell(tb)}</td>
            </tr>"""

    table = ""
    if show_results:
        table = f"""
  <div class="table-scroll" style="margin-top:16px;"><table>
    <tr><th>Pair</th><th>{gd.esc(name_a)} tag</th><th>{gd.esc(name_b)} tag</th>
        <th>{gd.esc(name_a)} opened</th><th>{gd.esc(name_b)} opened</th><th>Gap</th>
        <th>{gd.esc(name_a)} result</th><th>{gd.esc(name_b)} result</th></tr>
    {rows if rows else '<tr><td colspan="8" class="muted">No matched entries yet</td></tr>'}
  </table></div>"""

    return f"""
<div class="card">
  <h3>{title} <span class="muted" style="font-weight:400;font-size:0.75rem;">— same pair + direction, opened within {ov['window_minutes']} min</span></h3>
  <div class="grid" style="margin-bottom:0;">
    <div class="stat-card"><div class="label">Matched entries</div>
      <div class="value">{ov['matched']}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">{pa:.0f}% of {gd.esc(name_a)} · {pb:.0f}% of {gd.esc(name_b)}</div></div>
    <div class="stat-card"><div class="label">{gd.esc(name_a)}-only entries</div>
      <div class="value">{ov['a_only']}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">of {ov['n_a']} total</div></div>
    <div class="stat-card"><div class="label">{gd.esc(name_b)}-only entries</div>
      <div class="value">{ov['b_only']}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">of {ov['n_b']} total</div></div>
    <div class="stat-card"><div class="label">Same enter_tag too</div>
      <div class="value">{ov['same_tag']} / {ov['matched']}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">{tag_pct:.0f}% of matches</div></div>
  </div>
  {like}
  {table}
  {f'<p class="muted" style="font-size:0.8rem;margin:12px 0 0 0;">{note}</p>' if note else ''}
</div>"""


# ----------------------------------------------------------------------
# Per-market comparison block
# ----------------------------------------------------------------------

EQUITY_JS = """
new Chart(document.getElementById('__ID__'), {
  type: 'scatter',
  data: { datasets: [
    { label: 'X7ML', data: __D7__, borderColor: '#d29922', backgroundColor: '#d29922',
      showLine: true, stepped: 'after', pointRadius: 2, borderWidth: 2 },
    { label: 'X8', data: __D8__, borderColor: '#58a6ff', backgroundColor: '#58a6ff',
      showLine: true, stepped: 'after', pointRadius: 2, borderWidth: 2 }
  ]},
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      x: { type: 'linear', title: { display: true, text: 'Hours since X8 start', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
      y: { title: { display: true, text: 'Cumulative realized profit (USDT)', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
    },
    plugins: { legend: { labels: { color: '#c9d1d9' } } }
  }
});
"""


def build_market_block(key, title, subtitle_extra, start, source, trades7, trades8,
                       m7, m8, overlap, eq7, eq8):
    if start is None:
        return f"""
<h2 class="market-heading" id="cmp-{key}">{title}</h2>
<div class="card">
  <p class="muted">The X8 bot has no data yet (no state file, no start time, no trades) — nothing to compare.
  This fills in automatically once its first hourly state checkpoint reaches the state branch.</p>
</div>""", ""

    now_txt = gd.fmt_dt(start)
    carry = count_carry_over(trades7, start)
    days = (datetime.now(timezone.utc) - start).total_seconds() / 86400.0
    small = min(m7["n_closed"], m8["n_closed"]) < SMALL_SAMPLE_CLOSED

    caveats = f"""
  <p class="muted" style="font-size:0.8rem;margin:0 0 12px 0;">
    Window = trades opened since <b>{now_txt} UTC</b> ({days:.1f} days ago; start taken from {gd.esc(source)}).
    X7ML began this window with <b>{carry}</b> older trade(s) still open, occupying slots X8 had free.
  </p>
  {f'<p class="warn-note">⚠️ Small sample: {m7["n_closed"]} (X7ML) and {m8["n_closed"]} (X8) closed trades — below {SMALL_SAMPLE_CLOSED}, so differences here can easily be chance.</p>' if small else ''}"""

    chart_id = f"cmpEquity_{key}"
    html = f"""
<h2 class="market-heading" id="cmp-{key}">{title}</h2>
<div class="subtitle">{subtitle_extra}</div>

<div class="card">
  <h3>Head to head <span class="muted" style="font-weight:400;font-size:0.75rem;">— trades opened since the X8 bot started, both strategies</span></h3>
  {caveats}
  {metrics_table_html(m7, m8)}
</div>

<div class="card">
  <h3>Realized equity since X8 start <span class="muted" style="font-weight:400;font-size:0.75rem;">— cumulative closed-trade profit, USDT</span></h3>
  <div class="chart-wrap"><canvas id="{chart_id}"></canvas></div>
</div>

{overlap_html(overlap, "X7ML", "X8", "🔗 Signal Overlap — X7ML vs X8",
              note="A low match rate means the two strategies really do fire on different moments (or different pairs). "
                   "'Only' counts are entries with no counterpart on the other bot inside the window.")}

<div class="two-col">
  <div class="card"><h3>Entry tag families</h3>{family_table_html(trades7, trades8, start)}</div>
  <div class="card"><h3>Exit reasons (closed)</h3>{exit_reason_table_html(trades7, trades8, start)}</div>
</div>"""

    js = (EQUITY_JS.replace("__ID__", chart_id)
                   .replace("__D7__", json.dumps(eq7))
                   .replace("__D8__", json.dumps(eq8)))
    return html, js


# ----------------------------------------------------------------------
# Data plumbing
# ----------------------------------------------------------------------

def load_fills(db_path, trades):
    rows = gd.fetch_entry_fills(db_path, [t["id"] for t in trades])
    entry_fills, _ = gd.build_fill_maps(rows, {t["id"]: t for t in trades})
    return entry_fills


def prices_for(trades, options):
    open_pairs = list({t["pair"] for t in trades if t["is_open"]})
    return gd.fetch_live_prices(open_pairs, "okx", options) if open_pairs else {}


def main(spot_x7, fut_x7, spot_x8, fut_x8, history_db_path, output_path):
    install_patches()
    now = datetime.now(timezone.utc)

    # Read the X8 start times BEFORE anything else opens these files
    # (gd.fetch_trades would create an empty file if one were missing).
    raw_start_spot = fetch_bot_start_time(spot_x8)
    raw_start_fut = fetch_bot_start_time(fut_x8)

    b7s = gd.build_one_mode(spot_x7, history_db_path, "spot", "🟢 X7ML SPOT — OKX",
                            "OKX spot market · no leverage", "config_dryrun_telegram.json",
                            live_price_exchange_id="okx", live_price_ccxt_options=None)
    b7f = gd.build_one_mode(fut_x7, history_db_path, "futures", "🟣 X7ML FUTURES — OKX",
                            "OKX perpetual swaps · isolated margin · 3x leverage (default)",
                            "config_dryrun_futures.json",
                            live_price_exchange_id="okx", live_price_ccxt_options=SWAP_OPTIONS)
    b8s = build_x8_mode(spot_x8, history_db_path, "spot_x8", "🔵 X8 SPOT — OKX",
                        "OKX spot market · no leverage · NostalgiaForInfinityX8",
                        "config_dryrun_telegram_x8.json",
                        live_price_exchange_id="okx", live_price_ccxt_options=None)
    b8f = build_x8_mode(fut_x8, history_db_path, "futures_x8", "🟠 X8 FUTURES — OKX",
                        "OKX perpetual swaps · isolated margin · 3x leverage (X8 default — verified in upstream source v18.0.61)",
                        "config_dryrun_futures_x8.json",
                        live_price_exchange_id="okx", live_price_ccxt_options=SWAP_OPTIONS)

    wallet = gd.load_portfolio_config("config_dryrun_telegram_x8.json")["dry_run_wallet"]

    def market(key, title, sub, raw_start, bx7, bx8, path7, path8, options):
        start, source = determine_window_start(raw_start, bx8["trades"])
        if start is None:
            html, js = build_market_block(key, title, sub, None, None, [], [], None, None, None, None, None)
            return html, js, None
        f7, f8 = load_fills(path7, bx7["trades"]), load_fills(path8, bx8["trades"])
        m7 = compute_window_metrics(bx7["trades"], start, prices_for(bx7["trades"], options), f7, wallet, now)
        m8 = compute_window_metrics(bx8["trades"], start, prices_for(bx8["trades"], options), f8, wallet, now)
        ov = compute_entry_overlap(bx7["trades"], bx8["trades"], start)
        html, js = build_market_block(
            key, title, sub, start, source, bx7["trades"], bx8["trades"], m7, m8, ov,
            equity_series(bx7["trades"], start, now), equity_series(bx8["trades"], start, now))
        return html, js, start

    spot_html, spot_js, spot_start = market(
        "spot", "🟢 SPOT — X7ML vs X8", "Same OKX spot market, same pairlist rules, same config — only the strategy differs.",
        raw_start_spot, b7s, b8s, spot_x7, spot_x8, None)
    fut_html, fut_js, fut_start = market(
        "futures", "🟣 FUTURES — X7ML vs X8", "Same OKX perpetual-swap market, isolated margin, 3x leverage — only the strategy differs.",
        raw_start_fut, b7f, b8f, fut_x7, fut_x8, SWAP_OPTIONS)

    # X8 spot vs X8 futures (long entries), from the later of the two X8 starts
    x8_overlap_html = ""
    if spot_start is not None and fut_start is not None:
        ov = compute_entry_overlap(b8s["trades"], b8f["trades"], max(spot_start, fut_start), longs_only=True)
        x8_overlap_html = overlap_html(
            ov, "X8 spot", "X8 futures", "🔗 Signal Overlap — X8 Spot vs X8 Futures (long entries)",
            show_results=False,
            note="Futures-only short entries are excluded, since spot has no short side.")

    generated_at = now.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — X7ML vs X8</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{gd.PAGE_STYLES}
{EXTRA_CSS}
</style>
</head>
<body>

<h1>⚔️ NostalgiaForInfinity — X7ML vs X8</h1>
<div class="subtitle">Generated {generated_at} UTC · dry-run (simulated) — no real funds involved · all four bots on OKX</div>
<div class="nav">
  <a href="index.html">← Main dashboard (X7ML)</a>
  <a href="#cmp-spot">Spot comparison</a>
  <a href="#cmp-futures">Futures comparison</a>
  <a href="#x8-detail">X8 detail</a>
</div>

{spot_html}

{fut_html}

{x8_overlap_html}

<div id="x8-detail"></div>
{b8s['section_html']}

{b8f['section_html']}

<div class="subtitle">
  ⚠️ Small samples can look great or terrible by chance. X8 is still under active development
  upstream and the daily upstream sync can change its behavior mid-comparison — note the sync
  dates when reading results. Entries cancelled before filling aren't counted, only trades that opened.
</div>

<script>
{spot_js}
{fut_js}
{b8s['section_js']}
{b8f['section_js']}
</script>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Comparison page written to {output_path}")

    # Only the X8 modes are recorded here — spot/futures (X7ML) snapshots
    # are recorded by generate_dashboard.py itself, so nothing is doubled.
    gd.record_snapshot(history_db_path, "spot_x8", b8s["current_snapshot"])
    gd.record_snapshot(history_db_path, "futures_x8", b8f["current_snapshot"])


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:7])
