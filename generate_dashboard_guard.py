"""
generate_dashboard_guard.py

Builds docs/guard.html -- a THIRD, fully separate dashboard page that
compares the two X8 FUTURES bots head to head:

    NostalgiaForInfinityX8   vs   NostalgiaForInfinityX8Guard

It IMPORTS generate_dashboard.py and generate_dashboard_compare.py and
reuses their data-fetching and helper functions; neither file is edited,
so index.html and compare.html are unaffected (zero regression risk).
The SQLite state files are read READ-ONLY, like the other scripts.

SWITCH OF THE GUARD STRATEGY (v2 of this page)
----------------------------------------------
Until the switch time T the Guard differed from X8 in ONE thing: stale_range 15
(vs 7). From T on, the Guard is X8 plus the CUT RULE (close a long trade
completely when it reaches 3 filled entries, then lock the pair for 14 days).
T is read from "cut_start_date" in the Guard's own config file
(config_dryrun_futures_x8guard.json -> guard_parameters), the SAME value the
strategy uses, so the page and the bot can never disagree.

FAIRNESS RULES
--------------
* Head to head: only trades OPENED at/after T, for BOTH bots. Trades opened
  before T are inherited by both bots and are left out of the table; the
  "Inherited trades" card lists them side by side (booked profit only).
* Nothing is deleted: the period BEFORE T (Guard with stale_range 15 vs X8) is
  kept as its own card, and the history snapshots keep their mode id.
* True Total on this page = closed-trade profit + profit ALREADY REALIZED by
  partial exits of still-open trades + unrealized P/L of what is left. The
  compare script's own True Total ignores the middle part; without it a bot
  that cuts (realizes its loss) would look worse than a bot that holds the same
  loser after partial exits. The correction is applied here only.

If T is missing from the config (or lies in the future) the page behaves like
before: the window starts at the Guard bot's start time.

If the Guard bot's state file is missing or empty the page says so and
nothing else breaks.

Usage (unchanged):
    python generate_dashboard_guard.py \\
        <futures_x8.sqlite> <futures_x8guard.sqlite> \\
        <history.sqlite> <output_html_path>
"""

import os
import sys
import json
import traceback
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

import generate_dashboard as gd
import generate_dashboard_compare as gc


NAME_A = "X8"
NAME_B = "X8 Guard"
CONFIG_PATH = "config_dryrun_futures_x8guard.json"

EXTRA_CSS = """
  .legend-xg { color: #bc8cff; font-weight: 600; }
"""

EQUITY_JS = """
new Chart(document.getElementById('guardEquity'), {
  type: 'scatter',
  data: { datasets: [
    { label: 'X8', data: __DA__, borderColor: '#58a6ff', backgroundColor: '#58a6ff',
      showLine: true, stepped: 'after', pointRadius: 2, borderWidth: 2 },
    { label: 'X8 Guard', data: __DB__, borderColor: '#bc8cff', backgroundColor: '#bc8cff',
      showLine: true, stepped: 'after', pointRadius: 2, borderWidth: 2 }
  ]},
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: {
      x: { type: 'linear', title: { display: true, text: '__XLABEL__', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
      y: { title: { display: true, text: 'Cumulative realized profit (USDT)', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
    },
    plugins: { legend: { labels: { color: '#c9d1d9' } } }
  }
});
"""


# ----------------------------------------------------------------------
# Config: switch time and rule parameters
# ----------------------------------------------------------------------

def read_guard_parameters(config_path=CONFIG_PATH):
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("guard_parameters") or {}
    except Exception as e:
        print(f"  (could not read {config_path} for the cut parameters: {e})")
        return {}


def read_switch_time(guard_parameters):
    """UTC datetime of cut_start_date, or None (missing / unreadable)."""
    raw = (guard_parameters or {}).get("cut_start_date")
    if not raw:
        return None
    try:
        d = datetime.fromisoformat(str(raw).strip().replace("Z", "+00:00"))
    except ValueError:
        print(f"  (cut_start_date {raw!r} is not a valid date; ignoring it on this page)")
        return None
    return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)


def rule_text(gp):
    parts = []
    n = gp.get("cut_entries")
    if n:
        parts.append(f"{int(float(n))} filled entries")
    if gp.get("cut_age_hours"):
        parts.append(f"{gp['cut_age_hours']} h old and {gp.get('cut_adverse_pct', 12)}% against the first entry")
    what = " or ".join(parts) if parts else "no trigger set"
    side = "long trades only" if str(gp.get("cut_apply_to", "both")).lower() == "long" else "long and short trades"
    return f"close the whole position at {what} ({side}), then lock the pair for {gp.get('cut_lock_days', 3)} days"


# ----------------------------------------------------------------------
# Metrics with the realized-on-open correction
# ----------------------------------------------------------------------

def with_open_realized(m, trades, start, wallet):
    """Adds what partial exits of still-open trades already realized
    (trades.realized_profit) to True Total. `trades` must be the same list
    the metrics were computed from; only trades opened at/after `start` count."""
    extra = 0.0
    for t in trades:
        if t["is_open"] and gc.in_window(t, start):
            extra += gd.to_float(t.get("realized_profit"))
    m["realized_open"] = extra
    m["true_total"] = m["true_total"] + extra
    m["true_total_pct"] = (m["true_total"] / wallet) if wallet else None
    return m


def metrics_table(ma, mb):
    r = gc.cmp_row
    rows = [
        r("Trades opened (in window)", str(ma["n_opened"]), str(mb["n_opened"])),
        r("Closed / open now", f"{ma['n_closed']} / {ma['n_open']}", f"{mb['n_closed']} / {mb['n_open']}"),
        r("Win rate (closed)",
          gc.pct_s(ma["win_rate"]).lstrip("+") if ma["win_rate"] is not None else "—",
          gc.pct_s(mb["win_rate"]).lstrip("+") if mb["win_rate"] is not None else "—",
          ma["win_rate"], mb["win_rate"], True),
        r("Realized profit (closed trades)", gc.money(ma["realized"]), gc.money(mb["realized"]),
          ma["realized"], mb["realized"], True),
        r("Realized on open trades (partial exits)", gc.money(ma["realized_open"]), gc.money(mb["realized_open"]),
          ma["realized_open"], mb["realized_open"], True),
        r("Unrealized P/L (open)", gc.money(ma["unrealized"]), gc.money(mb["unrealized"]),
          ma["unrealized"], mb["unrealized"], True),
        r("<b>True Total</b> (all three above)",
          f"<b>{gc.money(ma['true_total'])}</b>", f"<b>{gc.money(mb['true_total'])}</b>",
          ma["true_total"], mb["true_total"], True),
        r("True Total (% of dry-run wallet)", gc.pct_s(ma["true_total_pct"]), gc.pct_s(mb["true_total_pct"]),
          ma["true_total_pct"], mb["true_total_pct"], True),
        r("Avg profit / closed trade", gc.pct_s(ma["avg_profit"]), gc.pct_s(mb["avg_profit"]),
          ma["avg_profit"], mb["avg_profit"], True),
        r("Avg closed duration", gc.days_s(ma["avg_dur_days"]), gc.days_s(mb["avg_dur_days"])),
        r("Worst closed trade", gc.trade_ref(ma["worst"]), gc.trade_ref(mb["worst"]),
          ma["worst"][0] if ma["worst"] else None, mb["worst"][0] if mb["worst"] else None, True),
        r(f"Open longer than {gc.OPEN_TOO_LONG_DAYS}d", str(ma["open_over"]), str(mb["open_over"]),
          ma["open_over"], mb["open_over"], False),
        r("Oldest open trade", gc.days_s(ma["oldest_open"]), gc.days_s(mb["oldest_open"]),
          ma["oldest_open"], mb["oldest_open"], False),
        r("Max drawdown (realized equity)", f"{ma['max_dd']:.2f} USDT", f"{mb['max_dd']:.2f} USDT",
          ma["max_dd"], mb["max_dd"], False),
    ]
    return f"""
  <div class="table-scroll"><table class="compare-table">
    <tr><th>Metric</th><th class="legend-x8">{NAME_A}</th><th class="legend-xg">{NAME_B}</th></tr>
    {''.join(rows)}
  </table></div>
  <p class="muted" style="font-size:0.8rem;margin:10px 0 0 0;">
    Highlighted cell = the better value where "better" is unambiguous.
    Unrealized P/L uses live OKX prices; open trades without a price are counted as 0
    ({ma['unpriced']} for {NAME_A}, {mb['unpriced']} for {NAME_B}).
    True Total includes what partial exits of open trades already realized, so a bot that cuts a
    loser and a bot that holds it are measured the same way.
  </p>"""


def exit_table(trades_a, trades_b, start):
    counts = defaultdict(lambda: [0, 0])
    for idx, trades in enumerate((trades_a, trades_b)):
        for t in trades:
            if gc.in_window(t, start) and not t["is_open"]:
                counts[t["exit_reason"] or "unknown"][idx] += 1
    if not counts:
        return '<p class="muted">No closed trades in the window yet.</p>'
    rows = "".join(
        f"<tr><td>{gd.esc(r)}</td><td>{c[0]}</td><td>{c[1]}</td></tr>"
        for r, c in sorted(counts.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:15]
    )
    return f"""
  <div class="table-scroll"><table class="compare-table">
    <tr><th>Exit reason</th><th class="legend-x8">{NAME_A}</th><th class="legend-xg">{NAME_B}</th></tr>
    {rows}
  </table></div>"""


def _exit_label(t):
    return "open" if t["is_open"] else (t["exit_reason"] or "unknown")


def divergence_html(ov, switched):
    rows = ""
    n = 0
    for ta, tb, gap in ov["matches"]:
        if _exit_label(ta) == _exit_label(tb):
            continue
        n += 1
        if n > 15:
            continue
        rows += f"""<tr>
          <td>{gd.esc(ta['pair'])}{' <span class="muted">(short)</span>' if ta['is_short'] else ''}</td>
          <td>{gd.fmt_dt(ta['open_date'])}</td>
          <td>{gc.result_cell(ta)} <span class="muted">{gd.esc(_exit_label(ta))}</span></td>
          <td>{gc.result_cell(tb)} <span class="muted">{gd.esc(_exit_label(tb))}</span></td>
        </tr>"""
    if n == 0:
        body = '<p class="muted">No matched entry has closed differently yet — the two bots behaved identically so far.</p>'
    else:
        body = f"""
  <div class="table-scroll"><table>
    <tr><th>Pair</th><th>Opened ({NAME_A})</th>
        <th class="legend-x8">{NAME_A} result / exit</th><th class="legend-xg">{NAME_B} result / exit</th></tr>
    {rows}
  </table></div>
  <p class="muted" style="font-size:0.75rem;margin:8px 0 0 0;">{n} matched entries closed differently (newest 15 shown). "open" = still open on that bot.</p>"""
    what = ("where the cut rule actually changed the outcome (look for exits starting with cut_)"
            if switched else "where the Guard's higher abandon threshold actually changed the outcome")
    return f"""
<div class="card">
  <h3>🔀 Same entry, different exit <span class="muted" style="font-weight:400;font-size:0.75rem;">— {what}</span></h3>
  {body}
</div>"""


# ----------------------------------------------------------------------
# Inherited trades (opened before the switch) and the "before" period
# ----------------------------------------------------------------------

def _booked(t):
    """Profit already booked: realized part of an open trade, or the final profit of a closed one."""
    if t["is_open"]:
        return gd.to_float(t.get("realized_profit"))
    return gd.to_float(t.get("close_profit_abs"))


def _trade_key(t):
    return (gd.normalize_pair(t["pair"]), t["open_date"].strftime("%Y-%m-%d %H:%M") if t["open_date"] else "?")


def inherited_html(trades_a, trades_b, switch):
    def inherited(trades):
        out = {}
        for t in trades:
            if t["open_date"] is None or gd.to_aware_utc(t["open_date"]) >= switch:
                continue
            closed_after = t["close_date"] is not None and gd.to_aware_utc(t["close_date"]) >= switch
            if t["is_open"] or closed_after:
                out[_trade_key(t)] = t
        return out

    ia, ib = inherited(trades_a), inherited(trades_b)
    keys = sorted(set(ia) | set(ib), key=lambda k: k[1])
    if not keys:
        return f"""
<div class="card">
  <h3>🧳 Inherited trades <span class="muted" style="font-weight:400;font-size:0.75rem;">— opened before the switch</span></h3>
  <p class="muted">No trade was open at the switch time on either bot.</p>
</div>"""

    def cell(t):
        if t is None:
            return '<span class="muted">—</span>'
        v = _booked(t)
        cls = "profit-pos" if v >= 0 else "profit-neg"
        state = "open" if t["is_open"] else f"closed ({gd.esc(t['exit_reason'] or 'unknown')})"
        return f'<span class="{cls}">{v:+.2f} USDT</span> <span class="muted">{state}</span>'

    rows = ""
    for k in keys:
        ta, tb = ia.get(k), ib.get(k)
        ref = ta or tb
        rows += f"""<tr>
          <td>{gd.esc(ref['pair'])}{' <span class="muted">(short)</span>' if ref['is_short'] else ''}</td>
          <td>{gd.fmt_tag(ref['enter_tag'])}</td>
          <td>{gd.fmt_dt(ref['open_date'])}</td>
          <td>{cell(ta)}</td><td>{cell(tb)}</td>
        </tr>"""
    return f"""
<div class="card">
  <h3>🧳 Inherited trades <span class="muted" style="font-weight:400;font-size:0.75rem;">— opened before the switch; the cut rule never touches them, so both bots handle them the same way</span></h3>
  <div class="table-scroll"><table>
    <tr><th>Pair</th><th>Tag</th><th>Opened</th>
        <th class="legend-x8">{NAME_A} booked profit</th><th class="legend-xg">{NAME_B} booked profit</th></tr>
    {rows}
  </table></div>
  <p class="muted" style="font-size:0.75rem;margin:8px 0 0 0;">"Booked" = profit already realized (partial exits of an open trade, or the final result of a closed one); unrealized P/L is not included here. If the two columns drift apart for the same trade, the bots did not handle it identically.</p>
</div>"""


def before_card(trades_a, trades_b, guard_start, switch, wallet, now, fa, fb, pa, pb):
    """Kept history: Guard with stale_range 15 vs X8, trades opened between the Guard's start and the switch."""
    if guard_start is None or guard_start >= switch:
        return ""
    sel_a = [t for t in trades_a if t["open_date"] and gd.to_aware_utc(t["open_date"]) < switch]
    sel_b = [t for t in trades_b if t["open_date"] and gd.to_aware_utc(t["open_date"]) < switch]
    ma = with_open_realized(gc.compute_window_metrics(sel_a, guard_start, pa, fa, wallet, now), sel_a, guard_start, wallet)
    mb = with_open_realized(gc.compute_window_metrics(sel_b, guard_start, pb, fb, wallet, now), sel_b, guard_start, wallet)
    return f"""
<div class="card">
  <h3>📦 Before the switch <span class="muted" style="font-weight:400;font-size:0.75rem;">— Guard with stale_range 15 vs X8, trades opened {gd.fmt_dt(guard_start)} to {gd.fmt_dt(switch)} UTC (kept as it was)</span></h3>
  <p class="muted" style="font-size:0.8rem;margin:0 0 12px 0;">
    Numbers are as of now for those trades (a trade opened before the switch can still be open or closed later).
  </p>
  {metrics_table(ma, mb)}
</div>"""


def placeholder_page(output_path, message):
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — X8 vs X8 Guard</title>
<style>{gd.PAGE_STYLES}{gc.EXTRA_CSS}{EXTRA_CSS}</style></head><body>
<h1>🛡️ NostalgiaForInfinity — X8 vs X8 Guard</h1>
<div class="nav"><a href="index.html">← Main dashboard</a><a href="compare.html">X7ML vs X8</a></div>
<div class="card"><p class="muted">{gd.esc(message)}</p></div>
</body></html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Guard page (placeholder) written to {output_path}: {message}")


def main(fut_x8, fut_guard, history_db_path, output_path):
    gc.install_patches()
    now = datetime.now(timezone.utc)

    if not fut_guard or not os.path.exists(fut_guard) or os.path.getsize(fut_guard) == 0:
        placeholder_page(output_path, "The X8 Guard bot has no state file yet — nothing to compare. "
                                      "This fills in automatically after its first hourly state checkpoint.")
        return

    # Read the Guard start time BEFORE anything else opens the file.
    raw_start = gc.fetch_bot_start_time(fut_guard)

    gp = read_guard_parameters()
    switch = read_switch_time(gp)
    switched = switch is not None and switch <= now
    guard_note = ("OKX perpetual swaps · isolated margin · 3x leverage · cut rule from "
                  f"{switch:%Y-%m-%d %H:%M} UTC" if switch else
                  "OKX perpetual swaps · isolated margin · 3x leverage")

    b8 = gc.build_x8_mode(fut_x8, history_db_path, "futures_x8", "🟠 X8 FUTURES — OKX",
                          "OKX perpetual swaps · isolated margin · 3x leverage",
                          "config_dryrun_futures_x8.json",
                          live_price_exchange_id="okx", live_price_ccxt_options=gc.SWAP_OPTIONS)
    try:
        bg = gc.build_x8_mode(fut_guard, history_db_path, "futures_x8guard", "🟣 X8 GUARD FUTURES — OKX",
                              guard_note,
                              "config_dryrun_futures_x8guard.json",
                              live_price_exchange_id="okx", live_price_ccxt_options=gc.SWAP_OPTIONS)
    except Exception:
        traceback.print_exc()
        placeholder_page(output_path, "The Guard bot's data could not be read (see the workflow log for the traceback).")
        return

    guard_start, source = gc.determine_window_start(raw_start, bg["trades"])
    if guard_start is None:
        placeholder_page(output_path, "The Guard bot has no start time or trades yet — nothing to compare.")
        return

    # Window of the head-to-head: from the switch on (both bots), else from the Guard's start.
    start = switch if switched else guard_start
    start_label = "the switch" if switched else "the Guard bot started"
    start_source = "cut_start_date in the Guard config" if switched else source

    wallet = gd.load_portfolio_config(CONFIG_PATH)["dry_run_wallet"]
    fa = gc.load_fills(fut_x8, b8["trades"])
    fb = gc.load_fills(fut_guard, bg["trades"])
    pa = gc.prices_for(b8["trades"], gc.SWAP_OPTIONS)
    pb = gc.prices_for(bg["trades"], gc.SWAP_OPTIONS)
    ma = with_open_realized(gc.compute_window_metrics(b8["trades"], start, pa, fa, wallet, now), b8["trades"], start, wallet)
    mb = with_open_realized(gc.compute_window_metrics(bg["trades"], start, pb, fb, wallet, now), bg["trades"], start, wallet)
    ov = gc.compute_entry_overlap(b8["trades"], bg["trades"], start)

    days = (now - start).total_seconds() / 86400.0
    small = min(ma["n_closed"], mb["n_closed"]) < gc.SMALL_SAMPLE_CLOSED
    warn = (f'<p class="warn-note">⚠️ Small sample: {ma["n_closed"]} ({NAME_A}) and {mb["n_closed"]} ({NAME_B}) '
            f'closed trades — below {gc.SMALL_SAMPLE_CLOSED}, so differences here can easily be chance.</p>') if small else ""

    carry_a = gc.count_carry_over(b8["trades"], start)
    carry_b = gc.count_carry_over(bg["trades"], start)
    if switched:
        window_note = (f"Window = trades opened since <b>{gd.fmt_dt(start)} UTC</b> ({days:.1f} days ago; {gd.esc(start_source)}). "
                       f"Trades opened earlier are inherited by both bots ({NAME_A}: <b>{carry_a}</b>, {NAME_B}: <b>{carry_b}</b> "
                       f"still open or closed after the switch) and are left out of this table — see 'Inherited trades' below.")
    else:
        window_note = (f"Window = trades opened since <b>{gd.fmt_dt(start)} UTC</b> ({days:.1f} days ago; start taken from {gd.esc(start_source)}). "
                       f"{NAME_A} began this window with <b>{carry_a}</b> older trade(s) still open, occupying slots the Guard had free.")
        if switch is not None:
            window_note += f" The cut rule is scheduled for {gd.fmt_dt(switch)} UTC and is not active yet."

    cuts = [t for t in bg["trades"] if not t["is_open"] and str(t["exit_reason"] or "").startswith("cut_")
            and gc.in_window(t, start)]
    cut_total = sum(gd.to_float(t["close_profit_abs"]) for t in cuts)
    cut_line = (f'<p style="font-size:0.85rem;margin:0 0 12px 0;">Cuts by {NAME_B} since {start_label}: <b>{len(cuts)}</b> '
                f'(booked <b>{cut_total:+.2f} USDT</b>).</p>') if switched else ""

    if switched:
        about = (f"The Guard is the SAME strategy as X8 with ONE addition since {gd.fmt_dt(switch)} UTC: the cut rule — "
                 f"{gd.esc(rule_text(gp))}. Backtests on 5 six-month windows: better on 4, worse on 1; it pays in crash-type "
                 f"markets and costs profit in quiet ones. Before the switch the Guard ran with stale_range 15; that period "
                 f"is kept as its own card below. All trades opened before the switch are untouched on both bots.")
    else:
        about = ("The Guard is the SAME strategy as X8 with ONE change: the 'abandon' exit's volatility "
                 "threshold (stale_max_daily_range_pct) is 15 instead of 7, so it can also close long-stuck trades on "
                 "volatile alts. Backtests (2 windows, 1 pairlist) showed this helps in bad markets and costs a few % in good ones.")

    xlabel = "Hours since the switch" if switched else "Hours since Guard start"
    js = (EQUITY_JS.replace("__DA__", json.dumps(gc.equity_series(b8["trades"], start, now)))
                   .replace("__DB__", json.dumps(gc.equity_series(bg["trades"], start, now)))
                   .replace("__XLABEL__", xlabel))

    inherited_card = inherited_html(b8["trades"], bg["trades"], switch) if switched else ""
    before = before_card(b8["trades"], bg["trades"], guard_start, switch, wallet, now, fa, fb, pa, pb) if switched else ""

    generated_at = now.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — X8 vs X8 Guard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{gd.PAGE_STYLES}
{gc.EXTRA_CSS}
{EXTRA_CSS}
</style>
</head>
<body>

<h1>🛡️ NostalgiaForInfinity — X8 vs X8 Guard</h1>
<div class="subtitle">Generated {generated_at} UTC · dry-run (simulated) — no real funds involved · both bots on OKX futures</div>
<div class="nav">
  <a href="index.html">← Main dashboard</a>
  <a href="compare.html">X7ML vs X8</a>
  <a href="#guard-detail">Guard detail</a>
</div>

<div class="subtitle">{about}</div>

<div class="card">
  <h3>Head to head <span class="muted" style="font-weight:400;font-size:0.75rem;">— trades opened since {start_label}, both bots</span></h3>
  <p class="muted" style="font-size:0.8rem;margin:0 0 12px 0;">
    {window_note}
  </p>
  {cut_line}
  {warn}
  {metrics_table(ma, mb)}
</div>

<div class="card">
  <h3>Realized equity since {start_label} <span class="muted" style="font-weight:400;font-size:0.75rem;">— cumulative closed-trade profit, USDT</span></h3>
  <div class="chart-wrap"><canvas id="guardEquity"></canvas></div>
</div>

{divergence_html(ov, switched)}

{inherited_card}

{before}

{gc.overlap_html(ov, NAME_A, NAME_B, "🔗 Signal Overlap — X8 vs X8 Guard",
                 note="Both bots run the same entry logic, so a high match rate is expected; 'only' counts come from different open slots or timing after earlier exits diverged.")}

<div class="card"><h3>Exit reasons (closed)</h3>{exit_table(b8["trades"], bg["trades"], start)}</div>

<div id="guard-detail"></div>
{bg['section_html']}

<div class="subtitle">
  ⚠️ Small samples can look great or terrible by chance — the cut rule only matters when a trade keeps
  rebuying (3+ entries), which is rare. Judge it by the "Same entry, different exit" table and by True Total
  over time, not by closed-trade profit alone.
</div>

<script>
{js}
{bg['section_js']}
</script>

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Guard comparison page written to {output_path}")

    try:
        gd.record_snapshot(history_db_path, "futures_x8guard", bg["current_snapshot"])
    except Exception:
        traceback.print_exc()
        print("Warning: could not record the Guard history snapshot (page was still written).")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:5])
