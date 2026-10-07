"""
generate_dashboard_guard.py

Builds docs/guard.html -- a THIRD, fully separate dashboard page that
compares the two X8 FUTURES bots head to head:

    NostalgiaForInfinityX8   vs   NostalgiaForInfinityX8Guard

It IMPORTS generate_dashboard.py and generate_dashboard_compare.py and
reuses their data-fetching and helper functions; neither file is edited,
so index.html and compare.html are unaffected (zero regression risk).
The SQLite state files are read READ-ONLY, like the other scripts.

FAIRNESS RULE (same idea as compare.html)
-----------------------------------------
Every head-to-head number is computed ONLY from trades OPENED at/after
the Guard bot's start time (freqtrade `bot_start_time`, falling back to
its first trade), for BOTH bots. Residual unfairness (shown on the page):
the X8 bot enters the window with older trades still open.

WHAT TO LOOK AT
---------------
The Guard differs from X8 in ONE thing: its stale_range threshold (15 vs 7)
for the 'abandon' exit. So most entries are identical; the interesting rows
are the ones listed in "Same entry, different exit" -- trades the Guard
closed earlier (exit_bad_trade_abandon_*) than X8 did.

If the Guard bot's state file is missing or empty the page says so and
nothing else breaks.

Usage:
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
      x: { type: 'linear', title: { display: true, text: 'Hours since Guard start', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
      y: { title: { display: true, text: 'Cumulative realized profit (USDT)', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#30363d' } }
    },
    plugins: { legend: { labels: { color: '#c9d1d9' } } }
  }
});
"""


def metrics_table(ma, mb):
    r = gc.cmp_row
    rows = [
        r("Trades opened (in window)", str(ma["n_opened"]), str(mb["n_opened"])),
        r("Closed / open now", f"{ma['n_closed']} / {ma['n_open']}", f"{mb['n_closed']} / {mb['n_open']}"),
        r("Win rate (closed)",
          gc.pct_s(ma["win_rate"]).lstrip("+") if ma["win_rate"] is not None else "—",
          gc.pct_s(mb["win_rate"]).lstrip("+") if mb["win_rate"] is not None else "—",
          ma["win_rate"], mb["win_rate"], True),
        r("Realized profit", gc.money(ma["realized"]), gc.money(mb["realized"]),
          ma["realized"], mb["realized"], True),
        r("Unrealized P/L (open)", gc.money(ma["unrealized"]), gc.money(mb["unrealized"]),
          ma["unrealized"], mb["unrealized"], True),
        r("<b>True Total</b> (realized + unrealized)",
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


def divergence_html(ov):
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
    return f"""
<div class="card">
  <h3>🔀 Same entry, different exit <span class="muted" style="font-weight:400;font-size:0.75rem;">— where the Guard's higher abandon threshold actually changed the outcome</span></h3>
  {body}
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

    b8 = gc.build_x8_mode(fut_x8, history_db_path, "futures_x8", "🟠 X8 FUTURES — OKX",
                          "OKX perpetual swaps · isolated margin · 3x leverage",
                          "config_dryrun_futures_x8.json",
                          live_price_exchange_id="okx", live_price_ccxt_options=gc.SWAP_OPTIONS)
    try:
        bg = gc.build_x8_mode(fut_guard, history_db_path, "futures_x8guard", "🟣 X8 GUARD FUTURES — OKX",
                              "OKX perpetual swaps · isolated margin · 3x leverage · stale_range 15",
                              "config_dryrun_futures_x8guard.json",
                              live_price_exchange_id="okx", live_price_ccxt_options=gc.SWAP_OPTIONS)
    except Exception:
        traceback.print_exc()
        placeholder_page(output_path, "The Guard bot's data could not be read (see the workflow log for the traceback).")
        return

    start, source = gc.determine_window_start(raw_start, bg["trades"])
    if start is None:
        placeholder_page(output_path, "The Guard bot has no start time or trades yet — nothing to compare.")
        return

    wallet = gd.load_portfolio_config("config_dryrun_futures_x8guard.json")["dry_run_wallet"]
    fa = gc.load_fills(fut_x8, b8["trades"])
    fb = gc.load_fills(fut_guard, bg["trades"])
    ma = gc.compute_window_metrics(b8["trades"], start, gc.prices_for(b8["trades"], gc.SWAP_OPTIONS), fa, wallet, now)
    mb = gc.compute_window_metrics(bg["trades"], start, gc.prices_for(bg["trades"], gc.SWAP_OPTIONS), fb, wallet, now)
    ov = gc.compute_entry_overlap(b8["trades"], bg["trades"], start)

    carry = gc.count_carry_over(b8["trades"], start)
    days = (now - start).total_seconds() / 86400.0
    small = min(ma["n_closed"], mb["n_closed"]) < gc.SMALL_SAMPLE_CLOSED
    warn = (f'<p class="warn-note">⚠️ Small sample: {ma["n_closed"]} ({NAME_A}) and {mb["n_closed"]} ({NAME_B}) '
            f'closed trades — below {gc.SMALL_SAMPLE_CLOSED}, so differences here can easily be chance.</p>') if small else ""

    js = (EQUITY_JS.replace("__DA__", json.dumps(gc.equity_series(b8["trades"], start, now)))
                   .replace("__DB__", json.dumps(gc.equity_series(bg["trades"], start, now))))

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

<div class="subtitle">The Guard is the SAME strategy as X8 with ONE change: the 'abandon' exit's volatility
  threshold (stale_max_daily_range_pct) is 15 instead of 7, so it can also close long-stuck trades on
  volatile alts. Backtests (2 windows, 1 pairlist) showed this helps in bad markets and costs a few % in good ones.</div>

<div class="card">
  <h3>Head to head <span class="muted" style="font-weight:400;font-size:0.75rem;">— trades opened since the Guard bot started, both bots</span></h3>
  <p class="muted" style="font-size:0.8rem;margin:0 0 12px 0;">
    Window = trades opened since <b>{gd.fmt_dt(start)} UTC</b> ({days:.1f} days ago; start taken from {gd.esc(source)}).
    {NAME_A} began this window with <b>{carry}</b> older trade(s) still open, occupying slots the Guard had free.
  </p>
  {warn}
  {metrics_table(ma, mb)}
</div>

<div class="card">
  <h3>Realized equity since Guard start <span class="muted" style="font-weight:400;font-size:0.75rem;">— cumulative closed-trade profit, USDT</span></h3>
  <div class="chart-wrap"><canvas id="guardEquity"></canvas></div>
</div>

{divergence_html(ov)}

{gc.overlap_html(ov, NAME_A, NAME_B, "🔗 Signal Overlap — X8 vs X8 Guard",
                 note="Both bots run the same entry logic, so a high match rate is expected; 'only' counts come from different open slots or timing after earlier exits diverged.")}

<div class="card"><h3>Exit reasons (closed)</h3>{exit_table(b8["trades"], bg["trades"], start)}</div>

<div id="guard-detail"></div>
{bg['section_html']}

<div class="subtitle">
  ⚠️ Small samples can look great or terrible by chance — the Guard only matters when a trade gets stuck
  for 14+ days, which is rare. Judge it by the "Same entry, different exit" table, not by total profit alone.
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
