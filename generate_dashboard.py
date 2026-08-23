"""
generate_dashboard.py

Builds a complete, self-contained HTML dashboard for the NFI dry-run bot:
  - Summary cards (open/closed counts, win rate, total profit)
  - Equity curve chart (cumulative profit over time, closed trades)
  - Open trades table, with LIVE unrealized P/L (fetched from KuCoin's
    public ticker API — no API key needed, read-only, doesn't touch the
    running bot)
  - Closed trades table
  - Per-pair performance breakdown
  - Exit-reason breakdown chart
  - The same statistical-significance checks as signal_stats.py

Reads from the same Postgres database the bot itself uses — this script
is entirely read-only and never writes to the database or the exchange.

Usage:
    python generate_dashboard.py "<postgres-connection-string>" <output_html_path>
"""

import sys
import json
from datetime import datetime, timezone

import numpy as np

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Missing dependency. Install with: pip install psycopg2-binary --break-system-packages")
    sys.exit(1)


# ----------------------------------------------------------------------
# Data fetching
# ----------------------------------------------------------------------

def fetch_trades(db_url: str):
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, pair, is_open, enter_tag, exit_reason,
                       open_date, close_date, open_rate, close_rate,
                       amount, stake_amount, close_profit, close_profit_abs
                FROM trades
                ORDER BY open_date ASC
            """)
            return cur.fetchall()
    finally:
        conn.close()


def fetch_live_prices(pairs):
    """Best-effort live price fetch for unrealized P/L on open trades.
    Returns {pair: last_price} — pairs that fail to fetch are simply
    omitted, and the dashboard shows 'live price unavailable' for those
    rather than crashing the whole report."""
    prices = {}
    try:
        import ccxt
        exchange = ccxt.kucoin()
        exchange.load_markets()
        for pair in pairs:
            try:
                ticker = exchange.fetch_ticker(pair)
                prices[pair] = ticker["last"]
            except Exception as e:
                print(f"  (could not fetch live price for {pair}: {e})")
    except Exception as e:
        print(f"Live price fetching unavailable this run: {e}")
    return prices


# ----------------------------------------------------------------------
# Stats (same methodology as signal_stats.py)
# ----------------------------------------------------------------------

def binomial_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    from math import comb
    def pmf(x):
        return comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    obs_p = pmf(k)
    total = sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs_p + 1e-12)
    return min(1.0, total)


def bootstrap_ci(profits: np.ndarray, n_boot: int = 10000, alpha: float = 0.05):
    n = len(profits)
    rng = np.random.default_rng(42)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(profits, size=n, replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ----------------------------------------------------------------------
# HTML building
# ----------------------------------------------------------------------

def esc(s):
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_dt(dt):
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M")


def fmt_pct(x, signed=True):
    if x is None:
        return "—"
    sign = "+" if signed and x > 0 else ""
    return f"{sign}{x*100:.2f}%"


def build_html(trades, live_prices, generated_at):
    open_trades = [t for t in trades if t["is_open"]]
    closed_trades = [t for t in trades if not t["is_open"]]
    closed_with_profit = [t for t in closed_trades if t["close_profit"] is not None]
    profits = np.array([float(t["close_profit"]) for t in closed_with_profit])
    n = len(profits)
    wins = int((profits > 0).sum()) if n else 0
    losses = n - wins
    win_rate = wins / n if n else 0

    # Equity curve (cumulative profit % over closed trades, in order)
    equity_labels = []
    equity_values = []
    cum = 0.0
    for t in closed_with_profit:
        cum += float(t["close_profit"]) * 100
        equity_labels.append(fmt_dt(t["close_date"]))
        equity_values.append(round(cum, 3))

    # Per-pair breakdown
    pair_stats = {}
    for t in closed_with_profit:
        pair_stats.setdefault(t["pair"], []).append(float(t["close_profit"]))
    pair_rows = []
    for pair, plist in sorted(pair_stats.items(), key=lambda x: -len(x[1])):
        arr = np.array(plist)
        w = int((arr > 0).sum())
        pair_rows.append((pair, len(arr), w / len(arr), arr.mean()))

    # Exit reason breakdown
    reason_counts = {}
    for t in closed_trades:
        r = t["exit_reason"] or "unknown"
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_labels = list(reason_counts.keys())
    reason_values = [reason_counts[k] for k in reason_labels]

    # Significance
    sig_html = ""
    if n >= 8:
        p_value = binomial_test_two_sided(wins, n, 0.5)
        lo, hi = bootstrap_ci(profits)
        sig_note = ("statistically distinguishable from 50/50" if p_value < 0.05
                     else "NOT yet distinguishable from chance")
        ci_note = ("does NOT include 0 — mean profit tentatively confirmed" if not (lo < 0 < hi)
                    else "includes 0 — mean profit NOT yet statistically confirmed")
        sig_html = f"""
        <div class="card">
          <h3>Statistical Significance</h3>
          <p>Win-rate p-value (two-sided binomial vs. 50%): <b>{p_value:.4f}</b> — {sig_note}</p>
          <p>95% bootstrap CI on mean profit/trade: <b>[{lo*100:.2f}%, {hi*100:.2f}%]</b> — {ci_note}</p>
        </div>"""
    else:
        sig_html = f"""
        <div class="card">
          <h3>Statistical Significance</h3>
          <p>⚠️ Only {n} closed trades — need at least ~8 for a meaningful significance test. Treat all numbers below as provisional.</p>
        </div>"""

    # Open trades rows (with live unrealized P/L)
    open_rows_html = ""
    for t in open_trades:
        live = live_prices.get(t["pair"])
        if live and t["open_rate"]:
            unreal_pct = (live - t["open_rate"]) / t["open_rate"]
            unreal_cls = "profit-pos" if unreal_pct > 0 else "profit-neg"
            unreal_str = f'<span class="{unreal_cls}">{fmt_pct(unreal_pct)}</span>'
            live_str = f"{live:.6g}"
        else:
            unreal_str = '<span class="muted">live price unavailable</span>'
            live_str = '<span class="muted">—</span>'
        open_rows_html += f"""
        <tr>
          <td>{esc(t['pair'])}</td>
          <td>{esc(t['enter_tag'])}</td>
          <td>{fmt_dt(t['open_date'])}</td>
          <td>{t['open_rate']:.6g}</td>
          <td>{live_str}</td>
          <td>{unreal_str}</td>
        </tr>"""

    # Closed trades rows
    closed_rows_html = ""
    for t in sorted(closed_trades, key=lambda x: x["close_date"] or datetime.min, reverse=True):
        profit = t["close_profit"]
        cls = "profit-pos" if (profit or 0) > 0 else "profit-neg"
        closed_rows_html += f"""
        <tr>
          <td>{esc(t['pair'])}</td>
          <td>{esc(t['enter_tag'])}</td>
          <td>{fmt_dt(t['open_date'])}</td>
          <td>{fmt_dt(t['close_date'])}</td>
          <td class="{cls}">{fmt_pct(profit)}</td>
          <td>{esc(t['exit_reason'])}</td>
        </tr>"""

    pair_rows_html = ""
    for pair, cnt, wr, avg in pair_rows:
        cls = "profit-pos" if avg > 0 else "profit-neg"
        pair_rows_html += f"""
        <tr>
          <td>{esc(pair)}</td>
          <td>{cnt}</td>
          <td>{wr*100:.0f}%</td>
          <td class="{cls}">{fmt_pct(avg)}</td>
        </tr>"""

    total_profit_abs = sum(float(t["close_profit_abs"] or 0) for t in closed_with_profit)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --pos: #3fb950; --neg: #f85149;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 24px; line-height: 1.5;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px;
  }}
  .stat-card .label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  .stat-card .value {{ font-size: 1.6rem; font-weight: 600; margin-top: 4px; }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 24px;
  }}
  .card h3 {{ margin-top: 0; font-size: 1.05rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.03em; }}
  tr:hover {{ background: rgba(255,255,255,0.02); }}
  .profit-pos {{ color: var(--pos); font-weight: 600; }}
  .profit-neg {{ color: var(--neg); font-weight: 600; }}
  .muted {{ color: var(--muted); }}
  .chart-wrap {{ position: relative; height: 300px; }}
  .two-col {{ display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }}
  @media (max-width: 800px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .table-scroll {{ overflow-x: auto; }}
</style>
</head>
<body>

<h1>🤖 NostalgiaForInfinity — Dry-Run Dashboard</h1>
<div class="subtitle">Generated {generated_at} UTC · dry-run (simulated) — no real funds involved · KuCoin market data</div>

<div class="grid">
  <div class="stat-card"><div class="label">Open Trades</div><div class="value">{len(open_trades)}</div></div>
  <div class="stat-card"><div class="label">Closed Trades</div><div class="value">{len(closed_trades)}</div></div>
  <div class="stat-card"><div class="label">Win Rate</div><div class="value">{win_rate*100:.1f}%</div></div>
  <div class="stat-card"><div class="label">Total Profit</div>
    <div class="value {'profit-pos' if total_profit_abs >= 0 else 'profit-neg'}">{total_profit_abs:+.2f} USDT</div></div>
</div>

<div class="two-col">
  <div class="card">
    <h3>Equity Curve (cumulative %, closed trades)</h3>
    <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
  </div>
  <div class="card">
    <h3>Exit Reasons</h3>
    <div class="chart-wrap"><canvas id="reasonChart"></canvas></div>
  </div>
</div>

{sig_html}

<div class="card">
  <h3>Open Trades ({len(open_trades)})</h3>
  <div class="table-scroll">
  <table>
    <tr><th>Pair</th><th>Enter Tag</th><th>Opened</th><th>Open Rate</th><th>Live Price</th><th>Unrealized P/L</th></tr>
    {open_rows_html if open_rows_html else '<tr><td colspan="6" class="muted">No open trades</td></tr>'}
  </table>
  </div>
</div>

<div class="card">
  <h3>Per-Pair Performance (closed trades)</h3>
  <div class="table-scroll">
  <table>
    <tr><th>Pair</th><th>Trades</th><th>Win Rate</th><th>Avg Profit</th></tr>
    {pair_rows_html if pair_rows_html else '<tr><td colspan="4" class="muted">No closed trades yet</td></tr>'}
  </table>
  </div>
</div>

<div class="card">
  <h3>Closed Trades ({len(closed_trades)})</h3>
  <div class="table-scroll">
  <table>
    <tr><th>Pair</th><th>Enter Tag</th><th>Opened</th><th>Closed</th><th>Profit</th><th>Exit Reason</th></tr>
    {closed_rows_html if closed_rows_html else '<tr><td colspan="6" class="muted">No closed trades yet</td></tr>'}
  </table>
  </div>
</div>

<div class="subtitle">
  ⚠️ Small sample sizes can look great or terrible by chance. Entries cancelled before filling (limit-order timeouts) aren't counted here — only trades that actually opened.
</div>

<script>
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_labels)},
    datasets: [{{
      label: 'Cumulative profit %',
      data: {json.dumps(equity_values)},
      borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
      fill: true, tension: 0.2, pointRadius: 2
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8 }}, grid: {{ color: '#30363d' }} }},
      y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});

new Chart(document.getElementById('reasonChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(reason_labels)},
    datasets: [{{
      data: {json.dumps(reason_values)},
      backgroundColor: ['#58a6ff','#3fb950','#f85149','#d29922','#a371f7','#39c5cf','#f778ba']
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#c9d1d9', font: {{ size: 10 }} }} }} }}
  }}
}});
</script>

</body>
</html>"""
    return html


def main(db_url: str, output_path: str):
    trades = fetch_trades(db_url)
    open_pairs = list({t["pair"] for t in trades if t["is_open"]})
    print(f"Fetched {len(trades)} trade records ({len(open_pairs)} open pairs).")

    live_prices = fetch_live_prices(open_pairs) if open_pairs else {}

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(trades, live_prices, generated_at)

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
