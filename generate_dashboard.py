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


def fetch_entry_fills(db_url: str, trade_ids):
    """Per-trade history of entry (DCA/rebuy) fills, so capital-days can
    use how much was ACTUALLY locked at each point in time — not just
    the trade's final/current stake_amount applied to its whole duration.
    A trade that grew from 100 -> 500 USDT over several rebuys had far
    less capital tied up in its early days than its final size suggests;
    using the final size for the entire holding period overstates
    capital-days (and understates the return-per-capital-day rate).

    Returns {trade_id: [(fill_time, cumulative_cost_after_this_fill), ...]}
    sorted by fill_time. Falls back to None (caller uses stake_amount ×
    full duration as before) if the orders table can't be read.
    """
    if not trade_ids:
        return {}
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ft_trade_id, order_filled_date, cost, average, filled
                FROM orders
                WHERE ft_trade_id = ANY(%s) AND ft_order_side = 'buy'
                  AND order_filled_date IS NOT NULL
                ORDER BY ft_trade_id, order_filled_date ASC
            """, (list(trade_ids),))
            rows = cur.fetchall()
    except Exception as e:
        print(f"  (could not read orders table for capital-over-time reconstruction: {e})")
        return {}
    finally:
        conn.close()

    by_trade = {}
    running_cost = {}
    for r in rows:
        tid = r["ft_trade_id"]
        fill_cost = to_float(r["cost"])
        if fill_cost <= 0:
            # Fallback if 'cost' wasn't populated for this fill: reconstruct
            # from average price × filled amount.
            fill_cost = to_float(r["average"]) * to_float(r["filled"])
        running_cost[tid] = running_cost.get(tid, 0.0) + fill_cost
        by_trade.setdefault(tid, []).append((r["order_filled_date"], running_cost[tid]))
    return by_trade


def capital_days_for_trade(fills, open_date, end_date, fallback_stake, fallback_days):
    """Integrate stake-locked × time using the ACTUAL step function of
    capital committed over the trade's life (each DCA fill raises the
    'locked capital' step). Falls back to a flat stake×duration estimate
    if no fill history is available."""
    if not fills:
        return fallback_stake * fallback_days

    total = 0.0
    prev_time = open_date
    prev_cost = 0.0
    for fill_time, cumulative_cost in fills:
        seg_days = duration_days(prev_time, fill_time)
        total += prev_cost * seg_days  # capital locked BEFORE this fill landed
        prev_time = fill_time
        prev_cost = cumulative_cost
    # Final segment: from the last fill to close/now, at the final size.
    total += prev_cost * duration_days(prev_time, end_date)
    return total


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
# Capital efficiency (opportunity cost of locked-up slots)
# ------------------------------------------------------------
# "Total profit" on its own hides how long capital sat idle to earn it.
# A trade that ties up a slot for 5 days to make +0.1% is a much worse
# use of capital than one that makes +0.1% in 10 minutes and frees the
# slot for the next signal. These helpers turn profit + duration into
# a single comparable rate, and flag trades whose slot has been stuck
# far longer than the strategy's typical hold time.
# ----------------------------------------------------------------------

STUCK_MULTIPLIER = 5    # flag an open trade as "stuck" once its age exceeds
                         # this many multiples of the median closed-trade
                         # holding time (falls back to a flat 2-day cutoff
                         # when there isn't enough closed-trade history yet).
STUCK_HARD_CAP_DAYS = 3  # ...but never let one slow outlier in the closed
                          # history push the threshold above this, or a
                          # single long-but-legitimate trade could mask a
                          # genuinely stuck one with too little data.


def to_aware_utc(dt):
    """Postgres can hand back naive or aware datetimes depending on the
    column type, and this script mixes values from several queries
    (trades.open_date, orders.order_filled_date, datetime.now()) — so
    every datetime gets normalized here rather than trusting call sites
    to do it consistently."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def duration_days(start, end):
    start = to_aware_utc(start)
    end = to_aware_utc(end)
    if start is None or end is None:
        return 0.0
    delta = end - start
    return max(delta.total_seconds() / 86400.0, 0.0)


def to_float(x, default=0.0):
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default


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


def build_html(trades, live_prices, entry_fills, generated_at):
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

    # --- Capital efficiency: closed trades ---
    # capital-days = stake tied up × how long it was tied up for. This is
    # the denominator that turns raw profit into a rate comparable across
    # trades of wildly different holding times.
    closed_capital_days = 0.0
    for t in closed_with_profit:
        fills = entry_fills.get(t["id"])
        closed_capital_days += capital_days_for_trade(
            fills, t["open_date"], t["close_date"],
            fallback_stake=to_float(t["stake_amount"]),
            fallback_days=duration_days(t["open_date"], t["close_date"]),
        )
    realized_abs = sum(to_float(t["close_profit_abs"]) for t in closed_with_profit)
    realized_rate_per_capital_day = (realized_abs / closed_capital_days) if closed_capital_days > 0 else None

    closed_durations_days = [
        duration_days(t["open_date"], t["close_date"]) for t in closed_with_profit
    ]
    median_closed_duration = float(np.median(closed_durations_days)) if closed_durations_days else None
    if median_closed_duration:
        stuck_threshold_days = min(max(median_closed_duration * STUCK_MULTIPLIER, 0.5), STUCK_HARD_CAP_DAYS)
    else:
        stuck_threshold_days = 2.0

    # --- Capital efficiency: open trades (unrealized) ---
    now_utc = datetime.now(timezone.utc)
    open_unrealized_abs_total = 0.0
    open_capital_days = 0.0
    open_capital_locked_total = 0.0
    stuck_trades = []

    open_rows_html = ""
    for t in open_trades:
        stake = to_float(t["stake_amount"])  # current/final size — shown as-is in the table
        open_date = t["open_date"]
        open_date_aware = open_date.replace(tzinfo=timezone.utc) if open_date and open_date.tzinfo is None else open_date
        age_days = duration_days(open_date_aware, now_utc)

        fills = entry_fills.get(t["id"])
        trade_capital_days = capital_days_for_trade(
            fills, open_date_aware, now_utc,
            fallback_stake=stake, fallback_days=age_days,
        )

        open_capital_locked_total += stake
        open_capital_days += trade_capital_days
        is_stuck = age_days > stuck_threshold_days
        if is_stuck:
            stuck_trades.append(t)

        live = live_prices.get(t["pair"])
        if live and t["open_rate"]:
            unreal_pct = (live - t["open_rate"]) / t["open_rate"]
            unreal_abs = stake * unreal_pct
            open_unrealized_abs_total += unreal_abs
            unreal_cls = "profit-pos" if unreal_pct > 0 else "profit-neg"
            unreal_str = f'<span class="{unreal_cls}">{fmt_pct(unreal_pct)} ({unreal_abs:+.2f} USDT)</span>'
            live_str = f"{live:.6g}"
        else:
            unreal_str = '<span class="muted">live price unavailable</span>'
            live_str = '<span class="muted">—</span>'

        age_str = f"{age_days:.1f}d"
        stuck_badge = ' <span class="stuck-badge" title="Open far longer than this strategy\'s typical hold time">🐌 stuck</span>' if is_stuck else ""

        open_rows_html += f"""
        <tr>
          <td>{esc(t['pair'])}</td>
          <td>{esc(t['enter_tag'])}</td>
          <td>{fmt_dt(t['open_date'])}</td>
          <td>{age_str}{stuck_badge}</td>
          <td>{stake:.2f} USDT</td>
          <td>{t['open_rate']:.6g}</td>
          <td>{live_str}</td>
          <td>{unreal_str}</td>
        </tr>"""

    true_total_abs = realized_abs + open_unrealized_abs_total
    total_capital_days = closed_capital_days + open_capital_days
    blended_rate_per_capital_day = (true_total_abs / total_capital_days) if total_capital_days > 0 else None

    def annualized(rate_per_day):
        if rate_per_day is None:
            return None
        return rate_per_day * 365 * 100  # simple (non-compounded) extrapolation, illustrative only

    stuck_capital = sum(to_float(t["stake_amount"]) for t in stuck_trades)
    stuck_pct_of_open_capital = (
        (stuck_capital / open_capital_locked_total * 100) if open_capital_locked_total > 0 else 0
    )

    capital_eff_html = f"""
    <div class="card">
      <h3>Capital Efficiency <span class="muted" style="font-weight:400;font-size:0.75rem;">— accounts for how long capital was tied up, not just the raw % per trade</span></h3>
      <div class="grid" style="margin-bottom:0;">
        <div class="stat-card">
          <div class="label">Realized Profit</div>
          <div class="value {'profit-pos' if realized_abs >= 0 else 'profit-neg'}">{realized_abs:+.2f} USDT</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">closed trades only</div>
        </div>
        <div class="stat-card">
          <div class="label">Unrealized P/L (open)</div>
          <div class="value {'profit-pos' if open_unrealized_abs_total >= 0 else 'profit-neg'}">{open_unrealized_abs_total:+.2f} USDT</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">{open_capital_locked_total:.2f} USDT currently locked in open trades</div>
        </div>
        <div class="stat-card">
          <div class="label">True Total (realized + unrealized)</div>
          <div class="value {'profit-pos' if true_total_abs >= 0 else 'profit-neg'}">{true_total_abs:+.2f} USDT</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">what you'd have if everything closed right now</div>
        </div>
        <div class="stat-card">
          <div class="label">Stuck Trades</div>
          <div class="value {'profit-neg' if stuck_trades else ''}">{len(stuck_trades)} / {len(open_trades)}</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">{stuck_pct_of_open_capital:.0f}% of open capital, open &gt;{stuck_threshold_days:.1f}d</div>
        </div>
      </div>
      <p style="margin-top:16px;margin-bottom:4px;">
        Realized return per capital-day: <b>{f'{realized_rate_per_capital_day*100:.4f}%' if realized_rate_per_capital_day is not None else '—'}</b>
        {f'(≈ {annualized(realized_rate_per_capital_day):.1f}%/yr if repeated — simple, non-compounded extrapolation)' if realized_rate_per_capital_day is not None else ''}
      </p>
      <p style="margin-bottom:4px;">
        Blended return per capital-day (incl. unrealized): <b>{f'{blended_rate_per_capital_day*100:.4f}%' if blended_rate_per_capital_day is not None else '—'}</b>
        {f'(≈ {annualized(blended_rate_per_capital_day):.1f}%/yr equivalent)' if blended_rate_per_capital_day is not None else ''}
      </p>
      <p class="muted" style="font-size:0.8rem;margin-top:12px;margin-bottom:0;">
        "Capital-day" = stake size × days held, tracked step-by-step through each DCA/rebuy
        fill (so a trade that grew from 100 → 500 USDT over several rebuys is charged the
        smaller amount for its early days, not its final size for the whole holding period).
        This is what lets a 4-day trade for +0.1% and a 9-minute trade for +0.1% be compared
        fairly, and it's why "Total Profit" alone can look fine while several slots are
        quietly stuck.
      </p>
    </div>"""

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
  .stuck-badge {{
    display: inline-block; background: rgba(248,81,73,0.15); color: var(--neg);
    border: 1px solid rgba(248,81,73,0.35); border-radius: 4px;
    font-size: 0.7rem; padding: 1px 6px; margin-left: 4px; white-space: nowrap;
  }}
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
  <div class="stat-card"><div class="label">Total Profit (closed only)</div>
    <div class="value {'profit-pos' if total_profit_abs >= 0 else 'profit-neg'}">{total_profit_abs:+.2f} USDT</div>
    <div class="muted" style="font-size:0.7rem;margin-top:2px;">excludes unrealized — see Capital Efficiency below</div></div>
</div>

{capital_eff_html}

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
    <tr><th>Pair</th><th>Enter Tag</th><th>Opened</th><th>Age</th><th>Capital Locked</th><th>Open Rate</th><th>Live Price</th><th>Unrealized P/L</th></tr>
    {open_rows_html if open_rows_html else '<tr><td colspan="8" class="muted">No open trades</td></tr>'}
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
    entry_fills = fetch_entry_fills(db_url, [t["id"] for t in trades])

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(trades, live_prices, entry_fills, generated_at)

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Dashboard written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
