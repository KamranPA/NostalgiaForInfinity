"""
generate_dashboard.py

Builds ONE combined, self-contained HTML dashboard covering BOTH the spot
(KuCoin) and futures (OKX) dry-run bots side by side:
  - A top "Spot vs Futures" comparison bar (win rate, True Total, capital
    efficiency, stuck trades) for an at-a-glance comparison
  - Then a full section per market (same content each side always had):
    summary cards, Capital Efficiency, Portfolio Utilization, equity
    curve, exit-reason chart, True-Total-over-time trend chart,
    statistical significance, open/closed trades tables, tag-family and
    per-pair breakdowns

Reads from the same Postgres database/project for both — spot's tables
live in the default `public` schema, futures' in a separate `futures`
schema (same database, fully isolated tables, no new project needed).
The only writes this script performs are appending one row per run to
each market's own nfi_dashboard_snapshots table (auto-created if
missing) — never touches `trades`, `orders`, or anything the running
bots read or write.

Usage:
    python generate_dashboard.py "<spot-db-url>" "<futures-db-url>" <output_html_path>
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


# ----------------------------------------------------------------------
# Historical snapshots (trend over time)
# ------------------------------------------------------------
# The dashboard is otherwise a pure snapshot — it can't show whether the
# "True Total" / stuck-trade situation is improving or getting worse
# without something to compare against. This adds one small extra table
# (per market's own schema/database) that a new row gets appended to on
# every dashboard run. This is the only thing this script ever writes —
# it only ever INSERTs into its own dedicated table and never touches
# `trades`/`orders`/anything the bot itself reads or writes.
# ----------------------------------------------------------------------

SNAPSHOT_HISTORY_LIMIT = 180  # ~7.5 days of hourly snapshots — plenty for
                               # a trend chart without the table or the
                               # page growing unbounded forever.


def ensure_snapshot_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS nfi_dashboard_snapshots (
                id SERIAL PRIMARY KEY,
                snapshot_time TIMESTAMP NOT NULL,
                open_trades INT NOT NULL,
                closed_trades INT NOT NULL,
                win_rate DOUBLE PRECISION,
                realized_profit_abs DOUBLE PRECISION,
                unrealized_pl_abs DOUBLE PRECISION,
                true_total_abs DOUBLE PRECISION,
                stuck_trades INT,
                open_capital_locked DOUBLE PRECISION
            )
        """)
    conn.commit()


def fetch_snapshot_history(db_url: str, limit: int = SNAPSHOT_HISTORY_LIMIT):
    """Past dashboard runs, oldest first, for the trend chart. Returns an
    empty list (never raises) if the table doesn't exist yet — e.g. the
    very first time this runs after deploying this feature."""
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"  (could not connect for snapshot history: {e})")
        return []
    try:
        ensure_snapshot_table(conn)
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT snapshot_time, true_total_abs, stuck_trades,
                       realized_profit_abs, unrealized_pl_abs
                FROM nfi_dashboard_snapshots
                ORDER BY snapshot_time DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
        return list(reversed(rows))  # oldest first, for left-to-right charting
    except Exception as e:
        print(f"  (could not read snapshot history, starting fresh: {e})")
        return []
    finally:
        conn.close()


def record_snapshot(db_url: str, metrics: dict):
    """Appends one row for this run. Failures here must never take down
    dashboard generation — the dashboard itself is far more important
    than the trend chart having an unbroken history."""
    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"  (could not connect to record snapshot: {e})")
        return
    try:
        ensure_snapshot_table(conn)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO nfi_dashboard_snapshots
                    (snapshot_time, open_trades, closed_trades, win_rate,
                     realized_profit_abs, unrealized_pl_abs, true_total_abs,
                     stuck_trades, open_capital_locked)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                metrics["snapshot_time"], metrics["open_trades"], metrics["closed_trades"],
                metrics["win_rate"], metrics["realized_profit_abs"], metrics["unrealized_pl_abs"],
                metrics["true_total_abs"], metrics["stuck_trades"], metrics["open_capital_locked"],
            ))
        conn.commit()
    except Exception as e:
        print(f"  (could not record snapshot, continuing anyway: {e})")
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


def fetch_live_prices(pairs, exchange_id="kucoin", ccxt_options=None):
    """Best-effort live price fetch for unrealized P/L on open trades.
    Returns {pair: last_price} — pairs that fail to fetch are simply
    omitted, and the dashboard shows 'live price unavailable' for those
    rather than crashing the whole report.

    exchange_id/ccxt_options let this work for either market: spot uses
    plain ccxt.kucoin(); futures uses ccxt.okx() with
    options={'defaultType': 'swap'} so symbols resolve against OKX's
    perpetual-swap market (matching freqtrade's own futures pair format,
    e.g. 'BTC/USDT:USDT') instead of its spot market.
    """
    prices = {}
    try:
        import ccxt
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({"options": ccxt_options}) if ccxt_options else exchange_class()
        exchange.load_markets()
        for pair in pairs:
            try:
                ticker = exchange.fetch_ticker(pair)
                prices[pair] = ticker["last"]
            except Exception as e:
                print(f"  (could not fetch live price for {pair} on {exchange_id}: {e})")
    except Exception as e:
        print(f"Live price fetching unavailable this run for {exchange_id}: {e}")
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


# Tag-family ranges, confirmed from NostalgiaForInfinityX7.py's own
# long_*_mode_tags lists. Purely cosmetic (labels a tag number with the
# human-readable mode it belongs to) — never used for any calculation.
# Futures-only short-side tags (501-671) are included too, since the
# futures market can hold short positions the spot market never can.
TAG_FAMILIES = [
    (1, 13, "Normal"),
    (21, 26, "Pump"),
    (41, 53, "Quick"),
    (61, 65, "Rebuy"),
    (81, 82, "High Profit"),
    (101, 110, "Rapid"),
    (120, 120, "Grind"),
    (121, 121, "BTC"),
    (141, 145, "Top Coins"),
    (161, 173, "Scalp"),
    (501, 513, "Short Normal"),
    (521, 526, "Short Pump"),
    (541, 553, "Short Quick"),
    (561, 565, "Short Rebuy"),
    (581, 582, "Short High Profit"),
    (601, 610, "Short Rapid"),
    (620, 620, "Short Grind"),
    (621, 621, "Short BTC"),
    (641, 645, "Short Top Coins"),
    (661, 673, "Short Scalp"),
]


def tag_family_name(enter_tag):
    try:
        n = int(str(enter_tag).strip())
    except (TypeError, ValueError):
        return None
    for lo, hi, name in TAG_FAMILIES:
        if lo <= n <= hi:
            return name
    return None


def fmt_tag(enter_tag):
    fam = tag_family_name(enter_tag)
    tag_str = esc(str(enter_tag)) if enter_tag is not None else "—"
    return f"{tag_str} <span class=\"muted\">({fam})</span>" if fam else tag_str


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


def annualized(rate_per_day):
    if rate_per_day is None:
        return None
    return rate_per_day * 365 * 100  # simple (non-compounded) extrapolation, illustrative only


PAGE_STYLES = """
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --muted: #8b949e; --accent: #58a6ff;
    --pos: #3fb950; --neg: #f85149;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 24px; line-height: 1.5;
  }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  h2.market-heading { font-size: 1.25rem; margin: 40px 0 4px 0; padding-top: 24px; border-top: 1px solid var(--border); }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px;
  }
  .stat-card .label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
  .stat-card .value { font-size: 1.6rem; font-weight: 600; margin-top: 4px; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 20px; margin-bottom: 24px;
  }
  .card h3 { margin-top: 0; font-size: 1.05rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.03em; }
  tr:hover { background: rgba(255,255,255,0.02); }
  .profit-pos { color: var(--pos); font-weight: 600; }
  .profit-neg { color: var(--neg); font-weight: 600; }
  .muted { color: var(--muted); }
  .stuck-badge {
    display: inline-block; background: rgba(248,81,73,0.15); color: var(--neg);
    border: 1px solid rgba(248,81,73,0.35); border-radius: 4px;
    font-size: 0.7rem; padding: 1px 6px; margin-left: 4px; white-space: nowrap;
  }
  .chart-wrap { position: relative; height: 300px; }
  .two-col { display: grid; grid-template-columns: 2fr 1fr; gap: 24px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } }
  .table-scroll { overflow-x: auto; }
  .compare-table td:not(:first-child), .compare-table th:not(:first-child) { text-align: right; }
"""


def build_mode_section(trades, live_prices, entry_fills, portfolio_cfg, snapshot_history,
                        mode_id, mode_title, market_note):
    """Builds the full report for ONE market (spot or futures): every
    card/table/chart this dashboard has always had. mode_id ('spot' /
    'futures') suffixes every canvas element id so both sections' charts
    can coexist on the same combined page without id collisions.

    Returns a dict: {section_html, section_js, current_snapshot, compare}
    — section_html/section_js get concatenated into the combined page by
    build_combined_html(); current_snapshot is what main() persists via
    record_snapshot(); compare is the small set of numbers used in the
    top-of-page Spot vs Futures comparison bar.
    """
    open_trades = [t for t in trades if t["is_open"]]
    closed_trades = [t for t in trades if not t["is_open"]]
    closed_with_profit = [t for t in closed_trades if t["close_profit"] is not None]
    profits = np.array([float(t["close_profit"]) for t in closed_with_profit])
    n = len(profits)
    wins = int((profits > 0).sum()) if n else 0
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

    # Tag-family breakdown (closed trades)
    open_tag_set = {str(t["enter_tag"]) for t in open_trades if t["enter_tag"] is not None}
    tag_stats = {}
    for t in closed_with_profit:
        tag_stats.setdefault(str(t["enter_tag"]), []).append(float(t["close_profit"]))
    tag_rows = []
    for tag, plist in sorted(tag_stats.items(), key=lambda x: -sum(x[1])):
        arr = np.array(plist)
        w = int((arr > 0).sum())
        tag_rows.append((tag, len(arr), w / len(arr), arr.mean(), arr.sum(), tag in open_tag_set))

    # Exit reason breakdown
    reason_counts = {}
    for t in closed_trades:
        r = t["exit_reason"] or "unknown"
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_labels = list(reason_counts.keys())
    reason_values = [reason_counts[k] for k in reason_labels]

    # Significance
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
        stake = to_float(t["stake_amount"])
        open_date = t["open_date"]
        open_date_aware = to_aware_utc(open_date)
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

        n_entry_fills = len(fills) if fills else 1
        n_rebuys = max(n_entry_fills - 1, 0)
        rebuy_str = f"{n_rebuys}" if fills else '<span class="muted">n/a</span>'

        open_rows_html += f"""
        <tr>
          <td>{esc(t['pair'])}</td>
          <td>{fmt_tag(t['enter_tag'])}</td>
          <td>{fmt_dt(t['open_date'])}</td>
          <td>{age_str}{stuck_badge}</td>
          <td>{rebuy_str}</td>
          <td>{stake:.2f} USDT</td>
          <td>{t['open_rate']:.6g}</td>
          <td>{live_str}</td>
          <td>{unreal_str}</td>
        </tr>"""

    true_total_abs = realized_abs + open_unrealized_abs_total
    total_capital_days = closed_capital_days + open_capital_days
    blended_rate_per_capital_day = (true_total_abs / total_capital_days) if total_capital_days > 0 else None

    stuck_capital = sum(to_float(t["stake_amount"]) for t in stuck_trades)
    stuck_pct_of_open_capital = (
        (stuck_capital / open_capital_locked_total * 100) if open_capital_locked_total > 0 else 0
    )

    current_snapshot = {
        "snapshot_time": datetime.now(timezone.utc),
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "win_rate": win_rate,
        "realized_profit_abs": realized_abs,
        "unrealized_pl_abs": open_unrealized_abs_total,
        "true_total_abs": true_total_abs,
        "stuck_trades": len(stuck_trades),
        "open_capital_locked": open_capital_locked_total,
    }

    trend_points = list(snapshot_history) + [current_snapshot]
    trend_labels = [
        (p["snapshot_time"].strftime("%m-%d %H:%M") if hasattr(p["snapshot_time"], "strftime") else str(p["snapshot_time"]))
        for p in trend_points
    ]
    trend_true_total = [round(to_float(p["true_total_abs"]), 2) for p in trend_points]
    trend_stuck = [int(p["stuck_trades"]) if p["stuck_trades"] is not None else 0 for p in trend_points]
    has_trend_history = len(trend_points) >= 2

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

    max_open_trades = portfolio_cfg.get("max_open_trades", 8)
    dry_run_wallet = portfolio_cfg.get("dry_run_wallet", 1000.0)
    slots_used = len(open_trades)
    slots_free = max(max_open_trades - slots_used, 0)
    slots_pct = (slots_used / max_open_trades * 100) if max_open_trades > 0 else 0
    wallet_pct = (open_capital_locked_total / dry_run_wallet * 100) if dry_run_wallet > 0 else 0

    portfolio_html = f"""
    <div class="card">
      <h3>Portfolio Utilization</h3>
      <div class="grid" style="margin-bottom:0;">
        <div class="stat-card">
          <div class="label">Slots Used</div>
          <div class="value">{slots_used} / {max_open_trades}</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">{slots_free} free — {slots_pct:.0f}% deployed</div>
        </div>
        <div class="stat-card">
          <div class="label">Capital Deployed</div>
          <div class="value">{open_capital_locked_total:.2f} / {dry_run_wallet:.0f} USDT</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">{wallet_pct:.0f}% of dry-run wallet</div>
        </div>
      </div>
    </div>"""

    closed_rows_html = ""
    for t in sorted(closed_trades, key=lambda x: x["close_date"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True):
        profit = t["close_profit"]
        cls = "profit-pos" if (profit or 0) > 0 else "profit-neg"
        closed_rows_html += f"""
        <tr>
          <td>{esc(t['pair'])}</td>
          <td>{fmt_tag(t['enter_tag'])}</td>
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

    tag_rows_html = ""
    for tag, cnt, wr, avg, total, in_use in tag_rows:
        cls = "profit-pos" if avg > 0 else "profit-neg"
        in_use_badge = ' <span class="stuck-badge" style="background:rgba(88,166,255,0.15);color:#58a6ff;border-color:rgba(88,166,255,0.35);">● open now</span>' if in_use else ""
        tag_rows_html += f"""
        <tr>
          <td>{fmt_tag(tag)}{in_use_badge}</td>
          <td>{cnt}</td>
          <td>{wr*100:.0f}%</td>
          <td class="{cls}">{fmt_pct(avg)}</td>
          <td class="{cls}">{fmt_pct(total, signed=True)}</td>
        </tr>"""

    total_profit_abs = sum(float(t["close_profit_abs"] or 0) for t in closed_with_profit)

    # Canvas ids are suffixed per-mode so both sections' Chart.js
    # instances can coexist on the same combined page.
    eq_id = f"equityChart_{mode_id}"
    reason_id = f"reasonChart_{mode_id}"
    trend_id = f"trendChart_{mode_id}"

    section_html = f"""
<h2 class="market-heading">{mode_title}</h2>
<div class="subtitle">{market_note}</div>

<div class="grid">
  <div class="stat-card"><div class="label">Open Trades</div><div class="value">{len(open_trades)}</div></div>
  <div class="stat-card"><div class="label">Closed Trades</div><div class="value">{len(closed_trades)}</div></div>
  <div class="stat-card"><div class="label">Win Rate</div><div class="value">{win_rate*100:.1f}%</div></div>
  <div class="stat-card"><div class="label">Total Profit (closed only)</div>
    <div class="value {'profit-pos' if total_profit_abs >= 0 else 'profit-neg'}">{total_profit_abs:+.2f} USDT</div>
    <div class="muted" style="font-size:0.7rem;margin-top:2px;">excludes unrealized — see Capital Efficiency below</div></div>
</div>

{capital_eff_html}

{portfolio_html}

<div class="two-col">
  <div class="card">
    <h3>Equity Curve (cumulative %, closed trades)</h3>
    <div class="chart-wrap"><canvas id="{eq_id}"></canvas></div>
  </div>
  <div class="card">
    <h3>Exit Reasons</h3>
    <div class="chart-wrap"><canvas id="{reason_id}"></canvas></div>
  </div>
</div>

<div class="card">
  <h3>True Total &amp; Stuck Trades Over Time <span class="muted" style="font-weight:400;font-size:0.75rem;">— is the real (realized+unrealized) position improving or getting worse?</span></h3>
  {f'<div class="chart-wrap"><canvas id="{trend_id}"></canvas></div>' if has_trend_history else '<p class="muted">Not enough history yet — this chart fills in as more dashboard runs are recorded (one point per run).</p>'}
</div>

{sig_html}

<div class="card">
  <h3>Open Trades ({len(open_trades)})</h3>
  <div class="table-scroll">
  <table>
    <tr><th>Pair</th><th>Enter Tag</th><th>Opened</th><th>Age</th><th>Rebuys</th><th>Capital Locked</th><th>Open Rate</th><th>Live Price</th><th>Unrealized P/L</th></tr>
    {open_rows_html if open_rows_html else '<tr><td colspan="9" class="muted">No open trades</td></tr>'}
  </table>
  </div>
</div>

<div class="card">
  <h3>Tag Family Performance (closed trades) <span class="muted" style="font-weight:400;font-size:0.75rem;">— ● open now marks a tag currently held by an open trade</span></h3>
  <div class="table-scroll">
  <table>
    <tr><th>Enter Tag</th><th>Trades</th><th>Win Rate</th><th>Avg Profit</th><th>Total Profit</th></tr>
    {tag_rows_html if tag_rows_html else '<tr><td colspan="5" class="muted">No closed trades yet</td></tr>'}
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
"""

    section_js = f"""
new Chart(document.getElementById('{eq_id}'), {{
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

new Chart(document.getElementById('{reason_id}'), {{
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

{f'''
new Chart(document.getElementById('{trend_id}'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(trend_labels)},
    datasets: [
      {{
        type: 'line', label: 'True Total (USDT)', yAxisID: 'y',
        data: {json.dumps(trend_true_total)},
        borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)',
        fill: false, tension: 0.2, pointRadius: 2, order: 1
      }},
      {{
        type: 'bar', label: 'Stuck Trades', yAxisID: 'y1',
        data: {json.dumps(trend_stuck)},
        backgroundColor: 'rgba(248,81,73,0.35)', order: 2
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8 }}, grid: {{ color: '#30363d' }} }},
      y: {{ position: 'left', ticks: {{ color: '#8b949e' }}, grid: {{ color: '#30363d' }},
           title: {{ display: true, text: 'USDT', color: '#8b949e' }} }},
      y1: {{ position: 'right', ticks: {{ color: '#8b949e', stepSize: 1 }}, grid: {{ display: false }},
            title: {{ display: true, text: 'Stuck Trades', color: '#8b949e' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }}
  }}
}});
''' if has_trend_history else ''}
"""

    compare = {
        "mode_title": mode_title,
        "open_trades": len(open_trades),
        "closed_trades": len(closed_trades),
        "win_rate": win_rate,
        "total_profit_abs": total_profit_abs,
        "true_total_abs": true_total_abs,
        "blended_rate_per_capital_day": blended_rate_per_capital_day,
        "stuck_trades": len(stuck_trades),
        "stuck_pct_of_open_capital": stuck_pct_of_open_capital,
    }

    return {
        "section_html": section_html,
        "section_js": section_js,
        "current_snapshot": current_snapshot,
        "compare": compare,
    }


def build_comparison_bar_html(spot_compare, futures_compare):
    """Top-of-page 'Spot vs Futures' quick comparison — the whole reason
    for combining these onto one page instead of two separate ones."""

    def row(label, spot_val, fut_val):
        return f"<tr><td>{label}</td><td>{spot_val}</td><td>{fut_val}</td></tr>"

    def blended_str(c):
        r = c["blended_rate_per_capital_day"]
        if r is None:
            return "—"
        return f"{r*100:.4f}%/day (≈{annualized(r):.1f}%/yr)"

    def true_total_str(c):
        v = c["true_total_abs"]
        cls = "profit-pos" if v >= 0 else "profit-neg"
        return f'<span class="{cls}">{v:+.2f} USDT</span>'

    rows_html = "".join([
        row("Open / Closed trades",
            f"{spot_compare['open_trades']} / {spot_compare['closed_trades']}",
            f"{futures_compare['open_trades']} / {futures_compare['closed_trades']}"),
        row("Win Rate",
            f"{spot_compare['win_rate']*100:.1f}%", f"{futures_compare['win_rate']*100:.1f}%"),
        row("Total Profit (closed only)",
            f"{spot_compare['total_profit_abs']:+.2f} USDT", f"{futures_compare['total_profit_abs']:+.2f} USDT"),
        row("True Total (realized+unrealized)",
            true_total_str(spot_compare), true_total_str(futures_compare)),
        row("Blended return per capital-day",
            blended_str(spot_compare), blended_str(futures_compare)),
        row("Stuck Trades",
            f"{spot_compare['stuck_trades']} ({spot_compare['stuck_pct_of_open_capital']:.0f}% of open capital)",
            f"{futures_compare['stuck_trades']} ({futures_compare['stuck_pct_of_open_capital']:.0f}% of open capital)"),
    ])

    return f"""
<div class="card">
  <h3>⚖️ Spot vs Futures — Quick Comparison</h3>
  <div class="table-scroll">
  <table class="compare-table">
    <tr><th>Metric</th><th>{esc(spot_compare['mode_title'])}</th><th>{esc(futures_compare['mode_title'])}</th></tr>
    {rows_html}
  </table>
  </div>
  <p class="muted" style="font-size:0.8rem;margin-top:12px;margin-bottom:0;">
    Both run the exact same NostalgiaForInfinityX7 strategy and pairlist rules — any
    difference here traces back to spot-vs-futures mechanics (leverage, funding,
    shortable pairs, exchange liquidity), not a different setup between the two.
  </p>
</div>"""


def build_combined_html(spot_bundle, futures_bundle, generated_at):
    comparison_html = build_comparison_bar_html(spot_bundle["compare"], futures_bundle["compare"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — Spot vs Futures</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{PAGE_STYLES}
</style>
</head>
<body>

<h1>🤖 NostalgiaForInfinity — Dry-Run Dashboard</h1>
<div class="subtitle">Generated {generated_at} UTC · dry-run (simulated) — no real funds involved · Spot: KuCoin market data · Futures: OKX market data</div>

{comparison_html}

{spot_bundle['section_html']}

{futures_bundle['section_html']}

<div class="subtitle">
  ⚠️ Small sample sizes can look great or terrible by chance. Entries cancelled before filling (limit-order timeouts) aren't counted here — only trades that actually opened.
</div>

<script>
{spot_bundle['section_js']}
{futures_bundle['section_js']}
</script>

</body>
</html>"""
    return html


def load_portfolio_config(config_path: str):
    """Reads max_open_trades / dry_run_wallet straight from the bot's own
    config so the dashboard never has these hardcoded and drifting out of
    sync. Falls back to sane defaults if the file has moved, been
    renamed, or the keys aren't there — never crashes the whole dashboard
    generation over a cosmetic stat."""
    defaults = {"max_open_trades": 8, "dry_run_wallet": 1000.0}
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            "max_open_trades": int(cfg.get("max_open_trades", defaults["max_open_trades"])),
            "dry_run_wallet": to_float(cfg.get("dry_run_wallet"), defaults["dry_run_wallet"]),
        }
    except Exception as e:
        print(f"  (could not read {config_path} for portfolio stats, using defaults: {e})")
        return defaults


def build_one_mode(db_url, mode_id, mode_title, market_note, config_path,
                    live_price_exchange_id, live_price_ccxt_options):
    trades = fetch_trades(db_url)
    open_pairs = list({t["pair"] for t in trades if t["is_open"]})
    print(f"[{mode_id}] Fetched {len(trades)} trade records ({len(open_pairs)} open pairs).")

    live_prices = (
        fetch_live_prices(open_pairs, live_price_exchange_id, live_price_ccxt_options)
        if open_pairs else {}
    )
    entry_fills = fetch_entry_fills(db_url, [t["id"] for t in trades])
    portfolio_cfg = load_portfolio_config(config_path)
    snapshot_history = fetch_snapshot_history(db_url)

    bundle = build_mode_section(
        trades, live_prices, entry_fills, portfolio_cfg, snapshot_history,
        mode_id, mode_title, market_note,
    )
    return bundle


def main(spot_db_url: str, futures_db_url: str, output_path: str):
    spot_bundle = build_one_mode(
        spot_db_url, "spot", "🟢 SPOT — KuCoin",
        "KuCoin spot market · no leverage",
        "config_dryrun_telegram.json",
        live_price_exchange_id="kucoin", live_price_ccxt_options=None,
    )
    futures_bundle = build_one_mode(
        futures_db_url, "futures", "🟣 FUTURES — OKX",
        "OKX perpetual swaps · isolated margin · 3x leverage (default)",
        "config_dryrun_futures.json",
        live_price_exchange_id="okx", live_price_ccxt_options={"defaultType": "swap"},
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = build_combined_html(spot_bundle, futures_bundle, generated_at)

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Combined dashboard written to {output_path}")

    record_snapshot(spot_db_url, spot_bundle["current_snapshot"])
    record_snapshot(futures_db_url, futures_bundle["current_snapshot"])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
