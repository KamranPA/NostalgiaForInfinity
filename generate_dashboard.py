"""
generate_dashboard.py

Builds ONE combined, self-contained HTML dashboard covering BOTH the spot
and futures dry-run bots side by side. Both now run on OKX (spot switched
from KuCoin on 2026-09-14 specifically so entry-signal timing/pairlist
differences between the two bots trace back to spot-vs-futures mechanics
alone, not to running on two different exchanges — this is what makes the
new Signal Overlap analysis below meaningful).

ARCHITECTURE (as of the SQLite migration):
Both bots now persist their live trading state as a LOCAL SQLite file,
carried across each ~5h45m restart via a dedicated git branch
(state-spot / state-futures) — see dry-run-telegram-signals.yml and
dry-run-telegram-futures.yml.

This script READS those two SQLite files (read-only — it never writes
back into either bot's live state). It maintains its OWN separate history
file (a third SQLite file, on its own `dashboard-history` git branch)
purely for the "True Total & Stuck Trades Over Time" trend chart.

MERGE NOTE (2026-09-14): this version absorbs everything that used to
live in the three standalone scripts signal_stats.py, signal_deep_stats.py
and ml_feature_export.py, which are now retired (along with their
nfi-signal-stats.yml / nfi-deep-stats.yml / nfi-ml-feature-summary.yml
workflows):
  - signal_stats.py's numbers were already fully covered by this
    dashboard (win rate, per-pair, exit reasons, significance).
  - signal_deep_stats.py's extra diagnostics are now their own cards
    per mode: Duration vs Outcome correlation, Time-Clustering of open
    trades, and DCA/Rebuy activity on open trades.
  - ml_feature_export.py's CSV export is UNCHANGED and still a separate
    script/workflow (its output is a downloadable file, not something
    that belongs on an HTML page) — but its daily Telegram summary is
    now redundant with the new "ML Data Readiness" card added here, so
    that summary step can be dropped from the export workflow if desired.

SIGNAL OVERLAP (added 2026-09-14): a new top-of-page card compares LONG
entries between the two bots — same pair, same enter_tag, opened within
a shared time window (default 15 min, both run the same 5m timeframe) —
to answer "when spot signals, does futures signal too?" Futures-only
short entries (tags 501-673) are excluded since spot has no short side.

FIX (2026-09-18): pair-name mismatch was silently zeroing out every
match. ccxt reports futures/swap symbols with a ':SETTLE' suffix (e.g.
'ARB/USDT:USDT') while spot symbols have none (e.g. 'ARB/USDT'), so the
old (pair, enter_tag) key never matched across the two bots even when a
trade was a perfect overlap. Every pair used in overlap matching now
goes through normalize_pair() first, which strips that suffix.

FIX (2026-09-21): unrealized P/L on open trades was computed as
stake_amount * price-change%. In FUTURES mode stake_amount is the margin
posted, not the position's notional value, so this understated
unrealized P/L by the leverage factor (e.g. off by ~3x at the default 3x
leverage) — and it ignored trade direction entirely, so open SHORTS had
their P/L sign backwards (a falling price looked like a loss instead of
a gain). Spot is unaffected (leverage 1x, no shorts), but futures'
"Unrealized P/L", "True Total", and per-trade unrealized figures were
wrong. Now computed as direction * (live_price - open_rate) * amount,
where direction is -1 for shorts and +1 for longs — this is leverage-
and direction-correct for both markets. Also added tag 68 (a valid
Rebuy tag in upstream) to TAG_FAMILIES, and linked the new compare.html
page (X7ML vs X8) from the header here.

VISUAL REFRESH (2026-09-24): restyled as a trading-terminal look
(condensed ticker strip for headline stats, tabular-figure numerals for
all numeric columns, sticky table headers, a real top nav instead of a
plain subtitle line) instead of the generic dark-SaaS/GitHub-clone look
the page started with. Purely CSS/markup — no computation logic changed.

Usage:
    python generate_dashboard.py <spot_sqlite_path> <futures_sqlite_path> <history_sqlite_path> <output_html_path>
"""

import sys
import json
import sqlite3
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np


# ----------------------------------------------------------------------
# Data fetching (SQLite — each bot's own local state file)
# ----------------------------------------------------------------------

def parse_sqlite_datetime(value):
    """freqtrade/SQLAlchemy stores datetimes in SQLite as plain text
    (e.g. '2026-09-06 15:01:05.140768' or without the microseconds) —
    unlike psycopg2, sqlite3 hands these back as raw strings, not
    datetime objects, so every date column needs to go through this."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def fetch_trades(db_path: str):
    if not db_path:
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, pair, is_open, is_short, enter_tag, exit_reason,
                   open_date, close_date, open_rate, close_rate,
                   amount, stake_amount, close_profit, close_profit_abs
            FROM trades
            ORDER BY open_date ASC
        """)
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"  (could not read trades from {db_path}: {e})")
        return []
    finally:
        conn.close()

    for r in rows:
        r["is_open"] = bool(r["is_open"])
        r["is_short"] = bool(r.get("is_short") or False)
        r["open_date"] = parse_sqlite_datetime(r["open_date"])
        r["close_date"] = parse_sqlite_datetime(r["close_date"])
    return rows


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


def fetch_entry_fills(db_path: str, trade_ids):
    """Per-trade history of entry (DCA/rebuy) fills, so capital-days can
    use how much was ACTUALLY locked at each point in time. Returns
    {trade_id: [(fill_time, cumulative_cost_after_this_fill), ...]}
    sorted by fill_time. Falls back to {} if the orders table can't be
    read. NOTE: entry side is 'buy' for longs, 'sell' for shorts — the
    caller passes is_short per trade separately (see fetch_all_fills)."""
    if not trade_ids or not db_path:
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(trade_ids))
        cur = conn.cursor()
        cur.execute(f"""
            SELECT ft_trade_id, order_filled_date, cost, average, filled, ft_order_side
            FROM orders
            WHERE ft_trade_id IN ({placeholders}) AND order_filled_date IS NOT NULL
            ORDER BY ft_trade_id, order_filled_date ASC
        """, list(trade_ids))
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"  (could not read orders table for capital-over-time reconstruction: {e})")
        return {}
    finally:
        conn.close()

    return rows


def build_fill_maps(all_fill_rows, trades_by_id):
    """Splits the raw orders rows into two maps this script needs:
      1. entry_fills: {trade_id: [(fill_time, cumulative_cost), ...]} —
         only ENTRY-side fills (buy for longs, sell for shorts), used
         for capital-days integration.
      2. fills_by_trade: {trade_id: [raw fill dict, ...]} — ALL fills
         (entry sides only, same filter), used by the DCA/rebuy activity
         card to just count them.
    """
    entry_fills = defaultdict(list)
    fills_by_trade = defaultdict(list)
    running_cost = {}

    for r in all_fill_rows:
        tid = r["ft_trade_id"]
        trade = trades_by_id.get(tid)
        if trade is None:
            continue
        entry_side = "sell" if trade["is_short"] else "buy"
        if r["ft_order_side"] != entry_side:
            continue

        fill_cost = to_float(r["cost"])
        if fill_cost <= 0:
            fill_cost = to_float(r["average"]) * to_float(r["filled"])
        running_cost[tid] = running_cost.get(tid, 0.0) + fill_cost
        fill_time = parse_sqlite_datetime(r["order_filled_date"])

        entry_fills[tid].append((fill_time, running_cost[tid]))
        fills_by_trade[tid].append(r)

    return dict(entry_fills), dict(fills_by_trade)


def fetch_live_prices(pairs, exchange_id="kucoin", ccxt_options=None):
    """Best-effort live price fetch for unrealized P/L on open trades."""
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
# ML Data Readiness (absorbed from ml_feature_export.py's summary)
# ----------------------------------------------------------------------

ML_MIN_FOR_MODEL = 100
# adx_14 added once its underlying column-name bug (needed "_4h" suffix,
# since ADX is only computed on the 4h informative timeframe) was fixed
# in NostalgiaForInfinityX7ML.py — before that it was always null, so
# there was no point tracking its health here.
ML_HEALTH_CHECK_KEYS = [
    "rsi_14", "btc_rsi_14", "ema_20", "adx_14",
    # added with the ML snapshot schema v2 (2026-09-18)
    "atr_14", "enter_tag", "exit_reason",
    "rsi_14_1h", "ema_12_1h", "ema_200_1h",
    "rsi_14_1d", "ema_50_1d", "ema_200_1d", "range_pct_14_1d",
    # btc_ema_20/btc_roc_3/btc_ema_200 renamed to _1h/_1h/_1d once
    # btc_info_timeframes was overridden to actually populate them
    # (previously always None on the 4h-only default)
    "btc_ema_20_1h", "btc_roc_3_1h", "btc_ema_200_1d",
]


def fetch_ml_readiness(db_path: str):
    """Reads entry_context custom_data rows to summarize how much labeled
    training data has accumulated — the dashboard-card version of what
    ml_feature_export.py's --telegram summary used to print. The full
    per-row CSV export itself is unchanged and stays a separate script,
    since a downloadable file isn't something an HTML card can replace."""
    result = {
        "available": False,
        "n_trades_with_data": 0,
        "n_closed": 0,
        "n_open": 0,
        "feature_health": {},
        "per_tag": [],  # [(tag, n_closed, n_total), ...]
    }
    if not db_path:
        return result

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT cd.ft_trade_id, cd.cd_value, t.enter_tag, t.is_open
            FROM trade_custom_data cd
            JOIN trades t ON t.id = cd.ft_trade_id
            WHERE cd.cd_key = 'entry_context'
        """)
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"  (could not read trade_custom_data for ML readiness: {e})")
        return result
    finally:
        conn.close()

    if not rows:
        return result

    result["available"] = True
    parsed = []
    for r in rows:
        try:
            snapshot = json.loads(r["cd_value"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict):
            continue
        parsed.append({
            "trade_id": r["ft_trade_id"],
            "enter_tag": r["enter_tag"],
            "is_open": bool(r["is_open"]),
            **snapshot,
        })

    result["n_trades_with_data"] = len({p["trade_id"] for p in parsed})
    closed = [p for p in parsed if not p["is_open"]]
    open_ = [p for p in parsed if p["is_open"]]
    result["n_closed"] = len(closed)
    result["n_open"] = len(open_)

    for key in ML_HEALTH_CHECK_KEYS:
        present = [p for p in parsed if key in p]
        if not present:
            result["feature_health"][key] = None  # missing column entirely
        else:
            non_null = sum(1 for p in present if p[key] is not None)
            result["feature_health"][key] = (non_null / len(parsed)) * 100 if parsed else 0

    by_tag = defaultdict(lambda: [0, 0])  # tag -> [n_closed, n_total]
    for p in parsed:
        tag = p["enter_tag"]
        by_tag[tag][1] += 1
        if not p["is_open"]:
            by_tag[tag][0] += 1
    result["per_tag"] = sorted(
        [(tag, n[0], n[1]) for tag, n in by_tag.items()],
        key=lambda x: -x[2],
    )

    return result


def build_ml_readiness_html(readiness: dict):
    if not readiness["available"]:
        return """
    <div class="card">
      <h3>ML Data Readiness</h3>
      <p class="muted">No entry_context rows yet — this means no trade has opened since the
      strategy's ml_snapshot logging was deployed. Data starts accumulating from the next
      new trade onward; existing trades from before the change won't have it retroactively.</p>
    </div>"""

    n_closed = readiness["n_closed"]
    progress_note = (
        f"{n_closed}/{ML_MIN_FOR_MODEL} — below the minimum sample size before any model "
        f"trained on this would be meaningful."
        if n_closed < ML_MIN_FOR_MODEL else
        f"{n_closed} — at/above the {ML_MIN_FOR_MODEL} minimum discussed earlier. "
        f"Worth revisiting whether meta-labeling is viable."
    )

    health_rows = ""
    for key, pct in readiness["feature_health"].items():
        if pct is None:
            health_rows += f'<tr><td data-label="Feature">{key}</td><td data-label="Coverage" class="profit-neg">MISSING COLUMN</td></tr>'
        else:
            cls = "profit-pos" if pct >= 95 else ("profit-neg" if pct < 50 else "")
            health_rows += f'<tr><td data-label="Feature">{key}</td><td data-label="Coverage" class="{cls}">{pct:.0f}%</td></tr>'

    tag_rows = ""
    for tag, n_closed_tag, n_total in readiness["per_tag"][:12]:
        tag_rows += f'<tr><td data-label="Tag">{esc(str(tag))}</td><td data-label="Closed/Total">{n_closed_tag}/{n_total}</td></tr>'

    return f"""
    <div class="card">
      <h3>ML Data Readiness <span class="muted" style="font-weight:400;font-size:0.75rem;">— labeled feature data logged via entry_context/ml_snapshot custom_data</span></h3>
      <div class="grid" style="margin-bottom:0;">
        <div class="stat-card">
          <div class="label">Trades With Logged Features</div>
          <div class="value">{readiness['n_trades_with_data']}</div>
        </div>
        <div class="stat-card">
          <div class="label">Labeled (Closed)</div>
          <div class="value">{n_closed}</div>
          <div class="muted" style="font-size:0.75rem;margin-top:4px;">{progress_note}</div>
        </div>
        <div class="stat-card">
          <div class="label">Unlabeled (Open)</div>
          <div class="value">{readiness['n_open']}</div>
        </div>
      </div>
      <div class="two-col" style="margin-top:16px;">
        <div>
          <p class="muted" style="font-size:0.8rem;margin-bottom:4px;">Feature health check (non-null rate)</p>
          <table><tr><th>Feature</th><th>Coverage</th></tr>{health_rows}</table>
        </div>
        <div>
          <p class="muted" style="font-size:0.8rem;margin-bottom:4px;">By enter_tag (closed / total logged)</p>
          <table><tr><th>Tag</th><th>Closed/Total</th></tr>{tag_rows if tag_rows else '<tr><td colspan="2" class="muted">No data</td></tr>'}</table>
        </div>
      </div>
    </div>"""


# ----------------------------------------------------------------------
# Signal Overlap — Spot vs Futures (added 2026-09-14)
# ----------------------------------------------------------------------
# Both bots share the exact same populate_entry_trend() long-entry logic
# and (as of the OKX switch) the same exchange, so if their pairlists and
# price data line up, a long signal SHOULD often fire on both at once.
# This answers "does it actually?" by matching trades on pair + enter_tag
# with an open_date within a shared time window of each other.

SIGNAL_OVERLAP_WINDOW_MINUTES = 15  # generous vs. the 5m timeframe, to
                                     # absorb per-exchange candle-close
                                     # timing jitter without over-matching


def normalize_pair(pair):
    """Strip the ':SETTLE' suffix ccxt adds to futures/swap symbols
    (e.g. 'ARB/USDT:USDT' -> 'ARB/USDT') so spot and futures pair names
    compare equal for signal-overlap matching. Spot symbols have no
    colon and pass through unchanged."""
    return pair.split(":")[0] if pair else pair


def is_long_tag(enter_tag):
    """Futures-only short tags are 501-673 (mirroring long tags 1-173) —
    exclude those, since spot has no short side to compare against."""
    try:
        n = int(str(enter_tag).strip())
    except (TypeError, ValueError):
        return True  # unknown/non-numeric tags: don't exclude, just can't match well
    return n < 500


def compute_signal_overlap(spot_trades, futures_trades, window_minutes=SIGNAL_OVERLAP_WINDOW_MINUTES):
    spot_long = [t for t in spot_trades if is_long_tag(t["enter_tag"]) and t["open_date"]]
    fut_long = [t for t in futures_trades if is_long_tag(t["enter_tag"]) and not t["is_short"] and t["open_date"]]

    window = window_minutes * 60  # seconds
    fut_by_key = defaultdict(list)
    for t in fut_long:
        fut_by_key[(normalize_pair(t["pair"]), str(t["enter_tag"]))].append(t)

    matched_pairs = []
    spot_matched = 0
    for s in spot_long:
        key = (normalize_pair(s["pair"]), str(s["enter_tag"]))
        candidates = fut_by_key.get(key, [])
        best = None
        best_gap = None
        for f in candidates:
            gap = abs((to_aware_utc(s["open_date"]) - to_aware_utc(f["open_date"])).total_seconds())
            if gap <= window and (best_gap is None or gap < best_gap):
                best, best_gap = f, gap
        if best is not None:
            spot_matched += 1
            matched_pairs.append((s["pair"], s["enter_tag"], s["open_date"], best["open_date"], best_gap))

    # Futures-side match count computed independently (a futures entry counts
    # as matched if ANY spot entry on the same pair+tag falls in its window).
    spot_by_key = defaultdict(list)
    for t in spot_long:
        spot_by_key[(normalize_pair(t["pair"]), str(t["enter_tag"]))].append(t)
    futures_matched = 0
    for f in fut_long:
        key = (normalize_pair(f["pair"]), str(f["enter_tag"]))
        for s in spot_by_key.get(key, []):
            gap = abs((to_aware_utc(f["open_date"]) - to_aware_utc(s["open_date"])).total_seconds())
            if gap <= window:
                futures_matched += 1
                break

    return {
        "n_spot_long": len(spot_long),
        "n_futures_long": len(fut_long),
        "spot_matched": spot_matched,
        "futures_matched": futures_matched,
        "matched_pairs": sorted(matched_pairs, key=lambda x: x[2] or datetime.min.replace(tzinfo=timezone.utc), reverse=True),
        "window_minutes": window_minutes,
    }


def build_signal_overlap_html(overlap: dict):
    n_spot = overlap["n_spot_long"]
    n_fut = overlap["n_futures_long"]
    if n_spot == 0 and n_fut == 0:
        return """
<div class="card">
  <h3>Signal Overlap — Spot vs Futures (long entries)</h3>
  <p class="muted">No long entries on either bot yet — nothing to compare.</p>
</div>"""

    spot_pct = (overlap["spot_matched"] / n_spot * 100) if n_spot else 0
    fut_pct = (overlap["futures_matched"] / n_fut * 100) if n_fut else 0

    rows = ""
    for pair, tag, s_date, f_date, gap in overlap["matched_pairs"][:15]:
        rows += f"""<tr>
          <td data-label="Pair">{esc(pair)}</td><td data-label="Tag">{fmt_tag(tag)}</td>
          <td data-label="Spot Opened">{fmt_dt(s_date)}</td><td data-label="Futures Opened">{fmt_dt(f_date)}</td>
          <td data-label="Gap">{gap/60:.1f}m</td>
        </tr>"""

    return f"""
<div class="card">
  <h3>Signal Overlap — Spot vs Futures (long entries) <span class="muted" style="font-weight:400;font-size:0.75rem;">— same pair + enter_tag, opened within {overlap['window_minutes']} min of each other</span></h3>
  <div class="grid" style="margin-bottom:0;">
    <div class="stat-card">
      <div class="label">Spot Long Entries Matched</div>
      <div class="value">{overlap['spot_matched']} / {n_spot}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">{spot_pct:.0f}% also fired on futures</div>
    </div>
    <div class="stat-card">
      <div class="label">Futures Long Entries Matched</div>
      <div class="value">{overlap['futures_matched']} / {n_fut}</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">{fut_pct:.0f}% also fired on spot</div>
    </div>
  </div>
  <div class="table-scroll" style="margin-top:16px;">
  <table>
    <tr><th>Pair</th><th>Tag</th><th>Spot Opened</th><th>Futures Opened</th><th>Gap</th></tr>
    {rows if rows else '<tr><td colspan="5" class="muted">No matches yet</td></tr>'}
  </table>
  </div>
  <p class="muted" style="font-size:0.8rem;margin-top:12px;margin-bottom:0;">
    Both bots share the same long-entry logic and (as of the OKX switch) the same
    exchange — a low match rate here points to pairlist divergence (different
    volume-ranked top-100/130 sets) or per-exchange price/candle timing, not a
    difference in the strategy's entry rules themselves.
  </p>
</div>"""


# ----------------------------------------------------------------------
# Duration/Clustering/DCA diagnostics (absorbed from signal_deep_stats.py)
# ----------------------------------------------------------------------

def build_duration_outcome_html(closed_with_profit):
    n = len(closed_with_profit)
    if n < 3:
        return """
    <div class="card">
      <h3>Duration vs Outcome</h3>
      <p class="muted">Not enough closed trades with duration data yet.</p>
    </div>"""

    durs = np.array([duration_days(t["open_date"], t["close_date"]) * 24 for t in closed_with_profit])
    profits = np.array([float(t["close_profit"]) for t in closed_with_profit])
    corr = float(np.corrcoef(durs, profits)[0, 1])

    if corr < -0.3:
        note = ("Negative correlation: trades that stayed open longer tended to profit less — "
                "consistent with slow-closing trades sitting in unusually weak territory.")
    elif corr > 0.3:
        note = "Positive correlation: longer-held trades actually did better — patience has paid off so far."
    else:
        note = "No strong linear relationship between duration and outcome — duration alone isn't a reliable signal here."

    fastest = ", ".join(f"{d:.1f}h" for d in np.sort(durs)[:3])
    slowest = ", ".join(f"{d:.1f}h" for d in np.sort(durs)[-3:])

    return f"""
    <div class="card">
      <h3>Duration vs Outcome <span class="muted" style="font-weight:400;font-size:0.75rem;">— across {n} closed trades</span></h3>
      <p>Correlation(duration, profit): <b>{corr:+.3f}</b></p>
      <p class="muted" style="font-size:0.85rem;">{note}</p>
      <p style="font-size:0.85rem;">Fastest closes: {fastest} &nbsp;|&nbsp; Slowest closes: {slowest}</p>
    </div>"""


def build_time_clustering_html(open_trades):
    dated = [(t["pair"], t["open_date"]) for t in open_trades if t["open_date"]]
    dated.sort(key=lambda x: x[1])
    if len(dated) < 2:
        return """
    <div class="card">
      <h3>Time-Clustering of Open Trades</h3>
      <p class="muted">Not enough open trades to check clustering.</p>
    </div>"""

    gaps_hours = [
        (dated[i][1] - dated[i - 1][1]).total_seconds() / 3600.0
        for i in range(1, len(dated))
    ]
    tight = [g for g in gaps_hours if g < 6]

    rows = "".join(f'<tr><td data-label="Pair">{esc(pair)}</td><td data-label="Opened">{fmt_dt(d)}</td></tr>' for pair, d in dated)

    if tight:
        note = (f"{len(tight)} gap(s) under 6h — suggests a cluster of entries fired off the same "
                f"short-lived market condition. If those clustered trades are underwater together, "
                f"that's more likely one correlated market move than {len(tight)} independent failures.")
    else:
        note = "Opens are spread out — no clustered entry burst; each trade was an independent signal."

    return f"""
    <div class="card">
      <h3>Time-Clustering of Open Trades</h3>
      <div class="table-scroll"><table><tr><th>Pair</th><th>Opened</th></tr>{rows}</table></div>
      <p class="muted" style="font-size:0.85rem;margin-top:8px;">{note}</p>
    </div>"""


def build_dca_activity_html(open_trades, fills_by_trade):
    if not fills_by_trade:
        return """
    <div class="card">
      <h3>DCA / Rebuy Activity (open trades)</h3>
      <p class="muted">Orders table not available/readable this run — skipped.</p>
    </div>"""

    rows = ""
    for t in open_trades:
        fills = fills_by_trade.get(t["id"], [])
        n_extra = max(0, len(fills) - 1)
        note = (
            "still on original entry; rebuy logic hasn't triggered yet"
            if n_extra == 0 else ""
        )
        rows += f"<tr><td data-label=\"Pair\">{esc(t['pair'])}</td><td data-label=\"Entry Fills\">{len(fills)}</td><td data-label=\"Extra DCA/Rebuy\">{n_extra}</td><td class=\"muted\">{note}</td></tr>"

    return f"""
    <div class="card">
      <h3>DCA / Rebuy Activity (open trades)</h3>
      <div class="table-scroll">
      <table><tr><th>Pair</th><th>Entry Fills</th><th>Extra DCA/Rebuy Fills</th><th></th></tr>
      {rows if rows else '<tr><td colspan="4" class="muted">No open trades</td></tr>'}
      </table>
      </div>
    </div>"""


# ----------------------------------------------------------------------
# Historical snapshots (trend over time) — the dashboard's OWN file
# ----------------------------------------------------------------------

SNAPSHOT_HISTORY_LIMIT = 180


def ensure_snapshot_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nfi_dashboard_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            snapshot_time TEXT NOT NULL,
            open_trades INTEGER NOT NULL,
            closed_trades INTEGER NOT NULL,
            win_rate REAL,
            realized_profit_abs REAL,
            unrealized_pl_abs REAL,
            true_total_abs REAL,
            stuck_trades INTEGER,
            open_capital_locked REAL
        )
    """)
    conn.commit()


def fetch_snapshot_history(history_db_path: str, mode_id: str, limit: int = SNAPSHOT_HISTORY_LIMIT):
    try:
        conn = sqlite3.connect(history_db_path)
        conn.row_factory = sqlite3.Row
    except Exception as e:
        print(f"  (could not open history file for {mode_id}: {e})")
        return []
    try:
        ensure_snapshot_table(conn)
        cur = conn.cursor()
        cur.execute("""
            SELECT snapshot_time, true_total_abs, stuck_trades,
                   realized_profit_abs, unrealized_pl_abs
            FROM nfi_dashboard_snapshots
            WHERE mode = ?
            ORDER BY snapshot_time DESC
            LIMIT ?
        """, (mode_id, limit))
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["snapshot_time"] = parse_sqlite_datetime(r["snapshot_time"])
        return list(reversed(rows))
    except Exception as e:
        print(f"  (could not read snapshot history for {mode_id}, starting fresh: {e})")
        return []
    finally:
        conn.close()


def record_snapshot(history_db_path: str, mode_id: str, metrics: dict):
    try:
        conn = sqlite3.connect(history_db_path)
    except Exception as e:
        print(f"  (could not open history file to record snapshot for {mode_id}: {e})")
        return
    try:
        ensure_snapshot_table(conn)
        conn.execute("""
            INSERT INTO nfi_dashboard_snapshots
                (mode, snapshot_time, open_trades, closed_trades, win_rate,
                 realized_profit_abs, unrealized_pl_abs, true_total_abs,
                 stuck_trades, open_capital_locked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mode_id, metrics["snapshot_time"].isoformat(),
            metrics["open_trades"], metrics["closed_trades"],
            metrics["win_rate"], metrics["realized_profit_abs"], metrics["unrealized_pl_abs"],
            metrics["true_total_abs"], metrics["stuck_trades"], metrics["open_capital_locked"],
        ))
        conn.commit()
    except Exception as e:
        print(f"  (could not record snapshot for {mode_id}, continuing anyway: {e})")
    finally:
        conn.close()


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
# Capital efficiency
# ----------------------------------------------------------------------

STUCK_MULTIPLIER = 5
STUCK_HARD_CAP_DAYS = 3


def to_aware_utc(dt):
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


TAG_FAMILIES = [
    (1, 13, "Normal"), (21, 26, "Pump"), (41, 53, "Quick"), (61, 68, "Rebuy"),
    (81, 82, "High Profit"), (101, 110, "Rapid"), (120, 120, "Grind"), (121, 121, "BTC"),
    (141, 145, "Top Coins"), (161, 173, "Scalp"),
    (501, 513, "Short Normal"), (521, 526, "Short Pump"), (541, 553, "Short Quick"),
    (561, 565, "Short Rebuy"), (581, 582, "Short High Profit"), (601, 610, "Short Rapid"),
    (620, 620, "Short Grind"), (621, 621, "Short BTC"), (641, 645, "Short Top Coins"),
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
    return rate_per_day * 365 * 100


PAGE_STYLES = """
  :root {
    --bg: #0b0c10; --bg-raised: #131620; --card: #131620; --border: #232838;
    --text: #dfe2ea; --muted: #838aa0; --accent: #d1a350; --accent-dim: #8a7040;
    --pos: #4fae7a; --neg: #d9645a; --pos-soft: rgba(79,174,122,0.12); --neg-soft: rgba(217,100,90,0.12);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; scroll-padding-top: 64px; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 0; line-height: 1.5; font-size: 15px;
  }
  main { max-width: 1180px; margin: 0 auto; padding: 24px 24px 48px; }

  /* ---- top bar ------------------------------------------------- */
  .topbar {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; gap: 20px;
    background: rgba(11,12,16,0.92); backdrop-filter: blur(6px);
    border-bottom: 1px solid var(--border);
    padding: 0 24px; height: 56px;
  }
  .topbar .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; letter-spacing: -0.01em; white-space: nowrap; }
  .topbar .brand span { color: var(--accent); }
  .topbar nav { display: flex; gap: 4px; flex: 1; }
  .topbar nav a {
    color: var(--muted); text-decoration: none; font-size: 0.85rem;
    padding: 6px 12px; border-radius: 6px; transition: color .15s, background .15s;
  }
  .topbar nav a:hover { color: var(--text); background: rgba(255,255,255,0.04); }
  .topbar .gen-time { color: var(--muted); font-size: 0.75rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .topbar .compare-link {
    color: var(--bg); background: var(--accent); font-weight: 600;
    padding: 6px 14px; border-radius: 6px; text-decoration: none; font-size: 0.82rem;
    white-space: nowrap;
  }
  .topbar .compare-link:hover { background: #e0b566; }

  .disclaimer {
    color: var(--muted); font-size: 0.78rem; padding: 10px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 28px;
  }

  h2.market-heading {
    font-family: 'Space Grotesk', sans-serif; font-weight: 600;
    font-size: 1.3rem; margin: 0 0 2px 0; padding-top: 40px; letter-spacing: -0.01em;
    display: flex; align-items: center; gap: 10px;
  }
  h2.market-heading .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  h2.market-heading.mode-spot .dot { background: #4fae7a; }
  h2.market-heading.mode-futures .dot { background: #7c8fd6; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 20px; }

  /* ---- ticker strip (headline stats) ---------------------------- */
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 0; margin-bottom: 24px; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
  .stat-card { background: var(--bg-raised); padding: 14px 16px; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); }
  .grid .stat-card:last-child { border-right: none; }
  .stat-card .label { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.02em; }
  .stat-card .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }

  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 20px; margin-bottom: 20px;
  }
  .card h3 { margin-top: 0; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 500; font-size: 0.72rem; letter-spacing: 0.02em; position: sticky; top: 56px; background: var(--card); }
  tr:hover td { background: rgba(255,255,255,0.025); }
  .profit-pos { color: var(--pos); font-weight: 600; }
  .profit-neg { color: var(--neg); font-weight: 600; }
  .muted { color: var(--muted); }
  .stuck-badge {
    display: inline-block; background: var(--neg-soft); color: var(--neg);
    border: 1px solid rgba(217,100,90,0.3); border-radius: 4px;
    font-size: 0.68rem; padding: 1px 6px; margin-left: 4px; white-space: nowrap;
  }
  .chart-wrap { position: relative; height: 280px; }
  .two-col { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .two-col { grid-template-columns: 1fr; } .topbar nav { display: none; } }
  @media (max-width: 600px) {
    .topbar { padding: 0 14px; gap: 10px; }
    .topbar .gen-time { display: none; }
    .topbar .brand { font-size: 0.95rem; }
    .topbar .compare-link { flex-shrink: 0; padding: 6px 10px; font-size: 0.78rem; }
    main { padding: 16px 14px 40px; }

    /* Stacked "card" tables instead of horizontal scrolling: each row
       becomes a small block, each cell a label:value line using the
       data-label attribute set on every <td> above. */
    .table-scroll { overflow-x: visible; }
    table { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
    table thead { display: none; }
    table, tbody, tr, td { display: block; width: 100%; }
    tr { border-bottom: 1px solid var(--border); padding: 10px 12px; }
    tr:last-child { border-bottom: none; }
    td {
      border: none; padding: 4px 0; display: flex;
      justify-content: space-between; align-items: baseline; gap: 14px;
      text-align: right;
    }
    td::before {
      content: attr(data-label); color: var(--muted); font-size: 0.7rem;
      flex-shrink: 0; text-align: left;
    }
    /* the comparison table's first column is the row's own heading, not
       a labeled value — render it full-width and bold instead */
    table.compare-table td:first-child {
      display: block; text-align: left; font-weight: 600; color: var(--text);
      padding-bottom: 6px; font-size: 0.82rem;
    }
  }
  .table-scroll { overflow-x: auto; }
  .compare-table td:not(:first-child), .compare-table th:not(:first-child) { text-align: right; }
"""


def build_mode_section(trades, live_prices, entry_fills, fills_by_trade, ml_readiness,
                        portfolio_cfg, snapshot_history, mode_id, mode_title, market_note):
    open_trades = [t for t in trades if t["is_open"]]
    closed_trades = [t for t in trades if not t["is_open"]]
    closed_with_profit = [t for t in closed_trades if t["close_profit"] is not None]
    profits = np.array([float(t["close_profit"]) for t in closed_with_profit])
    n = len(profits)
    wins = int((profits > 0).sum()) if n else 0
    win_rate = wins / n if n else 0

    equity_labels = []
    equity_values = []
    cum = 0.0
    for t in closed_with_profit:
        cum += float(t["close_profit"]) * 100
        equity_labels.append(fmt_dt(t["close_date"]))
        equity_values.append(round(cum, 3))

    pair_stats = {}
    for t in closed_with_profit:
        pair_stats.setdefault(t["pair"], []).append(float(t["close_profit"]))
    pair_rows = []
    for pair, plist in sorted(pair_stats.items(), key=lambda x: -len(x[1])):
        arr = np.array(plist)
        w = int((arr > 0).sum())
        pair_rows.append((pair, len(arr), w / len(arr), arr.mean()))

    open_tag_set = {str(t["enter_tag"]) for t in open_trades if t["enter_tag"] is not None}
    tag_stats = {}
    for t in closed_with_profit:
        tag_stats.setdefault(str(t["enter_tag"]), []).append(float(t["close_profit"]))
    tag_rows = []
    for tag, plist in sorted(tag_stats.items(), key=lambda x: -sum(x[1])):
        arr = np.array(plist)
        w = int((arr > 0).sum())
        tag_rows.append((tag, len(arr), w / len(arr), arr.mean(), arr.sum(), tag in open_tag_set))

    reason_counts = {}
    for t in closed_trades:
        r = t["exit_reason"] or "unknown"
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_labels = list(reason_counts.keys())
    reason_values = [reason_counts[k] for k in reason_labels]

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
          <p>Only {n} closed trades — need at least ~8 for a meaningful significance test. Treat all numbers below as provisional.</p>
        </div>"""

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
            # FIX (2026-09-21): leverage- and direction-correct unrealized
            # P/L. stake_amount is MARGIN in futures mode, not position
            # notional, and it carries no sign for shorts — so this must
            # be computed off amount (the actual position size) with an
            # explicit direction flip for shorts, not stake * price-change%.
            direction = -1.0 if t["is_short"] else 1.0
            amount = to_float(t["amount"])
            unreal_abs = direction * (live - t["open_rate"]) * amount
            unreal_pct = (unreal_abs / stake) if stake else 0.0
            open_unrealized_abs_total += unreal_abs
            unreal_cls = "profit-pos" if unreal_abs > 0 else "profit-neg"
            unreal_str = f'<span class="{unreal_cls}">{fmt_pct(unreal_pct)} ({unreal_abs:+.2f} USDT)</span>'
            live_str = f"{live:.6g}"
        else:
            unreal_str = '<span class="muted">live price unavailable</span>'
            live_str = '<span class="muted">—</span>'

        age_str = f"{age_days:.1f}d"
        stuck_badge = ' <span class="stuck-badge" title="Open far longer than this strategy\'s typical hold time">stuck</span>' if is_stuck else ""

        n_entry_fills = len(fills) if fills else 1
        n_rebuys = max(n_entry_fills - 1, 0)
        rebuy_str = f"{n_rebuys}" if fills else '<span class="muted">n/a</span>'

        open_rows_html += f"""
        <tr>
          <td data-label="Pair">{esc(t['pair'])}</td>
          <td data-label="Enter Tag">{fmt_tag(t['enter_tag'])}</td>
          <td data-label="Opened">{fmt_dt(t['open_date'])}</td>
          <td data-label="Age">{age_str}{stuck_badge}</td>
          <td data-label="Rebuys">{rebuy_str}</td>
          <td data-label="Capital Locked">{stake:.2f} USDT</td>
          <td data-label="Open Rate">{t['open_rate']:.6g}</td>
          <td data-label="Live Price">{live_str}</td>
          <td data-label="Unrealized P/L">{unreal_str}</td>
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
        fill. This is what lets a 4-day trade for +0.1% and a 9-minute trade for +0.1% be
        compared fairly, and it's why "Total Profit" alone can look fine while several slots
        are quietly stuck.
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
          <td data-label="Pair">{esc(t['pair'])}</td>
          <td data-label="Enter Tag">{fmt_tag(t['enter_tag'])}</td>
          <td data-label="Opened">{fmt_dt(t['open_date'])}</td>
          <td data-label="Closed">{fmt_dt(t['close_date'])}</td>
          <td data-label="Profit" class="{cls}">{fmt_pct(profit)}</td>
          <td data-label="Exit Reason">{esc(t['exit_reason'])}</td>
        </tr>"""

    pair_rows_html = ""
    for pair, cnt, wr, avg in pair_rows:
        cls = "profit-pos" if avg > 0 else "profit-neg"
        pair_rows_html += f"""
        <tr>
          <td data-label="Pair">{esc(pair)}</td>
          <td data-label="Trades">{cnt}</td>
          <td data-label="Win Rate">{wr*100:.0f}%</td>
          <td data-label="Avg Profit" class="{cls}">{fmt_pct(avg)}</td>
        </tr>"""

    tag_rows_html = ""
    for tag, cnt, wr, avg, total, in_use in tag_rows:
        cls = "profit-pos" if avg > 0 else "profit-neg"
        in_use_badge = ' <span class="stuck-badge" style="background:rgba(209,163,80,0.15);color:var(--accent);border-color:rgba(209,163,80,0.35);">open now</span>' if in_use else ""
        tag_rows_html += f"""
        <tr>
          <td data-label="Enter Tag">{fmt_tag(tag)}{in_use_badge}</td>
          <td data-label="Trades">{cnt}</td>
          <td data-label="Win Rate">{wr*100:.0f}%</td>
          <td data-label="Avg Profit" class="{cls}">{fmt_pct(avg)}</td>
          <td data-label="Total Profit" class="{cls}">{fmt_pct(total, signed=True)}</td>
        </tr>"""

    total_profit_abs = sum(float(t["close_profit_abs"] or 0) for t in closed_with_profit)

    eq_id = f"equityChart_{mode_id}"
    reason_id = f"reasonChart_{mode_id}"
    trend_id = f"trendChart_{mode_id}"

    duration_outcome_html = build_duration_outcome_html(closed_with_profit)
    time_clustering_html = build_time_clustering_html(open_trades)
    dca_activity_html = build_dca_activity_html(open_trades, fills_by_trade)
    ml_readiness_html = build_ml_readiness_html(ml_readiness)

    heading_cls = "mode-spot" if mode_id == "spot" else "mode-futures"

    section_html = f"""
<h2 class="market-heading {heading_cls}" id="{mode_id}"><span class="dot"></span>{mode_title}</h2>
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

{duration_outcome_html}

{time_clustering_html}

{dca_activity_html}

{ml_readiness_html}

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
  <h3>Tag Family Performance (closed trades) <span class="muted" style="font-weight:400;font-size:0.75rem;">— "open now" marks a tag currently held by an open trade</span></h3>
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
Chart.defaults.font.family = "'Public Sans', -apple-system, sans-serif";
new Chart(document.getElementById('{eq_id}'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(equity_labels)},
    datasets: [{{
      label: 'Cumulative profit %',
      data: {json.dumps(equity_values)},
      borderColor: '#d1a350', backgroundColor: 'rgba(209,163,80,0.10)',
      fill: true, tension: 0.2, pointRadius: 2
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ ticks: {{ color: '#838aa0', maxTicksLimit: 8 }}, grid: {{ color: '#232838' }} }},
      y: {{ ticks: {{ color: '#838aa0' }}, grid: {{ color: '#232838' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#dfe2ea' }} }} }}
  }}
}});

new Chart(document.getElementById('{reason_id}'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(reason_labels)},
    datasets: [{{
      data: {json.dumps(reason_values)},
      backgroundColor: ['#d1a350','#4fae7a','#d9645a','#7c8fd6','#a487c9','#4fa8ae','#c97fa8']
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#dfe2ea', font: {{ size: 10 }} }} }} }}
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
        borderColor: '#d1a350', backgroundColor: 'rgba(209,163,80,0.10)',
        fill: false, tension: 0.2, pointRadius: 2, order: 1
      }},
      {{
        type: 'bar', label: 'Stuck Trades', yAxisID: 'y1',
        data: {json.dumps(trend_stuck)},
        backgroundColor: 'rgba(217,100,90,0.35)', order: 2
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    scales: {{
      x: {{ ticks: {{ color: '#838aa0', maxTicksLimit: 8 }}, grid: {{ color: '#232838' }} }},
      y: {{ position: 'left', ticks: {{ color: '#838aa0' }}, grid: {{ color: '#232838' }},
           title: {{ display: true, text: 'USDT', color: '#838aa0' }} }},
      y1: {{ position: 'right', ticks: {{ color: '#838aa0', stepSize: 1 }}, grid: {{ display: false }},
            title: {{ display: true, text: 'Stuck Trades', color: '#838aa0' }} }}
    }},
    plugins: {{ legend: {{ labels: {{ color: '#dfe2ea' }} }} }}
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
    spot_label = esc(spot_compare["mode_title"])
    fut_label = esc(futures_compare["mode_title"])

    def row(label, spot_val, fut_val):
        return f'<tr><td>{label}</td><td data-label="{spot_label}">{spot_val}</td><td data-label="{fut_label}">{fut_val}</td></tr>'

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
  <h3>Spot vs Futures — Quick Comparison</h3>
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


def build_combined_html(spot_bundle, futures_bundle, generated_at, signal_overlap_html):
    comparison_html = build_comparison_bar_html(spot_bundle["compare"], futures_bundle["compare"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — Spot vs Futures</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{PAGE_STYLES}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">NFI <span>dry-run</span></div>
  <nav>
    <a href="#spot">Spot</a>
    <a href="#futures">Futures</a>
  </nav>
  <span class="gen-time">generated {generated_at} UTC</span>
  <a class="compare-link" href="compare.html">X7ML vs X8 →</a>
</div>

<main>
<div class="disclaimer">Dry-run (simulated) — no real funds involved · Spot: KuCoin market data · Futures: OKX market data</div>

{comparison_html}

{signal_overlap_html}

{spot_bundle['section_html']}

{futures_bundle['section_html']}

<div class="subtitle">
  Small sample sizes can look great or terrible by chance. Entries cancelled before filling (limit-order timeouts) aren't counted here — only trades that actually opened.
</div>
</main>

<script>
{spot_bundle['section_js']}
{futures_bundle['section_js']}
</script>

</body>
</html>"""
    return html


def load_portfolio_config(config_path: str):
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


def build_one_mode(sqlite_path, history_db_path, mode_id, mode_title, market_note, config_path,
                    live_price_exchange_id, live_price_ccxt_options):
    trades = fetch_trades(sqlite_path)
    trades_by_id = {t["id"]: t for t in trades}
    open_pairs = list({t["pair"] for t in trades if t["is_open"]})
    print(f"[{mode_id}] Fetched {len(trades)} trade records from {sqlite_path} ({len(open_pairs)} open pairs).")

    live_prices = (
        fetch_live_prices(open_pairs, live_price_exchange_id, live_price_ccxt_options)
        if open_pairs else {}
    )
    all_fill_rows = fetch_entry_fills(sqlite_path, [t["id"] for t in trades])
    entry_fills, fills_by_trade = build_fill_maps(all_fill_rows, trades_by_id)
    ml_readiness = fetch_ml_readiness(sqlite_path)
    portfolio_cfg = load_portfolio_config(config_path)
    snapshot_history = fetch_snapshot_history(history_db_path, mode_id)

    bundle = build_mode_section(
        trades, live_prices, entry_fills, fills_by_trade, ml_readiness,
        portfolio_cfg, snapshot_history, mode_id, mode_title, market_note,
    )
    bundle["trades"] = trades  # kept for cross-bot analyses (e.g. signal overlap) in main()
    return bundle


def main(spot_sqlite_path: str, futures_sqlite_path: str, history_db_path: str, output_path: str):
    spot_bundle = build_one_mode(
        spot_sqlite_path, history_db_path, "spot", "SPOT — OKX",
        "OKX spot market · no leverage",
        "config_dryrun_telegram.json",
        live_price_exchange_id="okx", live_price_ccxt_options=None,
    )
    futures_bundle = build_one_mode(
        futures_sqlite_path, history_db_path, "futures", "FUTURES — OKX",
        "OKX perpetual swaps · isolated margin · 3x leverage (default)",
        "config_dryrun_futures.json",
        live_price_exchange_id="okx", live_price_ccxt_options={"defaultType": "swap"},
    )

    overlap = compute_signal_overlap(spot_bundle["trades"], futures_bundle["trades"])
    signal_overlap_html = build_signal_overlap_html(overlap)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = build_combined_html(spot_bundle, futures_bundle, generated_at, signal_overlap_html)

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Combined dashboard written to {output_path}")

    record_snapshot(history_db_path, "spot", spot_bundle["current_snapshot"])
    record_snapshot(history_db_path, "futures", futures_bundle["current_snapshot"])


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
