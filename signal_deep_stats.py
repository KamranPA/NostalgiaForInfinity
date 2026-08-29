"""
signal_deep_stats.py

Extends signal_stats.py with deeper diagnostics aimed specifically at the
questions raised while reviewing the open trades (KAITO/SUI/ADA/AKE/MOVR):

  1. Enter-tag performance breakdown (closed trades) — are certain entry
     conditions (e.g. "rebuy_*", "normal_*", tag 144, 163...) systematically
     stronger or weaker than others?
  2. Enter-tag distribution of OPEN trades vs CLOSED trades — are the
     currently-stuck trades disproportionately using tags that historically
     underperform?
  3. Duration vs outcome — do trades that stay open longer tend to be
     losers? (fast closes vs slow closes)
  4. Time-clustering of open trades — did several of the currently-open
     losers get opened within the same narrow window, suggesting a single
     correlated market move rather than independent, unrelated signals?
  5. DCA/rebuy activity on currently open trades — has averaging-in
     actually triggered on the stuck trades, based on the `orders` table
     (if present) — helps tell "strategy hasn't needed to react yet" apart
     from "strategy's rebuy logic isn't kicking in as expected".

This script is READ-ONLY. It never writes to the database or the exchange.

Usage:
    pip install psycopg2-binary numpy --break-system-packages
    python signal_deep_stats.py "<postgres-connection-string>"
"""

import sys
from datetime import datetime, timezone
from collections import defaultdict

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


def fetch_orders(db_url: str, trade_ids):
    """Best-effort: freqtrade's `orders` table logs every fill, including
    DCA/rebuy fills. If the table/columns differ in this deployment, this
    degrades gracefully and DCA analysis is simply skipped."""
    if not trade_ids:
        return {}
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT ft_trade_id AS trade_id, ft_order_side, order_filled_date,
                       average, filled
                FROM orders
                WHERE ft_trade_id = ANY(%s)
                ORDER BY ft_trade_id, order_filled_date ASC
            """, (list(trade_ids),))
            rows = cur.fetchall()
    except Exception as e:
        print(f"(Skipping DCA/rebuy analysis — could not read `orders` table: {e})\n")
        return {}
    finally:
        conn.close()

    by_trade = defaultdict(list)
    for r in rows:
        by_trade[r["trade_id"]].append(r)
    return by_trade


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def duration_hours(open_date, close_date):
    if open_date is None or close_date is None:
        return None
    return (close_date - open_date).total_seconds() / 3600.0


def base_tag_family(enter_tag):
    """Group tags like 'exit_long_rebuy_w_5_7 ( 64 )' -> just the numeric
    tag id, and separately classify closed-trade exit reasons into
    families (rebuy / normal / tc / other) for readability."""
    if enter_tag is None:
        return "unknown"
    return str(enter_tag)


def exit_reason_family(exit_reason):
    if not exit_reason:
        return "unknown"
    r = exit_reason.lower()
    if "rebuy" in r:
        return "rebuy"
    if "normal" in r:
        return "normal"
    if "_tc_" in r or "trailing" in r:
        return "trailing/tc"
    return "other"


# ----------------------------------------------------------------------
# Analysis sections
# ----------------------------------------------------------------------

def section_tag_performance(closed):
    print("=" * 70)
    print("1. ENTER-TAG PERFORMANCE (closed trades)")
    print("=" * 70)
    by_tag = defaultdict(list)
    for t in closed:
        if t["close_profit"] is None:
            continue
        by_tag[base_tag_family(t["enter_tag"])].append(float(t["close_profit"]))

    if not by_tag:
        print("No closed trades with profit data.\n")
        return

    rows = []
    for tag, profits in by_tag.items():
        arr = np.array(profits)
        wins = int((arr > 0).sum())
        rows.append((tag, len(arr), wins / len(arr), arr.mean(), arr.sum()))

    rows.sort(key=lambda r: -r[4])
    print(f"{'tag':>6}  {'n':>3}  {'win%':>6}  {'avg':>8}  {'total':>8}")
    for tag, n, wr, avg, total in rows:
        flag = "  <-- currently used by an OPEN trade" if tag in OPEN_TAGS else ""
        print(f"{tag:>6}  {n:>3}  {wr:>5.0%}  {avg:>+7.3%}  {total:>+7.3%}{flag}")
    print()
    print("Read this as: tags with few samples (n=1-2) are anecdotal, not")
    print("proof. The interesting signal is if an open trade's tag has a")
    print("track record meaningfully worse than the strategy's overall average.\n")


def section_open_vs_closed_tags(open_trades, closed):
    print("=" * 70)
    print("2. OPEN-TRADE TAGS vs HISTORICAL TAG PERFORMANCE")
    print("=" * 70)
    by_tag = defaultdict(list)
    for t in closed:
        if t["close_profit"] is None:
            continue
        by_tag[base_tag_family(t["enter_tag"])].append(float(t["close_profit"]))

    for t in open_trades:
        tag = base_tag_family(t["enter_tag"])
        hist = by_tag.get(tag)
        if hist:
            arr = np.array(hist)
            print(f"  {t['pair']:14s} tag={tag:>5}  "
                  f"history: n={len(arr)} win_rate={((arr>0).mean()):.0%} avg={arr.mean():+.3%}")
        else:
            print(f"  {t['pair']:14s} tag={tag:>5}  history: NO closed trades yet with this tag "
                  f"(first time this entry condition has fired)")
    print()


def section_duration_vs_outcome(closed):
    print("=" * 70)
    print("3. DURATION vs OUTCOME (closed trades)")
    print("=" * 70)
    durs, profits = [], []
    for t in closed:
        d = duration_hours(t["open_date"], t["close_date"])
        if d is None or t["close_profit"] is None:
            continue
        durs.append(d)
        profits.append(float(t["close_profit"]))

    if len(durs) < 3:
        print("Not enough closed trades with duration data yet.\n")
        return

    durs = np.array(durs)
    profits = np.array(profits)
    corr = np.corrcoef(durs, profits)[0, 1]

    print(f"Correlation(duration_hours, profit) across {len(durs)} closed trades: {corr:+.3f}")
    if corr < -0.3:
        print("Negative correlation: trades that stayed open longer tended to")
        print("profit less. Consistent with the currently-open trades (which")
        print("have been open ~1 week) being in unusually weak territory for")
        print("this strategy's normal operating pattern.")
    elif corr > 0.3:
        print("Positive correlation: longer-held trades actually did better —")
        print("patience has historically paid off for this strategy so far.")
    else:
        print("No strong linear relationship between duration and outcome —")
        print("duration alone isn't a reliable signal here.")

    print(f"\nFastest closes: {np.sort(durs)[:3].round(2)} hours")
    print(f"Slowest closes: {np.sort(durs)[-3:].round(2)} hours")
    print()


def section_time_clustering(open_trades):
    print("=" * 70)
    print("4. TIME-CLUSTERING OF OPEN TRADES")
    print("=" * 70)
    dated = [(t["pair"], t["open_date"]) for t in open_trades if t["open_date"]]
    dated.sort(key=lambda x: x[1])
    if len(dated) < 2:
        print("Not enough open trades to check clustering.\n")
        return

    print("Open dates in order:")
    for pair, d in dated:
        print(f"  {pair:14s} {d}")

    gaps_hours = [
        (dated[i][1] - dated[i - 1][1]).total_seconds() / 3600.0
        for i in range(1, len(dated))
    ]
    print(f"\nGaps between consecutive opens (hours): {[round(g, 1) for g in gaps_hours]}")
    tight = [g for g in gaps_hours if g < 6]
    if tight:
        print(f"{len(tight)} gap(s) under 6h — suggests a cluster of entries fired off")
        print("the same short-lived market condition. If those clustered trades are")
        print("now all underwater together, that's more likely one correlated market")
        print(f"move (e.g. a broad altcoin dip) than {len(tight)} independent strategy failures.")
    else:
        print("Opens are spread out — the current drawdown isn't from one clustered")
        print("entry burst; each trade was an independent signal.")
    print()


def section_dca_activity(open_trades, orders_by_trade):
    print("=" * 70)
    print("5. DCA / REBUY ACTIVITY ON CURRENTLY-OPEN TRADES")
    print("=" * 70)
    if not orders_by_trade:
        print("(orders table not available/readable — skipped; see message above)\n")
        return

    for t in open_trades:
        fills = orders_by_trade.get(t["id"], [])
        # These are long-only spot trades, so entry fills are 'buy' orders
        # (ft_is_entry isn't a real DB column — it's computed in freqtrade
        # from ft_order_side == trade.entry_side, which is 'buy' for longs).
        entries = [f for f in fills if f.get("ft_order_side") == "buy"]
        n_extra = max(0, len(entries) - 1)
        print(f"  {t['pair']:14s} total entry fills={len(entries)}  "
              f"extra DCA/rebuy fills={n_extra}")
        if n_extra == 0:
            print(f"      -> still on the original single entry; rebuy logic hasn't")
            print(f"         triggered yet (either hasn't dropped enough, or rebuy")
            print(f"         conditions for this tag require something else).")
    print()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

OPEN_TAGS = set()


def analyze(db_url: str):
    global OPEN_TAGS
    trades = fetch_trades(db_url)
    if not trades:
        print("No trades found in the database yet — nothing to analyze.")
        return

    open_trades = [t for t in trades if t["is_open"]]
    closed = [t for t in trades if not t["is_open"]]
    OPEN_TAGS = {base_tag_family(t["enter_tag"]) for t in open_trades}

    print("NFI Dry-Run — Deep Signal Diagnostics")
    print(f"open={len(open_trades)}  closed={len(closed)}\n")

    section_tag_performance(closed)
    section_open_vs_closed_tags(open_trades, closed)
    section_duration_vs_outcome(closed)
    section_time_clustering(open_trades)

    orders_by_trade = fetch_orders(db_url, [t["id"] for t in open_trades])
    section_dca_activity(open_trades, orders_by_trade)

    print("=" * 70)
    print("CAVEATS")
    print("=" * 70)
    print("- Sample sizes here are still small (single digits per tag in many")
    print("  cases). Treat every 'pattern' above as a hypothesis to keep an eye")
    print("  on, not a conclusion — re-run this after more trades close.")
    print("- This only looks at trades that filled. Cancelled/timed-out entry")
    print("  orders aren't in the `trades` table at all.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])
