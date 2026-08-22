"""
signal_stats.py

Connects directly to the same Postgres database (Supabase) that the NFI
dry-run bot uses, and summarizes the REAL simulated trade history — not a
backtest, the actual signals this specific bot has generated over time.

Usage:
    python signal_stats.py "<postgres-connection-string>"

Note on scope: this reads the `trades` table, which only contains orders
that actually got INSERTED — i.e. entries that filled (or are still open).
Entries that were cancelled before filling (like the DASH/USDT timeout
case) never became a trade row, so they aren't counted here. This script
answers "how did the trades that actually happened perform", not "how
many signals were fired in total including ones that never filled".
"""

import sys
from datetime import datetime, timezone

import numpy as np

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Missing dependency. Install with: pip install psycopg2-binary --break-system-packages")
    sys.exit(1)


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
            rows = cur.fetchall()
        return rows
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


def fmt_duration(open_date, close_date):
    if open_date is None or close_date is None:
        return "n/a"
    delta = close_date - open_date
    total_min = delta.total_seconds() / 60
    if total_min < 60:
        return f"{total_min:.0f}m"
    return f"{total_min/60:.1f}h"


def analyze(db_url: str):
    trades = fetch_trades(db_url)
    n_total = len(trades)

    if n_total == 0:
        print("No trades found in the database yet — nothing to analyze.")
        return

    open_trades = [t for t in trades if t["is_open"]]
    closed_trades = [t for t in trades if not t["is_open"]]

    print("=== NFI Dry-Run Signal Statistics (live from Supabase) ===\n")
    print(f"Total trade records: {n_total}")
    print(f"  Open (in progress): {len(open_trades)}")
    print(f"  Closed:              {len(closed_trades)}\n")

    if open_trades:
        print("--- Currently open ---")
        for t in open_trades:
            print(f"  {t['pair']:14s}  opened {t['open_date']}  enter_tag={t['enter_tag']}")
        print()

    if not closed_trades:
        print("No closed trades yet — win rate / profit stats need at least one completed trade.")
        return

    profits = np.array([float(t["close_profit"]) for t in closed_trades if t["close_profit"] is not None])
    n = len(profits)
    wins = int((profits > 0).sum())
    losses = int((profits <= 0).sum())
    win_rate = wins / n if n else 0

    print("--- Closed trades ---")
    print(f"n = {n}")
    print(f"Win rate: {win_rate:.2%} ({wins}W / {losses}L)")
    print(f"Mean profit/trade: {profits.mean():.4%}")
    print(f"Std dev/trade: {profits.std():.4%}")
    print(f"Best trade: {profits.max():+.4%}")
    print(f"Worst trade: {profits.min():+.4%}")

    print("\n--- Sample size check ---")
    if n < 30:
        print(f"⚠️  n={n} is very small. Treat any win-rate/profit number as provisional — "
              f"not yet enough data to draw real conclusions.")
    elif n < 100:
        print(f"⚠️  n={n} is still small. Directionally informative, not yet statistically solid.")
    else:
        print(f"n={n} — reasonable sample size floor. Still watch for regime changes over time.")

    if n >= 8:
        print("\n--- Win-rate significance (two-sided binomial test vs. p=0.5) ---")
        p_value = binomial_test_two_sided(wins, n, 0.5)
        print(f"p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("Win rate is statistically distinguishable from 50/50 at alpha=0.05.")
        else:
            print("⚠️  Cannot yet reject the hypothesis that this win rate is due to chance.")

        print("\n--- Bootstrap 95% CI on mean profit per trade ---")
        lo, hi = bootstrap_ci(profits)
        print(f"[{lo:.4%}, {hi:.4%}]")
        if lo < 0 < hi:
            print("⚠️  The confidence interval includes 0 — mean profitability not yet statistically confirmed.")
    else:
        print("\n(Skipping significance tests — need at least ~8 closed trades for a meaningful binomial test.)")

    # Per-pair breakdown
    print("\n--- Per-pair breakdown (closed trades) ---")
    pairs = {}
    for t in closed_trades:
        if t["close_profit"] is None:
            continue
        pairs.setdefault(t["pair"], []).append(float(t["close_profit"]))
    for pair, plist in sorted(pairs.items(), key=lambda x: -len(x[1])):
        arr = np.array(plist)
        w = int((arr > 0).sum())
        print(f"  {pair:14s}  n={len(arr):3d}  win_rate={w/len(arr):.0%}  avg={arr.mean():+.3%}")

    # Exit reason breakdown
    print("\n--- Exit reason breakdown ---")
    reasons = {}
    for t in closed_trades:
        r = t["exit_reason"] or "unknown"
        reasons[r] = reasons.get(r, 0) + 1
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}  {count}")

    print("\n--- Reminder ---")
    print("This reflects only trades that actually filled — entries cancelled before filling")
    print("(e.g. limit-order timeouts) aren't in this table and aren't counted here.")
    print("Small sample sizes can look great or terrible by chance — keep collecting data")
    print("before treating any of these numbers as a real verdict on the strategy.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    analyze(sys.argv[1])
