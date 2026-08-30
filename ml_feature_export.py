"""
ml_feature_export.py

Reads the ml_snapshot_* and entry_context custom_data entries logged by
the modified order_filled() in NostalgiaForInfinityX7.py, and flattens
them into a single wide table — one row per fill (entry, every rebuy
step, and exits) — ready to load into pandas/sklearn/whatever later.

This is purely a READ tool. It never touches the bot's decision-making.

Requires the strategy change that logs `entry_context` and
`ml_snapshot_fill_*` keys into custom_data (via trade.set_custom_data)
to already be deployed and running — rows only exist for trades opened
AFTER that change went live. Trades opened before it won't have this
data (there was nowhere to get it from retroactively).

Usage:
    pip install psycopg2-binary pandas --break-system-packages
    python ml_feature_export.py "<postgres-connection-string>" [output.csv]

If output.csv is omitted, prints a summary to the terminal instead of
writing a file.
"""

import sys
import json
from datetime import datetime

import pandas as pd

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Missing dependency. Install with: pip install psycopg2-binary pandas --break-system-packages")
    sys.exit(1)


def fetch_snapshots(db_url: str) -> pd.DataFrame:
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Join custom_data back to trades so every row also carries
            # the pair, enter_tag, and (if closed) final outcome — this
            # is what turns "conditions at signal time" into a labeled
            # training row: features (indicators/context) + label (did
            # this trade end up profitable).
            cur.execute("""
                SELECT
                    cd.ft_trade_id,
                    cd.cd_key,
                    cd.cd_value,
                    cd.created_at AS logged_at,
                    t.pair,
                    t.enter_tag,
                    t.is_open,
                    t.open_date,
                    t.close_date,
                    t.close_profit
                FROM trade_custom_data cd
                JOIN trades t ON t.id = cd.ft_trade_id
                WHERE cd.cd_key LIKE 'ml_snapshot_%%' OR cd.cd_key = 'entry_context'
                ORDER BY cd.ft_trade_id, cd.created_at ASC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame()

    flat_rows = []
    for r in rows:
        try:
            snapshot = json.loads(r["cd_value"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(snapshot, dict):
            continue

        flat = {
            "trade_id": r["ft_trade_id"],
            "pair": r["pair"],
            "enter_tag": r["enter_tag"],
            "is_open": r["is_open"],
            "trade_open_date": r["open_date"],
            "trade_close_date": r["close_date"],
            "final_close_profit_pct": (
                round(float(r["close_profit"]) * 100.0, 4) if r["close_profit"] is not None else None
            ),
            "custom_data_key": r["cd_key"],
        }
        flat.update(snapshot)
        flat_rows.append(flat)

    return pd.DataFrame(flat_rows)


def summarize(df: pd.DataFrame):
    print(f"Loaded {len(df)} snapshot rows across {df['trade_id'].nunique()} trades.\n")

    entry_rows = df[df["custom_data_key"] == "entry_context"]
    print(f"  entry_context rows (one per trade, first entry only): {len(entry_rows)}")
    print(f"  fill-level rows (every entry/rebuy/exit fill):        {len(df) - len(entry_rows)}\n")

    if entry_rows.empty:
        print("No entry_context rows yet — this means no trade has been opened since the")
        print("strategy's ml_snapshot logging was deployed. Data will start accumulating")
        print("from the next new trade onward; existing open trades from before the change")
        print("won't have it retroactively.")
        return

    print("Feature columns available per entry (for future model training):")
    feature_cols = [
        c for c in entry_rows.columns
        if c not in ("trade_id", "pair", "enter_tag", "is_open", "trade_open_date",
                     "trade_close_date", "final_close_profit_pct", "custom_data_key")
    ]
    for c in feature_cols:
        non_null = entry_rows[c].notna().sum()
        print(f"    {c:26s} ({non_null}/{len(entry_rows)} non-null)")

    closed = entry_rows[entry_rows["is_open"] == False]  # noqa: E712
    if len(closed) >= 1:
        print(f"\nClosed trades with a labeled outcome so far: {len(closed)}")
        print("(This is the number of labeled training rows you'd currently have for")
        print(" a meta-labeling model — compare against the ~100-200 minimum discussed")
        print(" earlier before treating any model trained on this as reliable.)")


def build_telegram_summary(df: pd.DataFrame) -> str:
    """Short, Telegram-friendly summary — not the full column dump."""
    lines = []
    lines.append("🧠 ML Feature Export — Daily Summary")
    lines.append("")

    entry_rows = df[df["custom_data_key"] == "entry_context"]
    n_trades_with_data = entry_rows["trade_id"].nunique()
    n_fill_rows = len(df) - len(entry_rows)

    lines.append(f"Trades with logged features: {n_trades_with_data}")
    lines.append(f"Total fill snapshots (entries+rebuys+exits): {n_fill_rows}")

    if entry_rows.empty:
        lines.append("")
        lines.append("No entry_context rows yet. Logging only applies to trades opened")
        lines.append("after the ml_snapshot change went live — still waiting on the")
        lines.append("first new trade to open.")
        return "\n".join(lines)

    closed = entry_rows[entry_rows["is_open"] == False]  # noqa: E712
    open_ = entry_rows[entry_rows["is_open"] == True]  # noqa: E712
    lines.append(f"  closed (labeled): {len(closed)}   open (unlabeled yet): {len(open_)}")

    MIN_FOR_MODEL = 100
    lines.append("")
    if len(closed) < MIN_FOR_MODEL:
        lines.append(
            f"Labeled rows: {len(closed)}/{MIN_FOR_MODEL} — still below the minimum "
            f"sample size before any model trained on this would be meaningful."
        )
    else:
        lines.append(
            f"Labeled rows: {len(closed)} — now at/above the {MIN_FOR_MODEL} minimum "
            f"discussed earlier. Worth revisiting whether meta-labeling is viable."
        )

    # Per-tag coverage, since that's what earlier analysis cared about most.
    if "enter_tag" in entry_rows.columns:
        lines.append("")
        lines.append("By enter_tag (closed / total logged):")
        for tag, grp in entry_rows.groupby("enter_tag"):
            n_closed = (grp["is_open"] == False).sum()  # noqa: E712
            lines.append(f"  tag {tag}: {n_closed}/{len(grp)}")

    # Basic feature completeness check — flags if BTC/indicator columns
    # are coming back empty, which would mean the logging silently broke.
    key_cols = ["rsi_14", "btc_rsi_14", "ema_20"]
    lines.append("")
    lines.append("Feature health check (non-null rate):")
    for c in key_cols:
        if c in entry_rows.columns:
            rate = entry_rows[c].notna().mean() * 100 if len(entry_rows) else 0
            lines.append(f"  {c}: {rate:.0f}%")
        else:
            lines.append(f"  {c}: MISSING COLUMN")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    db_url = sys.argv[1]
    remaining = sys.argv[2:]
    telegram_mode = "--telegram" in remaining
    out_path = next((a for a in remaining if not a.startswith("--")), None)

    df = fetch_snapshots(db_url)
    if df.empty:
        msg = (
            "No ml_snapshot/entry_context custom_data found yet.\n"
            "Make sure the updated NostalgiaForInfinityX7.py is deployed and at least\n"
            "one trade has opened since then."
        )
        print(msg)
        sys.exit(0)

    if telegram_mode:
        print(build_telegram_summary(df))
    else:
        summarize(df)
        if out_path:
            df.to_csv(out_path, index=False)
            print(f"\nWrote {len(df)} rows to {out_path}")
