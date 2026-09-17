# NostalgiaForInfinity — Dry-Run + ML Data Collection Fork

This is a fork of [iterativv/NostalgiaForInfinity](https://github.com/iterativv/NostalgiaForInfinity),
a trading strategy for the [Freqtrade](https://www.freqtrade.io/) crypto bot.

This fork does **not** change the trading logic itself. It adds:
- two always-on **dry-run** bots (spot + futures) running entirely on GitHub Actions,
- Telegram notifications for every simulated entry/exit,
- structured **ML feature logging** on every order fill, for later meta-labeling,
- an automatic **updater** that keeps the strategy file(s) in sync with upstream.

No real orders are ever placed. `dry_run: true` in both configs, always.

---

## Repo layout

| File | Purpose |
|---|---|
| `NostalgiaForInfinityX7.py` | Upstream strategy, unmodified. Kept in sync automatically — see [Updater](#updater) below. Do not hand-edit this file; edits will be silently overwritten. |
| `NostalgiaForInfinityX7ML.py` | **This project's own file.** A thin subclass of `NostalgiaForInfinityX7` that adds ML-snapshot logging on every order fill. This is what both bots actually run. Never touched by the updater. |
| `NostalgiaForInfinityX8.py` | Upstream's newer strategy version, pulled in automatically once it appeared. **Not currently used by either bot** — as of writing, upstream itself hasn't confirmed X8 is out of testing. Kept around for a possible future third, separate dry-run bot. |
| `config_dryrun_telegram.json` | Spot bot config (OKX, no leverage). Never touched by the updater. |
| `config_dryrun_futures.json` | Futures bot config (OKX, isolated margin, 3x default leverage). Never touched by the updater. |
| `.github/workflows/dry-run-telegram-signals.yml` | Runs the spot bot. Self-queueing restart chain (~5h45m per session, see comments in the file for why). |
| `.github/workflows/dry-run-telegram-futures.yml` | Runs the futures bot. Same architecture, separate config/state/Telegram bot. |
| `.github/workflows/sync-upstream-strategy.yml` | The updater — see below. |

State (SQLite trade databases) is **not** stored in this repo's working tree. Each bot persists its database to its own dedicated git branch (`state-spot` / `state-futures`) between restarts.

---

## Why two bots

- **Spot** (`config_dryrun_telegram.json`) — no leverage.
- **Futures** (`config_dryrun_futures.json`) — isolated margin, 3x leverage, can go both long and short.

Both run the same strategy logic (`NostalgiaForInfinityX7ML`) on the same exchange (OKX), so their entries/exits can be compared directly (a "Signal Overlap" analysis: does a spot long fire on futures too, and vice versa) without the comparison being confounded by two exchanges' different price/volume data.

---

## ML data collection

`NostalgiaForInfinityX7ML.py` overrides `order_filled()` to write a snapshot of indicator values to freqtrade's built-in `trade_custom_data` table (in the same SQLite file as `trades`/`orders`) on every fill — entry, rebuy, and exit. This is meant to let a later ML model see the *conditions at the time of the signal*, not just the final profit/loss outcome.

Each snapshot includes: fill metadata (time, type, order tag, fill price), pair-level indicators (RSI, ADX/DI, EMA20/50/200, close, volume — note ADX/DI only exist on the 4h informative timeframe, hence the `_4h` suffix), BTC market context (RSI/EMA/ROC, also 4h), and portfolio pressure (open trade count vs. `max_open_trades`). The first entry's snapshot is also kept under a separate always-current `entry_context` key.

Query it directly via Telegram: `/list_custom_data <trade_id>`.

Logging is wrapped in a broad `try/except` so a logging failure can never break live (simulated) trading — if something goes wrong, it's silently skipped and a warning is written to the freqtrade log.

---

## Updater

`.github/workflows/sync-upstream-strategy.yml` runs daily (and on manual trigger). It:

1. Clones the upstream repo (`iterativv/NostalgiaForInfinity`) fresh.
2. Copies every **new or changed** file into this repo (never deletes anything, even if removed upstream).
3. **Never touches**: `.github/`, `NostalgiaForInfinityX7ML.py`, both `config_dryrun_*.json` files, or this `README.md`.
4. Runs a `py_compile` syntax check on every changed Python file. If anything fails to compile, nothing is committed.
5. Commits straight to `main` (no PR/review step — see the risk note in the workflow file itself) and posts a Telegram notification with the commit link.

**The bots do not restart when this runs.** They only pick up whatever is on `main` at the start of their *next* self-queued session. So an upstream update can sit on `main` for up to ~5h45m before it actually takes effect — check the Telegram notification and the commit diff in that window if you want to catch something before it goes live.

If you ever add another custom file at the repo root, add it to the exclude list near the top of `sync-upstream-strategy.yml`, or the updater will silently overwrite it on its next run.

---

## Known gotchas (learned the hard way)

- **`refresh_period` vs. `lookback_days` on pairlist filters.** Any pairlist entry (`VolumePairList`, `RangeStabilityFilter`, etc.) that sets `lookback_days` needs `refresh_period` at least `86400` (one day). A smaller value makes freqtrade refuse to start — and because this happens right after pairlist resolution, before anything else logs, it can look like the bot is "running" for hours while actually stuck. Always check Telegram `/count` and `/status table` after any pairlist change — `trader is not running` means it never got past this check.
- **OKX exchange support notes**, from this project's own configs: OKX's live-ticker data doesn't expose `quoteVolume` in the format `VolumePairList` expects by default, so `lookback_days` (candle-based volume) is required for both spot and futures on OKX.
- **`NostalgiaForInfinityX7ML.py` must ship together with `NostalgiaForInfinityX7.py`** in `user_data/strategies/` — it imports the base class directly. Both dry-run workflows copy both files; if you ever add a third bot/workflow, remember to do the same.
