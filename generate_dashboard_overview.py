#
# generate_dashboard_overview.py
#
# FUTURES-ONLY version, WITHOUT X7ML (2026-10-25): the spot bots are off and
# the team migrated from X7ML to X8, so the overview shows only the
# 2 futures bots: X8 and X8 Guard.
#
# RESET OF THE NUMBERS (v2 of this script)
# ----------------------------------------
# From the switch time T on, the MAIN page (output_html_path) counts both bots
# from ZERO: only trades OPENED at/after T are included in its cards and charts.
# T is read from "cut_start_date" in config_dryrun_futures_x8guard.json
# (guard_parameters) -- the SAME value the Guard strategy and guard.html use.
#
# Nothing is deleted or changed:
#   * the DETAILS page (optional 8th argument) keeps showing the full ALL-TIME
#     numbers, tables and charts exactly as before;
#   * the bots' databases and the history file are untouched; the old history
#     snapshots stay under their old mode ids (futures_x8, futures_x8guard);
#   * the reset view records its own snapshots under NEW mode ids
#     (futures_x8_since, futures_x8guard_since), so its "True Total over time"
#     chart simply starts at the reset.
#
# Trades opened BEFORE T (still open, or closed after T) are carried over by
# both bots. They are NOT counted in the reset numbers, but are listed side by
# side in a "Carried over" card, so a difference in what each bot still holds is
# visible instead of hidden.
#
# If T is missing from the config or lies in the future, both pages behave as
# before (all-time numbers) and a note says when the reset is scheduled.
#
# TRUE TOTAL BOX (v3 of this script)
# ----------------------------------
# A small box at the TOP of the futures section shows the True Total of both
# bots side by side (X8 and X8 Guard) and the difference (Guard minus X8). The
# numbers are the same ones as in each bot's Capital Efficiency card: the
# "since reset" numbers when the reset is active, the all-time numbers
# otherwise. A tiny script copies the live (browser re-priced) value of each
# bot's own True Total card into the box every 2 seconds, so the box stays in
# step with the page without a second price request.
#
# It IMPORTS generate_dashboard.py and generate_dashboard_compare.py and
# reuses their builders; neither file is edited. The existing section HTML
# is re-ordered by capturing the diagnostic cards (monkeypatch, same
# technique compare.py uses) and splitting at a marker.
#
# It writes history snapshots ONLY for the two new "_since" mode ids; the other
# scripts keep recording the old ids, so nothing is double-recorded.
#
# COMMAND-LINE NOTE: the argument list is UNCHANGED on purpose, so the
# workflow file does not need editing. The two spot sqlite arguments
# (spot_x7ml, spot_x8) and the X7ML futures argument (futures_x7ml) are still
# accepted but IGNORED.
#
# Usage:
#     python generate_dashboard_overview.py \
#         <spot_x7ml.sqlite (ignored)> <futures_x7ml.sqlite (ignored)> \
#         <spot_x8.sqlite (ignored)> <futures_x8.sqlite> <futures_x8guard.sqlite> \
#         <history.sqlite> <main_html_path> [<details_html_path>]
#

import os
import re
import sys
import traceback
from datetime import datetime, timezone

import generate_dashboard as gd
import generate_dashboard_compare as gc
from generate_dashboard_guard import read_guard_parameters, read_switch_time, inherited_html

SPLIT = "<!--NFI_SPLIT-->"
DIAG = (
    "build_duration_outcome_html",
    "build_time_clustering_html",
    "build_dca_activity_html",
    "build_ml_readiness_html",
)

EXTRA_CSS = """
  .group-title { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700;
                 margin:48px 0 0; padding-bottom:8px; border-bottom:1px solid var(--border); }
  .group-title:first-of-type { margin-top:8px; }
  details.fold { margin-bottom:12px; border:1px solid var(--border); border-radius:10px; background:var(--bg-raised); }
  details.fold > summary { cursor:pointer; padding:14px 18px; font-weight:600; }
  details.fold[open] > summary { border-bottom:1px solid var(--border); }
  details.fold > .inner { padding:16px; }
  .skip-note { color:var(--muted); font-size:0.85rem; }
"""

# Plain string (NOT an f-string) so the JS braces need no escaping.
# Copies the live True Total text/colour of each bot's own Capital Efficiency
# card into the box at the top, and recomputes the difference.
TT_SYNC_JS = """
(function () {
  try {
    var pairs = [["futures_x8", "futures_x8_since"], ["futures_x8guard", "futures_x8guard_since"]];
    function sync() {
      var vals = {};
      pairs.forEach(function (p) {
        var src = document.getElementById('live-total-' + p[1]) || document.getElementById('live-total-' + p[0]);
        var dst = document.getElementById('box-total-' + p[0]);
        if (src && dst) {
          dst.textContent = src.textContent;
          dst.className = src.className;
          var m = src.textContent.match(/[-+]?[0-9]+(\\.[0-9]+)?/);
          if (m) { vals[p[0]] = parseFloat(m[0]); }
        }
      });
      var d = document.getElementById('box-diff');
      if (d && vals.futures_x8 !== undefined && vals.futures_x8guard !== undefined) {
        var diff = vals.futures_x8guard - vals.futures_x8;
        d.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(2) + ' USDT';
        d.className = 'value ' + (diff >= 0 ? 'profit-pos' : 'profit-neg');
      }
    }
    sync();
    setInterval(sync, 2000);
  } catch (e) { /* never break the rest of the page */ }
})();
"""


def build_and_split(builder, *args, **kwargs):
    """Runs a gd.build_one_mode-style builder with the diagnostic card
    builders captured, then returns (title, top_html, detail_html,
    analytics_html, section_js, bundle)."""
    cap, saved = {}, {n: getattr(gd, n) for n in DIAG}

    def wrap(name, ret):
        orig = saved[name]

        def w(*a, **k):
            cap[name] = orig(*a, **k)
            return ret
        return w

    gd.build_duration_outcome_html = wrap("build_duration_outcome_html", SPLIT)
    for n in DIAG[1:]:
        setattr(gd, n, wrap(n, ""))
    try:
        b = builder(*args, **kwargs)
    finally:
        for n, f in saved.items():
            setattr(gd, n, f)

    before, _, detail = b["section_html"].partition(SPLIT)
    idx = before.find("Statistical Significance</h3>")
    if idx != -1:
        cut = before.rfind('<div class="card">', 0, idx)
        top, sig = before[:cut], before[cut:]
    else:
        top, sig = before, ""
    analytics = sig + "".join(cap.get(n, "") for n in DIAG)
    return b["compare"]["mode_title"], top, detail, analytics, b["section_js"], b


def slim_top(top):
    """Keep only: heading, stats grid, Capital Efficiency, Equity Curve,
    True Total & Stuck. Portfolio Utilization and Exit Reasons are hidden
    (not removed) so their chart canvases still exist for the JS."""
    top = re.sub(r'<div class="card">(\s*<h3>(?:Portfolio Utilization|Exit Reasons)</h3>)',
                 r'<div class="card" style="display:none">\1', top)
    return top.replace('<div class="two-col">', '<div>')


def build_since(path, history_db, mode_id, title, note, config_path, reset=None, **kw):
    """Same as gc.build_x8_mode, but only trades OPENED at/after `reset` are loaded
    (gd.fetch_trades is wrapped for the duration of the call; the database is not touched)."""
    orig = gd.fetch_trades

    def filtered(db_path):
        return [t for t in orig(db_path)
                if t["open_date"] is not None and gd.to_aware_utc(t["open_date"]) >= reset]

    gd.fetch_trades = filtered
    try:
        return gc.build_x8_mode(path, history_db, mode_id, title, note, config_path, **kw)
    finally:
        gd.fetch_trades = orig


def snapshot_numbers(bundle):
    """(true_total_abs, open_trades, closed_trades) from a bundle's current_snapshot,
    or None when it can't be read."""
    try:
        snap = bundle["current_snapshot"]
        return (float(snap["true_total_abs"]), int(snap["open_trades"]), int(snap["closed_trades"]))
    except Exception:
        traceback.print_exc()
        return None


def build_true_total_box(items, scope_txt):
    """items: list of (mode_id, label, true_total, n_open, n_closed) in display order.
    Returns the HTML of the box at the top of the futures section."""
    if not items:
        return ""

    def cls(v):
        return "profit-pos" if v >= 0 else "profit-neg"

    cards = ""
    for mode_id, label, total, n_open, n_closed in items:
        cards += f"""
    <div class="stat-card">
      <div class="label">{gd.esc(label)}</div>
      <div id="box-total-{mode_id}" class="value {cls(total)}">{total:+.2f} USDT</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">{n_open} open / {n_closed} closed</div>
    </div>"""

    by_id = {i[0]: i[2] for i in items}
    if "futures_x8" in by_id and "futures_x8guard" in by_id:
        diff = by_id["futures_x8guard"] - by_id["futures_x8"]
        cards += f"""
    <div class="stat-card">
      <div class="label">Guard minus X8</div>
      <div id="box-diff" class="value {cls(diff)}">{diff:+.2f} USDT</div>
      <div class="muted" style="font-size:0.75rem;margin-top:4px;">positive = Guard is ahead</div>
    </div>"""

    return f"""
<div class="card">
  <h3>True Total &mdash; X8 vs X8 Guard <span class="muted" style="font-weight:400;font-size:0.75rem;">&mdash; {gd.esc(scope_txt)}</span></h3>
  <div class="grid" style="margin-bottom:0;">{cards}
  </div>
  <p class="muted" style="font-size:0.8rem;margin-top:12px;margin-bottom:0;">
    True Total = closed trades + profit already booked by partial exits of open trades + unrealized P/L
    of the remaining amount. Re-priced live in the browser, same numbers as each bot's Capital Efficiency card below.
  </p>
</div>"""


def main(spot_x7, fut_x7, spot_x8, fut_x8, fut_guard, history_db, output_path, details_path=None):
    # spot_x7, fut_x7 and spot_x8 are intentionally unused
    # (spot bots are off; X7ML was retired on 2026-10-25).
    gc.install_patches()
    swap = gc.SWAP_OPTIONS
    now = datetime.now(timezone.utc)

    reset = read_switch_time(read_guard_parameters())
    reset_active = reset is not None and reset <= now
    reset_txt = f"{reset:%Y-%m-%d %H:%M} UTC" if reset else ""
    base_note = "OKX perpetual swaps \u00b7 isolated margin \u00b7 3x leverage"
    guard_note = base_note + (f" \u00b7 cut rule from {reset_txt}" if reset else "")

    jobs = [("futures_x8", "\U0001f7e0 X8 FUTURES \u2014 OKX", base_note, fut_x8, "config_dryrun_futures_x8.json")]
    if fut_guard and os.path.exists(fut_guard) and os.path.getsize(fut_guard) > 0:
        jobs.append(("futures_x8guard", "\U0001f6e1\ufe0f X8 GUARD FUTURES \u2014 OKX", guard_note,
                     fut_guard, "config_dryrun_futures_x8guard.json"))

    kw = dict(live_price_exchange_id="okx", live_price_ccxt_options=swap)
    tops = {"futures": ""}
    details, analytics, fulls = "", "", ""
    js_main, js_all = "", ""
    all_trades = {}          # mode_id -> all-time trade list (for the carried-over card)
    box_items = []           # (mode_id, label, true_total, n_open, n_closed) for the True Total box
    box_labels = {"futures_x8": "X8", "futures_x8guard": "X8 Guard"}

    for mode_id, title_txt, note, path, cfg_path in jobs:
        try:
            title, top_all, detail, ana, section_js_all, b_all = build_and_split(
                gc.build_x8_mode, path, history_db, mode_id, title_txt, note, cfg_path, **kw)
        except Exception:
            traceback.print_exc()
            tops["futures"] += f'<div class="card skip-note">Could not build {gd.esc(title_txt)} (see workflow log).</div>'
            continue
        all_trades[mode_id] = b_all["trades"]

        # --- MAIN page block: since the reset (new trades only), else all-time ---
        main_top, main_js = top_all, section_js_all
        main_numbers = snapshot_numbers(b_all)
        if reset_active:
            try:
                s_title, s_top, _d, _a, s_js, s_bundle = build_and_split(
                    build_since, path, history_db, mode_id + "_since",
                    title_txt.replace(" \u2014 OKX", " \u2014 since reset \u2014 OKX"),
                    note + f" \u00b7 numbers restart at {reset_txt}", cfg_path, reset=reset, **kw)
                main_top, main_js = s_top, s_js
                main_numbers = snapshot_numbers(s_bundle) or main_numbers
                try:
                    gd.record_snapshot(history_db, mode_id + "_since", s_bundle["current_snapshot"])
                except Exception:
                    traceback.print_exc()
                    print("Warning: could not record the since-reset snapshot (page still written).")
            except Exception:
                traceback.print_exc()
                tops["futures"] += (f'<div class="card skip-note">Could not build the since-reset view of '
                                    f'{gd.esc(title_txt)} - showing all-time numbers instead (see workflow log).</div>')
        tops["futures"] += slim_top(main_top)
        js_main += main_js
        if main_numbers is not None:
            box_items.append((mode_id, box_labels.get(mode_id, mode_id), main_numbers[0],
                              main_numbers[1], main_numbers[2]))

        # --- DETAILS page: always the full all-time numbers (unchanged) ---
        fulls += f'<details class="fold"><summary>{gd.esc(title)}</summary><div class="inner">{top_all}</div></details>'
        details += f'<details class="fold"><summary>{gd.esc(title)}</summary><div class="inner">{detail}</div></details>'
        if ana.strip():
            analytics += f'<details class="fold"><summary>{gd.esc(title)}</summary><div class="inner">{ana}</div></details>'
        js_all += section_js_all

    # ---- True Total box (top of the futures section, main page only) ----
    scope_txt = f"since reset ({reset_txt})" if reset_active else "all-time (the reset has not started yet)"
    true_total_box = build_true_total_box(box_items, scope_txt)
    if true_total_box:
        js_main += TT_SYNC_JS

    # ---- reset note + carried-over card (main page only) ----
    reset_cards = ""
    if reset_active:
        counts = []
        for mode_id, label in (("futures_x8", "X8"), ("futures_x8guard", "X8 Guard")):
            trs = all_trades.get(mode_id)
            if trs is None:
                continue
            carried_open = sum(1 for t in trs if t["is_open"] and t["open_date"]
                               and gd.to_aware_utc(t["open_date"]) < reset)
            counts.append(f"{label}: <b>{carried_open}</b> open")
        reset_cards = f"""
<div class="card">
  <h3>\U0001f504 Numbers restart at {reset_txt}</h3>
  <p class="muted" style="font-size:0.85rem;margin:0;">
    Both bots are counted from zero below: only trades opened after the reset are included.
    Trades opened earlier are carried over (list at the bottom of this section) and are not part of these
    numbers; nothing was deleted &mdash; the full all-time numbers and trade lists are on the
    <a href="details.html">Details page</a>. Still-open carried-over trades (they keep occupying slots): {' \u00b7 '.join(counts) if counts else 'none'}.
  </p>
</div>"""
        if "futures_x8" in all_trades and "futures_x8guard" in all_trades:
            reset_cards_after = inherited_html(all_trades["futures_x8"], all_trades["futures_x8guard"], reset)
        else:
            reset_cards_after = ""
    else:
        reset_cards_after = ""
        if reset is not None:
            reset_cards = (f'<div class="card"><p class="muted" style="margin:0;font-size:0.85rem;">'
                           f'The numbers will restart at {reset_txt} (not reached yet); until then this page shows all-time numbers.</p></div>')

    now_txt = now.strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard \u2014 Overview</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{gd.PAGE_STYLES}
{gc.EXTRA_CSS}
{EXTRA_CSS}
</style>
</head>
<body>

<div class="topbar">
  <div class="brand">NFI <span>dry-run</span></div>
  <nav>
    <a href="#grp-futures">Futures</a>
  </nav>
  <span class="gen-time">generated {now_txt} UTC</span>
  <a class="compare-link" href="details.html">Details \u2192</a>
</div>

<main>
<div class="disclaimer">Dry-run (simulated) \u2014 no real funds involved \u00b7 futures only (spot bots are off, X7ML retired) \u00b7 all bots on OKX \u00b7
  <a href="guard.html">X8 vs Guard</a> \u00b7 <a href="details.html">Trade details &amp; analytics (all-time)</a></div>

<h2 class="group-title" id="grp-futures">\U0001f7e0 FUTURES</h2>
{true_total_box}
{reset_cards}
{tops['futures']}
{reset_cards_after}

<div class="subtitle">Small sample sizes can look great or terrible by chance.</div>
</main>

<script>
{js_main}
</script>

</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Main dashboard written to {output_path}")

    if details_path:
        d = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard \u2014 Details</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
{gd.PAGE_STYLES}
{gc.EXTRA_CSS}
{EXTRA_CSS}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">NFI <span>dry-run</span></div>
  <nav><a href="#summary">Full summary</a><a href="#details">Trade details</a><a href="#analytics">Analytics</a></nav>
  <span class="gen-time">generated {now_txt} UTC</span>
  <a class="compare-link" href="index.html">\u2190 Main</a>
</div>
<main>
<h2 class="group-title" id="summary">Full summary (all-time)</h2>
<div class="subtitle">Everything since the bots started, never reset{f' (the Main page restarted its numbers at {reset_txt})' if reset_active else ''} \u2014 plus Portfolio Utilization and Exit Reasons; nothing is dropped.</div>
{fulls}
<h2 class="group-title" id="details">Trade details</h2>
<div class="subtitle">Open / closed trades, tag and pair performance \u2014 tap a bot to expand.</div>
{details}
<h2 class="group-title" id="analytics">Analytics</h2>
<div class="subtitle">Significance, duration, clustering, DCA activity.</div>
{analytics}
</main>
<script>
{js_all}
</script>
</body>
</html>"""
        with open(details_path, "w") as f:
            f.write(d)
        print(f"Details page written to {details_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (8, 9):
        print("Usage: python generate_dashboard_overview.py <8 or 9 arguments, see the comments at the top of this file>")
        sys.exit(1)
    main(*sys.argv[1:])
