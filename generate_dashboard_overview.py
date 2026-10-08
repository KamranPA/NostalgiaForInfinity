"""
generate_dashboard_overview.py

TWO pages from one run:

  MAIN page (output_html_path) -- ONLY the essentials, for all 5 bots:
     SPOT: X7ML, X8   |   FUTURES: X7ML, X8, X8 Guard
     per bot = headline stats grid, Capital Efficiency card,
               Equity Curve, True Total & Stuck Trades chart.

  DETAILS page (optional 8th arg) -- everything else, collapsed per bot:
     trade tables, tag/pair performance, significance, duration,
     clustering, DCA, ML readiness.

It IMPORTS generate_dashboard.py and generate_dashboard_compare.py and
reuses their builders; neither file is edited. The existing section HTML
is re-ordered by capturing the diagnostic cards (monkeypatch, same
technique compare.py uses) and splitting at a marker.

It does NOT write history snapshots (the other scripts already do), so
running it never double-records anything.

Usage:
    python generate_dashboard_overview.py \\
        <spot_x7ml.sqlite> <futures_x7ml.sqlite> \\
        <spot_x8.sqlite> <futures_x8.sqlite> <futures_x8guard.sqlite> \\
        <history.sqlite> <main_html_path> [<details_html_path>]
"""

import os
import re
import sys
import traceback
from datetime import datetime, timezone

import generate_dashboard as gd
import generate_dashboard_compare as gc

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


def build_and_split(builder, *args, **kwargs):
    """Runs a gd.build_one_mode-style builder with the diagnostic card
    builders captured, then returns (title, top_html, detail_html,
    analytics_html, section_js)."""
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
    return b["compare"]["mode_title"], top, detail, analytics, b["section_js"]


def slim_top(top):
    """Keep only: heading, stats grid, Capital Efficiency, Equity Curve,
    True Total & Stuck. Portfolio Utilization and Exit Reasons are hidden
    (not removed) so their chart canvases still exist for the JS."""
    top = re.sub(r'<div class="card">(\s*<h3>(?:Portfolio Utilization|Exit Reasons)</h3>)',
                 r'<div class="card" style="display:none">\1', top)
    return top.replace('<div class="two-col">', '<div>')


def main(spot_x7, fut_x7, spot_x8, fut_x8, fut_guard, history_db, output_path, details_path=None):
    gc.install_patches()
    swap = gc.SWAP_OPTIONS

    jobs = [
        ("spot", gd.build_one_mode, (spot_x7, history_db, "spot", "🟢 X7ML SPOT — OKX",
            "OKX spot market · no leverage", "config_dryrun_telegram.json"),
            dict(live_price_exchange_id="okx", live_price_ccxt_options=None)),
        ("spot", gc.build_x8_mode, (spot_x8, history_db, "spot_x8", "🔵 X8 SPOT — OKX",
            "OKX spot market · no leverage · NostalgiaForInfinityX8", "config_dryrun_telegram_x8.json"),
            dict(live_price_exchange_id="okx", live_price_ccxt_options=None)),
        ("futures", gd.build_one_mode, (fut_x7, history_db, "futures", "🟣 X7ML FUTURES — OKX",
            "OKX perpetual swaps · isolated margin · 3x leverage (default)", "config_dryrun_futures.json"),
            dict(live_price_exchange_id="okx", live_price_ccxt_options=swap)),
        ("futures", gc.build_x8_mode, (fut_x8, history_db, "futures_x8", "🟠 X8 FUTURES — OKX",
            "OKX perpetual swaps · isolated margin · 3x leverage", "config_dryrun_futures_x8.json"),
            dict(live_price_exchange_id="okx", live_price_ccxt_options=swap)),
    ]
    if fut_guard and os.path.exists(fut_guard) and os.path.getsize(fut_guard) > 0:
        jobs.append(("futures", gc.build_x8_mode, (fut_guard, history_db, "futures_x8guard",
            "🛡️ X8 GUARD FUTURES — OKX", "OKX perpetual swaps · isolated margin · 3x leverage · stale_range 15",
            "config_dryrun_futures_x8guard.json"),
            dict(live_price_exchange_id="okx", live_price_ccxt_options=swap)))

    tops = {"spot": "", "futures": ""}
    details, analytics, js = "", "", ""
    for group, builder, args, kwargs in jobs:
        try:
            title, top, detail, ana, section_js = build_and_split(builder, *args, **kwargs)
        except Exception:
            traceback.print_exc()
            tops[group] += f'<div class="card skip-note">Could not build {gd.esc(args[3])} (see workflow log).</div>'
            continue
        tops[group] += slim_top(top)
        details += f'<details class="fold"><summary>{gd.esc(title)}</summary><div class="inner">{detail}</div></details>'
        if ana.strip():
            analytics += f'<details class="fold"><summary>{gd.esc(title)}</summary><div class="inner">{ana}</div></details>'
        js += section_js

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NFI Dry-Run Dashboard — Overview</title>
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
    <a href="#grp-spot">Spot</a>
    <a href="#grp-futures">Futures</a>
  </nav>
  <span class="gen-time">generated {now} UTC</span>
  <a class="compare-link" href="details.html">Details →</a>
</div>

<main>
<div class="disclaimer">Dry-run (simulated) — no real funds involved · all bots on OKX ·
  <a href="compare.html">X7ML vs X8</a> · <a href="guard.html">X8 vs Guard</a> · <a href="details.html">Trade details &amp; ML</a></div>

<h2 class="group-title" id="grp-spot">🟢 SPOT</h2>
{tops['spot']}

<h2 class="group-title" id="grp-futures">🟣 FUTURES</h2>
{tops['futures']}

<div class="subtitle">Small sample sizes can look great or terrible by chance.</div>
</main>

<script>
{js}
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
<title>NFI Dry-Run Dashboard — Details</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
{gd.PAGE_STYLES}
{gc.EXTRA_CSS}
{EXTRA_CSS}
</style>
</head>
<body>
<div class="topbar">
  <div class="brand">NFI <span>dry-run</span></div>
  <nav><a href="#details">Trade details</a><a href="#ml">ML &amp; analytics</a></nav>
  <span class="gen-time">generated {now} UTC</span>
  <a class="compare-link" href="index.html">← Main</a>
</div>
<main>
<h2 class="group-title" id="details">Trade details</h2>
<div class="subtitle">Open / closed trades, tag and pair performance — tap a bot to expand.</div>
{details}
<h2 class="group-title" id="ml">ML data &amp; analytics</h2>
<div class="subtitle">ML readiness (X7ML only), significance, duration, clustering, DCA activity.</div>
{analytics}
</main>
</body>
</html>"""
        with open(details_path, "w") as f:
            f.write(d)
        print(f"Details page written to {details_path}")


if __name__ == "__main__":
    if len(sys.argv) not in (8, 9):
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:])
