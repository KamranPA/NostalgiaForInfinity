#!/bin/bash
# SAVE THIS FILE AS: .github/workflows/scripts/watchdog_check.sh
# (create the "scripts" folder inside .github/workflows/ if it doesn't
# exist yet, and put this file there)
#
# Usage: watchdog_check.sh <workflow-file-name> <owner/repo> <ref>
# Requires GH_TOKEN in the environment.
#
# Checks the given workflow's most recent runs. If NONE of them are
# currently "queued" or "in_progress", dispatches a new run — this is
# the safety net for a self-queueing dry-run chain that stopped without
# managing to queue its own successor. If a run IS already active
# (including one this same chain already queued), does nothing, so this
# can safely run on a tight schedule (e.g. every 30 min) without ever
# stacking up duplicate queued runs.

set -e

WORKFLOW_FILE="$1"
REPO="$2"
REF="$3"

if [ -z "$WORKFLOW_FILE" ] || [ -z "$REPO" ] || [ -z "$REF" ]; then
  echo "Usage: $0 <workflow-file-name> <owner/repo> <ref>" >&2
  exit 1
fi

RESPONSE=$(curl -s -H "Authorization: Bearer ${GH_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/runs?per_page=5")

ACTIVE_COUNT=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print(-1)  # signals 'could not parse response' to the shell below
    sys.exit(0)
runs = data.get('workflow_runs', [])
active = [r for r in runs if r.get('status') in ('queued', 'in_progress')]
print(len(active))
")

echo "Workflow: $WORKFLOW_FILE — active (queued/in_progress) runs found: $ACTIVE_COUNT"

if [ "$ACTIVE_COUNT" = "-1" ]; then
  echo "::warning::Could not parse the GitHub API response for $WORKFLOW_FILE — skipping this cycle rather than guessing. Raw response (first 300 chars):"
  echo "$RESPONSE" | head -c 300
  exit 0
fi

if [ "$ACTIVE_COUNT" -eq 0 ]; then
  echo "No active run found for $WORKFLOW_FILE — it appears to have stopped without queueing its own successor. Dispatching a new run now."
  DISPATCH_HTTP_CODE=$(curl -s -o /tmp/dispatch_resp.txt -w "%{http_code}" -X POST \
    -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches" \
    -d "{\"ref\": \"${REF}\"}")

  if [ "$DISPATCH_HTTP_CODE" = "204" ]; then
    echo "Dispatch accepted (HTTP 204)."
    echo "$WORKFLOW_FILE" >> /tmp/watchdog_restarted.txt
  else
    echo "::warning::Dispatch for $WORKFLOW_FILE returned unexpected HTTP $DISPATCH_HTTP_CODE — response:"
    cat /tmp/dispatch_resp.txt
  fi
else
  echo "$WORKFLOW_FILE is already active — nothing to do this cycle."
fi
