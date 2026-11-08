#!/usr/bin/env bash
# Does a do-nothing endpoint show the same tail latency? Run from the project root inside WSL.
set -uo pipefail

summary() { grep -E "http_req_duration|dropped" || true; }

docker compose up -d --no-build api worker sink 2>&1 | tail -1
sleep 15

for round in 1 2 3; do
  echo "=== /health (no work), 200 rps, round $round ==="
  { docker compose run --rm --no-deps -v "$PWD/loadtest:/scripts:ro" \
      -e API_URL=http://api:8000/health -e RATE=200 k6 run /scripts/health.js 2>&1 || true; } | summary
  echo "=== /v1/score, 200 rps, round $round ==="
  { docker compose run --rm --no-deps -e RATE=200 -e DURATION=20s k6 2>&1 || true; } | summary
done

echo "=== host load while idle ==="
uptime
cat /proc/pressure/cpu 2>/dev/null
