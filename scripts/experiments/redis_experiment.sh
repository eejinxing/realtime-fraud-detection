#!/usr/bin/env bash
# Is the API tail latency caused by Redis disk persistence? Run from the project root inside WSL.
set -uo pipefail

redis() { docker compose exec -T redis redis-cli "$@"; }
summary() { grep -E "http_req_duration|dropped" || true; }
run_k6() { { docker compose run --rm --no-deps -e RATE=200 -e DURATION=20s k6 2>&1 || true; } | summary; }

docker compose up -d --no-build api worker sink 2>&1 | tail -1
sleep 15
redis config set latency-monitor-threshold 10 >/dev/null

for mode in yes no yes no; do
  redis config set appendonly "$mode" >/dev/null
  sleep 2
  redis latency reset >/dev/null
  echo "=== appendonly=$mode ==="
  run_k6
  echo "  redis latency events over 10 ms (event, last_ts, last_ms, max_ms):"
  redis latency latest | paste - - - - | sed 's/^/    /'
  redis info persistence | grep -E "aof_delayed_fsync|aof_rewrite_in_progress" | tr -d '\r' | tr '\n' ' '
  echo
done
redis config set appendonly yes >/dev/null
