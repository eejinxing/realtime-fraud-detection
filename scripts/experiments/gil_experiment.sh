#!/usr/bin/env bash
# Does Python's thread switch interval explain the API tail latency? Run from the project root inside WSL.
set -uo pipefail

summary() { grep -E "http_req_duration|dropped" || true; }
run_k6() { { docker compose run --rm --no-deps -e API_URL="$1" -e RATE=200 -e DURATION=20s k6 2>&1 || true; } | summary; }

docker compose build bootstrap 2>&1 | tail -1
docker compose up -d --no-build --force-recreate api worker sink 2>&1 | tail -1
sleep 25
bash scripts/e2e_check.sh 10000 1000 0 | tail -1

docker rm -f api-fast >/dev/null 2>&1
docker compose run -d --name api-fast --no-deps -e FRAUD_SWITCH_INTERVAL=0.0002 -e KAFKA_BOOTSTRAP= api >/dev/null
docker rm -f api-default >/dev/null 2>&1
docker compose run -d --name api-default --no-deps -e KAFKA_BOOTSTRAP= api >/dev/null
sleep 10

for round in 1 2 3; do
  echo "=== default switch interval (5 ms), round $round ==="
  run_k6 http://api-default:8000/v1/score
  echo "=== switch interval 0.2 ms, round $round ==="
  run_k6 http://api-fast:8000/v1/score
done
docker rm -f api-fast api-default >/dev/null 2>&1
