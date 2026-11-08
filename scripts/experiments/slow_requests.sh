#!/usr/bin/env bash
# Where do the slow API requests spend their time? Run from the project root inside WSL.
set -uo pipefail

summary() { grep -E "http_req_duration|http_reqs|dropped" || true; }

docker compose build bootstrap 2>&1 | tail -1
docker compose up -d --no-build --force-recreate api worker sink 2>&1 | tail -1
sleep 25
bash scripts/e2e_check.sh 10000 1000 0 | tail -1

docker compose logs api --since 1s >/dev/null 2>&1
mark=$(date -u +%Y-%m-%dT%H:%M:%S)
echo "=== k6, 200 rps, 20 s ==="
{ docker compose run --rm --no-deps -e RATE=200 -e DURATION=20s k6 2>&1 || true; } | summary

echo "=== slow requests seen inside the handler (over 30 ms) ==="
docker compose logs api --since "$mark" 2>&1 | grep -c "slow request" || true
echo "=== slow Redis store calls (over 20 ms) ==="
docker compose logs api --since "$mark" 2>&1 | grep -c "slow store" || true
docker compose logs api --since "$mark" 2>&1 | grep "slow store" | sed 's/^api-1 *| //' | cut -c1-120 | head -14
