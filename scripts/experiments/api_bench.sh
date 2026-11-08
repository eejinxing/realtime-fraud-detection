#!/usr/bin/env bash
# Latency of the synchronous API under a fixed arrival rate. Run from the project root inside WSL.
# Usage: bash scripts/api_bench.sh [requests_per_second] [rounds]
set -uo pipefail

RATE=${1:-200}
ROUNDS=${2:-3}

docker compose down -v --remove-orphans 2>&1 | tail -1
docker compose up -d --build api worker sink 2>&1 | tail -1
sleep 30

echo "=== Warm-up, 20,000 events through Kafka, not a measurement ==="
bash scripts/e2e_check.sh 20000 1000 0 | tail -1

for round in $(seq 1 "$ROUNDS"); do
  echo "=== API at $RATE requests/s, round $round ==="
  { docker compose run --rm -e RATE="$RATE" k6 2>&1 || true; } \
    | grep -E "http_req_duration|http_reqs|http_req_failed|dropped"
done
docker compose exec -T api python -c "import uvloop, httptools; print('uvloop', uvloop.__version__, 'httptools', httptools.__version__)"
