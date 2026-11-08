#!/usr/bin/env bash
# Finds where the API's tail latency comes from. Run from the project root inside WSL.
set -uo pipefail

summary() { grep -E "http_req_duration|dropped" || true; }
run_k6() { { docker compose run --rm --no-deps -e API_URL="$1" -e RATE="$2" -e DURATION=20s k6 2>&1 || true; } | summary; }

docker compose up -d --no-build api worker sink 2>&1 | tail -1
sleep 20
echo "=== Warm-up, 10,000 events, not a measurement ==="
bash scripts/e2e_check.sh 10000 1000 0 | tail -1

docker compose run -d --name api-nokafka --no-deps -e KAFKA_BOOTSTRAP= api >/dev/null
sleep 8

for round in 1 2; do
  echo "=== A. normal API (publishes to Kafka), 200 rps, round $round ==="
  run_k6 http://api:8000/v1/score 200
  echo "=== B. API without Kafka publishing, 200 rps, round $round ==="
  run_k6 http://api-nokafka:8000/v1/score 200
done
echo "=== C. normal API, 100 rps ==="
run_k6 http://api:8000/v1/score 100
echo "=== D. API without Kafka, 100 rps ==="
run_k6 http://api-nokafka:8000/v1/score 100

docker rm -f api-nokafka >/dev/null
