#!/usr/bin/env bash
# Full stage-two run from a clean state. Run from the project root inside WSL.
# Warning: this deletes the local Redis and PostgreSQL volumes of this project.
#
# The order matters:
# * Warm-up comes first. The Kafka JVM compiles its hot paths only once traffic
#   arrives, and that burns several cores for a minute or two.
# * k6 comes last. It writes made-up transactions onto real cards, so anything
#   compared against offline scores has to run before it.
set -euo pipefail

k6_summary() { grep -E "http_req_duration|http_reqs|http_req_failed|dropped" || true; }

docker compose --profile '*' down -v --remove-orphans 2>&1 | tail -1
docker compose up -d --build api worker sink 2>&1 | tail -1
sleep 30

echo; echo "=== Warm-up, 20,000 events, not a measurement ==="
bash scripts/e2e_check.sh 20000 1000 0 | tail -1

echo; echo "=== Kafka path, 500 events/s ==="
bash scripts/e2e_check.sh 15000 500 20000

echo; echo "=== Kafka path, 1000 events/s ==="
bash scripts/e2e_check.sh 15000 1000 35000

echo; echo "=== Failover: kill one worker mid-stream ==="
# Wait until decisions are really flowing, so the kill lands mid-stream.
(
  rows() { docker compose exec -T postgres psql -U fraud -d fraud -At -c "SELECT count(*) FROM decisions"; }
  start=$(rows)
  until [ "$(rows)" -ge $((start + 1500)) ]; do sleep 0.2; done
  docker kill realtime-fraud-detection-worker-2 >/dev/null
  echo "worker-2 killed after $(($(rows) - start)) decisions of this run"
) &
bash scripts/e2e_check.sh 6000 500 50000
wait
docker compose up -d --no-build worker 2>&1 | tail -1
sleep 10

for round in 1 2 3; do
  echo; echo "=== Synchronous API, 200 requests/s, round $round ==="
  # k6 exits non-zero when a threshold fails, and that is a result, not a crash.
  { docker compose run --rm k6 2>&1 || true; } | k6_summary
done
