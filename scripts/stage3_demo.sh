#!/usr/bin/env bash
# The whole stage-three story on a clean stack. Run from the project root inside WSL.
# Warning: this deletes the local Redis, PostgreSQL and MLflow volumes of this project.
#
#   1. Train v1 and replay twelve days of traffic through Kafka
#   2. Release the chargebacks that have arrived by day 112
#   3. Look at what the monitors say
#   4. Retrain, and let the gate decide
#   5. Switch the live model by hand, watch the running services follow, and roll back
set -euo pipefail

sql() { docker compose exec -T postgres psql -U fraud -d fraud -c "$1"; }
sql_value() { docker compose exec -T postgres psql -U fraud -d fraud -At -c "$1"; }
ops() { docker compose --profile ops run --rm -T ops "$@"; }
stream() { docker compose run --rm -T producer python -m frauddetect.stream "$@" 2>&1; }

replay() {  # replay FROM_DAY TO_DAY: send that range through Kafka and wait for the decisions
  local from=$1 to=$2 before sent stored
  before=$(sql_value "SELECT count(*) FROM decisions")
  sent=$(stream produce --from-day "$from" --to-day "$to" --n 1000000 --rate 1500 \
         | grep -oE 'Published [0-9,]+' | tr -dc '0-9')
  for _ in $(seq 1 300); do
    stored=$(( $(sql_value 'SELECT count(*) FROM decisions') - before ))
    [ "$stored" -ge "$sent" ] && break
    sleep 1
  done
  echo "days $from to $to: $sent transactions sent, $stored decisions stored"
  if [ "$stored" -ne "$sent" ]; then
    echo "Replay did not store the expected number of decisions" >&2
    return 1
  fi
}

step() { echo; echo "=== $* ==="; }

docker compose --profile '*' down -v --remove-orphans 2>&1 | tail -1
# The images bake in the code. Rebuilding is quick when only code changed.
docker compose --profile ops build ops 2>&1 | tail -1

step "1. Train v1 (the model in production before merchant history existed)"
ops python -m frauddetect.train 2>&1 | grep -E "labels already|low_and_slow|test_pr_auc|test_recall"

docker compose up --build --wait api worker sink
step "1. Replay days 100 to 112 under v1"
replay 100 112

step "2. Chargebacks that have arrived by day 112"
stream labels --as-of-day 112 | grep Published
sleep 8
echo "labels stored: $(sql_value 'SELECT count(*) FROM labels')"

step "3. What the monitors say"
ops python -m frauddetect.monitor report --since-day 100 --until-day 112 --text
echo
echo "Share of confirmed fraud caught, by day:"
sql "SELECT day, decisions, confirmed_fraud, caught, recall_pct FROM cohort_recall WHERE day >= 104 ORDER BY day"

step "4. Retrain on the labels known at day 112"
ops python -m frauddetect.mlops retrain --as-of-day 112 --check-future --text

step "5. Switch the live model by hand: v2 is live"
ops python -m frauddetect.mlops promote v2
sleep 3
replay 112 113
echo "API answers with:"
docker compose exec -T api python - <<'PY'
import json, urllib.request
# Inside the day just replayed, so it does not move the system's idea of "today".
body = {"transaction_id": "demo-1", "ts": 112.9 * 86400.0, "card_id": "demo-card", "merchant_id": "m1",
        "merchant_category": 1, "amount": 25.0, "lat": 40.0, "lon": -75.0, "device_id": "d1",
        "channel": "pos"}
request = urllib.request.Request("http://localhost:8000/v1/score", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
print(" ", json.loads(urllib.request.urlopen(request).read()))
PY

step "5. Roll back: v1 is live again"
ops python -m frauddetect.mlops promote v1
sleep 3
replay 113 114

echo
echo "Decisions by day and model version (the services switched without a restart):"
sql "SELECT floor(event_ts / 86400)::int AS day, model_version, count(*) FROM decisions WHERE event_ts >= 111 * 86400 GROUP BY 1, 2 ORDER BY 1, 2"
