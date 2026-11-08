#!/usr/bin/env bash
# Checks Dagster and Flink against the stack that scripts/stage3_demo.sh leaves behind,
# so the retraining job runs against real decisions and chargebacks. It keeps the data.
# Run from the project root inside WSL.
set -uo pipefail

sql_value() { docker compose exec -T postgres psql -U fraud -d fraud -At -c "$1"; }
step() { echo; echo "=== $* ==="; }

docker compose --profile '*' up -d --build api worker sink 2>&1 | tail -1
sleep 20

step "Dagster: start it and ask what it knows"
docker compose --profile orchestration up -d dagster 2>&1 | tail -1
for _ in $(seq 1 90); do
  curl -sf localhost:3000/server_info >/dev/null && break
  sleep 2
done
python3 - <<'PY'
import json, urllib.request
query = """{ repositoriesOrError { ... on RepositoryConnection { nodes { name
  jobs { name } sensors { name sensorState { status } } } } } }"""
request = urllib.request.Request("http://localhost:3000/graphql", json.dumps({"query": query}).encode(),
                                 {"Content-Type": "application/json"})
for node in json.loads(urllib.request.urlopen(request, timeout=30).read())["data"]["repositoriesOrError"]["nodes"]:
    print("jobs:", [j["name"] for j in node["jobs"] if not j["name"].startswith("__")])
    print("sensors:", [(s["name"], s["sensorState"]["status"]) for s in node["sensors"]])
PY

step "Dagster: would the sensor start a retrain right now?"
docker compose exec -T dagster python -c "
import os, psycopg
from frauddetect.monitor import retrain_trigger
with psycopg.connect(os.environ['DATABASE_URL']) as connection:
    print(retrain_trigger(connection))
"

step "Dagster: run the retraining job against the real database"
docker compose exec -T dagster sh -c 'cat > /tmp/retrain.yaml' <<'YAML'
ops:
  plan_retrain:
    config:
      as_of_day: 112
      n_trials: 3
YAML
docker compose exec -T dagster dagster job execute -m frauddetect.orchestration -j retrain_job \
  -c /tmp/retrain.yaml 2>&1 | grep -E "Gate:|Keeping|is now live|Trained|Confirmed fraud|RUN_SUCCESS|RUN_FAILURE|Error" | sed 's/^[0-9-]* [0-9:,]* //' | cut -c1-200
docker compose --profile ops run --rm -T ops python -m frauddetect.mlops status

step "Flink: start the cluster and submit the job"
docker compose --profile flink up -d flink-jobmanager flink-taskmanager 2>&1 | tail -1
for _ in $(seq 1 60); do
  slots=$(docker compose exec -T flink-jobmanager curl -s localhost:8081/overview 2>/dev/null | grep -o '"slots-total":[0-9]*' | cut -d: -f2)
  [ "${slots:-0}" -ge 1 ] && break
  sleep 2
done
docker compose --profile flink run --rm flink-submit 2>&1 | grep -E "Job ID|Exception" | head -2
sleep 10

step "Flink: replay real traffic and compare the totals"
decisions_before=$(sql_value "SELECT count(*) FROM decisions")
windowed_before=$(sql_value "SELECT coalesce(sum(n), 0) FROM decision_rate")
docker compose run --rm -T producer python -m frauddetect.stream produce \
  --from-day 114 --to-day 115 --n 100000 --rate 300 2>&1 | grep Published
sleep 35
decisions_added=$(( $(sql_value "SELECT count(*) FROM decisions") - decisions_before ))
windowed_added=$(( $(sql_value "SELECT coalesce(sum(n), 0) FROM decision_rate") - windowed_before ))
echo "decisions stored in PostgreSQL: $decisions_added"
echo "decisions counted by Flink:     $windowed_added"

step "Grafana: every panel again, now with real data behind it"
docker compose --profile monitoring up -d prometheus grafana 2>&1 | tail -1
sleep 25
python3 scripts/grafana_check.py
