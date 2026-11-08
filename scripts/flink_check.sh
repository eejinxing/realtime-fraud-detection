#!/usr/bin/env bash
# Checks the Flink job on its own: known decisions go into Kafka, and the totals
# the job writes to PostgreSQL must match exactly. Run from the project root inside WSL.
# Warning: this deletes the local Redis, PostgreSQL and MLflow volumes of this project.
set -uo pipefail

sql() { docker compose exec -T postgres psql -U fraud -d fraud -At -c "$1"; }

# Writes COUNT decisions of one kind into the decisions topic.
send() {  # send DECISION COUNT START_INDEX
  local decision=$1 count=$2 start=$3
  for i in $(seq "$start" $((start + count - 1))); do
    echo "{\"transaction_id\":\"flink-$i\",\"decision\":\"$decision\",\"score\":0.5,\"model_version\":\"v1\",\"card_id\":\"c$i\",\"amount\":10.0,\"features\":[1.0]}"
  done | docker compose exec -T kafka /opt/kafka/bin/kafka-console-producer.sh \
        --bootstrap-server kafka:29092 --topic decisions
}

docker compose --profile '*' down -v --remove-orphans 2>&1 | tail -1
docker compose up -d kafka postgres 2>&1 | tail -1
until [ "$(docker inspect -f '{{.State.Health.Status}}' realtime-fraud-detection-kafka-1 2>/dev/null)" = healthy ]; do sleep 2; done
docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:29092 \
  --create --if-not-exists --topic decisions --partitions 4 --replication-factor 1 | tail -1

docker compose --profile flink up -d flink-jobmanager flink-taskmanager 2>&1 | tail -2
echo "waiting for the Flink cluster"
for _ in $(seq 1 60); do
  slots=$(docker compose exec -T flink-jobmanager curl -s localhost:8081/overview 2>/dev/null | grep -o '"slots-total":[0-9]*' | cut -d: -f2)
  [ "${slots:-0}" -ge 1 ] && break
  sleep 2
done
echo "task slots available: ${slots:-0}"

echo "submitting the SQL job"
docker compose --profile flink run --rm flink-submit 2>&1 | grep -iE "job id|error|exception|submitted" | head -5
sleep 10
docker compose exec -T flink-jobmanager curl -s localhost:8081/jobs/overview | grep -o '"name":"[^"]*","state":"[A-Z]*"' | head -3

echo "sending 300 approve, 40 review, 12 decline over about 25 seconds"
for round in 1 2 3 4 5; do
  send approve 60 $((round * 1000)) >/dev/null 2>&1
  send review 8 $((round * 1000 + 100)) >/dev/null 2>&1
  [ "$round" -le 2 ] && send decline 6 $((round * 1000 + 200)) >/dev/null 2>&1
  sleep 5
done
echo "waiting for the last window to close"
sleep 25

echo "totals counted by Flink, against what was sent:"
sql "SELECT decision, sum(n), CASE decision WHEN 'approve' THEN 300 WHEN 'review' THEN 40 ELSE 12 END AS sent FROM decision_rate GROUP BY decision ORDER BY decision"
echo "windows written: $(sql 'SELECT count(DISTINCT window_start) FROM decision_rate')"
