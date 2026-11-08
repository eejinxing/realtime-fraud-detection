#!/usr/bin/env bash
# Sends transactions through Kafka and reports what reached PostgreSQL.
# Run from the project root inside WSL, after `docker compose up -d api worker sink`.
# Usage: bash scripts/e2e_check.sh [count] [events_per_second] [rows_to_skip]
set -euo pipefail

N=${1:-20000}
RATE=${2:-500}
SKIP=${3:-0}

sql() { docker compose exec -T postgres psql -U fraud -d fraud -At -c "$1"; }

before=$(sql "SELECT count(*) FROM decisions")
started=$(sql "SELECT now()")
docker compose run --rm producer python -m frauddetect.stream produce \
  --n "$N" --rate "$RATE" --skip "$SKIP" 2>&1 | grep -E "Published|Error" || true

target=$((before + N))
for _ in $(seq 1 180); do
  count=$(sql "SELECT count(*) FROM decisions")
  [ "$count" -ge "$target" ] && break
  sleep 1
done

echo "rows in decisions: $count (expected $target)"
echo "--- decisions in this run"
sql "SELECT decision, count(*) FROM decisions WHERE stored_at >= '$started' GROUP BY 1 ORDER BY 2 DESC"
echo "--- Kafka to decision latency (ms): p50 | p99 | max"
sql "SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY pipeline_ms)::numeric, 1),
            round(percentile_cont(0.99) WITHIN GROUP (ORDER BY pipeline_ms)::numeric, 1),
            round(max(pipeline_ms)::numeric, 1)
     FROM decisions WHERE pipeline_ms IS NOT NULL AND stored_at >= '$started'"

sql "COPY (SELECT transaction_id, score FROM decisions WHERE pipeline_ms IS NOT NULL)
     TO STDOUT WITH CSV HEADER" > /tmp/stream_scores.csv
docker compose run --rm -T -v /tmp/stream_scores.csv:/tmp/stream_scores.csv producer python -c "
import pandas as pd
live = pd.read_csv('/tmp/stream_scores.csv').set_index('transaction_id')['score']
offline = pd.read_parquet('artifacts/test_scores.parquet').set_index('transaction_id')['score']
gap = (live - offline.reindex(live.index)).abs()
print(f'compared {gap.notna().sum():,} scores, max gap vs offline {gap.max():.2e}')
" 2>&1 | tail -1
