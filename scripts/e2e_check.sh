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
docker compose run --rm -T producer python -m frauddetect.stream produce \
  --n "$N" --rate "$RATE" --skip "$SKIP"

target=$((before + N))
for _ in $(seq 1 180); do
  count=$(sql "SELECT count(*) FROM decisions")
  [ "$count" -ge "$target" ] && break
  sleep 1
done

echo "rows in decisions: $count (expected $target)"
if [ "$count" -ne "$target" ]; then
  echo "Expected $N new decisions. Use an unused replay range on the baseline model." >&2
  exit 1
fi
echo "--- decisions in this run"
sql "SELECT decision, count(*) FROM decisions WHERE stored_at >= '$started' GROUP BY 1 ORDER BY 2 DESC"
echo "--- Kafka to decision latency (ms): p50 | p99 | max"
sql "SELECT round(percentile_cont(0.5) WITHIN GROUP (ORDER BY pipeline_ms)::numeric, 1),
            round(percentile_cont(0.99) WITHIN GROUP (ORDER BY pipeline_ms)::numeric, 1),
            round(max(pipeline_ms)::numeric, 1)
     FROM decisions WHERE pipeline_ms IS NOT NULL AND stored_at >= '$started'"

scores_file=$(mktemp)
trap 'rm -f "$scores_file"' EXIT
sql "COPY (SELECT transaction_id, score FROM decisions
           WHERE pipeline_ms IS NOT NULL AND stored_at >= '$started')
     TO STDOUT WITH CSV HEADER" > "$scores_file"
docker compose run --rm -T -v "$scores_file:/tmp/stream_scores.csv:ro" producer python -c "
import sys

import pandas as pd
live = pd.read_csv('/tmp/stream_scores.csv').set_index('transaction_id')['score']
offline = pd.read_parquet('artifacts/test_scores.parquet').set_index('transaction_id')['score']
gap = (live - offline.reindex(live.index)).abs()
print(f'compared {gap.notna().sum():,} scores, max gap vs offline {gap.max():.2e}')
if gap.notna().sum() != int(sys.argv[1]) or gap.max() > 1e-5:
    raise SystemExit('Stream scores diverged from the offline baseline')
" "$N"
