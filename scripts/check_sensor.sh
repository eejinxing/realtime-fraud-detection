#!/usr/bin/env bash
# Asks the retrain sensor's logic about the stack that scripts/stage3_demo.sh leaves behind.
# Rebuilds both images first, because they bake in the code. Run from the project root inside WSL.
set -uo pipefail

docker compose --profile ops build ops bootstrap 2>&1 | tail -1
docker compose --profile '*' up -d --build api worker sink 2>&1 | tail -1
docker compose --profile orchestration up -d --force-recreate dagster 2>&1 | tail -1
sleep 20
docker compose exec -T dagster python -c "
import os, psycopg
from frauddetect.monitor import retrain_trigger
with psycopg.connect(os.environ['DATABASE_URL']) as connection:
    print('sensor would request:', retrain_trigger(connection))
"
