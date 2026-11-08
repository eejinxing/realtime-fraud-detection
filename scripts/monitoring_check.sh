#!/usr/bin/env bash
# Starts the system with Prometheus and Grafana, sends traffic down both paths, and checks
# that the metrics and every dashboard panel work. Run from the project root inside WSL.
# Warning: this deletes the local Redis, PostgreSQL and MLflow volumes of this project.
set -uo pipefail

docker compose --profile '*' down -v --remove-orphans 2>&1 | tail -1
# --build: the image bakes in the code, so a stale image silently tests old code.
docker compose --profile monitoring up -d --build api worker sink prometheus grafana 2>&1 | tail -1
sleep 40

echo "sending traffic down the Kafka path and the API path at the same time"
( docker compose run --rm -T producer python -m frauddetect.stream produce \
    --n 15000 --rate 500 --from-day 100 --to-day 105 >/dev/null 2>&1 ) &
docker compose run --rm -T -e RATE=100 -e DURATION=25s k6 >/dev/null 2>&1 || true
wait
sleep 20  # a few scrape intervals, so the numbers settle

python3 scripts/grafana_check.py
