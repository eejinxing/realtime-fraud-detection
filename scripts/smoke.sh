#!/usr/bin/env bash
# Quick end-to-end check on a clean stack. Run from the project root inside WSL.
# Warning: this deletes the local Redis and PostgreSQL volumes of this project.
set -uo pipefail

docker compose --profile '*' down -v --remove-orphans 2>&1 | tail -1
docker compose up -d --build api worker sink 2>&1 | tail -3
echo "up-exit=$?"
sleep 30
docker compose ps -a --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'
bash scripts/e2e_check.sh 3000 200 0 2>&1 | tail -9
echo "e2e-exit=$?"
docker compose down 2>&1 | tail -1
