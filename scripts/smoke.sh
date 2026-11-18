#!/usr/bin/env bash
# Quick end-to-end check on a clean stack. Run from the project root inside WSL.
# Warning: this deletes the local Redis and PostgreSQL volumes of this project.
set -euo pipefail

docker compose --profile '*' down -v --remove-orphans
docker compose up --build --wait api worker sink
bash scripts/e2e_check.sh 3000 200 0
docker compose down
