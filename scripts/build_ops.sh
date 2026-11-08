#!/usr/bin/env bash
# Builds the ops image (training and Dagster dependencies). The first build downloads a lot
# and takes a while. The full log goes to build_ops.log. Run from the project root inside WSL.
set -uo pipefail

docker compose --profile ops build --progress=plain ops 2>&1 | tee build_ops.log | tail -15
echo "build exit: ${PIPESTATUS[0]}"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E 'frauddetect:ops' || echo "no ops image"
