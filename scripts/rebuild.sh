#!/usr/bin/env bash
# Rebuilds the serving image after code changes. Dependencies are cached, so it takes seconds.
# Run from the project root inside WSL.
set -euo pipefail

docker compose build bootstrap 2>&1 | tail -2
docker run --rm --entrypoint python frauddetect:local -c \
  "import frauddetect.metrics as m; print('metrics module present, content type:', m.render()[1][:30])"
