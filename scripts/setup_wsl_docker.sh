#!/usr/bin/env bash
# Installs the open-source Docker Engine inside WSL Ubuntu, from Ubuntu's own
# repositories. It is free, and asks once for your Ubuntu sudo password.
set -euo pipefail

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker

sudo docker version --format 'Docker Engine {{.Server.Version}} is running'
docker compose version
echo "Done. You can close this tab."
