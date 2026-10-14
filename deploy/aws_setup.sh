#!/usr/bin/env bash
# One-time setup of FloodSight on a fresh Ubuntu 24.04 EC2 instance. Run from the floodsight folder:
#   bash deploy/aws_setup.sh
# Needs, in this folder: .env (secrets, never committed), models/siamese_unet.pt, and optionally showcase/.
set -euo pipefail

[ -f run.py ] || { echo "Run this from the floodsight folder (the one with run.py)."; exit 1; }
[ -f .env ] || { echo "Missing .env (copy it with scp; see deploy/AWS.md)."; exit 1; }
[ -f models/siamese_unet.pt ] || { echo "Missing models/siamese_unet.pt (copy it with scp)."; exit 1; }
grep -q '^FLOODSIGHT_ACCESS_CODE=.\+' .env || { echo "Add FLOODSIGHT_ACCESS_CODE=<long code> to .env first."; exit 1; }

# 1. Docker
if ! command -v docker >/dev/null; then
  sudo apt-get update -y
  sudo apt-get install -y docker.io
  sudo systemctl enable --now docker
fi

# 2. Swap, so a small instance does not run out of memory on a large area (slower, never crashes)
if ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 6G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# 3. Build and start. --restart unless-stopped: it comes back by itself after the instance is started again.
sudo docker build -t floodsight .
sudo docker rm -f floodsight 2>/dev/null || true
sudo docker run -d --name floodsight --restart unless-stopped \
  -p 80:7860 --env-file .env -e FLOODSIGHT_PUBLIC=1 \
  -v floodsight_runs:/app/runs floodsight

echo
echo "FloodSight is starting. Open http://$(curl -s https://checkip.amazonaws.com) in a browser in about a minute."
echo "Logs: sudo docker logs -f floodsight"
