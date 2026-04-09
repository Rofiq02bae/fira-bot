#!/bin/bash
set -e
cd /opt/fira-bot

git pull origin main  # ← explicit dari main
source venv/bin/activate
pip install --no-cache-dir -r requirements.api.txt -q
deactivate

sudo systemctl restart fira-bot-api
sleep 20
sudo systemctl restart fira-bot-telegram
echo "✅ Done: $(date)"