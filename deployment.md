# Deployment Guide — Systemd
# VPS: Ubuntu 22.04 | Python 3.10.12
# Sekarang: Polling | Nanti: Webhook setelah domain siap

## TAHAP 1 — Keamanan Dasar

```bash
apt update && apt upgrade -y
apt install -y curl git ufw fail2ban nginx python3.10-venv python3-pip

adduser deploy
usermod -aG sudo deploy

ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 80
ufw allow 443
ufw enable

systemctl enable fail2ban && systemctl start fail2ban
```

## TAHAP 2 — Setup Project

```bash
su - deploy
cd /home/deploy
git clone <repo_url> chatbot
cd chatbot

# Setup venv — torch CPU wajib duluan
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu"
pip install --no-cache-dir -r requirements.api.txt
pip install --no-cache-dir "faiss-cpu>=1.7.4" "openai>=1.0.0" "numpy>=1.24.0"
pip install --no-cache-dir "sentence-transformers>=2.2.2"
pip install --no-cache-dir -r requirements.bot.txt
deactivate

mkdir -p logs data/dataset data/lstm_models data/bert_model data/rag
```

### .env (isi sesuai kondisi)
```env
DATASET_PATH=/home/deploy/chatbot/data/dataset/dataset_training.csv
LSTM_MODEL_PATH=/home/deploy/chatbot/data/lstm_models/chatbot_model.h5
LSTM_TOKENIZER_PATH=/home/deploy/chatbot/data/lstm_models/tokenizer.pkl
LSTM_LABEL_ENCODER_PATH=/home/deploy/chatbot/data/lstm_models/label_encoder.pkl
BERT_MODEL_PATH=/home/deploy/chatbot/data/bert_model
USE_RAG=true
FAISS_INDEX_PATH=/home/deploy/chatbot/data/rag/faiss.index
FAISS_METADATA_PATH=/home/deploy/chatbot/data/rag/metadata.pkl
SIMILARITY_THRESHOLD=0.3
TOP_K_DOCUMENTS=5
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-xxxxxxxxxx
LLM_MODEL=google/gemini-2.0-flash-exp:free
TELEGRAM_BOT_TOKEN=isi_token_bot
WEBHOOK_URL=
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=False
```

## TAHAP 3 — Upload Model dari Local

```bash
# Jalankan dari local machine
scp -r ./data/lstm_models deploy@IP_VPS:/home/deploy/chatbot/data/
scp -r ./data/bert_model  deploy@IP_VPS:/home/deploy/chatbot/data/
scp -r ./data/dataset     deploy@IP_VPS:/home/deploy/chatbot/data/
scp -r ./data/rag         deploy@IP_VPS:/home/deploy/chatbot/data/
```

## TAHAP 4 — Systemd Services

### /etc/systemd/system/fira-bot-api.service
```ini
[Unit]
Description=Fira Bot API
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/home/deploy/chatbot
EnvironmentFile=/home/deploy/chatbot/.env
ExecStart=/home/deploy/chatbot/venv/bin/python -m uvicorn app:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fira-bot-api
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### /etc/systemd/system/fira-bot-telegram.service
```ini
[Unit]
Description=Fira Bot Telegram
After=network.target fira-bot-api.service
Requires=fira-bot-api.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/home/deploy/chatbot
EnvironmentFile=/home/deploy/chatbot/.env
Environment=API_BASE_URL=http://127.0.0.1:8000
ExecStartPre=/bin/sleep 15
ExecStart=/home/deploy/chatbot/venv/bin/python interfaces/telegram_bot.py
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=fira-bot-telegram

[Install]
WantedBy=multi-user.target
```

### Aktifkan
```bash
sudo systemctl daemon-reload
sudo systemctl enable fira-bot-api fira-bot-telegram
sudo systemctl start fira-bot-api
sleep 30
sudo systemctl start fira-bot-telegram

# Verifikasi
curl http://127.0.0.1:8000/health
journalctl -u fira-bot-api -f
journalctl -u fira-bot-telegram -f
```

## TAHAP 5 — Script Update

### /home/deploy/update.sh
```bash
#!/bin/bash
set -e
cd /home/deploy/chatbot
echo "📥 Pull latest..."
git pull
echo "📦 Update deps..."
source venv/bin/activate
pip install --no-cache-dir -r requirements.api.txt -q
deactivate
echo "🔄 Restart services..."
sudo systemctl restart fira-bot-api
sleep 20
sudo systemctl restart fira-bot-telegram
echo "✅ Done: $(date)"
```

```bash
chmod +x /home/deploy/update.sh

# Izinkan deploy restart tanpa password
sudo visudo
# Tambahkan:
# deploy ALL=(ALL) NOPASSWD: /bin/systemctl restart fira-bot-api, /bin/systemctl restart fira-bot-telegram, /bin/systemctl status fira-bot-api, /bin/systemctl status fira-bot-telegram
```

## TAHAP 6 — Migrasi Webhook (Setelah Domain Siap)

### Install SSL
```bash
apt install -y certbot python3-certbot-nginx
certbot --nginx -d domain.kamu.com
```

### /etc/nginx/sites-available/fira-bot
```nginx
server {
    listen 80;
    server_name domain.kamu.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name domain.kamu.com;

    ssl_certificate     /etc/letsencrypt/live/domain.kamu.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/domain.kamu.com/privkey.pem;

    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;

    limit_req_zone $binary_remote_addr zone=webhook:10m rate=30r/m;

    location /webhook {
        limit_req zone=webhook burst=10 nodelay;
        proxy_pass http://127.0.0.1:8000/webhook;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }

    location / {
        return 403;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/fira-bot /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
```

### Aktifkan webhook — cukup update .env
```env
WEBHOOK_URL=https://domain.kamu.com
```

```bash
sudo systemctl restart fira-bot-telegram
```

## Perintah Sehari-hari

```bash
# Status
sudo systemctl status fira-bot-api
sudo systemctl status fira-bot-telegram

# Log live
journalctl -u fira-bot-api -f
journalctl -u fira-bot-telegram -f
journalctl -u fira-bot-api -u fira-bot-telegram -f

# Restart
sudo systemctl restart fira-bot-api
sudo systemctl restart fira-bot-telegram

# Health check
curl http://127.0.0.1:8000/health

# Update
/home/deploy/update.sh
```

## Checklist

```
[ ] ufw aktif, port 8000 tidak terbuka publik
[ ] fira-bot-api health check OK
[ ] fira-bot-telegram bot merespons di Telegram
[ ] Auto-start enabled
[ ] .env tidak di-commit ke git
--- setelah domain ---
[ ] SSL valid
[ ] Nginx config OK
[ ] Bot merespons via webhook
```