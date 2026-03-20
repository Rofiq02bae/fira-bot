# 📘 Fira Bot — Command Reference

> Panduan lengkap command untuk development, deployment, dan maintenance Fira Bot.
> Target environment: Cloud VM (AWS/GCP/Azure) dengan Ubuntu/Debian.

---

## 📑 Daftar Isi

1. [Setup Awal](#1-setup-awal)
2. [Dataset & Preprocessing](#2-dataset--preprocessing)
3. [Docker — Build & Run](#3-docker--build--run)
4. [Docker — Monitoring & Debug](#4-docker--monitoring--debug)
5. [Push ke Docker Hub](#5-push-ke-docker-hub)
6. [Git Workflow](#6-git-workflow)
7. [Maintenance](#7-maintenance)

---

## 1. Setup Awal

### Jalankan installer otomatis
Script `install.sh` akan menginstall semua dependency yang dibutuhkan (Docker, Python 3.11, Git).

```bash
chmod +x install.sh
sudo ./install.sh
```

### Verifikasi instalasi
```bash
docker --version
docker compose version
python3.11 --version
git --version
```

### Clone repository
```bash
git clone https://github.com/rofiq02bae/fira-bot.git
cd fira-bot
git checkout build
```

### Setup file `.env`
```bash
cp .env.example .env
nano .env
```

Variabel penting yang harus diisi:
```env
# Telegram
TELEGRAM_BOT_TOKEN=your_token_here

# Path model & data
DATA_PATH=./data
HF_CACHE_PATH=/home/ubuntu/.cache/huggingface

# Docker Hub
DOCKER_USERNAME=your_dockerhub_username
DOCKER_PAT=your_personal_access_token

# LLM (opsional)
LLM_PROVIDER=openrouter
LLM_API_KEY=your_api_key
LLM_MODEL=x-ai/grok-4-beta
```

---

## 2. Dataset & Preprocessing

### Expand prosedur dataset (path default)
Menggunakan path input/output yang sudah dikonfigurasi di script.

```bash
python3 data/dataset/expand_prosedur.py
```

### Expand prosedur dataset (path custom)
Gunakan flag `--input` dan `--output` untuk menentukan path sendiri.

```bash
python3 data/dataset/expand_prosedur.py \
  --input /path/ke/input.csv \
  --output /path/ke/output.csv
```

**Contoh:**
```bash
python3 data/dataset/expand_prosedur.py \
  --input data/dataset/data_mentah.csv \
  --output data/dataset/dataset_training.csv
```

---

## 3. Docker — Build & Run

### Build semua image (pertama kali atau setelah perubahan kode)
```bash
docker compose build --no-cache
```

### Build image API saja dengan log detail
Berguna untuk debug jika build gagal di layer tertentu.

```bash
docker compose build --no-cache --progress=plain api 2>&1 | tee build.log
```

### Jalankan semua service
```bash
docker compose up -d
```

### Build ulang sekaligus jalankan
```bash
docker compose up -d --build
```

### Rebuild total dari nol
Gunakan ini jika ada dependency yang berubah drastis atau build stuck.

```bash
docker system prune -f
docker compose down --rmi all
docker compose build --no-cache
docker compose up -d
```

### Hentikan semua service
```bash
docker compose down
```

### Restart service tertentu
```bash
docker compose restart api
docker compose restart bot
```

---

## 4. Docker — Monitoring & Debug

### Monitor startup API (tunggu model selesai loading ~60-90 detik)
```bash
docker logs -f fira-bot-api
```

### Lihat logs bot Telegram
```bash
docker logs -f fira-bot-telegram
```

### Cek status semua container
```bash
docker compose ps
```

### Validasi health check API
```bash
curl http://localhost:8000/health
```

### Masuk ke dalam container untuk debug
```bash
docker exec -it fira-bot-api bash
docker exec -it fira-bot-telegram bash
```

### Lihat penggunaan resource real-time
```bash
docker stats
```

---

## 5. Push ke Docker Hub

### Jalankan script push otomatis
Script membaca `DOCKER_USERNAME` dan `DOCKER_PAT` dari `.env`, lalu login, tag, push, dan logout otomatis.

```bash
chmod +x push_to_dockerhub.sh
./push_to_dockerhub.sh
```

### Manual dengan tag versi spesifik
```bash
export $(grep -E '^(DOCKER_PAT|DOCKER_USERNAME)' .env | xargs)
echo "$DOCKER_PAT" | docker login -u "$DOCKER_USERNAME" --password-stdin

docker tag fira-bot-api:latest $DOCKER_USERNAME/fira-bot-api:v3.0.0
docker tag fira-bot-telegram:latest $DOCKER_USERNAME/fira-bot-telegram:v3.0.0

docker push $DOCKER_USERNAME/fira-bot-api:v3.0.0
docker push $DOCKER_USERNAME/fira-bot-telegram:v3.0.0

docker logout
```

---

## 6. Git Workflow

### Pull branch build dengan rebase (recommended)
```bash
git pull --rebase origin build
```

### Push ke branch build
```bash
git push origin build:build
```

### Jika divergent branch
```bash
# Rebase — history bersih
git pull --rebase origin build

# Setelah resolve conflict
git add <file-conflict>
git rebase --continue

# Batalkan rebase
git rebase --abort
```

### Cek file besar di git history
```bash
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  grep blob | sort -k3 -rn | head -20 | \
  awk '{printf "%.2f MB\t%s\n", $3/1048576, $4}'
```

### Hapus file besar dari git history
Gunakan jika file model/zip tidak sengaja ter-commit.

```bash
pip install git-filter-repo

git filter-repo --path-glob '*.zip' --invert-paths --force

echo -e "*.zip\n*.h5\n*.bin\ndata/bert_model/\ndata/lstm_models/" >> .gitignore
git add .gitignore
git commit -m "chore: exclude large files"
git push origin build --force
```

### Naikkan buffer HTTP untuk push repo besar
```bash
git config http.postBuffer 524288000
```

---

## 7. Maintenance

### Bersihkan Docker image tidak terpakai
```bash
docker image prune -f
```

### Bersihkan semua resource Docker tidak terpakai
```bash
docker system prune -af
```

### Lihat ukuran image
```bash
docker images | grep fira-bot
```

---

> **Last Updated:** March 2026
> **Project:** Fira Bot v3.0.0 — Hybrid NLU (LSTM + IndoBERT + RAG)
