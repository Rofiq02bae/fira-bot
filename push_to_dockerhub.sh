#!/bin/bash

set -e

# ── Load .env ──────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "❌ File .env tidak ditemukan. Jalankan script ini dari direktori proyek."
  exit 1
fi

export $(grep -E '^(DOCKER_PAT|DOCKER_USERNAME)' .env | xargs)

if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PAT" ]; then
  echo "❌ DOCKER_USERNAME atau DOCKER_PAT tidak ditemukan di .env"
  exit 1
fi

echo "👤 Username  : $DOCKER_USERNAME"
echo "🐳 Image API : $DOCKER_USERNAME/fira-bot-api:latest"
echo "🤖 Image Bot : $DOCKER_USERNAME/fira-bot-telegram:latest"
echo ""

# ── Login ──────────────────────────────────────────────────────────────────
echo "🔐 Login ke Docker Hub..."
echo "$DOCKER_PAT" | docker login -u "$DOCKER_USERNAME" --password-stdin
echo ""

# ── Tag ────────────────────────────────────────────────────────────────────
echo "🏷️  Tagging images..."
docker tag fira-bot-api:latest "$DOCKER_USERNAME/fira-bot-api:latest"
docker tag fira-bot-telegram:latest "$DOCKER_USERNAME/fira-bot-telegram:latest"
echo ""

# ── Push ───────────────────────────────────────────────────────────────────
echo "📤 Push fira-bot-api..."
docker push "$DOCKER_USERNAME/fira-bot-api:latest"
echo ""

echo "📤 Push fira-bot-telegram..."
docker push "$DOCKER_USERNAME/fira-bot-telegram:latest"
echo ""

# ── Done ───────────────────────────────────────────────────────────────────
echo "✅ Selesai! Images berhasil di-push ke Docker Hub."
echo "   🔗 https://hub.docker.com/u/$DOCKER_USERNAME"

docker logout