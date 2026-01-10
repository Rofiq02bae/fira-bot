#!/usr/bin/env bash

set -euo pipefail

# Konfigurasi
DOCKER_USER="rofiq02bae" # <--- GANTI INI
IMAGE_NAME="fira-bot"
TAG="latest"

IMAGE_REF="$DOCKER_USER/$IMAGE_NAME:$TAG"

echo "--- Cek Docker daemon & context ---"

try_docker_info() {
	docker info >/dev/null 2>&1
}

switch_context_if_needed() {
	# If docker is already reachable, do nothing.
	if try_docker_info; then
		return 0
	fi

	# Try common Docker Desktop contexts.
	for ctx in default desktop-linux; do
		if docker context inspect "$ctx" >/dev/null 2>&1; then
			echo "Mencoba docker context: $ctx"
			docker context use "$ctx" >/dev/null 2>&1 || true
			if try_docker_info; then
				echo "Docker OK pada context: $ctx"
				return 0
			fi
		fi
	done

	echo "❌ Docker daemon tidak terhubung. Pastikan Docker Desktop berjalan dan WSL2 backend aktif." >&2
	echo "   Coba jalankan di PowerShell: docker context ls ; docker info" >&2
	return 1
}

switch_context_if_needed

echo "--- Melakukan Login ---"
docker login

echo "--- Memulai Build untuk Platform Linux AMD64 ---"
# Menggunakan buildx (lebih stabil untuk cross-platform). Jika buildx tidak tersedia, fallback ke docker build.
if docker buildx version >/dev/null 2>&1; then
	# Prefer the 'docker' driver (no separate buildkit container). The docker-container driver
	# can fail on some Windows/WSL setups with: "bind source path does not exist: /usr/lib/wsl".
	if docker buildx inspect fira-builder >/dev/null 2>&1; then
		docker buildx rm fira-builder >/dev/null 2>&1 || true
	fi

	if docker buildx create --name fira-builder --use --driver docker >/dev/null 2>&1; then
		docker buildx build --platform linux/amd64 -t "$IMAGE_REF" --load .
	else
		echo "Buildx builder gagal dibuat; fallback ke docker build biasa"
		docker build --platform linux/amd64 -t "$IMAGE_REF" .
	fi
else
	docker build --platform linux/amd64 -t "$IMAGE_REF" .
fi

echo "--- Mengunggah ke Docker Hub ---"
docker push "$IMAGE_REF"

echo "--- SELESAI! ---"
echo "Di VM kamu, jalankan: docker pull $IMAGE_REF"
