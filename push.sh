#!/bin/bash

# Konfigurasi
DOCKER_USER="rofiq02bae" # <--- GANTI INI
IMAGE_NAME="fira-bot"
TAG="latest"

echo "--- Melakukan Login ---"
docker login

echo "--- Memulai Build untuk Platform Linux AMD64 ---"
# Menggunakan buildx untuk memastikan kompatibilitas dengan VM Linux
docker build --platform linux/amd64 -t $DOCKER_USER/$IMAGE_NAME:$TAG .

echo "--- Mengunggah ke Docker Hub ---"
docker push $DOCKER_USER/$IMAGE_NAME:$TAG

echo "--- SELESAI! ---"
echo "Di VM kamu, jalankan: docker pull $DOCKER_USER/$IMAGE_NAME:$TAG"
