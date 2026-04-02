#!/bin/bash

# ============================================================
#  Fira Bot — Installer untuk Cloud VM (AWS / GCP / Azure)
#  Ubuntu 20.04 / 22.04 / 24.04 LTS
#  Installs: Docker, Docker Compose, Python 3.11, Git
# ============================================================

set -e

# ── Warna output ─────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log()     { echo -e "${BLUE}[INFO]${NC}  $1"; }
success() { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ── Cek root ─────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
  error "Jalankan script ini sebagai root: sudo ./install.sh"
fi

# ── Deteksi OS ───────────────────────────────────────────────
if [ -f /etc/os-release ]; then
  . /etc/os-release
  OS=$ID
  OS_VERSION=$VERSION_ID
else
  error "Tidak dapat mendeteksi OS. Script ini hanya mendukung Ubuntu/Debian."
fi

if [[ "$OS" != "ubuntu" && "$OS" != "debian" ]]; then
  error "OS tidak didukung: $OS. Gunakan Ubuntu atau Debian."
fi

log "Terdeteksi OS: $OS $OS_VERSION"
echo ""

# ════════════════════════════════════════════════════════════
#  STEP 1 — Update sistem
# ════════════════════════════════════════════════════════════
log "Step 1/4 — Update & upgrade sistem..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
  curl \
  wget \
  gnupg \
  ca-certificates \
  lsb-release \
  software-properties-common \
  apt-transport-https \
  unzip \
  htop
success "Sistem berhasil diupdate."
echo ""

# ════════════════════════════════════════════════════════════
#  STEP 2 — Install Git
# ════════════════════════════════════════════════════════════
log "Step 2/4 — Install Git..."
if command -v git &>/dev/null; then
  warn "Git sudah terinstall: $(git --version)"
else
  apt-get install -y -qq git
  success "Git berhasil diinstall: $(git --version)"
fi
echo ""

# ════════════════════════════════════════════════════════════
#  STEP 3 — Install Python 3.11
# ════════════════════════════════════════════════════════════
log "Step 3/4 — Install Python 3.11..."
if command -v python3.11 &>/dev/null; then
  warn "Python 3.11 sudah terinstall: $(python3.11 --version)"
else
  # Tambah deadsnakes PPA untuk Ubuntu
  if [[ "$OS" == "ubuntu" ]]; then
    add-apt-repository -y ppa:deadsnakes/ppa -q
    apt-get update -qq
  fi

  apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip

  # Set python3.11 sebagai default python3
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 || true

  success "Python berhasil diinstall: $(python3.11 --version)"
fi

# Upgrade pip
python3.11 -m pip install --upgrade pip -q
success "pip diupgrade: $(python3.11 -m pip --version)"
echo ""

# ════════════════════════════════════════════════════════════
#  STEP 4 — Install Docker & Docker Compose
# ════════════════════════════════════════════════════════════
log "Step 4/4 — Install Docker & Docker Compose..."

if command -v docker &>/dev/null; then
  warn "Docker sudah terinstall: $(docker --version)"
else
  # Hapus versi lama kalau ada
  apt-get remove -y -qq docker docker-engine docker.io containerd runc 2>/dev/null || true

  # Tambah Docker GPG key & repo resmi
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/$OS/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg

  echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/$OS \
    $(lsb_release -cs) stable" \
    | tee /etc/apt/sources.list.d/docker.list > /dev/null

  apt-get update -qq
  apt-get install -y -qq \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

  # Enable & start Docker
  systemctl enable docker
  systemctl start docker

  success "Docker berhasil diinstall: $(docker --version)"
fi

# Verifikasi Docker Compose
if docker compose version &>/dev/null; then
  success "Docker Compose tersedia: $(docker compose version)"
else
  error "Docker Compose tidak ditemukan. Cek instalasi Docker."
fi

# Tambah user saat ini ke grup docker (supaya tidak perlu sudo)
CURRENT_USER="${SUDO_USER:-$USER}"
if [ -n "$CURRENT_USER" ] && [ "$CURRENT_USER" != "root" ]; then
  usermod -aG docker "$CURRENT_USER"
  warn "User '$CURRENT_USER' ditambahkan ke grup docker."
  warn "Logout & login kembali agar perubahan grup berlaku."
fi
echo ""

# ════════════════════════════════════════════════════════════
#  SELESAI
# ════════════════════════════════════════════════════════════
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ✅ Instalasi selesai!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Git        : $(git --version)"
echo "  Python     : $(python3.11 --version)"
echo "  Docker     : $(docker --version)"
echo "  Compose    : $(docker compose version)"
echo ""
echo -e "${YELLOW}  Langkah selanjutnya:${NC}"
echo "  1. Logout & login kembali (agar docker tanpa sudo)"
echo "  2. cd fira-bot"
echo "  3. cp .env.example .env && nano .env"
echo "  4. docker compose build --no-cache"
echo "  5. docker compose up -d"
echo ""
