#!/bin/bash
# setup_venv.sh
# Jalankan SEKALI di host server sebelum docker compose up
# Host harus Ubuntu 22.04 + Python 3.10

set -e

VENV_DIR=${VENV_DIR:-./.venv}

# Cek OS
if ! grep -q "Ubuntu 22.04" /etc/os-release 2>/dev/null; then
    echo "⚠️  WARNING: Host bukan Ubuntu 22.04"
    echo "   Strategi mount venv mungkin tidak kompatibel dengan container"
    echo "   Lanjutkan? (y/N)"
    read -r confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || exit 1
fi

# Cek Python version
PY_VERSION=$(python3.12 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
echo "🐍 Python version: $PY_VERSION"

echo "🔧 Creating venv at $VENV_DIR..."
python3.12 -m venv $VENV_DIR

source $VENV_DIR/bin/activate
pip install --upgrade pip --quiet

echo "🔥 Installing PyTorch CPU-only..."
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.5.1+cpu" -q

echo "📦 Installing API dependencies..."
pip install --no-cache-dir -r requirements.api.txt -q

echo "🔍 Installing RAG dependencies..."
pip install --no-cache-dir \
    "faiss-cpu>=1.7.4" \
    "openai>=1.0.0" \
    "numpy>=1.24.0" -q

# sentence-transformers terakhir — torch CPU sudah ada, tidak pull CUDA
pip install --no-cache-dir "sentence-transformers>=2.2.2" -q

echo ""
echo "✅ Verifikasi:"
$VENV_DIR/bin/python -c "
import torch
print(f'  torch     : {torch.__version__} | CUDA: {torch.cuda.is_available()}')
import tensorflow as tf
print(f'  tensorflow: {tf.__version__}')
import faiss
print(f'  faiss     : OK')
import sentence_transformers
print(f'  sent-trans: {sentence_transformers.__version__}')
"

echo ""
echo "📁 Ukuran venv: $(du -sh $VENV_DIR | cut -f1)"
echo ""

# Simpan info versi untuk verifikasi di container
cat > $VENV_DIR/venv_info.txt << INFO
python_version=$PY_VERSION
os=Ubuntu 22.04
created=$(date)
torch=2.5.1+cpu
INFO

echo "🚀 Setup selesai! Jalankan: docker compose up -d"
