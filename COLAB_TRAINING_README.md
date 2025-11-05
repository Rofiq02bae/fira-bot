# 🚀 Training BERT Model dengan Google Colab

Panduan lengkap untuk melatih model BERT menggunakan GPU gratis dari Google Colab.

## 📋 Mengapa Menggunakan Google Colab?

Melatih model BERT memerlukan:
- **GPU yang lumayan besar** (minimum 8GB VRAM)
- **Waktu training yang lama** (30-90 menit tergantung dataset)
- **Resource komputasi tinggi**

Jika Anda **tidak memiliki GPU**, gunakan **Google Colab** dengan GPU **Tesla T4 gratis**!

---

## 🎯 Langkah-langkah Training

### **a) Buat Notebook Baru**

1. Buka [Google Colab](https://colab.research.google.com/)
2. Klik **File > New notebook**
3. Atau upload file `BERT_TRAINING_COLAB.ipynb` dari repository

### **b) Klik 'Runtime' > 'Change runtime type' > Ubah ke GPU T4**

1. Klik menu **Runtime** di atas
2. Pilih **Change runtime type**
3. Di dropdown **Hardware accelerator**, pilih **GPU**
4. Pilih GPU Type: **T4** (gratis)
5. Klik **Save**

![GPU Setup](https://i.imgur.com/example.png)

✅ **Verifikasi GPU:** Jalankan `!nvidia-smi` untuk memastikan GPU aktif

### **c) Copy paste isi file `collab_requirements.txt` ke cell pertama**

1. Buka file [`collab_requirements.txt`](collab_requirements.txt)
2. Copy semua isi file
3. Paste ke **cell pertama** di Colab notebook
4. Atau gunakan code ini:

```python
# ==================== INSTALL DEPENDENCIES ====================
print("🔄 Installing dependencies...")
print("⏱️  Estimasi waktu: 2-3 menit\n")

!pip install -q transformers datasets accelerate torch pandas
!pip install -q scikit-learn numpy tqdm

print("\n" + "="*60)
print("🚀 Colab environment ready for BERT training!")
print("✅ All dependencies installed successfully!")
print("="*60)
print("\n⚠️  PENTING: Restart Session sekarang!")
print("    Klik: Runtime > Restart session")
print("="*60)
```

### **d) Running cell pertama, tunggu semua dependensi selesai di install, lalu restart session**

1. Klik tombol **Play (▶️)** di cell pertama atau tekan **Shift+Enter**
2. Tunggu hingga semua package terinstall (~2-3 menit)
3. Setelah muncul pesan "Restart Session sekarang!"
4. Klik **Runtime > Restart session**
5. Konfirmasi dengan klik **Restart**

⚠️ **PENTING:** Restart session wajib dilakukan untuk menerapkan perubahan library!

### **e) Copy paste file `bert_training.py` atau `bert_train.ipynb` ke cell berikutnya**

**Option 1: Menggunakan Notebook Lengkap (Recommended)**

Upload file `BERT_TRAINING_COLAB.ipynb` yang sudah disediakan - sudah include semua code!

**Option 2: Manual Copy-Paste**

1. Buka file [`training/bert_training.py`](training/bert_training.py)
2. Copy seluruh code
3. Paste ke **cell kedua** (setelah restart session)

**Option 3: Clone Repository**

```python
# Clone repository
!git clone https://github.com/Rofiq02bae/fira-bot.git
%cd fira-bot

# Import training script
from training import bert_training
```

### **f) Pilih model yang akan digunakan, ada 3 pilihan**

Ubah variable `MODEL_CHOICE` di code:

```python
# 🎯 PILIH MODEL (1, 2, atau 3)
MODEL_CHOICE = 1  # Default: IndoBERT Base
```

**Pilihan Model:**

| Model | Nama | Ukuran | Kecepatan | Akurasi | Waktu Training |
|-------|------|--------|-----------|---------|----------------|
| **1** | IndoBERT Base | ~500MB | Sedang | **Tinggi** ⭐ | 6-7 min |
| **2** | IndoBERT Lite | ~200MB | **Cepat** ⚡ | Sedang | 5-6 min |
| **3** | bert-base-indonesian-522M | ~700MB | Lambat | Tinggi | 7-8 min |

**Rekomendasi:**
- ✅ **Model 1 (IndoBERT Base)** - Untuk production/deployment
- ⚡ **Model 2 (IndoBERT Lite)** - Untuk testing/cepat
- 🌍 **Model 3 (bert-base-indonesian-522M)** - Jika perlu bahasa kompleks

### **g) Import `dataset_training.csv` ke Google Colab**

**Method 1: Upload Manual (Recommended)**

```python
from google.colab import files

# Upload dataset
print("📂 Upload file dataset_training.csv")
uploaded = files.upload()

# Verify
import pandas as pd
df = pd.read_csv('dataset_training.csv')
print(f"✅ Dataset loaded: {len(df)} rows")
```

**Method 2: From Google Drive**

```python
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Copy dataset
!cp /content/drive/MyDrive/dataset_training.csv ./
```

**Method 3: Clone dari GitHub**

Jika dataset sudah di-commit ke repository:

```python
!git clone https://github.com/Rofiq02bae/fira-bot.git
%cd fira-bot
# Dataset sudah ada di: data/dataset/dataset_training.csv
```

### **h) Running cell kedua dan tunggu sekitar 7 menit**

1. Klik **Play (▶️)** di cell training atau tekan **Shift+Enter**
2. Training akan dimulai dengan progress bar


**Proses Training:**
```
📂 Loading dataset...
✅ Dataset loaded: 2847 samples
   Unique intents: 156
   Training samples: 2277
   Validation samples: 570

🔄 Loading model: indobenchmark/indobert-base-p1
✅ Model loaded successfully!

🔄 Tokenizing dataset...
✅ Tokenization complete!

🚀 TRAINING STARTED!
⏱️  Estimasi waktu: ~30-60 min

Epoch 1/3:
  Training: 100%|██████████| 143/143 [05:23<00:00]
  Loss: 0.3421, Accuracy: 0.8956
  
Epoch 2/3:
  Training: 100%|██████████| 143/143 [05:18<00:00]
  Loss: 0.1842, Accuracy: 0.9412

Epoch 3/3:
  Training: 100%|██████████| 143/143 [05:15<00:00]
  Loss: 0.0921, Accuracy: 0.9756

✅ TRAINING COMPLETED!
⏱️  Total time: 27.43 minutes
```

💡 **Tips:**
- Jangan tutup tab browser selama training
- Bisa minimize tab tapi jangan close
- Monitor progress di output cell

### **i) Running cell ketiga untuk download model**

Setelah training selesai:

```python
# ==================== DOWNLOAD MODEL ====================
from google.colab import files
import shutil

print("📦 Preparing model for download...")

# Zip the model
shutil.make_archive('bert_model', 'zip', './bert_model')

# Download
files.download('bert_model.zip')

print("✅ MODEL DOWNLOADED!")
```

File `bert_model.zip` akan otomatis terdownload (~500-700 MB)

⏱️ Waktu download: **2-5 menit** (tergantung koneksi internet)

### **j) Ekstrak model ke `data/bert_model`**

**Di komputer local Anda:**

1. **Extract ZIP file:**
   ```bash
   # Windows
   Extract bert_model.zip menggunakan Windows Explorer
   # Atau dengan PowerShell
   Expand-Archive bert_model.zip -DestinationPath bert_model
   ```

2. **Copy ke project directory:**
   ```bash

   # Copy semua isi folder
   Copy-Item -Path "bert_model\*" -Destination "data\bert_model\" -Recurse
   ```

3. **Verify struktur folder:**
   ```
   data/bert_model/
   ├── config.json
   ├── model.safetensors
   ├── tokenizer.json
   ├── tokenizer_config.json
   ├── special_tokens_map.json
   ├── vocab.txt
   └── label_encoder.pkl
   ```

4. **Test model berhasil di-load:**
   ```bash
   python -c "from main import get_hybrid_nlu; service = get_hybrid_nlu(); print('✅ Model loaded!')"
   ```

### **k) Jalankan server**

Setelah model berhasil di-extract dan di-copy:

```bash
# Aktifkan virtual environment (jika ada)
.\.venv\Scripts\Activate.ps1

# Jalankan FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Output yang diharapkan:**
```
🚀 Starting Hybrid Chatbot API (Modular Version)...
📂 Checking model files...
✅ lstm_model: data/lstm_models/chatbot_model.h5 - EXISTS
✅ bert_folder: data/bert_model - EXISTS
✅ dataset: data/dataset/dataset_training.csv - EXISTS

🔄 Initializing modular hybrid service...
✅ LSTM Model loaded successfully!
✅ Fine-tuned BERT loaded successfully!
✅ Modular hybrid service initialized!

INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Test API:**

1. Buka browser: http://localhost:8000/docs
2. Test endpoint `/api/chat`:
   ```json
   {
     "text": "hai"
   }
   ```

3. Response:
   ```json
   {
     "original_text": "hai",
     "predicted_intent": "salam",
     "confidence": 0.956,
     "response": "halo, ada yang bisa saya bantu",
     "method_used": "hybrid_fusion"
   }
   ```

✅ **SELESAI!** Model BERT Anda sudah berjalan di server local!

---

## 📊 Monitoring Training

### GPU Usage

Monitor GPU selama training:

```python
# Run di cell terpisah
!nvidia-smi -l 2  # Update setiap 2 detik
```

Stop dengan: Runtime > Interrupt execution

### Training Metrics

Yang perlu diperhatikan:

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| **Loss** | Menurun setiap epoch | Stagnan | Naik |
| **Accuracy** | Naik setiap epoch | Stagnan | Turun |
| **Val Loss** | Similar to Train Loss | Sedikit lebih tinggi | Jauh lebih tinggi (overfit) |

**Contoh Good Training:**
```
Epoch 1: Loss=0.45, Acc=0.82, Val_Loss=0.48
Epoch 2: Loss=0.28, Acc=0.89, Val_Loss=0.31
Epoch 3: Loss=0.18, Acc=0.93, Val_Loss=0.22
```

✅ Loss menurun, Accuracy naik, Validation loss mengikuti

---

## 🐛 Troubleshooting

### ❌ GPU Not Available

**Problem:** GPU tidak terdeteksi

**Solution:**
1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. Save
4. Runtime > Restart runtime

### ❌ CUDA Out of Memory

**Problem:** GPU memory habis

**Solution:**
```python
# Kurangi batch size
BATCH_SIZE = 8  # dari 16 ke 8
# Atau
BATCH_SIZE = 4  # jika masih error
```

### ❌ Session Disconnected

**Problem:** Colab disconnect saat training

**Solution:**
- Reconnect dan re-run cells
- Backup model ke Google Drive secara berkala
- Gunakan code untuk keep alive (lihat tips di bawah)

### ❌ Dataset Not Found

**Problem:** File dataset tidak ditemukan

**Solution:**
```python
# Upload ulang dataset
from google.colab import files
uploaded = files.upload()

# Verify
!ls -lh dataset_training.csv
```

### ❌ Download Failed

**Problem:** Download model gagal/incomplete

**Solution:**
```python
# Backup ke Google Drive dulu
from google.colab import drive
drive.mount('/content/drive')

!cp -r bert_model /content/drive/MyDrive/

# Download dari Drive nanti
```

---

## 💡 Tips & Tricks

### 1. Keep Colab Alive

Prevent auto-disconnect dengan JavaScript di browser console (F12):

```javascript
function ClickConnect(){
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(ClickConnect, 60000);  // Every 60 seconds
```

### 2. Auto-Backup ke Google Drive

Tambahkan di akhir training:

```python
from google.colab import drive
drive.mount('/content/drive')

# Backup otomatis
!cp -r bert_model /content/drive/MyDrive/fira-bot-backup/
print("✅ Backup complete!")
```

### 3. Monitor via Email

Get notifikasi saat training selesai:

```python
# Install
!pip install -q notify-send-py

# At end of training
from notify_send import notify
notify("Training Complete!", "BERT model training finished!")
```

### 4. Multiple Model Training

Train beberapa konfigurasi sekaligus:

```python
configs = [
    {'model': 1, 'epochs': 3, 'batch': 16},  # IndoBERT Base
    {'model': 2, 'epochs': 3, 'batch': 16},  # IndoBERT Lite
]

for config in configs:
    print(f"Training: {config}")
    # Run training with config
    # Save with different names
```

---

## 📈 Hasil Training yang Baik

### Model Size

File yang dihasilkan:

| File | Size | Description |
|------|------|-------------|
| `model.safetensors` | ~400-600 MB | Model weights |
| `config.json` | ~1 KB | Model configuration |
| `tokenizer.json` | ~500 KB | Tokenizer |
| `vocab.txt` | ~200 KB | Vocabulary |
| `label_encoder.pkl` | ~5-50 KB | Intent encoder |
| **Total** | **~500-700 MB** | Full model |

---

## ✅ Checklist

### Before Training

- [ ] Akun Google sudah login
- [ ] GPU T4 sudah aktif di Colab
- [ ] Dependencies sudah terinstall
- [ ] Session sudah di-restart
- [ ] Dataset CSV sudah di-upload
- [ ] Model choice sudah dipilih

### During Training

- [ ] Monitor training progress
- [ ] Check GPU usage (`!nvidia-smi`)
- [ ] Pastikan loss menurun
- [ ] Pastikan accuracy naik
- [ ] Jangan tutup tab browser

### After Training

- [ ] Model sudah di-download
- [ ] Ekstrak ZIP file
- [ ] Copy ke `data/bert_model/`
- [ ] Verify file structure
- [ ] Test model loading
- [ ] Run server dan test API
- [ ] Backup model (optional)

---

## 📚 Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [IndoBERT Model](https://huggingface.co/indobenchmark/indobert-base-p1)
- [File Lengkap: COLAB_GUIDE.md](COLAB_GUIDE.md)

---

## 🎉 Success!

Jika semua langkah berhasil:

✅ Model BERT sudah ter-training  
✅ Model sudah di-download  
✅ Model sudah di-copy ke project  
✅ Server berjalan dengan baik  
✅ API bisa memprediksi dengan akurat  

**Congratulations!** 🎊

Anda sekarang punya chatbot dengan:
- 🚀 LSTM (fast response)
- 🎯 BERT (high accuracy)
- 🤖 Hybrid (best of both worlds)

---

**Made with ❤️ for fira-bot project**

Last updated: November 2025 oleh Rofiq02bae 
please add star
