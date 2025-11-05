# 🚀 Google Colab Training Guide

Panduan lengkap untuk training BERT model menggunakan Google Colab GPU gratis.

## 📋 Prerequisites

- Akun Google (untuk akses Colab)
- Repository GitHub sudah ter-upload
- Dataset CSV siap (atau sudah ada di repo)
- Browser modern (Chrome/Firefox recommended)

## 🎯 Quick Start

### Option 1: Menggunakan Notebook (Recommended)

1. **Upload Notebook ke Colab**
   - Buka [Google Colab](https://colab.research.google.com/)
   - File > Upload notebook
   - Upload file `COLAB_TRAINING.ipynb` dari repository
   - Atau direct link: `https://colab.research.google.com/github/Rofiq02bae/fira-bot/blob/main/COLAB_TRAINING.ipynb`

2. **Enable GPU**
   - Runtime > Change runtime type
   - Hardware accelerator: **GPU** (T4)
   - Click **Save**

3. **Run All Cells**
   - Runtime > Run all
   - Atau run cell by cell (Ctrl+Enter / Shift+Enter)

4. **Download Model**
   - Setelah training selesai, model akan di-download otomatis
   - Extract `bert_model.zip` ke folder `data/bert_model/` di local

### Option 2: Manual Setup

Jika prefer manual setup, ikuti langkah di file `collab_requirements.txt`.

## 📊 Training Process

### Step-by-Step Workflow

```
1. Install Dependencies (2-3 menit)
   ↓
2. Verify GPU (< 1 menit)
   ↓
3. Clone Repository (1-2 menit)
   ↓
4. Check Dataset (< 1 menit)
   ↓
5. Train BERT (1-3 jam) ⏰
   ↓
6. Test Model (2-5 menit)
   ↓
7. Download Model (3-5 menit)
```

### Expected Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Setup Dependencies | 2-3 min | One-time installation |
| GPU Verification | < 1 min | Check GPU availability |
| Clone Repository | 1-2 min | Download from GitHub |
| Dataset Check | < 1 min | Verify data exists |
| **BERT Training** | **1-3 hours** | **Main process** |
| Model Testing | 2-5 min | Test predictions |
| Model Download | 3-5 min | Download to local |
| **Total** | **~1.5-3.5 hours** | Mostly training time |

## 🔥 GPU Information

### Free Tier GPU Options

Google Colab Free provides:
- **GPU Type:** NVIDIA Tesla T4 (most common)
- **GPU Memory:** 15-16 GB VRAM
- **System RAM:** 12-13 GB
- **Disk Space:** ~100 GB

### GPU Limits (Free Tier)

- **Session Duration:** Max 12 hours continuous
- **Daily Limit:** ~12-24 GPU hours per day (varies)
- **Idle Timeout:** 90 minutes (if no code running)
- **Background Timeout:** 12 hours max

### Tips untuk Maximize GPU Time

1. **Enable GPU early** - Jangan lupa enable sebelum training
2. **Close unused tabs** - Fokus hanya di Colab tab
3. **Monitor progress** - Check logs secara berkala
4. **Save checkpoints** - Backup ke Google Drive
5. **Use Colab Pro** - Jika perlu unlimited (consider upgrade)

## 📦 Dataset Requirements

### Dataset Format

File: `data/dataset/dataset_training.csv`

Required columns:
```csv
patterns,tag,responses,response_type
"Apa itu KTP?",ktp_info,"KTP adalah...",static
"Cara buat KTP?",ktp_prosedur,"Prosedur pembuatan KTP...",static
```

### Dataset Size Recommendations

| Dataset Size | Training Time | GPU Recommended |
|--------------|---------------|-----------------|
| < 1,000 rows | 30-60 min | T4 (Free) |
| 1,000-5,000 rows | 1-2 hours | T4 (Free) |
| 5,000-10,000 rows | 2-3 hours | T4 Pro |
| > 10,000 rows | 3+ hours | A100 (Pro+) |

### If Dataset Not in Repo

**Option 1: Upload manually in Colab**
```python
from google.colab import files
uploaded = files.upload()  # Upload dataset_training.csv
```

**Option 2: Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
# Then copy from Drive to Colab
!cp /content/drive/MyDrive/dataset_training.csv data/dataset/
```

**Option 3: Add to GitHub repo** (Recommended)
- Commit dataset ke repository
- Will auto-download saat clone

## ⚙️ Configuration Options

### Default Training Config

Located in: `training/bert_training.py`

```python
config = {
    'model_name': 'indobenchmark/indobert-base-p1',
    'num_epochs': 3,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'max_length': 128,
    'warmup_steps': 500,
    'weight_decay': 0.01
}
```

### Modify for Your Needs

**For faster training (testing):**
```python
num_epochs = 1  # Quick test
batch_size = 32  # Larger batches if GPU allows
```

**For better accuracy:**
```python
num_epochs = 5  # More epochs
learning_rate = 1e-5  # Lower learning rate
batch_size = 8  # Smaller batches for stability
```

**If GPU memory error:**
```python
batch_size = 4  # Reduce batch size
max_length = 64  # Reduce sequence length
```

## 🐛 Troubleshooting

### Common Errors & Solutions

#### 1. GPU Not Available

**Error:**
```
⚠️ GPU not available! Training will be slow.
```

**Solution:**
1. Runtime > Change runtime type
2. Hardware accelerator: GPU
3. Click Save
4. Runtime > Restart runtime

---

#### 2. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Edit training/bert_training.py
batch_size = 4  # Reduce from 16 to 4
# Or
batch_size = 8  # Try 8 first
```

---

#### 3. Session Disconnected

**Error:**
```
Session crashed or disconnected
```

**Solution:**
- Reconnect to runtime
- Re-run all cells
- Enable checkpointing to Google Drive (see below)

---

#### 4. Dataset Not Found

**Error:**
```
FileNotFoundError: data/dataset/dataset_training.csv
```

**Solution:**
```python
# Upload dataset manually
from google.colab import files
uploaded = files.upload()

# Move to correct location
!mv dataset_training.csv data/dataset/
```

---

#### 5. Model Download Failed

**Error:**
```
Download failed or incomplete
```

**Solution:**
```python
# Try manual download
!zip -r bert_model.zip data/bert_model/
files.download('bert_model.zip')

# Or backup to Google Drive
!cp -r data/bert_model /content/drive/MyDrive/fira-bot-backup/
```

---

#### 6. Git Clone Failed

**Error:**
```
Repository not found or access denied
```

**Solution:**
```bash
# Ensure repository is public
# Or use correct URL:
!git clone https://github.com/Rofiq02bae/fira-bot.git

# If already cloned, pull updates:
%cd fira-bot
!git pull origin main
```

## 💾 Save Progress to Google Drive

### Setup Auto-Backup

Add this at the beginning of training:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create backup directory
!mkdir -p /content/drive/MyDrive/fira-bot-backup
```

### During Training

Modify `training/bert_training.py` to save checkpoints:

```python
# Add after each epoch
model.save_pretrained('/content/drive/MyDrive/fira-bot-backup/checkpoint-epoch-{epoch}')
```

### After Training

```python
# Backup final model
!cp -r data/bert_model /content/drive/MyDrive/fira-bot-backup/
print("✅ Model backed up to Google Drive!")
```

## 📊 Monitor Training

### Option 1: Watch Logs

Training akan menampilkan progress:
```
Epoch 1/3:
  Training: 100%|██████████| 50/50 [05:23<00:00,  6.47s/it]
  Loss: 0.3421, Accuracy: 0.8956
  
Epoch 2/3:
  Training: 100%|██████████| 50/50 [05:18<00:00,  6.37s/it]
  Loss: 0.1842, Accuracy: 0.9412
```

### Option 2: GPU Usage

Run in separate cell:
```python
# Monitor GPU (updates every 2 seconds)
!nvidia-smi -l 2
```

Stop dengan: Runtime > Interrupt execution

### Option 3: TensorBoard (if enabled)

```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

## 🎯 After Training

### 1. Download Model

```python
# Automatic download
files.download('bert_model.zip')
```

### 2. Extract & Copy

```bash
# On your local machine
unzip bert_model.zip
cp -r bert_model/* "d:/bot/New folder/data/bert_model/"
```

### 3. Test Locally

```bash
# Test import
python -c "from main import get_hybrid_nlu; print('✅ Model loaded')"

# Run API
uvicorn app:app --reload
```

### 4. Verify Model Works

```python
from main import get_hybrid_nlu

service = get_hybrid_nlu()
result = service.process_query("jam buka bappenda")
print(result)
```

## 📈 Performance Expectations

### Accuracy Targets

| Dataset Quality | Expected Accuracy |
|-----------------|-------------------|
| Well-balanced | 85-95% |
| Moderate | 75-85% |
| Imbalanced | 60-75% |

### Training Metrics

Good training shows:
- **Loss decreasing** each epoch
- **Accuracy increasing** each epoch
- **Validation similar** to training (not overfitting)

Example:
```
Epoch 1: Loss=0.45, Acc=0.82
Epoch 2: Loss=0.28, Acc=0.89
Epoch 3: Loss=0.18, Acc=0.93  ✅ Good!
```

## 🚀 Advanced Tips

### 1. Keep Colab Alive

Prevent auto-disconnect:

```javascript
// Paste in browser console (F12)
function ClickConnect(){
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(ClickConnect, 60000);  // Every 60 seconds
```

### 2. Parallel Training

Train multiple configurations:

```python
configs = [
    {'epochs': 3, 'batch_size': 16},
    {'epochs': 5, 'batch_size': 8},
]

for config in configs:
    print(f"Training with {config}")
    # train_model(config)
```

### 3. Experiment Tracking

Log all experiments:

```python
import json
from datetime import datetime

experiment = {
    'date': datetime.now().isoformat(),
    'config': config,
    'accuracy': final_accuracy,
    'loss': final_loss
}

with open('experiments.json', 'a') as f:
    f.write(json.dumps(experiment) + '\n')
```

### 4. Early Stopping

Save time if not improving:

```python
# In training loop
if current_loss < best_loss:
    best_loss = current_loss
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= 3:
        print("Early stopping!")
        break
```

## 💰 Colab Pro Comparison

| Feature | Free | Pro | Pro+ |
|---------|------|-----|------|
| Price | $0 | $10/mo | $50/mo |
| GPU Time | 12-24h/day | ~unlimited | ~unlimited |
| Session | 12h max | 24h max | 24h max |
| GPU Type | T4 | T4/P100/V100 | T4/V100/A100 |
| RAM | 12 GB | 32 GB | 52 GB |
| Priority | Low | Medium | High |

**Recommendation:**
- **Free:** Perfect untuk dataset kecil (<5k rows)
- **Pro:** Good untuk production training
- **Pro+:** Overkill unless dataset huge (>50k rows)

## 📝 Checklist

Before starting:
- [ ] Dataset CSV ready
- [ ] GitHub repo accessible
- [ ] Google account logged in
- [ ] GPU enabled in Colab
- [ ] Enough time (1-3 hours)

During training:
- [ ] Monitor GPU usage
- [ ] Check loss/accuracy
- [ ] Save checkpoints to Drive
- [ ] Keep session alive

After training:
- [ ] Download model
- [ ] Backup to Drive (optional)
- [ ] Test model locally
- [ ] Deploy to production

## 🆘 Support

Jika masih ada masalah:

1. **Check Logs** - Baca error message dengan teliti
2. **Google Error** - Search error di Google/Stack Overflow
3. **Restart Runtime** - Often fixes weird issues
4. **Re-run Cells** - Try running from scratch
5. **Ask Community** - Colab forum, Reddit, Discord

## 📚 Additional Resources

- [Google Colab Docs](https://colab.research.google.com/notebooks/intro.ipynb)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [IndoBERT Model](https://huggingface.co/indobenchmark/indobert-base-p1)
- [PyTorch CUDA Guide](https://pytorch.org/docs/stable/cuda.html)

## ✅ Success Criteria

Training berhasil jika:
- ✅ No errors during training
- ✅ Loss decreases consistently
- ✅ Accuracy > 80% (target tergantung dataset)
- ✅ Model files generated in `data/bert_model/`
- ✅ Model works in local testing
- ✅ Predictions make sense

---

**Happy Training!** 🚀

Made with ❤️ for fira-bot project
