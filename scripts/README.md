# 🚀 Chatbot Training Scripts

Koleksi script untuk menjalankan pipeline training chatbot dengan berbagai platform dan opsi.

## 📋 Available Scripts

### 1. `train.sh` - Basic Bash Script
Script bash sederhana untuk training dasar:
```bash
bash train.sh
```

### 2. `train_pipeline.sh` - Advanced Bash Script  
Script bash lengkap dengan berbagai opsi:
```bash
# Full pipeline dengan advanced deduplication
bash train_pipeline.sh

# Simple deduplication saja
bash train_pipeline.sh --simple-dedup

# Advanced dengan threshold custom
bash train_pipeline.sh --advanced-dedup --threshold 0.9

# Hanya clean data, skip training
bash train_pipeline.sh --skip-training

# Show help
bash train_pipeline.sh --help
```

### 3. `train_pipeline.bat` - Windows Batch File
Script batch untuk Windows:
```cmd
REM Full pipeline
train_pipeline.bat

REM Simple deduplication
train_pipeline.bat simple

REM Advanced dengan threshold custom
train_pipeline.bat advanced 0.9
```

### 4. `train_pipeline.ps1` - PowerShell Script
Script PowerShell untuk Windows dengan fitur lengkap:
```powershell
# Full pipeline
.\train_pipeline.ps1

# Simple deduplication
.\train_pipeline.ps1 -Mode simple

# Advanced dengan threshold custom  
.\train_pipeline.ps1 -Mode advanced -Threshold 0.9

# Hanya clean data
.\train_pipeline.ps1 -SkipTraining

# Show help
.\train_pipeline.ps1 -Help
```

## 🔧 Pipeline Steps

Semua script menjalankan pipeline berikut:

1. **🔍 Prerequisites Check** - Validasi file yang diperlukan
2. **🧹 Data Deduplication** - Hapus duplikasi dengan mode simple/advanced
3. **🔧 Data Validation** - Validasi format CSV
4. **📊 Data Splitting** - Bagi data untuk training/validation  
5. **🚀 Model Training** - Training model LSTM + IndoBERT

## 📊 Deduplication Modes

### Simple Mode
- Hanya hapus duplikasi eksak
- Lebih cepat
- Cocok untuk dataset kecil

### Advanced Mode
- Hapus duplikasi eksak + serupa
- Menggunakan similarity threshold
- Reduksi data lebih besar (60%+)
- Cocok untuk dataset besar dengan banyak variasi

## 🎛️ Similarity Thresholds

| Threshold | Behavior | Use Case |
|-----------|----------|-----------|
| 0.95 | Sangat ketat | Hapus hanya yang hampir identik |
| 0.85 | Seimbang (default) | Training normal |
| 0.75 | Agresif | Dataset dengan banyak variasi |
| 0.65 | Sangat agresif | Tidak disarankan |

## 📁 Input/Output Files

### Input:
- `../dataset/data_mentah.csv` - Dataset mentah

### Output:
- `../dataset/dataset_training.csv` - Data bersih untuk training
- `../model/chatbot_model.h5` - Model yang sudah dilatih
- `../model/hybrid_config.json` - Konfigurasi model

## ⚙️ Requirements

1. **Python** dengan packages:
   - pandas
   - tensorflow/keras
   - transformers
   - scikit-learn

2. **Platform Support**:
   - Linux/Mac: bash scripts
   - Windows: batch file atau PowerShell
   - Git Bash: bash scripts di Windows

## 🚨 Error Handling

- Semua script memiliki error handling
- Exit dengan kode error jika ada step yang gagal
- Logging detail untuk debugging

## 💡 Tips

1. **Untuk development**: Gunakan `--skip-training` untuk cepat test cleaning
2. **Untuk production**: Gunakan advanced mode dengan threshold 0.85
3. **Untuk dataset besar**: Gunakan simple mode jika advanced terlalu lambat
4. **Backup data**: Selalu backup `data_mentah.csv` sebelum processing

## 🔍 Troubleshooting

### Error: "File not found"
- Pastikan menjalankan dari direktori `scripts/`
- Check apakah `data_mentah.csv` ada di `../dataset/`

### Error: "Python command not found"
- Pastikan Python terinstall dan ada di PATH
- Atau gunakan full path: `/usr/bin/python3` atau `C:\Python\python.exe`

### Training gagal
- Check apakah virtual environment aktif
- Pastikan semua dependencies terinstall
- Check log error untuk detail

---

🎯 **Quick Start**: Untuk training cepat, gunakan:
```bash
bash train_pipeline.sh --advanced-dedup --threshold 0.85
```