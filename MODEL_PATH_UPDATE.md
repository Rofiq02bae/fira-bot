# Struktur Folder Model - Update Summary

## 📁 Perubahan Path Model

Semua model sekarang disimpan dalam struktur yang terorganisir di folder `data/`:

### ✅ Struktur Baru

```
data/
├── models/                      # LSTM models (diubah dari data/lstm/)
│   ├── chatbot_model.h5        # LSTM model
│   ├── tokenizer.pkl           # LSTM tokenizer
│   ├── label_encoder.pkl       # Label encoder
│   └── hybrid_config.json      # Konfigurasi hybrid system
├── bert_model/                  # BERT models (diubah dari bert_simple_finetuned/)
│   ├── config.json             # BERT config
│   ├── pytorch_model.bin       # BERT weights
│   ├── tokenizer_config.json   # BERT tokenizer config
│   ├── vocab.txt               # BERT vocabulary
│   ├── label_encoder.pkl       # BERT label encoder
│   └── bert_info.json          # Training info
└── dataset/
    └── dataset_training.csv    # Training dataset
```

### ❌ Path Lama (DEPRECATED)

```
data/lstm/           → DIGANTI data/models/
bert_simple_finetuned/ → DIGANTI data/bert_model/
model/               → DIGANTI data/models/
```

## 🔧 File yang Diupdate

### 1. Training Scripts

#### `training/lstm_trainng.py`
- ✅ Model disimpan ke: `data/models/chatbot_model.h5`
- ✅ Tokenizer disimpan ke: `data/models/tokenizer.pkl`
- ✅ Label encoder disimpan ke: `data/models/label_encoder.pkl`
- ✅ Config disimpan ke: `data/models/hybrid_config.json`

#### `training/bert_training.py`
- ✅ Model disimpan ke: `data/bert_model/`
- ✅ Label encoder disimpan ke: `data/bert_model/label_encoder.pkl`
- ✅ Info disimpan ke: `data/bert_model/bert_info.json`

### 2. Configuration Files

#### `config/settings.py`
```python
@dataclass
class ModelConfig:
    lstm_model_path: str = "../data/models/chatbot_model.h5"
    lstm_tokenizer_path: str = "../data/models/tokenizer.pkl"
    lstm_label_encoder_path: str = "../data/models/label_encoder.pkl"
    bert_model_path: str = "../data/bert_model"
    dataset_path: str = "../data/dataset/dataset_training.csv"
```

### 3. Application Files

#### `app.py`
```python
lstm_model_path = os.environ.get('LSTM_MODEL_PATH', 'data/models/chatbot_model.h5')
lstm_tokenizer_path = os.environ.get('LSTM_TOKENIZER_PATH', 'data/models/tokenizer.pkl')
lstm_label_encoder_path = os.environ.get('LSTM_LABEL_ENCODER_PATH', 'data/models/label_encoder.pkl')
bert_model_path = os.environ.get('BERT_MODEL_PATH', 'data/bert_model')
```

#### `main.py`
```python
def initialize_hybrid_service(
    dataset_path: str = "data/dataset/dataset_training.csv",
    lstm_model_path: str = "data/models/chatbot_model.h5",
    lstm_tokenizer_path: str = "data/models/tokenizer.pkl",
    lstm_label_encoder_path: str = "data/models/label_encoder.pkl",
    bert_model_path: str = "data/bert_model"
)
```

## 🚀 Cara Penggunaan

### Training LSTM Model

```powershell
# Jalankan dari root directory
python .\training\lstm_trainng.py
```

Output akan tersimpan di:
- `data/models/chatbot_model.h5`
- `data/models/tokenizer.pkl`
- `data/models/label_encoder.pkl`
- `data/models/hybrid_config.json`

### Training BERT Model

```powershell
# Jalankan dari root directory
python .\training\bert_training.py
```

Output akan tersimpan di:
- `data/bert_model/` (semua file model BERT)

### Menjalankan Application

```powershell
# Menggunakan path default (otomatis menggunakan data/models/ dan data/bert_model/)
python app.py
```

atau dengan environment variables:

```powershell
$env:LSTM_MODEL_PATH="data/models/chatbot_model.h5"
$env:BERT_MODEL_PATH="data/bert_model"
python app.py
```

## 📝 Migration Notes

Jika Anda memiliki model lama di path lama:

### Migrasi LSTM Models

```powershell
# Buat folder baru
New-Item -ItemType Directory -Force -Path "data\models"

# Copy dari path lama
Copy-Item "data\lstm\*" -Destination "data\models\" -Recurse
# ATAU
Copy-Item "model\*" -Destination "data\models\" -Recurse
```

### Migrasi BERT Models

```powershell
# Buat folder baru
New-Item -ItemType Directory -Force -Path "data\bert_model"

# Copy dari path lama
Copy-Item "bert_simple_finetuned\*" -Destination "data\bert_model\" -Recurse
```

## ✅ Keuntungan Struktur Baru

1. **Terorganisir**: Semua model dalam satu parent folder `data/`
2. **Konsisten**: Naming yang jelas dan konsisten
3. **Maintainable**: Mudah di-backup dan di-version control
4. **Scalable**: Mudah menambahkan model baru di masa depan
5. **Clean**: Tidak ada folder model tersebar di root directory

## 🔍 Verifikasi

Cek apakah struktur sudah benar:

```powershell
# Check struktur folder
tree /F data

# Atau menggunakan PowerShell
Get-ChildItem -Path data -Recurse | Select-Object FullName
```

Expected output:
```
data/
├── models/
│   ├── chatbot_model.h5
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── hybrid_config.json
├── bert_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── dataset/
    └── dataset_training.csv
```

## 🆘 Troubleshooting

### Error: Model files not found

Pastikan Anda menjalankan training terlebih dahulu:
```powershell
python .\training\lstm_trainng.py
python .\training\bert_training.py
```

### Error: Permission denied

Pastikan folder `data/models/` dan `data/bert_model/` memiliki write permission.

### Path tidak sesuai

Double check bahwa Anda menjalankan script dari root directory project (`d:\bot\New folder\`).
