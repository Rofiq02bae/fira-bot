# Data Cleaning Pipeline

Pipeline otomatis untuk menjalankan seluruh proses pembersihan data secara berurutan.

## Urutan Eksekusi

Pipeline menjalankan script berikut secara berurutan:

1. **1hapus_duplikat.py** - Menghapus data duplikat
2. **2fix_csv.py** - Memperbaiki format CSV
3. **3fix_csv_malformed.py** - Memperbaiki CSV yang rusak
4. **4validate_csv.py** - Validasi struktur CSV
5. **5data_splitter.py** - Memisahkan data training/validation
6. **6data_leak.py** - Analisis data leakage

## Cara Penggunaan

### Jalankan Semua Script (Berhenti Saat Error)
```powershell
python .\scripts\pipeline.py
```

### Lanjutkan Meskipun Ada Error
```powershell
python .\scripts\pipeline.py --continue-on-error
```

### Preview Tanpa Eksekusi (Dry Run)
```powershell
python .\scripts\pipeline.py --dry-run
```

### Lihat Daftar Script
```powershell
python .\scripts\pipeline.py --list
```

## Output

- **Log Console**: Output real-time di terminal
- **Log File**: `logs/pipeline.log` - Log lengkap dengan timestamp

## Contoh Output

```
2025-10-22 14:07:48,633 [INFO] Starting data cleaning pipeline…
------------------------------------------------------------
[INFO] Step 1/6: 1hapus_duplikat.py
[INFO] ➡️ Running step: 1hapus_duplikat (1hapus_duplikat.py)
[INFO] ✅ Step succeeded: 1hapus_duplikat
------------------------------------------------------------
...
[INFO] Pipeline finished successfully. ✅
```

## Troubleshooting

### Import Error pada Script
Jika terjadi error `ModuleNotFoundError: No module named 'core'`:
- Pastikan menjalankan dari root directory project (`d:\bot\New folder`)
- Script pipeline sudah handle path secara otomatis

### Missing Dependencies
Install dependencies yang diperlukan:
```powershell
pip install pandas numpy scikit-learn
```

### File Dataset Tidak Ditemukan
Pastikan file dataset berada di:
```
data/dataset/dataset_training.csv
```

## Catatan Penting

1. **Urutan Penting**: Script harus dijalankan sesuai urutan karena output satu script menjadi input script berikutnya
2. **Backup Data**: Disarankan backup data sebelum menjalankan pipeline
3. **Logging**: Semua output disimpan di `logs/pipeline.log` untuk audit trail
4. **Error Handling**: Gunakan `--continue-on-error` jika ingin melihat hasil semua script meskipun ada yang gagal

## Struktur File

```
scripts/
├── pipeline.py              # Pipeline runner utama
├── 1hapus_duplikat.py
├── 2fix_csv.py
├── 3fix_csv_malformed.py
├── 4validate_csv.py
├── 5data_splitter.py
└── 6data_leak.py

logs/
└── pipeline.log             # Log hasil eksekusi
```
