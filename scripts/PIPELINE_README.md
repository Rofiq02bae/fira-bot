# Data Cleaning Pipeline

Pipeline otomatis untuk menjalankan seluruh proses pembersihan data secara berurutan.

Update terbaru:
- Pipeline lama berbasis banyak file bernomor (1hapus_duplikat.py, 2fix_csv.py, dst) tetap ada.
- Pipeline baru yang lebih ringkas ada di `scripts/dataset_pipeline.py`.

## Urutan Eksekusi (Pipeline Ringkas)

Pipeline ringkas menjalankan langkah berikut:

1. **Clean dataset** - normalisasi kolom + delimiter pattern
2. **Deduplicate patterns** - hapus pattern duplikat (case-insensitive) di dalam sel pattern
3. **Split patterns** - 1 pattern = 1 baris (format training)
4. **Validate** - cek struktur dataset hasil akhir

## Cara Penggunaan

### Opsi 1 (Direkomendasikan): pipeline ringkas

Jalankan semua step:
```bash
python scripts/dataset_pipeline.py all
```

Jika `response` di `data_mentah.csv` masih berupa teks biasa (bukan JSON), aktifkan konversi:
```bash
python scripts/dataset_pipeline.py all --convert-response-json --validate-response-json
```

Jalankan step tertentu:
```bash
python scripts/dataset_pipeline.py clean
python scripts/dataset_pipeline.py dedup
python scripts/dataset_pipeline.py split
python scripts/dataset_pipeline.py validate
```

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
├── dataset_pipeline.py      # Pipeline ringkas (baru)
├── pipeline.py              # Wrapper runner (ringkas)
├── 1hapus_duplikat.py       # Legacy
├── 2fix_csv.py              # Legacy
├── 3fix_csv_malformed.py    # Legacy
├── 4validate_csv.py         # Legacy
├── 5data_splitter.py        # Legacy
└── 6data_leak.py            # Analisis leakage (terpisah)

logs/
└── pipeline.log             # Log hasil eksekusi
```

& "D:/bot/New folder/.venv/Scripts/python.exe" "D:/bot/New folder/scripts/visualize_test_results.py" "D:/bot/New folder/logs/test_results_20260113_210545.csv"