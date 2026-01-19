#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk split data patterns.
"""

# Fix encoding HARUS di awal
import encoding_fix

import pandas as pd
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "dataset" / "data_jadi.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "dataset" / "dataset_training.csv"

# Baca file CSV awal
df = pd.read_csv(INPUT_FILE)

# Buat list baru untuk menampung baris hasil split
new_rows = []

for _, row in df.iterrows():
    # Pisahkan pattern berdasarkan koma
    patterns = [p.strip() for p in row['pattern'].split('|')]
    
    for pattern in patterns:
        new_rows.append({
            'intent': row['intent'],
            'pattern': pattern,
            'response_type': row['response_type'],
            'response': row['response']
        })

# Buat DataFrame baru dari hasil split
df_new = pd.DataFrame(new_rows)

# Simpan ke CSV baru
df_new.to_csv(OUTPUT_FILE, index=False)
print(f"Selesai! File '{OUTPUT_FILE}' sudah dibuat.")
