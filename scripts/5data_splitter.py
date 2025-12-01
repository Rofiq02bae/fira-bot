#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk split data patterns.
"""

# Fix encoding HARUS di awal
from encoding_fix import get_data_path

import pandas as pd

# Baca file CSV awal
df = pd.read_csv(get_data_path("dataset_clean.csv"))

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
df_new.to_csv(get_data_path("dataset_training.csv"), index=False)
print("Selesai! File 'dataset_training.csv' sudah dibuat.")
