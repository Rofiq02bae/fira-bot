#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk validasi CSV.
"""

# Fix encoding HARUS di awal
import encoding_fix

import csv
from pathlib import Path
import pandas as pd

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "dataset" / "data_jadi.csv"

def main():
    df = pd.read_csv(DATA_FILE, dtype=str)
    bad = df[df.isnull().any(axis=1)]
    print(f"Total rows: {len(df)}")
    print(f"Bad rows: {len(bad)}")
    
    if len(bad) > 0:
        print("\nBaris bermasalah:")
        for i, row in bad.iterrows():
            print(f"Row {i}: {row.tolist()[:3]}")
    
    # Validasi dengan csv.reader
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        bad_rows = []
        total = 0
        
        for i, row in enumerate(reader, start=1):
            total += 1
            if len(row) != 4:
                bad_rows.append((i, len(row), row[:3]))
        
        print(f"\nValidasi dengan csv.reader:")
        print(f"Total rows: {total}")
        print(f"Bad rows: {len(bad_rows)}")
        
        if bad_rows:
            print("\n20 baris pertama yang bermasalah:")
            for item in bad_rows[:20]:
                print(item)


if __name__ == '__main__':
    main()
