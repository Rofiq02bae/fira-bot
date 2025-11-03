#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk validasi CSV.
"""

# Fix encoding HARUS di awal
import encoding_fix
from encoding_fix import get_data_path

import csv
from pathlib import Path
import pandas as pd

def main():
    df = pd.read_csv(get_data_path("data_jadi.csv"), dtype=str)
    bad = df[df.isnull().any(axis=1)]
    print(f"Total rows: {len(df)}")
    print(f"Bad rows: {len(bad)}")
    for i, row in bad.iterrows():
        print((i, row.tolist()[:3]))
        total = 0
        for i, row in enumerate(reader, start=1):
            total += 1
            if len(row) != 4:
                bad.append((i, len(row), row[:3]))
        print(f"Total rows: {total}")
        print(f"Bad rows: {len(bad)}")
        for item in bad[:20]:
            print(item)


if __name__ == '__main__':
    main()
