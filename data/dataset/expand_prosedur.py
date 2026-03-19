#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand dataset_prosedur pattern column from pipe-separated to one-pattern-per-row.

Default behavior:
- input  : data/dataset/dataset_prosedur.csv
- output : data/dataset/dataset_prosedur_expanded.csv

Other columns are preserved as-is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_COLUMNS = ["intent", "pattern", "response_type", "is_master", "response"]


def split_patterns(input_csv: Path, output_csv: Path) -> tuple[int, int]:
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        rows = list(reader)

    expanded_rows: list[dict[str, str]] = []
    for row in rows:
        raw = str(row.get("pattern", ""))
        patterns = [p.strip() for p in raw.split("|") if p.strip()]

        if not patterns:
            # Keep row if pattern is empty, preserving original data.
            expanded_rows.append(row)
            continue

        for p in patterns:
            new_row = dict(row)
            new_row["pattern"] = p
            expanded_rows.append(new_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded_rows)

    return len(rows), len(expanded_rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pisahkan kolom pattern (dipisah '|') menjadi 1 row per pattern."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir / "dataset_prosedur.csv",
        help="Path file input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "dataset_prosedur_expanded.csv",
        help="Path file output CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_rows, out_rows = split_patterns(args.input, args.output)
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows  : {src_rows} -> {out_rows}")
