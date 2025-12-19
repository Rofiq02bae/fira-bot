#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset pipeline (clean -> deduplicate -> split -> validate).

Tujuan:
- Mengganti script bernomor (1hapus_duplikat, 2fix_csv, dst) dengan pipeline yang
  lebih ringkas dan penamaan jelas.
- Tetap kompatibel dengan format dataset yang sudah ada (kolom intent/pattern/
  response_type/response).

Default I/O (project-root relative):
- input:  data/dataset/data_mentah.csv
- clean:  data/dataset/dataset_clean.csv
- dedup:  data/dataset/data_tanpa_duplikat.csv
- train:  data/dataset/dataset_training.csv

Catatan:
- Script ini tidak melakukan training model; hanya menyiapkan dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency: pandas. Install it with: 'python3 -m pip install pandas' "
        "(or install full requirements.txt)."
    ) from e


REQUIRED_COLUMNS = ["intent", "pattern", "response_type", "response"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelinePaths:
    input_raw: Path
    output_clean: Path
    output_dedup: Path
    output_train: Path

    @staticmethod
    def defaults() -> "PipelinePaths":
        root = _project_root()
        return PipelinePaths(
            input_raw=root / "data" / "dataset" / "data_mentah.csv",
            output_clean=root / "data" / "dataset" / "dataset_clean.csv",
            output_dedup=root / "data" / "dataset" / "data_tanpa_duplikat.csv",
            output_train=root / "data" / "dataset" / "dataset_training.csv",
        )


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read CSV in a way that tolerates minor issues.

    Strategy:
    1) Try normal pandas read (expects header).
    2) Fallback: python engine, skip bad lines.
    3) If still looks like no header, retry with header=None + assign columns.
    """

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")
        return df
    except Exception:
        pass

    # last resort: no-header parsing into 4 columns
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        header=None,
        names=REQUIRED_COLUMNS,
        engine="python",
        on_bad_lines="skip",
    )
    return df


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # tolerate legacy column name
    if "patterns" in df.columns and "pattern" not in df.columns:
        df = df.rename(columns={"patterns": "pattern"})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found: {list(df.columns)}")

    # keep only required columns (ignore extras)
    return df[REQUIRED_COLUMNS].copy()


def _normalize_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["intent", "pattern", "response_type", "response"]:
        out[col] = out[col].astype(str).fillna("")

    out["intent"] = out["intent"].str.strip()
    out["pattern"] = out["pattern"].str.strip()
    out["response_type"] = out["response_type"].str.strip()
    out["response"] = out["response"].astype(str)

    # Drop empty essentials
    out = out[(out["intent"] != "") & (out["pattern"] != "")]
    return out


def _convert_response_to_json(response_text: str) -> str:
    """Convert plain response text into JSON string.

    Rule (same spirit as scripts/convert_json.py):
    - If response contains '|': treat as list (title + items)
    - Else: treat as text

    If response already looks like JSON, keep as-is.
    """

    text = str(response_text).strip()
    if not text:
        return json.dumps({"type": "text", "body": ""}, ensure_ascii=False)

    # already json
    if text.startswith("{") and text.endswith("}"):
        return text

    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if not parts:
            return json.dumps({"type": "text", "body": ""}, ensure_ascii=False)
        return json.dumps(
            {
                "type": "list",
                "title": parts[0],
                "items": parts[1:],
            },
            ensure_ascii=False,
        )

    return json.dumps({"type": "text", "body": text}, ensure_ascii=False)


def clean_dataset(input_csv: Path, output_csv: Path, *, convert_response_json: bool) -> pd.DataFrame:
    """Clean + normalize dataset (doesn't split patterns yet)."""

    df = _read_csv_flexible(input_csv)
    df = _ensure_columns(df)
    df = _normalize_fields(df)

    # Normalize delimiter in pattern (commas are a common accidental delimiter)
    df["pattern"] = df["pattern"].str.replace(r"\s*,\s*", "|", regex=True)

    if convert_response_json:
        df["response"] = df["response"].apply(_convert_response_to_json)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def deduplicate_patterns(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Deduplicate patterns inside each row (pipe-separated list)."""

    df = _read_csv_flexible(input_csv)
    df = _ensure_columns(df)
    df = _normalize_fields(df)

    def dedup_cell(cell: str) -> str:
        raw = str(cell)
        parts = [p.strip() for p in raw.split("|") if p.strip()]
        seen = set()
        kept: list[str] = []
        for p in parts:
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
        return "|".join(kept)

    df["pattern"] = df["pattern"].apply(dedup_cell)
    df = df[df["pattern"] != ""]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def split_patterns(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    """Split each pipe-separated pattern into one row (training format)."""

    df = _read_csv_flexible(input_csv)
    df = _ensure_columns(df)
    df = _normalize_fields(df)

    rows = []
    for _, row in df.iterrows():
        patterns = [p.strip() for p in str(row["pattern"]).split("|") if p.strip()]
        for p in patterns:
            rows.append(
                {
                    "intent": row["intent"].strip(),
                    "pattern": p,
                    "response_type": row["response_type"].strip() or "static",
                    "response": row["response"],
                }
            )

    out = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    out = _normalize_fields(out)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    return out


def validate_dataset(path: Path, *, validate_response_json: bool) -> None:
    df = _read_csv_flexible(path)
    df = _ensure_columns(df)
    df = _normalize_fields(df)

    if df.empty:
        raise ValueError(f"Dataset is empty after normalization: {path}")

    if validate_response_json:
        bad = 0
        for i, val in enumerate(df["response"].tolist(), start=1):
            try:
                json.loads(val)
            except Exception:
                bad += 1
                if bad <= 5:
                    raise ValueError(
                        f"Invalid JSON in response at row {i} of {path}. "
                        f"Example value starts with: {str(val)[:60]}"
                    )
        # if >0 but none raised (shouldn't happen), still fail
        if bad:
            raise ValueError(f"Found {bad} invalid JSON responses in {path}")


def run_all(paths: PipelinePaths, *, convert_response_json: bool, validate_response_json: bool) -> None:
    # Convenience: if user asks to validate JSON responses, ensure we also convert them
    # during the clean step (unless the dataset already contains JSON strings).
    if validate_response_json and not convert_response_json:
        print("ℹ️  --validate-response-json enabled → auto-enabling --convert-response-json")
        convert_response_json = True

    print("🚀 DATASET PIPELINE")
    print(f"Input : {paths.input_raw}")
    print(f"Clean : {paths.output_clean}")
    print(f"Dedup : {paths.output_dedup}")
    print(f"Train : {paths.output_train}")

    print("\n1) Cleaning dataset...")
    df_clean = clean_dataset(paths.input_raw, paths.output_clean, convert_response_json=convert_response_json)
    print(f"   ✅ Clean rows: {len(df_clean)} | intents: {df_clean['intent'].nunique()}")

    print("\n2) Deduplicating patterns...")
    df_dedup = deduplicate_patterns(paths.output_clean, paths.output_dedup)
    print(f"   ✅ Dedup rows: {len(df_dedup)} | intents: {df_dedup['intent'].nunique()}")

    print("\n3) Splitting patterns into training rows...")
    df_train = split_patterns(paths.output_dedup, paths.output_train)
    print(f"   ✅ Train rows: {len(df_train)} | intents: {df_train['intent'].nunique()}")

    print("\n4) Validating final dataset...")
    validate_dataset(paths.output_train, validate_response_json=validate_response_json)
    print("   ✅ Validation OK")


def _parse_args() -> argparse.Namespace:
    defaults = PipelinePaths.defaults()

    parser = argparse.ArgumentParser(description="Dataset preparation pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", default=str(defaults.input_raw), help="Input raw dataset CSV")
        p.add_argument("--clean", default=str(defaults.output_clean), help="Output cleaned CSV")
        p.add_argument("--dedup", default=str(defaults.output_dedup), help="Output deduplicated CSV")
        p.add_argument("--train", default=str(defaults.output_train), help="Output training CSV")
        p.add_argument(
            "--convert-response-json",
            action="store_true",
            help="Convert plain response text to JSON format (text/list)",
        )
        p.add_argument(
            "--validate-response-json",
            action="store_true",
            help="Validate that response column is valid JSON",
        )

    p_all = sub.add_parser("all", help="Run clean -> dedup -> split -> validate")
    add_common(p_all)

    p_clean = sub.add_parser("clean", help="Normalize dataset + optional response JSON conversion")
    add_common(p_clean)

    p_dedup = sub.add_parser("dedup", help="Deduplicate pipe-separated patterns")
    add_common(p_dedup)

    p_split = sub.add_parser("split", help="Split patterns to one row per pattern")
    add_common(p_split)

    p_validate = sub.add_parser("validate", help="Validate dataset format")
    add_common(p_validate)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    paths = PipelinePaths(
        input_raw=Path(args.input),
        output_clean=Path(args.clean),
        output_dedup=Path(args.dedup),
        output_train=Path(args.train),
    )

    if args.cmd == "all":
        run_all(paths, convert_response_json=args.convert_response_json, validate_response_json=args.validate_response_json)
        return

    if args.cmd == "clean":
        df = clean_dataset(paths.input_raw, paths.output_clean, convert_response_json=args.convert_response_json)
        print(f"✅ Clean done: {paths.output_clean} (rows={len(df)})")
        return

    if args.cmd == "dedup":
        df = deduplicate_patterns(paths.output_clean, paths.output_dedup)
        print(f"✅ Dedup done: {paths.output_dedup} (rows={len(df)})")
        return

    if args.cmd == "split":
        df = split_patterns(paths.output_dedup, paths.output_train)
        print(f"✅ Split done: {paths.output_train} (rows={len(df)})")
        return

    if args.cmd == "validate":
        validate_dataset(paths.output_train, validate_response_json=args.validate_response_json)
        print(f"✅ Validate OK: {paths.output_train}")
        return


if __name__ == "__main__":
    main()
