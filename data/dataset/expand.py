#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate and inject 10 patterns per intent into dataset_prosedur.csv.

Rules:
- Every intent gets exactly 10 common-question patterns.
- Patterns for one intent are stored in one cell, separated by '|'.
- The CSV pattern column is replaced automatically from pattern_dict.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "dataset_prosedur.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset_prosedur.csv"


def _clean_topic(intent: str) -> str:
    topic = intent.replace("_", " ")
    topic = re.sub(r"\s+", " ", topic).strip().lower()

    # Keep abbreviations readable
    replacements = {
        " api ": " API ",
        " uat ": " UAT ",
        " jipd ": " JIPD ",
        " splp ": " SPLP ",
    }

    topic = f" {topic} "
    for src, dst in replacements.items():
        topic = topic.replace(src, f" {dst} ")
    return re.sub(r"\s+", " ", topic).strip()


def _generate_10_questions(intent: str) -> list[str]:
    topic = _clean_topic(intent)

    questions = [
        f"apa itu {topic}?",
        f"bagaimana prosedur {topic}?",
        f"apa syarat untuk {topic}?",
        f"dokumen apa yang dibutuhkan untuk {topic}?",
        f"berapa lama proses {topic}?",
        f"apakah {topic} bisa diajukan online?",
        f"dimana layanan {topic} dilakukan?",
        f"berapa biaya untuk {topic}?",
        f"bagaimana alur lengkap {topic} dari awal sampai selesai?",
        f"siapa yang bisa membantu jika ada kendala {topic}?",
    ]

    # Ensure exactly 10 unique patterns while preserving order
    seen = set()
    unique_questions: list[str] = []
    for q in questions:
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_questions.append(q.strip())

    if len(unique_questions) != 10:
        raise ValueError(f"Generated pattern count for intent '{intent}' is {len(unique_questions)}, expected 10")

    return unique_questions


def build_pattern_dict(rows: list[dict[str, str]]) -> dict[str, str]:
    pattern_dict: dict[str, str] = {}
    for row in rows:
        intent = str(row.get("intent", "")).strip()
        if not intent:
            continue
        if intent not in pattern_dict:
            pattern_dict[intent] = "|".join(_generate_10_questions(intent))
    return pattern_dict


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    required = {"intent", "pattern", "response_type", "is_master", "response"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    return fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def inject_patterns(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> dict[str, str]:
    fieldnames, rows = load_csv(input_path)
    pattern_dict = build_pattern_dict(rows)

    for row in rows:
        intent = str(row.get("intent", "")).strip()
        if intent in pattern_dict:
            row["pattern"] = pattern_dict[intent]

    write_csv(output_path, fieldnames, rows)
    return pattern_dict


if __name__ == "__main__":
    pattern_dict = inject_patterns()
    total_intent = len(pattern_dict)
    print(f"Updated CSV: {DEFAULT_OUTPUT}")
    print(f"Total intent injected: {total_intent}")
    for intent, pattern in list(pattern_dict.items())[:5]:
        print(f"- {intent}: {len(pattern.split('|'))} patterns")
