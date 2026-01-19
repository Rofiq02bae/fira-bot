#!/usr/bin/env python3
"""Generate laporan pertanyaan testing per intent.

Outputs:
- logs/testing_questions_by_intent.md

Sumber data:
- data/dataset/dataset_training.csv
- data/dataset/data_mentah.csv
- tests/test_auto.py
"""

from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_dataset_training(path: Path) -> Dict[str, List[str]]:
    intent_to_patterns: Dict[str, List[str]] = defaultdict(list)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = (row.get("intent") or "").strip()
            pattern = norm(row.get("pattern") or "")
            if intent and pattern:
                intent_to_patterns[intent].append(pattern)

    return {k: dedupe_preserve_order(v) for k, v in intent_to_patterns.items()}


def load_data_mentah(path: Path) -> Dict[str, List[str]]:
    intent_to_patterns: Dict[str, List[str]] = defaultdict(list)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = (row.get("intent") or "").strip()
            field = row.get("pattern") or ""
            parts = [norm(p) for p in field.split("|")]
            parts = [p for p in parts if p]
            if intent and parts:
                intent_to_patterns[intent].extend(parts)

    return {k: dedupe_preserve_order(v) for k, v in intent_to_patterns.items()}


def extract_test_suite(test_file: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (test_cases, fallback_hint_texts).

    fallback_hint_texts is extracted from inline comment 'Should fallback' lines.
    """
    src = test_file.read_text(encoding="utf-8")
    module = ast.parse(src)

    tests: List[Dict[str, Any]] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_test_suite":
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    raw = ast.literal_eval(stmt.value)
                    if isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict):
                                tests.append(item)
                    break
            break

    fallback_hints: List[str] = []
    for line in src.splitlines():
        if "Should fallback" in line:
            m = re.search(r'\{\s*"text"\s*:\s*"([^"]+)"', line)
            if m:
                fallback_hints.append(norm(m.group(1)))

    return tests, dedupe_preserve_order(fallback_hints)


def flatten_patterns(intent_to_patterns: Dict[str, List[str]]) -> set[str]:
    return {p for pats in intent_to_patterns.values() for p in pats}


def write_report(
    out_path: Path,
    dataset_training_path: Path,
    data_mentah_path: Path,
    test_file: Path,
) -> None:
    train = load_dataset_training(dataset_training_path)
    mentah = load_data_mentah(data_mentah_path)

    train_intents = set(train.keys())
    mentah_intents = set(mentah.keys())

    train_patterns = flatten_patterns(train)
    mentah_patterns = flatten_patterns(mentah)

    test_cases_raw, fallback_hints = extract_test_suite(test_file)

    normalized_tests: List[Dict[str, Optional[str]]] = []
    for tc in test_cases_raw:
        text = norm(str(tc.get("text", "")))
        exp = tc.get("expected_intent")
        exp = exp.strip() if isinstance(exp, str) else None
        if text:
            normalized_tests.append({"text": text, "expected_intent": exp})

    tests_by_intent: Dict[str, List[str]] = defaultdict(list)
    no_expected: List[str] = []

    for t in normalized_tests:
        if t["expected_intent"]:
            tests_by_intent[t["expected_intent"]].append(t["text"])
        else:
            no_expected.append(t["text"])  # intended fallback-ish tests

    wrong_vs_training: List[Dict[str, str]] = []
    for t in normalized_tests:
        exp = t["expected_intent"]
        if not exp:
            continue
        if exp not in train_intents:
            wrong_vs_training.append(
                {
                    "text": t["text"],
                    "expected_intent": exp,
                    "reason": "expected_intent tidak ada di dataset_training.csv",
                }
            )
        elif t["text"] not in train_patterns:
            wrong_vs_training.append(
                {
                    "text": t["text"],
                    "expected_intent": exp,
                    "reason": "text tidak ada sebagai pattern di dataset_training.csv",
                }
            )

    wrong_vs_mentah: List[Dict[str, str]] = []
    for t in normalized_tests:
        exp = t["expected_intent"]
        if not exp:
            continue
        if exp not in mentah_intents:
            wrong_vs_mentah.append(
                {
                    "text": t["text"],
                    "expected_intent": exp,
                    "reason": "expected_intent tidak ada di data_mentah.csv",
                }
            )
        elif t["text"] not in mentah_patterns:
            wrong_vs_mentah.append(
                {
                    "text": t["text"],
                    "expected_intent": exp,
                    "reason": "text tidak ada sebagai pattern di data_mentah.csv",
                }
            )

    fallback_triggers = dedupe_preserve_order([*no_expected, *fallback_hints])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as w:
        w.write("# Testing questions per intent\n\n")
        w.write("Report ini menggabungkan:\n")
        w.write(f"- dataset training: `{dataset_training_path.name}` (1 baris = 1 pattern)\n")
        w.write(f"- dataset mentah: `{data_mentah_path.name}` (1 baris = banyak pattern dipisah `|`)\n")
        w.write(f"- test suite: `tests/{test_file.name}`\n\n")

        w.write("## Ringkasan\n\n")
        w.write(f"- Total intent di dataset_training.csv: {len(train_intents)}\n")
        w.write(f"- Total pattern unik di dataset_training.csv: {len(train_patterns)}\n")
        w.write(f"- Total intent di data_mentah.csv: {len(mentah_intents)}\n")
        w.write(f"- Total pattern unik di data_mentah.csv: {len(mentah_patterns)}\n")
        w.write(f"- Total test case di test_auto.py: {len(normalized_tests)}\n\n")

        w.write("## Pertanyaan testing (dari test_auto.py) per expected_intent\n\n")
        for intent in sorted(tests_by_intent.keys()):
            w.write(f"### {intent}\n")
            for q in tests_by_intent[intent]:
                w.write(f"- {q}\n")
            w.write("\n")

        w.write("## Pertanyaan salah (tidak match dataset)\n\n")
        w.write("### Dibandingkan dataset_training.csv\n")
        if wrong_vs_training:
            for item in wrong_vs_training:
                w.write(
                    f"- {item['text']} (expected: {item['expected_intent']}) — {item['reason']}\n"
                )
        else:
            w.write("- (tidak ada)\n")
        w.write("\n")

        w.write("### Dibandingkan data_mentah.csv\n")
        if wrong_vs_mentah:
            for item in wrong_vs_mentah:
                w.write(
                    f"- {item['text']} (expected: {item['expected_intent']}) — {item['reason']}\n"
                )
        else:
            w.write("- (tidak ada)\n")
        w.write("\n")

        w.write("## Pertanyaan yang memicu fallback\n\n")
        w.write(
            "Definisi praktis di report ini: item yang **tidak punya** `expected_intent` di `test_auto.py` atau diberi komentar `Should fallback`.\n\n"
        )
        for q in fallback_triggers:
            w.write(f"- {q}\n")
        w.write("\n")

        w.write("## Semua pertanyaan/pattern dari dataset per intent\n\n")

        w.write("### Sumber: dataset_training.csv\n\n")
        for intent in sorted(train.keys()):
            pats = train[intent]
            w.write(f"#### {intent} ({len(pats)} pattern)\n")
            for p in pats:
                w.write(f"- {p}\n")
            w.write("\n")

        w.write("### Sumber: data_mentah.csv\n\n")
        for intent in sorted(mentah.keys()):
            pats = mentah[intent]
            w.write(f"#### {intent} ({len(pats)} pattern)\n")
            for p in pats:
                w.write(f"- {p}\n")
            w.write("\n")


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    dataset_training_path = root / "data" / "dataset" / "dataset_training.csv"
    data_mentah_path = root / "data" / "dataset" / "data_mentah.csv"
    test_file = root / "tests" / "test_auto.py"
    out_path = root / "logs" / "testing_questions_by_intent.md"

    write_report(out_path, dataset_training_path, data_mentah_path, test_file)
    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
