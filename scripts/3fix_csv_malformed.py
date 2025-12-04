#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk fix malformed CSV.
"""

# Fix encoding HARUS di awal
import encoding_fix

import csv
import shutil
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CSV_PATH = PROJECT_ROOT / "data" / "dataset" / "data_jadi.csv"
BACKUP_PATH = CSV_PATH.with_suffix('.csv.bak')



def count_fields(line: str) -> int:
    # Use csv.reader on a single-line string to count top-level fields
    try:
        row = next(csv.reader([line]))
        return len(row)
    except Exception:
        return -1


def try_fix_line(line: str) -> str | None:
    # If line starts like: token"...  (no comma between token and opening quote)
    # insert a comma after the token. Token is alnum/_ characters.
    import re

    m = re.match(r"^([A-Za-z0-9_]+)\"", line)
    if m:
        fixed = line[: m.end(1)] + "," + line[m.end(1) :]
        # confirm the fixed line now parses to >= 4 fields
        if count_fields(fixed) >= 4:
            return fixed
    return None


def scan_top_level_commas(line: str) -> list:
    # Return indices of commas that are not inside double quotes
    idxs = []
    in_q = False
    i = 0
    L = len(line)
    while i < L:
        c = line[i]
        if c == '"':
            # handle escaped double-quote "" inside quoted field
            if in_q and i + 1 < L and line[i + 1] == '"':
                i += 2
                continue
            in_q = not in_q
            i += 1
            continue
        if c == ',' and not in_q:
            idxs.append(i)
        i += 1
    return idxs


def reconstruct_row(line: str) -> str | None:
    # Use first top-level comma and last two top-level commas to split into 4 fields
    idxs = scan_top_level_commas(line)
    if len(idxs) < 3:
        return None
    first = idxs[0]
    last2 = idxs[-2]
    last1 = idxs[-1]
    intent = line[:first].strip()
    pattern = line[first + 1 : last2]
    response_type = line[last2 + 1 : last1].strip()
    response = line[last1 + 1 :].strip()

    # clean up possible stray commas/spaces
    # Escape any double quotes in fields by doubling them
    def esc(s: str) -> str:
        return s.replace('"', '""')

    pattern_q = '"' + esc(pattern.strip()) + '"'
    response_q = '"' + esc(response.strip()) + '"'

    new = f"{intent},{pattern_q},{response_type},{response_q}"
    return new


def heuristic_split_on_response_type(line: str) -> str | None:
    # Common response types in this dataset: static, dynamic, list, etc.
    for token in [',static,', ',dynamic,', ',list,']:
        idx = line.find(token)
        if idx != -1:
            left = line[:idx]
            right = line[idx + len(token) :]
            # left should contain intent,pattern (pattern may contain commas)
            if ',' not in left:
                continue
            intent, pattern = left.split(',', 1)
            response_type = token.strip(',')
            response = right
            # escape quotes
            pattern_q = '"' + pattern.replace('"', '""').strip() + '"'
            response_q = '"' + response.replace('"', '""').strip() + '"'
            return f"{intent},{pattern_q},{response_type},{response_q}"
    return None


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return

    # backup
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"Backup created at {BACKUP_PATH}")

    lines = CSV_PATH.read_text(encoding='utf-8').splitlines()
    fixed_lines = []
    fixes = []
    bad = []

    i = 0
    n = len(lines)
    while i < n:
        i += 1
        buf = lines[i - 1]
        num = count_fields(buf)
        # If this physical line doesn't parse as 4 fields, try merging following lines
        j = i
        while num != 4 and j < n:
            # append next physical line with newline (CSV may contain embedded newlines)
            buf = buf + "\n" + lines[j]
            j += 1
            num = count_fields(buf)
        if num == 4:
            # If buffer parses to 4 fields but needs minor fix (missing comma after intent), try that
            if count_fields(buf) != 4:
                # should not happen, but keep safe
                pass
            fixed_lines.append(buf)
            # record if we merged multiple lines
            if j - i > 1:
                fixes.append((i, j, 'merged'))
            i = j
            continue
        # try auto-fix on original single line
        orig_line = lines[i - 1]
        new = try_fix_line(orig_line)
        if new and count_fields(new) == 4:
            fixed_lines.append(new)
            fixes.append((i, i, 'fix_comma'))
            i = i
            continue
        # try heuristic split on response type
        heur = heuristic_split_on_response_type(orig_line)
        if heur and count_fields(heur) == 4:
            fixed_lines.append(heur)
            fixes.append((i, i, 'heuristic'))
            i = i
            continue
        # fallback: keep original and mark as bad
        fixed_lines.append(orig_line)
        bad.append((i, num, orig_line[:200]))

    # write fixed file
    CSV_PATH.write_text('\n'.join(fixed_lines) + '\n', encoding='utf-8')

    print(f"Total lines: {len(lines)}")
    print(f"Auto-fixed lines: {len(fixes)}")
    if fixes:
        for ln, old, new in fixes[:20]:
            print(f"Fixed line {ln}: -> now has {count_fields(new)} fields")

    print(f"Remaining malformed lines: {len(bad)} (line_no, field_count, sample...) )")
    if bad:
        for ln, fc, sample in bad[:20]:
            print(f"Line {ln}: fields={fc} sample={sample}")


if __name__ == '__main__':
    main()
