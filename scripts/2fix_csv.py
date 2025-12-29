#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk fix CSV formatting.
"""

# Fix encoding HARUS di awal
import encoding_fix

import pandas as pd
import re
import csv
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "dataset" / "data_tanpa_duplikat.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "dataset" / "data_jadi.csv"

# Read the original file line by line and manually parse it
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_data = []
header = ['intent', 'pattern', 'response_type','is_master' , 'response']

for i, line in enumerate(lines):
    line = line.strip()
    if i == 0 or not line:  # Skip header and empty lines
        continue
    
    # Manually parse each line to extract the 4 fields
    # Expected format: intent, "pattern", response_type, "response"
    
    # Find the first comma (after intent)
    first_comma = line.find(',')
    if first_comma == -1:
        continue
    
    intent = line[:first_comma].strip()
    remainder = line[first_comma + 1:].strip()
    
    # Find the pattern (quoted string)
    if remainder.startswith('"'):
        # Find the closing quote for pattern
        quote_end = remainder.find('"', 1)
        pattern_end = quote_end + 1
        
        # Keep finding quotes until we find the real end (handle escaped quotes)
        while quote_end != -1 and pattern_end < len(remainder):
            if pattern_end >= len(remainder):
                break
            # Check if there's content after this quote that looks like next field
            after_quote = remainder[pattern_end:].strip()
            if after_quote.startswith(',') and ('static' in after_quote[:20] or '"' in after_quote[:20]):
                break
            # Look for next quote
            next_quote = remainder.find('"', pattern_end)
            if next_quote == -1:
                break
            quote_end = next_quote
            pattern_end = quote_end + 1
        
        pattern = remainder[1:quote_end]  # Remove outer quotes
        # Replace commas with pipes in pattern
        pattern = re.sub(r'\s*,\s*', '|', pattern)
        
        remainder = remainder[pattern_end:].strip()
        if remainder.startswith(','):
            remainder = remainder[1:].strip()
    else:
        # Pattern without quotes - find next comma
        next_comma = remainder.find(',')
        if next_comma == -1:
            continue
        pattern = remainder[:next_comma].strip()
        remainder = remainder[next_comma + 1:].strip()
    
    # Find response_type 
    if remainder.startswith('"'):
        # Response type is quoted
        quote_end = remainder.find('"', 1)
        response_type = remainder[1:quote_end]
        remainder = remainder[quote_end + 1:].strip()
        if remainder.startswith(','):
            remainder = remainder[1:].strip()
    else:
        # Response type not quoted - find next comma
        next_comma = remainder.find(',')
        if next_comma == -1:
            # Rest is response_type, no is_master or response
            response_type = remainder.strip()
            is_master = "false"
            response = ""
            remainder = ""
        else:
            response_type = remainder[:next_comma].strip()
            remainder = remainder[next_comma + 1:].strip()
            
    # Find is_master
    if remainder:
        next_comma = remainder.find(',')
        if next_comma == -1:
            # Rest is is_master, no response
            is_master = remainder.strip()
            response = ""
            remainder = ""
        else:
            is_master = remainder[:next_comma].strip()
            remainder = remainder[next_comma + 1:].strip()
    else:
        is_master = "false" if 'response' not in locals() else is_master # fallback

    # Rest is response
    if remainder:
        if remainder.startswith('"') and remainder.endswith('"'):
            response = remainder[1:-1]  # Remove outer quotes
        else:
            response = remainder
    else:
        if 'response' not in locals():
            response = ""
    
    # Clean up fields
    intent = intent.strip()
    response_type = response_type.strip()
    is_master = is_master.strip().lower() or "false"
    response = response.strip()
    
    fixed_data.append([intent, pattern, response_type, is_master, response])

# Write the fixed CSV
with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(fixed_data)

print(f"Fixed CSV created with {len(fixed_data)} rows")

# Test if the fixed file can be parsed
try:
    df = pd.read_csv(OUTPUT_FILE)
    print(f"✅ Fixed CSV parsed successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Show first few rows
    print(f"\nFirst 3 rows:")
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"Row {i+1}:")
        print(f"  Intent: {row['intent']}")
        print(f"  Pattern: {row['pattern'][:100]}{'...' if len(row['pattern']) > 100 else ''}")
        print(f"  Response Type: {row['response_type']}")
        print(f"  Response: {row['response'][:100]}{'...' if len(row['response']) > 100 else ''}")
        print()
        
except Exception as e:
    print(f"❌ Error parsing fixed CSV: {str(e)}")