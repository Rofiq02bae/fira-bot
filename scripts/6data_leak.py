#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk check data leakage.
"""

# Fix encoding HARUS di awal
import encoding_fix

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import sys
import os

# Add parent directory to path BEFORE importing
parent_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, parent_dir)

# Import directly from the module file to avoid loading the entire core package
import importlib.util
text_normalizer_path = os.path.join(parent_dir, "core", "processors", "text_normalizer.py")
spec = importlib.util.spec_from_file_location("text_normalizer", text_normalizer_path)

if spec is None or spec.loader is None:
    raise ImportError(f"Could not load text_normalizer from {text_normalizer_path}")

text_normalizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_normalizer_module)
TextNormalizer = text_normalizer_module.TextNormalizer

print("🚨 ADVANCED DATA LEAKAGE DEBUG")
print("=" * 60)

# Load cleaned dataset
try:
    # Build absolute path to dataset
    dataset_path = os.path.join(parent_dir, 'data', 'dataset', 'dataset_training.csv')
    df = pd.read_csv(dataset_path)
    normalizer = TextNormalizer()

    print(f"📊 Dataset: {len(df)} samples, {df['intent'].nunique()} intents")

    # 1. Simulate EXACT same split seperti di training
    print("\n🔍 1. REPLICATING TRAINING SPLIT...")

    # Encode labels sama seperti training
    le = LabelEncoder()
    encoded_labels = le.fit_transform(df['intent'])

    # Split dengan parameter SAMA seperti training
    train_idx, val_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=encoded_labels
    )

    train_patterns = [str(df.iloc[i]['pattern']) for i in train_idx]  # NO NORMALIZATION
    val_patterns = [str(df.iloc[i]['pattern']) for i in val_idx]      # NO NORMALIZATION

    train_intents = [df.iloc[i]['intent'] for i in train_idx]
    val_intents = [df.iloc[i]['intent'] for i in val_idx]

    print(f"   Train: {len(train_patterns)} samples")
    print(f"   Val: {len(val_patterns)} samples")

    # 2. Check for pattern overlap (case-sensitive)
    print("\n🔍 2. PATTERN OVERLAP ANALYSIS (CASE-SENSITIVE):")
    train_set = set(train_patterns)
    val_set = set(val_patterns)
    overlap = train_set.intersection(val_set)

    print(f"   Exact pattern overlap: {len(overlap)}")
    if overlap:
        print("   🚨 OVERLAP FOUND! First 5 overlapping patterns:")
        for pattern in list(overlap):
            print(f"      '{pattern}'")

    # 3. Check intent distribution in validation set
    print("\n🔍 3. VALIDATION SET INTENT DISTRIBUTION:")
    val_intent_counts = Counter(val_intents)
    total_val = len(val_patterns)

    print("   Top intents in validation set:")
    for intent, count in val_intent_counts.most_common(10):
        percentage = (count / total_val) * 100
        print(f"      {intent}: {count} samples ({percentage:.1f}%)")

    # Check if validation set is dominated by few intents
    top_3_percentage = sum([count for _, count in val_intent_counts.most_common(3)]) / total_val * 100
    print(f"   Top 3 intents cover: {top_3_percentage:.1f}% of validation set")

    if top_3_percentage > 50:
        print("   🚨 PROBLEM: Validation set dominated by few intents!")

    # 4. Check pattern similarity (fuzzy matching)
    print("\n🔍 4. FUZZY PATTERN SIMILARITY:")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()


def simple_similarity(pattern1, pattern2):
    """Simple word overlap similarity"""
    words1 = set(str(pattern1).lower().split())
    words2 = set(str(pattern2).lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

# Check for very similar patterns between train and val
similar_patterns = []
similarity_threshold = 0.8

for i, val_pattern in enumerate(val_patterns[:100]):  # Check first 100 untuk performance
    for train_pattern in train_patterns[:100]:
        similarity = simple_similarity(val_pattern, train_pattern)
        if similarity >= similarity_threshold:
            similar_patterns.append((val_pattern, train_pattern, similarity))

print(f"   Very similar patterns (≥{similarity_threshold:.0%}): {len(similar_patterns)}")
if similar_patterns:
    print("   Examples of similar patterns:")
    for val_p, train_p, sim in similar_patterns[:5]:
        print(f"      Val:  '{val_p}'")
        print(f"      Train: '{train_p}' ({sim:.1%})")
        print()

# 5. Check if some intents have very simple patterns
print("\n🔍 5. PATTERN COMPLEXITY ANALYSIS:")
def analyze_pattern_complexity(patterns):
    complexities = []
    for pattern in patterns:
        words = str(pattern).split()
        complexities.append(len(words))
    return np.mean(complexities)

train_complexity = analyze_pattern_complexity(train_patterns)
val_complexity = analyze_pattern_complexity(val_patterns)

print(f"   Average words per pattern:")
print(f"      Train: {train_complexity:.2f}")
print(f"      Val: {val_complexity:.2f}")

# Check for very short patterns in validation
short_val_patterns = [p for p in val_patterns if len(str(p).split()) <= 3]
print(f"   Very short patterns in val (≤3 words): {len(short_val_patterns)}/{len(val_patterns)} ({len(short_val_patterns)/len(val_patterns):.1%})")

# 6. Check specific intent patterns
print("\n🔍 6. INTENT-SPECIFIC ANALYSIS:")
# Ambil intent yang paling banyak di validation
top_val_intent = val_intent_counts.most_common(1)[0][0]
top_intent_patterns = df[df['intent'] == top_val_intent]['pattern'].tolist()

print(f"   Patterns for top validation intent '{top_val_intent}':")
for pattern in top_intent_patterns[:10]:
    print(f"      '{pattern}'")

print(f"   Total patterns for this intent: {len(top_intent_patterns)}")
print(f"   Unique patterns: {len(set(top_intent_patterns))}")
print(f"   Diversity: {len(set(top_intent_patterns))/len(top_intent_patterns):.1%}")

print("\n" + "=" * 60)
print("🎯 LEAKAGE DIAGNOSIS:")
if len(overlap) > 0:
    print("❌ CRITICAL: Exact pattern overlap between train/val")
elif top_3_percentage > 50:
    print("❌ CRITICAL: Validation set imbalance")
elif len(similar_patterns) > 50:
    print("❌ CRITICAL: Too many similar patterns")
elif val_complexity < train_complexity * 0.8:
    print("⚠️  WARNING: Validation patterns are simpler")
else:
    print("✅ No obvious leakage detected - need deeper investigation")

print("=" * 60)