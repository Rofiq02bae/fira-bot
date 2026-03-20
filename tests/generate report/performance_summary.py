"""
performance_summary.py — Fira Bot Overall Performance Summary
==============================================================
Membaca evaluation_results.json dan menampilkan ringkasan performa
overall untuk LSTM, BERT, Hybrid (LSTM+BERT), dan response time.

Usage:
    python3 performance_summary.py
    python3 performance_summary.py --input evaluation_results.json --output summary.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fira Bot Performance Summary")
parser.add_argument("--input",  default="evaluation_results.json", help="Path ke file evaluation_results.json")
parser.add_argument("--output", default=None,                       help="Path output file (opsional, jika tidak diisi akan print ke console)")
args = parser.parse_args()

INPUT_PATH = Path(args.input)
if not INPUT_PATH.exists():
    print(f"ERROR: {INPUT_PATH} tidak ditemukan. Jalankan evaluate.py terlebih dahulu.")
    sys.exit(1)

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

clf = data.get("classification", {})
lat = data.get("latency", {})
fus = data.get("fusion", {})
rag = data.get("rag", {})
res = data.get("resource", {})
meta = data.get("metadata", {})

# ── Helper functions ──────────────────────────────────────────────────────
def fmt_pct(v):
    """Format sebagai persentase"""
    if v is None:
        return "N/A"
    return f"{v*100:.2f}%"

def fmt_ms(v):
    """Format sebagai milidetik"""
    if v is None:
        return "N/A"
    return f"{v:.1f}ms"

def fmt_num(v):
    """Format sebagai angka"""
    if v is None:
        return "N/A"
    return str(v)

# ── Build summary ─────────────────────────────────────────────────────────
summary_lines = []
summary_lines.append("=" * 70)
summary_lines.append("  FIRA BOT — OVERALL PERFORMANCE SUMMARY")
summary_lines.append("=" * 70)
summary_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
summary_lines.append(f"  Dataset: {meta.get('dataset_path', 'N/A')}")
summary_lines.append(f"  Test Samples: {meta.get('test_samples', 'N/A')}")
summary_lines.append(f"  Total Intents: {meta.get('intent_count', 'N/A')}")
summary_lines.append("")

# Classification Performance Table
summary_lines.append("CLASSIFICATION PERFORMANCE")
summary_lines.append("-" * 70)
summary_lines.append("<10"    "<10"    "<12"    "<12"    "<12"    "<12")
summary_lines.append("-" * 70)

models = ["lstm", "bert", "hybrid"]
for model in models:
    if model in clf:
        m = clf[model]
        summary_lines.append("<10"            "<10"            "<12"            "<12"            "<12"            "<12")
    else:
        summary_lines.append("<10"            "<10"            "<12"            "<12"            "<12"            "<12")

summary_lines.append("")

# Response Time Table
summary_lines.append("RESPONSE TIME (LATENCY)")
summary_lines.append("-" * 70)
summary_lines.append("<10"    "<10"    "<10"    "<10"    "<10"    "<10")
summary_lines.append("-" * 70)

for model in models:
    if model in lat:
        l = lat[model]
        summary_lines.append("<10"            "<10"            "<10"            "<10"            "<10"            "<10")
    else:
        summary_lines.append("<10"            "<10"            "<10"            "<10"            "<10"            "<10")

summary_lines.append("")

# Fusion Analysis
if fus:
    summary_lines.append("HYBRID FUSION ANALYSIS")
    summary_lines.append("-" * 70)
    summary_lines.append(f"  Agreement Rate (LSTM vs BERT):     {fmt_pct(fus.get('agreement_rate'))}")
    summary_lines.append(f"  Disagreement Rate:                 {fmt_pct(fus.get('disagreement_rate'))}")
    summary_lines.append(f"  LSTM Weight:                       {fus.get('lstm_weight', 'N/A')}")
    summary_lines.append(f"  BERT Weight:                       {fus.get('bert_weight', 'N/A')}")
    summary_lines.append("")

# RAG Status
if rag:
    summary_lines.append("RAG (RETRIEVAL-AUGMENTED GENERATION)")
    summary_lines.append("-" * 70)
    summary_lines.append(f"  Enabled:                           {rag.get('enabled', False)}")
    if rag.get('enabled'):
        summary_lines.append(f"  Index Size:                        {rag.get('index_size', 'N/A')}")
        summary_lines.append(f"  Avg Retrieval Time:                {fmt_ms(rag.get('avg_retrieval_ms'))}")
        summary_lines.append(f"  P95 Retrieval Time:                {fmt_ms(rag.get('p95_retrieval_ms'))}")
        summary_lines.append(f"  Avg Similarity Score:              {fmt_pct(rag.get('avg_similarity_score'))}")
    else:
        summary_lines.append(f"  Reason:                            {rag.get('reason', 'N/A')}")
    summary_lines.append("")

# Resource Usage
if res:
    summary_lines.append("RESOURCE USAGE (DURING EVALUATION)")
    summary_lines.append("-" * 70)
    summary_lines.append(f"  RAM Usage:                         {fmt_num(res.get('ram_mb'))} MB")
    summary_lines.append(f"  CPU Usage:                         {fmt_pct(res.get('cpu_percent'))}")
    summary_lines.append(f"  Total RAM:                         {fmt_num(res.get('total_ram_gb'))} GB")
    summary_lines.append("")

# Recommendations
summary_lines.append("PERFORMANCE INSIGHTS & RECOMMENDATIONS")
summary_lines.append("-" * 70)

# Find best model
best_acc = max([(model, clf[model].get('accuracy', 0)) for model in models if model in clf], key=lambda x: x[1], default=(None, 0))
best_lat = min([(model, lat[model].get('avg_ms', float('inf'))) for model in models if model in lat], key=lambda x: x[1], default=(None, float('inf')))

if best_acc[0]:
    summary_lines.append(f"  • Best Accuracy: {best_acc[0].upper()} ({fmt_pct(best_acc[1])})")
if best_lat[0]:
    summary_lines.append(f"  • Fastest Response: {best_lat[0].upper()} ({fmt_ms(best_lat[1])})")

# General recommendations
if fus.get('agreement_rate', 0) < 0.8:
    summary_lines.append("  • Consider improving model agreement (currently < 80%)")
if any(lat[model].get('avg_ms', 0) > 500 for model in models if model in lat):
    summary_lines.append("  • High latency detected (> 500ms) - consider optimization")
if any(clf[model].get('accuracy', 0) < 0.7 for model in models if model in clf):
    summary_lines.append("  • Some models have low accuracy (< 70%) - review training data")

summary_lines.append("=" * 70)

# ── Output ─────────────────────────────────────────────────────────────────
summary_text = "\n".join(summary_lines)

if args.output:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary saved to: {output_path}")
else:
    print(summary_text)