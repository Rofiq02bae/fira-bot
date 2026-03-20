"""
generate_report.py — Fira Bot Report Generator
===============================================
Membaca evaluation_results.json, menghasilkan:
  - charts/  (PNG charts)
  - fira_bot_performance_report.docx

Usage:
    python3 generate_report.py
    python3 generate_report.py --input evaluation_results.json --out-dir reports/
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",   default="evaluation_results.json")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

INPUT_PATH = Path(args.input)
OUT_DIR    = Path(args.out_dir)
CHART_DIR  = OUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_PATH.exists():
    print(f"ERROR: {INPUT_PATH} tidak ditemukan. Jalankan evaluate.py terlebih dahulu.")
    sys.exit(1)

with open(INPUT_PATH, encoding="utf-8") as f:
    data = json.load(f)

clf  = data.get("classification", {})
lat  = data.get("latency", {})
rag  = data.get("rag", {})
fus  = data.get("fusion", {})
res  = data.get("resource", {})
meta = data.get("metadata", {})

COLORS = {
    "lstm":   "#5B8DEF",
    "bert":   "#3CC88C",
    "hybrid": "#F5A623",
    "rag":    "#9B59B6",
}
STYLE = {
    "font":      "DejaVu Sans",
    "bg":        "#FAFAFA",
    "grid":      "#E5E5E5",
    "text":      "#2C2C2A",
    "subtext":   "#6B6B68",
    "header_bg": "#1A3A5C",
}

plt.rcParams.update({
    "font.family": STYLE["font"],
    "axes.facecolor": STYLE["bg"],
    "figure.facecolor": "white",
    "axes.edgecolor": STYLE["grid"],
    "axes.grid": True,
    "grid.color": STYLE["grid"],
    "grid.linewidth": 0.6,
    "text.color": STYLE["text"],
    "xtick.color": STYLE["subtext"],
    "ytick.color": STYLE["subtext"],
})

PNG_PATHS = {}

# ══════════════════════════════════════════════════════════════════════════
#  CHART 1 — Model comparison bar chart (accuracy, F1, precision, recall)
# ══════════════════════════════════════════════════════════════════════════
def chart_model_comparison():
    models   = [k for k in ["lstm","bert","hybrid"] if k in clf]
    metrics  = ["accuracy","f1_weighted","precision","recall"]
    labels   = ["Accuracy","F1 (weighted)","Precision","Recall"]

    if not models:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.25
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    for i, model in enumerate(models):
        vals = [clf[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model.upper(),
                      color=COLORS[model], alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=STYLE["text"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Classification Performance Comparison", fontsize=13, fontweight="bold",
                 color=STYLE["text"], pad=14)
    ax.legend(fontsize=10, framealpha=0.7)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()

    path = CHART_DIR / "01_model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ══════════════════════════════════════════════════════════════════════════
#  CHART 2 — Response time comparison
# ══════════════════════════════════════════════════════════════════════════
def chart_latency():
    models = [k for k in ["lstm","bert","hybrid"] if k in lat]
    if not models:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: avg/p95/p99 grouped bar
    ax = axes[0]
    lat_metrics = ["avg_ms", "p95_ms", "p99_ms"]
    lat_labels  = ["Avg", "P95", "P99"]
    x = np.arange(len(lat_labels))
    width = 0.25
    offsets = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * width

    for i, model in enumerate(models):
        vals = [lat[model].get(m, 0) for m in lat_metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model.upper(),
                      color=COLORS[model], alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(lat_labels, fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Response Time per Model", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.7)
    ax.set_axisbelow(True)

    # Right: horizontal bar avg only
    ax2 = axes[1]
    avgs   = [lat[m]["avg_ms"] for m in models]
    colors = [COLORS[m] for m in models]
    bars   = ax2.barh([m.upper() for m in models], avgs, color=colors, alpha=0.88, zorder=3)
    for bar, val in zip(bars, avgs):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f} ms", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Avg Response Time (ms)", fontsize=11)
    ax2.set_title("Avg Latency Comparison", fontsize=12, fontweight="bold")
    ax2.set_axisbelow(True)

    plt.suptitle("Latency Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = CHART_DIR / "02_latency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ══════════════════════════════════════════════════════════════════════════
#  CHART 3 — Per-class F1 heatmap table
# ══════════════════════════════════════════════════════════════════════════
def chart_per_class():
    models_avail = [k for k in ["lstm","bert","hybrid"] if k in clf and "per_class" in clf[k]]
    if not models_avail:
        return None

    # Collect all intents
    all_intents = set()
    for m in models_avail:
        all_intents.update(clf[m]["per_class"].keys())
    all_intents = sorted(all_intents - {"accuracy","macro avg","weighted avg"})

    if not all_intents:
        return None

    data_matrix = np.zeros((len(all_intents), len(models_avail)))
    for j, model in enumerate(models_avail):
        for i, intent in enumerate(all_intents):
            data_matrix[i, j] = clf[model]["per_class"].get(intent, {}).get("f1-score", 0)

    fig_h = max(5, len(all_intents) * 0.38 + 2)
    fig, ax = plt.subplots(figsize=(7, fig_h))
    im = ax.imshow(data_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(models_avail)))
    ax.set_xticklabels([m.upper() for m in models_avail], fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(all_intents)))
    ax.set_yticklabels(all_intents, fontsize=9)

    for i in range(len(all_intents)):
        for j in range(len(models_avail)):
            val = data_matrix[i, j]
            color = "white" if val < 0.5 else "#1A1A1A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("F1-score", fontsize=10)
    ax.set_title("Per-Intent F1-Score Heatmap", fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()

    path = CHART_DIR / "03_per_class_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ══════════════════════════════════════════════════════════════════════════
#  CHART 4 — Radar chart overall
# ══════════════════════════════════════════════════════════════════════════
def chart_radar():
    models = [k for k in ["lstm","bert","hybrid"] if k in clf]
    if not models:
        return None

    cats   = ["Accuracy","F1","Precision","Recall","Confidence"]
    keys   = ["accuracy","f1_weighted","precision","recall","avg_confidence"]
    N      = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for model in models:
        vals = [clf[model].get(k, 0) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COLORS[model], label=model.upper())
        ax.fill(angles, vals, color=COLORS[model], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8, color=STYLE["subtext"])
    ax.set_title("Overall Model Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
    plt.tight_layout()

    path = CHART_DIR / "04_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

# ══════════════════════════════════════════════════════════════════════════
#  CHART 5 — Confusion matrix (hybrid only)
# ══════════════════════════════════════════════════════════════════════════
def chart_confusion():
    model = "hybrid" if "hybrid" in clf else ("bert" if "bert" in clf else "lstm")
    if model not in clf or "confusion_matrix" not in clf[model]:
        return None

    cm     = np.array(clf[model]["confusion_matrix"])
    labels = clf[model].get("cm_labels", [str(i) for i in range(cm.shape[0])])

    # Normalize
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig_size = max(7, len(labels) * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model.upper()} (normalized)", fontsize=12, fontweight="bold")

    thresh = 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > thresh else STYLE["text"]
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = CHART_DIR / "05_confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path

print("[1/3] Generating charts...")
PNG_PATHS["comparison"] = chart_model_comparison()
PNG_PATHS["latency"]    = chart_latency()
PNG_PATHS["per_class"]  = chart_per_class()
PNG_PATHS["radar"]      = chart_radar()
PNG_PATHS["confusion"]  = chart_confusion()

for name, path in PNG_PATHS.items():
    if path:
        print(f"  {name}: {path}")
    else:
        print(f"  {name}: skipped (data tidak tersedia)")

# ══════════════════════════════════════════════════════════════════════════
#  DOCX REPORT via docx-js (Node.js)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/3] Generating DOCX report...")

def fmt(v, pct=True):
    if v is None:
        return "N/A"
    return f"{v*100:.2f}%" if pct else str(v)

def ms(v):
    if v is None:
        return "N/A"
    return f"{v} ms"

# Build summary table rows for JS
def model_table_rows():
    rows = []
    for model in ["lstm","bert","hybrid"]:
        if model not in clf:
            continue
        m = clf[model]
        rows.append({
            "model":      model.upper(),
            "accuracy":   fmt(m.get("accuracy")),
            "f1":         fmt(m.get("f1_weighted")),
            "precision":  fmt(m.get("precision")),
            "recall":     fmt(m.get("recall")),
            "confidence": fmt(m.get("avg_confidence")),
        })
    return rows

def latency_table_rows():
    rows = []
    for model in ["lstm","bert","hybrid"]:
        if model not in lat:
            continue
        l = lat[model]
        rows.append({
            "model":   model.upper(),
            "avg":     ms(l.get("avg_ms")),
            "median":  ms(l.get("median_ms")),
            "p95":     ms(l.get("p95_ms")),
            "p99":     ms(l.get("p99_ms")),
            "samples": str(l.get("samples","-")),
        })
    return rows

clf_rows = model_table_rows()
lat_rows = latency_table_rows()

# PNG as base64 for embedding
import base64

def png_b64(path):
    if path and Path(path).exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

imgs = {k: png_b64(v) for k, v in PNG_PATHS.items()}

# Build JS script
report_date = datetime.now().strftime("%B %d, %Y")
docx_out    = OUT_DIR / "fira_bot_performance_report.docx"

js_rows_clf = json.dumps(clf_rows)
js_rows_lat = json.dumps(lat_rows)
js_imgs     = json.dumps({k: v for k, v in imgs.items() if v})

js_script = f"""
const fs = require('fs');
const path = require('path');
const {{
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, LevelFormat,
}} = require('docx');

const REPORT_DATE  = "{report_date}";
const CLF_ROWS     = {js_rows_clf};
const LAT_ROWS     = {js_rows_lat};
const IMGS         = {js_imgs};

// ── Design tokens ──────────────────────────────────────────────
const C_PRIMARY    = "1A3A5C";
const C_ACCENT     = "2E75B6";
const C_HEADER_BG  = "1A3A5C";
const C_ROW_ALT    = "EAF3FB";
const C_ROW_WHITE  = "FFFFFF";
const C_BORDER     = "CCCCCC";

// ── Helpers ────────────────────────────────────────────────────
const border = (color = C_BORDER) => ({{
  style: BorderStyle.SINGLE, size: 1, color,
}});
const cellBorders = {{
  top: border(), bottom: border(), left: border(), right: border(),
}};
const cellMargins = {{ top: 80, bottom: 80, left: 120, right: 120 }};

function hCell(text, widthDxa, bg = C_HEADER_BG) {{
  return new TableCell({{
    width: {{ size: widthDxa, type: WidthType.DXA }},
    shading: {{ fill: bg, type: ShadingType.CLEAR }},
    borders: cellBorders,
    margins: cellMargins,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({{
      alignment: AlignmentType.CENTER,
      children: [new TextRun({{ text, bold: true, color: "FFFFFF", size: 20, font: "Arial" }})],
    }})],
  }});
}}

function dCell(text, widthDxa, alt = false, align = AlignmentType.CENTER) {{
  return new TableCell({{
    width: {{ size: widthDxa, type: WidthType.DXA }},
    shading: {{ fill: alt ? C_ROW_ALT : C_ROW_WHITE, type: ShadingType.CLEAR }},
    borders: cellBorders,
    margins: cellMargins,
    children: [new Paragraph({{
      alignment: align,
      children: [new TextRun({{ text, size: 19, font: "Arial" }})],
    }})],
  }});
}}

function heading1(text) {{
  return new Paragraph({{
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({{ text, font: "Arial", size: 32, bold: true, color: C_PRIMARY }})],
    spacing: {{ before: 360, after: 160 }},
    border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 6, color: C_ACCENT, space: 1 }} }},
  }});
}}

function heading2(text) {{
  return new Paragraph({{
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({{ text, font: "Arial", size: 26, bold: true, color: C_ACCENT }})],
    spacing: {{ before: 280, after: 120 }},
  }});
}}

function body(text) {{
  return new Paragraph({{
    children: [new TextRun({{ text, size: 22, font: "Arial", color: "2C2C2A" }})],
    spacing: {{ after: 100 }},
  }});
}}

function spacer() {{
  return new Paragraph({{ children: [new TextRun("")], spacing: {{ after: 120 }} }});
}}

function imgPara(b64Key, widthEMU = 5400000) {{
  const b64 = IMGS[b64Key];
  if (!b64) return spacer();
  const buf = Buffer.from(b64, 'base64');
  const heightEMU = Math.round(widthEMU * 0.5);
  return new Paragraph({{
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({{ data: buf, transformation: {{ width: Math.round(widthEMU/9144), height: Math.round(heightEMU/9144) }}, type: "png" }})],
    spacing: {{ after: 180 }},
  }});
}}

// ── Classification table ───────────────────────────────────────
const CLF_COLS = [1400, 1500, 1500, 1500, 1500, 1500];
const clfTable = new Table({{
  width: {{ size: 8900, type: WidthType.DXA }},
  columnWidths: CLF_COLS,
  rows: [
    new TableRow({{
      tableHeader: true,
      children: [
        hCell("Model",      CLF_COLS[0]),
        hCell("Accuracy",   CLF_COLS[1]),
        hCell("F1",         CLF_COLS[2]),
        hCell("Precision",  CLF_COLS[3]),
        hCell("Recall",     CLF_COLS[4]),
        hCell("Confidence", CLF_COLS[5]),
      ],
    }}),
    ...CLF_ROWS.map((r, i) => new TableRow({{
      children: [
        dCell(r.model,      CLF_COLS[0], i%2===1, AlignmentType.LEFT),
        dCell(r.accuracy,   CLF_COLS[1], i%2===1),
        dCell(r.f1,         CLF_COLS[2], i%2===1),
        dCell(r.precision,  CLF_COLS[3], i%2===1),
        dCell(r.recall,     CLF_COLS[4], i%2===1),
        dCell(r.confidence, CLF_COLS[5], i%2===1),
      ],
    }})),
  ],
}});

// ── Latency table ──────────────────────────────────────────────
const LAT_COLS = [1400, 1500, 1500, 1500, 1500, 1500];
const latTable = new Table({{
  width: {{ size: 8900, type: WidthType.DXA }},
  columnWidths: LAT_COLS,
  rows: [
    new TableRow({{
      tableHeader: true,
      children: [
        hCell("Model",   LAT_COLS[0]),
        hCell("Avg",     LAT_COLS[1]),
        hCell("Median",  LAT_COLS[2]),
        hCell("P95",     LAT_COLS[3]),
        hCell("P99",     LAT_COLS[4]),
        hCell("Samples", LAT_COLS[5]),
      ],
    }}),
    ...LAT_ROWS.map((r, i) => new TableRow({{
      children: [
        dCell(r.model,   LAT_COLS[0], i%2===1, AlignmentType.LEFT),
        dCell(r.avg,     LAT_COLS[1], i%2===1),
        dCell(r.median,  LAT_COLS[2], i%2===1),
        dCell(r.p95,     LAT_COLS[3], i%2===1),
        dCell(r.p99,     LAT_COLS[4], i%2===1),
        dCell(r.samples, LAT_COLS[5], i%2===1),
      ],
    }})),
  ],
}});

// ── Cover / title page ─────────────────────────────────────────
const coverSection = [
  new Paragraph({{ spacing: {{ before: 2880 }} }}),
  new Paragraph({{
    alignment: AlignmentType.CENTER,
    children: [new TextRun({{ text: "FIRA BOT", font: "Arial", size: 60, bold: true, color: C_PRIMARY }})],
  }}),
  new Paragraph({{
    alignment: AlignmentType.CENTER,
    children: [new TextRun({{ text: "Performance Evaluation Report", font: "Arial", size: 36, color: C_ACCENT }})],
    spacing: {{ after: 200 }},
  }}),
  new Paragraph({{
    alignment: AlignmentType.CENTER,
    children: [new TextRun({{ text: "Hybrid NLU System: LSTM + IndoBERT + RAG", font: "Arial", size: 26, color: "555555" }})],
    spacing: {{ after: 160 }},
  }}),
  new Paragraph({{
    alignment: AlignmentType.CENTER,
    children: [new TextRun({{ text: `Generated: ${{REPORT_DATE}}`, font: "Arial", size: 22, color: "888888", italics: true }})],
  }}),
  new Paragraph({{ spacing: {{ before: 2880 }} }}),
];

// ── Build document children ────────────────────────────────────
const children = [
  ...coverSection,

  // 1. Overview
  heading1("1. Executive Overview"),
  body("Laporan ini menyajikan hasil evaluasi komprehensif terhadap sistem Fira Bot v3.0.0 yang menggunakan arsitektur hybrid NLU (LSTM + IndoBERT) dengan opsi RAG (Retrieval-Augmented Generation). Evaluasi mencakup 8 dimensi utama: klasifikasi intent, hybrid fusion, RAG quality, latency, response quality, error analysis, resource usage, dan A/B comparison."),
  spacer(),

  // 2. Classification
  heading1("2. Classification Performance"),
  heading2("2.1 Overall Metrics per Model"),
  body("Tabel berikut merangkum performa klasifikasi intent dari masing-masing model pada test set."),
  spacer(),
  clfTable,
  spacer(),
  heading2("2.2 Visual Comparison"),
  IMGS.comparison ? imgPara("comparison", 5800000) : body("(Chart tidak tersedia)"),
  heading2("2.3 Radar Chart"),
  IMGS.radar ? imgPara("radar", 4200000) : body("(Radar chart tidak tersedia)"),

  // 3. Hybrid Fusion
  heading1("3. Hybrid Fusion Analysis"),
  body(`Agreement rate antara LSTM dan BERT: ${js_fus_agree * 100).toFixed(2)}% — artinya kedua model sepakat pada prediksi sebesar angka tersebut. Disagreement analysis memberikan insight tentang kasus-kasus di mana model perlu fallback ke RAG.`),
  spacer(),

  // 4. RAG Quality
  heading1("4. RAG Quality"),
  body("RAG (Retrieval-Augmented Generation) diaktifkan untuk meningkatkan akurasi pada intent dengan confidence rendah."),
  spacer(),

  // 5. Latency
  heading1("5. Latency & Response Time"),
  heading2("5.1 Response Time Table"),
  body("Pengukuran latency dilakukan per-sample (single inference) untuk mencerminkan kondisi produksi."),
  spacer(),
  latTable,
  spacer(),
  heading2("5.2 Latency Chart"),
  IMGS.latency ? imgPara("latency", 5800000) : body("(Chart tidak tersedia)"),

  // 6. Per-intent
  heading1("6. Per-Intent Breakdown"),
  body("Heatmap berikut menampilkan F1-score per intent untuk setiap model. Warna merah menandakan performa rendah, hijau tinggi."),
  IMGS.per_class ? imgPara("per_class", 4000000) : body("(Heatmap tidak tersedia)"),

  // 7. Error Analysis
  heading1("7. Error Analysis"),
  heading2("7.1 Confusion Matrix"),
  body("Confusion matrix dinormalisasi per baris (true label). Nilai diagonal menunjukkan recall per kelas."),
  IMGS.confusion ? imgPara("confusion", 4200000) : body("(Confusion matrix tidak tersedia)"),

  // 8. Resource Usage
  heading1("8. Resource Usage"),
  body("Penggunaan resource saat evaluasi berjalan pada host machine."),
  spacer(),

  // 9. Recommendations
  heading1("9. Recommendations"),
  body("Berdasarkan hasil evaluasi, beberapa rekomendasi untuk peningkatan sistem:"),
  body("1. Tingkatkan dataset untuk intent dengan F1-score rendah (< 0.7) yang terlihat pada heatmap per-intent."),
  body("2. Tuning confidence threshold pada fusion layer untuk mengurangi disagreement rate."),
  body("3. Jika latency hybrid melebihi 500ms, pertimbangkan ONNX quantization untuk BERT."),
  body("4. Monitor fallback rate secara berkala untuk mendeteksi drift pada distribusi input."),
  spacer(),
];

// Substitute fusion agreement placeholder
const fusionAgree = {fus.get("agreement_rate", 0)};
const childrenFinal = children.map(c => c);

const doc = new Document({{
  styles: {{
    default: {{ document: {{ run: {{ font: "Arial", size: 22 }} }} }},
    paragraphStyles: [
      {{ id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 32, bold: true, font: "Arial", color: C_PRIMARY }},
         paragraph: {{ spacing: {{ before: 360, after: 160 }}, outlineLevel: 0 }} }},
      {{ id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
         run: {{ size: 26, bold: true, font: "Arial", color: C_ACCENT }},
         paragraph: {{ spacing: {{ before: 280, after: 120 }}, outlineLevel: 1 }} }},
    ],
  }},
  sections: [{{
    properties: {{
      page: {{
        size: {{ width: 11906, height: 16838 }},
        margin: {{ top: 1134, right: 1134, bottom: 1134, left: 1134 }},
      }},
    }},
    headers: {{
      default: new Header({{
        children: [new Paragraph({{
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({{ text: "Fira Bot — Performance Report", size: 18, color: "888888", font: "Arial" }})],
          border: {{ bottom: {{ style: BorderStyle.SINGLE, size: 4, color: C_BORDER, space: 1 }} }},
        }})],
      }}),
    }},
    footers: {{
      default: new Footer({{
        children: [new Paragraph({{
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({{ text: "Page ", size: 18, color: "888888", font: "Arial" }}),
            new TextRun({{ children: [new PageNumber()], size: 18, color: "888888", font: "Arial" }}),
            new TextRun({{ text: `  |  ${{REPORT_DATE}}`, size: 18, color: "888888", font: "Arial" }}),
          ],
          border: {{ top: {{ style: BorderStyle.SINGLE, size: 4, color: C_BORDER, space: 1 }} }},
        }})],
      }}),
    }},
    children: childrenFinal,
  }}],
}});

Packer.toBuffer(doc).then(buf => {{
  fs.writeFileSync("{docx_out}", buf);
  console.log("DOCX saved:", "{docx_out}");
}}).catch(e => {{ console.error(e); process.exit(1); }});
"""

# Substitute fusion agreement value directly
js_script = js_script.replace(
    "{js_fus_agree}",
    str(fus.get("agreement_rate", 0))
)

# Write JS to temp file and run
tmp_js = OUT_DIR / "_report_gen.js"
with open(tmp_js, "w", encoding="utf-8") as f:
    f.write(js_script)

# Install docx if needed
try:
    result = subprocess.run(
        ["node", "-e", "require('docx')"],
        capture_output=True, timeout=10
    )
    if result.returncode != 0:
        raise Exception("docx not found")
except Exception:
    print("  Installing docx npm package...")
    subprocess.run(["npm", "install", "-g", "docx"], check=True, capture_output=True)

result = subprocess.run(
    ["node", str(tmp_js)],
    capture_output=True, text=True, cwd=str(OUT_DIR)
)

tmp_js.unlink(missing_ok=True)

if result.returncode != 0:
    print(f"  ERROR generating DOCX:\n{result.stderr}")
    sys.exit(1)

print(f"  {result.stdout.strip()}")

# ══════════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════════
print(f"\n[3/3] Done!")
print(f"{'='*60}")
print(f"  Charts : {CHART_DIR}/")
for k, p in PNG_PATHS.items():
    if p:
        print(f"           {p.name}")
print(f"  Report : {docx_out}")
print(f"{'='*60}")