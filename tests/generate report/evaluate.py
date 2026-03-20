"""
evaluate.py — Fira Bot Performance Evaluator
=============================================
Mengevaluasi LSTM, BERT, dan LSTM+BERT (hybrid) dari model yang sudah ada.
Output: evaluation_results.json (dibaca oleh generate_report.py)

Usage:
    python3 evaluate.py
    python3 evaluate.py --config .env --output results/evaluation_results.json
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Argument parser ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fira Bot Evaluator")
parser.add_argument("--config",  default=".env",                         help="Path ke file .env")
parser.add_argument("--output",  default="evaluation_results.json",      help="Path output JSON")
parser.add_argument("--split",   default="0.2", type=float,              help="Test split ratio (default 0.2)")
parser.add_argument("--samples", default="200", type=int,                help="Jumlah sampel untuk latency test")
args = parser.parse_args()

load_dotenv(args.config)

# ── Path dari .env ─────────────────────────────────────────────────────────
DATASET_PATH        = os.getenv("DATASET_PATH",        "/media/aas/New Volume1/bot/New folder/data/dataset/bert/dataset_training_bert.csv")
LSTM_MODEL_PATH     = os.getenv("LSTM_MODEL_PATH",     "/media/aas/New Volume1/bot/New folder/data/lstm_models/chatbot_model.h5")
LSTM_TOKENIZER_PATH = os.getenv("LSTM_TOKENIZER_PATH", "/media/aas/New Volume1/bot/New folder/data/lstm_models/tokenizer.pkl")
LSTM_LABEL_ENC_PATH = os.getenv("LSTM_LABEL_ENCODER_PATH", "/media/aas/New Volume1/bot/New folder/data/lstm_models/label_encoder.pkl")
BERT_MODEL_PATH     = os.getenv("BERT_MODEL_PATH",     "/media/aas/New Volume1/bot/New folder/data/bert_model")
FAISS_INDEX_PATH    = os.getenv("FAISS_INDEX_PATH",    "/media/aas/New Volume1/bot/New folder/services/rag/faiss.index")
FAISS_META_PATH     = os.getenv("FAISS_METADATA_PATH", "/media/aas/New Volume1/bot/New folder/services/rag/metadata.pkl")
USE_RAG             = os.getenv("USE_RAG", "false").lower() == "true"

OUTPUT_PATH = Path(args.output)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Fira Bot — Performance Evaluator")
print(f"  Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

results = {
    "metadata": {
        "generated_at": datetime.now().isoformat(),
        "dataset_path": DATASET_PATH,
        "test_split": args.split,
    },
    "classification": {},
    "latency": {},
    "rag": {},
    "fusion": {},
    "resource": {},
}

# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load dataset & split
# ══════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading dataset...")

try:
    df = pd.read_csv(DATASET_PATH)
    # Deteksi nama kolom otomatis
    text_col   = next((c for c in df.columns if c.lower() in ["text","question","input","message","utterance"]), df.columns[0])
    intent_col = next((c for c in df.columns if c.lower() in ["intent","label","category","tag"]),               df.columns[1])
    df = df[[text_col, intent_col]].dropna()
    df.columns = ["text", "intent"]
    print(f"  Dataset: {len(df)} rows, {df['intent'].nunique()} intents")
except Exception as e:
    print(f"  ERROR loading dataset: {e}")
    sys.exit(1)

from sklearn.model_selection import train_test_split

# Check if stratification is possible (all classes must have at least 2 members)
min_class_count = df["intent"].value_counts().min()
if min_class_count < 2:
    print(f"  WARNING: Some classes have only {min_class_count} member(s). Disabling stratification.")
    stratify_param = None
else:
    stratify_param = df["intent"]

_, df_test = train_test_split(df, test_size=args.split, random_state=42, stratify=stratify_param)
X_test = df_test["text"].tolist()
y_true = df_test["intent"].tolist()
print(f"  Test set: {len(X_test)} samples")
results["metadata"]["test_samples"] = len(X_test)
results["metadata"]["intent_count"] = df["intent"].nunique()
results["metadata"]["intents"] = sorted(df["intent"].unique().tolist())

# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load LSTM model
# ══════════════════════════════════════════════════════════════════════════
print("\n[2/6] Loading LSTM model...")

lstm_loaded = False
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    lstm_model = load_model(LSTM_MODEL_PATH)
    with open(LSTM_TOKENIZER_PATH, "rb") as f:
        lstm_tokenizer = pickle.load(f)
    with open(LSTM_LABEL_ENC_PATH, "rb") as f:
        lstm_label_enc = pickle.load(f)

    MAX_LEN = lstm_model.input_shape[1]

    # ── Deteksi format tokenizer ──────────────────────────────────
    # Format 1: Keras Tokenizer object  → punya .texts_to_sequences()
    # Format 2: dict word→index         → {"word": idx, ...}
    # Format 3: dict dengan key "word_index" → {"word_index": {"word": idx}}
    if isinstance(lstm_tokenizer, dict):
        word_index = lstm_tokenizer.get("word_index", lstm_tokenizer)
        print(f"  Tokenizer: dict format, vocab size={len(word_index)}")

        def _tokenize(texts):
            result = []
            for text in texts:
                tokens = str(text).lower().split()
                seq = [word_index.get(t, 0) for t in tokens]
                result.append(seq)
            return result
        lstm_tokenizer_fn = _tokenize
    else:
        # Keras Tokenizer
        print(f"  Tokenizer: Keras format, vocab size={len(lstm_tokenizer.word_index)}")
        lstm_tokenizer_fn = lstm_tokenizer.texts_to_sequences

    lstm_loaded = True
    print(f"  LSTM model loaded. Input shape: {lstm_model.input_shape}")
except Exception as e:
    print(f"  WARNING: LSTM tidak bisa diload — {e}")

def predict_lstm_batch(texts):
    seqs   = lstm_tokenizer_fn(texts)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs  = lstm_model.predict(padded, verbose=0)
    preds  = lstm_label_enc.inverse_transform(np.argmax(probs, axis=1))
    confs  = np.max(probs, axis=1)
    return preds.tolist(), confs.tolist(), probs

# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — Load BERT model
# ══════════════════════════════════════════════════════════════════════════
print("\n[3/6] Loading BERT (IndoBERT) model...")

bert_loaded = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    bert_model_hf  = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_model_hf.eval()

    # Label mapping dari model config
    bert_id2label = bert_model_hf.config.id2label
    bert_loaded = True
    print(f"  BERT model loaded. Labels: {len(bert_id2label)}")
except Exception as e:
    print(f"  WARNING: BERT tidak bisa diload — {e}")

def predict_bert_batch(texts, batch_size=16):
    all_preds, all_confs, all_probs = [], [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = bert_tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=128)
        with torch.no_grad():
            logits = bert_model_hf(**enc).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = [bert_id2label[idx] for idx in np.argmax(probs, axis=1)]
        confs = np.max(probs, axis=1).tolist()
        all_preds.extend(preds)
        all_confs.extend(confs)
        all_probs.append(probs)
    return all_preds, all_confs, np.vstack(all_probs)

# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — Evaluate classification metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n[4/6] Evaluating classification metrics...")

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

def compute_metrics(y_true, y_pred, label):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    print(f"  {label}: acc={acc:.4f} | f1={f1:.4f} | prec={prec:.4f} | rec={rec:.4f}")
    return {
        "accuracy":  round(acc, 4),
        "f1_weighted": round(f1, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "per_class": {k: v for k, v in report.items() if k not in ["accuracy","macro avg","weighted avg"]},
        "confusion_matrix": cm.tolist(),
        "cm_labels": sorted(set(y_true)),
    }

# — LSTM
if lstm_loaded:
    lstm_preds, lstm_confs, lstm_probs = predict_lstm_batch(X_test)
    results["classification"]["lstm"] = compute_metrics(y_true, lstm_preds, "LSTM")
    results["classification"]["lstm"]["avg_confidence"] = round(float(np.mean(lstm_confs)), 4)

# — BERT
if bert_loaded:
    bert_preds, bert_confs, bert_probs = predict_bert_batch(X_test)
    results["classification"]["bert"] = compute_metrics(y_true, bert_preds, "BERT")
    results["classification"]["bert"]["avg_confidence"] = round(float(np.mean(bert_confs)), 4)

# — Hybrid fusion (LSTM + BERT weighted average)
if lstm_loaded and bert_loaded:
    LSTM_WEIGHT = 0.4
    BERT_WEIGHT = 0.6

    # Align label space
    all_labels = sorted(set(y_true))
    label2idx  = {l: i for i, l in enumerate(all_labels)}

    def align_probs(probs, model_labels, target_labels):
        aligned = np.zeros((len(probs), len(target_labels)))
        for i, lbl in enumerate(model_labels):
            if lbl in label2idx:
                aligned[:, label2idx[lbl]] += probs[:, i]
        return aligned

    lstm_labels = lstm_label_enc.classes_.tolist()
    bert_labels = [bert_id2label[i] for i in range(len(bert_id2label))]

    lstm_aligned = align_probs(lstm_probs, lstm_labels, all_labels)
    bert_aligned = align_probs(bert_probs, bert_labels, all_labels)
    hybrid_probs = LSTM_WEIGHT * lstm_aligned + BERT_WEIGHT * bert_aligned
    hybrid_preds = [all_labels[i] for i in np.argmax(hybrid_probs, axis=1)]
    hybrid_confs = np.max(hybrid_probs, axis=1)

    results["classification"]["hybrid"] = compute_metrics(y_true, hybrid_preds, "Hybrid (LSTM+BERT)")
    results["classification"]["hybrid"]["avg_confidence"] = round(float(np.mean(hybrid_confs)), 4)
    results["classification"]["hybrid"]["lstm_weight"] = LSTM_WEIGHT
    results["classification"]["hybrid"]["bert_weight"] = BERT_WEIGHT

    # Agreement rate
    agree  = sum(l == b for l, b in zip(lstm_preds, bert_preds))
    results["fusion"]["agreement_rate"]     = round(agree / len(X_test), 4)
    results["fusion"]["disagreement_rate"]  = round(1 - agree / len(X_test), 4)
    results["fusion"]["lstm_weight"]        = LSTM_WEIGHT
    results["fusion"]["bert_weight"]        = BERT_WEIGHT

# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — Latency measurement
# ══════════════════════════════════════════════════════════════════════════
print("\n[5/6] Measuring latency...")

latency_samples = X_test[:args.samples]

def measure_latency(fn, texts, label, warmup=5):
    # Warmup
    fn(texts[:warmup])
    times = []
    for text in texts:
        t0 = time.perf_counter()
        fn([text])
        times.append((time.perf_counter() - t0) * 1000)  # ms
    arr = np.array(times)
    stats = {
        "avg_ms":    round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms":    round(float(np.percentile(arr, 95)), 2),
        "p99_ms":    round(float(np.percentile(arr, 99)), 2),
        "min_ms":    round(float(arr.min()), 2),
        "max_ms":    round(float(arr.max()), 2),
        "std_ms":    round(float(arr.std()), 2),
        "samples":   len(texts),
    }
    print(f"  {label}: avg={stats['avg_ms']}ms | p95={stats['p95_ms']}ms | p99={stats['p99_ms']}ms")
    return stats

if lstm_loaded:
    results["latency"]["lstm"] = measure_latency(predict_lstm_batch, latency_samples, "LSTM")

if bert_loaded:
    results["latency"]["bert"] = measure_latency(predict_bert_batch, latency_samples, "BERT")

if lstm_loaded and bert_loaded:
    def hybrid_infer(texts):
        lp, _, lprobs = predict_lstm_batch(texts)
        bp, _, bprobs = predict_bert_batch(texts)
        la = align_probs(lprobs, lstm_labels, all_labels)
        ba = align_probs(bprobs, bert_labels, all_labels)
        hp = LSTM_WEIGHT * la + BERT_WEIGHT * ba
        return [all_labels[i] for i in np.argmax(hp, axis=1)]
    results["latency"]["hybrid"] = measure_latency(hybrid_infer, latency_samples, "Hybrid")

# ══════════════════════════════════════════════════════════════════════════
#  STEP 6 — RAG evaluation (jika USE_RAG=true)
# ══════════════════════════════════════════════════════════════════════════
print("\n[6/6] RAG evaluation...")

if USE_RAG:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            metadata = pickle.load(f)

        emb_model = SentenceTransformer("all-MiniLM-L6-v2")

        rag_times = []
        rag_scores = []
        for text in latency_samples[:50]:
            t0 = time.perf_counter()
            emb = emb_model.encode([text])
            D, I = index.search(emb, k=3)
            rag_times.append((time.perf_counter() - t0) * 1000)
            rag_scores.append(float(np.mean(D[0])))

        results["rag"] = {
            "enabled": True,
            "index_size": index.ntotal,
            "avg_retrieval_ms": round(float(np.mean(rag_times)), 2),
            "p95_retrieval_ms": round(float(np.percentile(rag_times, 95)), 2),
            "avg_similarity_score": round(float(np.mean(rag_scores)), 4),
        }
        print(f"  RAG: avg={results['rag']['avg_retrieval_ms']}ms, index_size={results['rag']['index_size']}")
    except Exception as e:
        print(f"  RAG skipped: {e}")
        results["rag"] = {"enabled": False, "reason": str(e)}
else:
    print("  RAG dinonaktifkan (USE_RAG=false)")
    results["rag"] = {"enabled": False, "reason": "USE_RAG=false"}

# ══════════════════════════════════════════════════════════════════════════
#  Resource usage
# ══════════════════════════════════════════════════════════════════════════
try:
    import psutil
    proc = psutil.Process()
    results["resource"]["ram_mb"]       = round(proc.memory_info().rss / 1e6, 1)
    results["resource"]["cpu_percent"]  = round(psutil.cpu_percent(interval=1), 1)
    results["resource"]["total_ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
except Exception:
    pass

# ── Save results ───────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  Evaluasi selesai!")
print(f"  Output: {OUTPUT_PATH}")
print(f"{'='*60}\n")