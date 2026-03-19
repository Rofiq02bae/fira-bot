"""
Script ingest dokumen ke FAISS vector store.
v3:
- CSV parser disesuaikan dengan format dataset_training_bert.csv
  (kolom: intent, pattern, response_type, is_master, response)
- Hanya ingest baris yang eligible untuk RAG runtime:
  response_type = static dan is_master = false
- Response JSON diubah menjadi plain text agar retrieval + LLM prompt lebih natural.

Cara pakai:
  python scripts/ingest_documents_rag.py \
    --input data/dataset/bert/dataset_training_bert.csv \
    --output-index data/rag/faiss.index \
    --output-metadata data/rag/metadata.pkl \
    --text-column response

  # Atau dari file knowledge base JSON
  python scripts/ingest_documents_rag.py \
    --input data/knowledge_base.json \
    --output-index data/rag/faiss.index \
    --output-metadata data/rag/metadata.pkl
"""

import argparse
import logging
import pickle
import json
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_embedding_model(model_name: str = "indobenchmark/indobert-base-p1"):
    """Load sentence-transformers atau IndoBERT untuk embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        logger.info(f"✅ Loaded embedding model via sentence-transformers: {model_name}")
        return model, "sentence_transformers"
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        logger.info(f"✅ Loaded embedding model via transformers: {model_name}")
        return (tokenizer, model), "transformers"
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise


def encode_texts(texts, model, model_type):
    """Encode list of texts menjadi embedding matrix."""
    if model_type == "sentence_transformers":
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    elif model_type == "transformers":
        import torch
        tokenizer, bert_model = model
        all_embeddings = []

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            all_embeddings.append(embedding.astype(np.float32))

        return np.stack(all_embeddings)

    raise ValueError(f"Unknown model_type: {model_type}")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings agar scoring cosine konsisten di FAISS L2."""
    if embeddings.size == 0:
        return embeddings.astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (embeddings / norms).astype(np.float32)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _response_json_to_text(raw_response: Any) -> str:
    """Konversi response JSON-string dari dataset_training_bert.csv ke plain text."""
    text = str(raw_response).strip()
    if not text or text.lower() == "nan":
        return ""

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            response_type = str(data.get("type", "")).strip().lower()

            if response_type == "text":
                body = str(data.get("body", "")).strip()
                return body

            if response_type == "list":
                title = str(data.get("title", "")).strip()
                items = [str(item).strip() for item in data.get("items", []) if str(item).strip()]
                if items:
                    bullet_items = "\n".join(f"- {item}" for item in items)
                    return f"{title}\n{bullet_items}".strip() if title else bullet_items
                return title

            # Fallback untuk format JSON lain
            body = str(data.get("body", "")).strip()
            if body:
                return body
            return text
    except Exception:
        return text

    return text


def load_documents_from_csv(csv_path: str, text_column: str = "response", embedding_model_name: str = "indobenchmark/indobert-base-p1") -> list:
    """
    Load dokumen dari CSV dataset_training_bert.csv.
    Hanya ingest baris:
      - response_type == 'static'
      - is_master == false
    """
    df = pd.read_csv(csv_path, encoding="utf-8", keep_default_na=False)
    logger.info(f"📊 CSV loaded: {len(df)} rows")

    # Normalisasi nama kolom
    df.columns = [str(c).strip() for c in df.columns]
    required = {"intent", "pattern", "response_type", "is_master", text_column}
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.error(f"❌ CSV missing required columns: {missing}")
        return []

    # Buang baris header yang kadang ikut kebaca sebagai data
    before_header_cleanup = len(df)
    df = df[
        (df["intent"].astype(str).str.strip().str.lower() != "intent")
        & (df["response_type"].astype(str).str.strip().str.lower() != "response_type")
    ]
    if len(df) != before_header_cleanup:
        logger.info(f"   Cleanup duplicated-header rows: {before_header_cleanup} → {len(df)}")

    # Filter eligibility RAG runtime: static + non-master
    before_filter = len(df)
    df["response_type_norm"] = df["response_type"].astype(str).str.strip().str.lower()
    df["is_master_norm"] = df["is_master"].apply(_parse_bool)
    df = df[(df["response_type_norm"] == "static") & (~df["is_master_norm"])]
    logger.info(f"   Filter eligible RAG docs (static & non-master): {before_filter} → {len(df)} rows")

    if df.empty:
        logger.warning("⚠️ Tidak ada dokumen eligible untuk RAG")
        return []

    documents: List[Dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        response_raw = row.get(text_column, "")
        response_text = _response_json_to_text(response_raw)
        if not response_text:
            continue

        intent = str(row.get("intent", "")).strip()
        pattern = str(row.get("pattern", "")).strip()
        response_type = str(row.get("response_type_norm", "static")).strip().lower()
        is_master = bool(row.get("is_master_norm", False))

        # Embed gabungan query-example + jawaban agar retrieval lebih relevan
        doc_text = f"Intent: {intent}\nPattern: {pattern}\nJawaban: {response_text}".strip()

        documents.append({
            "document": doc_text,
            "intent": intent,
            "pattern": pattern,
            "response": response_text,
            "response_type": response_type,
            "is_master": is_master,
            "kb_id": intent,  # dipakai untuk filter RAG berdasarkan intent terprediksi
            "domain": "dataset_static_non_master",
            "metadata": {
                "source": os.path.basename(csv_path),
                "row": str(row_idx),
                "response_type": response_type,
                "is_master": is_master,
                "embedding_model": embedding_model_name,
                "embedding_normalized": True
            },
        })

    logger.info(f"✅ {len(documents)} dokumen siap diindex")
    return documents


def load_documents_from_json(json_path: str, embedding_model_name: str = "indobenchmark/indobert-base-p1") -> list:
    """
    Load dokumen dari file knowledge base JSON.
    Format yang diharapkan:
    [
      {
        "kb_id": "sop_splp_pengajuan_api",
        "title": "SOP PENGAJUAN API DARI SPLP",
        "content": "...",
        "domain": "sop_splp"
      },
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        kb_id   = str(item.get("kb_id", "")).strip()
        title   = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        domain  = str(item.get("domain", "")).strip()

        if not content:
            continue

        # Teks yang di-embed adalah gabungan title + content untuk retrieval lebih baik
        embed_text = f"{title}\n{content}" if title else content

        documents.append({
            "document": embed_text,
            "intent":   kb_id,
            "pattern":  title,
            "response": content,
            "domain":   domain,
            "kb_id":    kb_id,
            "metadata": {
                "source": os.path.basename(json_path),
                "title":  title,
                "embedding_model": embedding_model_name,
                "embedding_normalized": True
            }
        })

    logger.info(f"✅ {len(documents)} dokumen dari JSON siap diindex")
    return documents


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index dari embedding matrix."""
    import faiss

    dimension = embeddings.shape[1]
    logger.info(f"📐 Embedding dimension: {dimension}")

    # Gunakan IndexFlatL2 untuk akurasi maksimal
    # Untuk dataset besar (>100k), pertimbangkan IndexIVFFlat
    index = faiss.IndexFlatL2(dimension)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    getattr(index, "add")(embeddings)

    logger.info(f"✅ FAISS index built: {index.ntotal} vectors")
    return index


def main():
    parser = argparse.ArgumentParser(description="Ingest documents ke FAISS vector store")
    parser.add_argument("--input",           required=True, help="Path ke CSV atau JSON dokumen")
    parser.add_argument("--output-index",    required=True, help="Path output FAISS index (.index)")
    parser.add_argument("--output-metadata", required=True, help="Path output metadata (.pkl)")
    parser.add_argument("--text-column",     default="response", help="Kolom teks di CSV (default: response)")
    parser.add_argument("--batch-size",      type=int, default=32, help="Batch size encoding")
    parser.add_argument("--embedding-model", default=os.getenv("RAG_EMBEDDING_MODEL", "indobenchmark/indobert-base-p1"), help="Embedding model name (default: env RAG_EMBEDDING_MODEL)")
    args = parser.parse_args()

    # Pastikan output directory ada
    Path(args.output_index).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metadata).parent.mkdir(parents=True, exist_ok=True)

    # Load dokumen
    input_path = args.input
    if input_path.endswith(".json"):
        documents = load_documents_from_json(input_path, args.embedding_model)
    else:
        documents = load_documents_from_csv(input_path, args.text_column, args.embedding_model)

    if not documents:
        logger.error("❌ Tidak ada dokumen untuk diindex. Selesai.")
        return

    # Load embedding model
    model, model_type = load_embedding_model(args.embedding_model)

    # Encode semua teks
    texts = [doc["document"] for doc in documents]
    logger.info(f"🔢 Encoding {len(texts)} dokumen...")
    embeddings = encode_texts(texts, model, model_type)
    embeddings = normalize_embeddings(embeddings)

    # Build FAISS index
    import faiss
    index = build_faiss_index(embeddings)

    # Simpan index
    faiss.write_index(index, args.output_index)
    logger.info(f"💾 FAISS index disimpan: {args.output_index}")

    # Simpan metadata
    with open(args.output_metadata, "wb") as f:
        pickle.dump(documents, f)
    logger.info(f"💾 Metadata disimpan: {args.output_metadata}")

    # Ringkasan
    domains = {}
    kb_ids  = set()
    for doc in documents:
        d = doc.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
        kb_ids.add(doc.get("kb_id", ""))

    print("\n── Ringkasan Ingest ──")
    print(f"Total dokumen : {len(documents)}")
    print(f"Unique kb_id  : {len(kb_ids)}")
    print("Per domain    :")
    for d, c in sorted(domains.items()):
        print(f"  {d:30s}: {c} dokumen")


if __name__ == "__main__":
    main()
