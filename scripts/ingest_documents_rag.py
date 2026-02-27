#!/usr/bin/env python3
"""
Document ingestion script for RAG.
Loads CSV, generates embeddings, creates FAISS index.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import pickle

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_bert_model(config: ModelConfig):
    """Load BERT model for embeddings."""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        logger.info(f"Loading BERT model from {config.bert_model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(config.bert_model_path)
        # Use AutoModel instead of AutoModelForSequenceClassification for embeddings
        model = AutoModel.from_pretrained(config.bert_model_path, ignore_mismatched_sizes=True)
        model.eval()
        
        logger.info("✅ BERT model loaded")
        return tokenizer, model
    except Exception as e:
        logger.error(f"❌ Failed to load BERT model: {e}")
        return None, None


def generate_embedding(text: str, tokenizer, model) -> np.ndarray:
    """Generate embedding for text using BERT."""
    try:
        import torch
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding.astype(np.float32)
    except Exception as e:
        logger.error(f"❌ Embedding error: {e}")
        return np.zeros(768, dtype=np.float32)


def ingest_documents(
    csv_path: str,
    output_index: str,
    output_metadata: str,
    text_column: str = "response"
):
    """Main ingestion pipeline."""
    try:
        # Load dataset
        logger.info(f"📂 Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path, encoding="utf-8")
        logger.info(f"✅ Loaded {len(df)} rows")
        
        # Validate
        if text_column not in df.columns:
            logger.error(f"❌ Column '{text_column}' not found")
            return False
        
        # Load BERT model
        config = ModelConfig()
        tokenizer, model = load_bert_model(config)
        if tokenizer is None or model is None:
            return False
        
        # Build documents
        documents = []
        embeddings = []
        
        logger.info("🔄 Generating embeddings...")
        for idx, row in df.iterrows():
            doc = {
                "document": str(row.get(text_column, "")),
                "intent": str(row.get("intent", "unknown")),
                "pattern": str(row.get("pattern", "")),
                "response": str(row.get(text_column, "")),
                "metadata": {
                    "source": "csv",
                    "row_index": idx
                }
            }
            documents.append(doc)
            
            # Generate embedding
            embedding = generate_embedding(doc["document"], tokenizer, model)
            embeddings.append(embedding)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"   Processed {idx + 1}/{len(df)}...")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        logger.info(f"✅ Generated {len(embeddings)} embeddings ({embeddings.shape})")
        
        # Create FAISS index
        try:
            import faiss
            
            logger.info("🔄 Creating FAISS index...")
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            # Save index
            os.makedirs(os.path.dirname(output_index), exist_ok=True)
            logger.info(f"💾 Saving index to {output_index}...")
            faiss.write_index(index, output_index)
            
            # Save metadata
            logger.info(f"💾 Saving metadata to {output_metadata}...")
            with open(output_metadata, "wb") as f:
                pickle.dump(documents, f)
            
            logger.info("✅ FAISS index created successfully!")
            logger.info(f"   - Documents: {len(documents)}")
            logger.info(f"   - Dimension: {dimension}")
            logger.info(f"   - Index: {output_index}")
            logger.info(f"   - Metadata: {output_metadata}")
            
            return True
            
        except ImportError:
            logger.error("❌ FAISS not installed. Install with: pip install faiss-cpu")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents for RAG")
    parser.add_argument("--input", default="data/dataset/dataset_training.csv", help="Input CSV file")
    parser.add_argument("--output-index", default="data/rag/faiss.index", help="Output FAISS index")
    parser.add_argument("--output-metadata", default="data/rag/metadata.pkl", help="Output metadata file")
    parser.add_argument("--text-column", default="response", help="Column name for document text")
    
    args = parser.parse_args()
    
    success = ingest_documents(
        csv_path=args.input,
        output_index=args.output_index,
        output_metadata=args.output_metadata,
        text_column=args.text_column
    )
    
    sys.exit(0 if success else 1)
