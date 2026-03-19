"""
Vector Store implementation using FAISS for semantic search.
Supports loading embeddings and retrieving similar documents.
v2: tambah filter by domain dan kb_id
"""

import logging
import os
import numpy as np
import pickle
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic document retrieval.
    Uses pre-computed embeddings from BERT model.
    """

    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self._load()

    def _load(self):
        """Load FAISS index and metadata from disk."""
        try:
            import faiss

            if not os.path.exists(self.index_path):
                logger.warning(f"⚠️ FAISS index not found at {self.index_path}")
                return False

            if not os.path.exists(self.metadata_path):
                logger.warning(f"⚠️ Metadata file not found at {self.metadata_path}")
                return False

            logger.info(f"📂 Loading FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)

            logger.info(f"📂 Loading metadata from {self.metadata_path}...")
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

            logger.info(f"✅ Vector store loaded: {self.index.ntotal} documents")
            return True

        except ImportError:
            logger.error("❌ FAISS library not installed. Install with: pip install faiss-cpu")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            return False

    def is_available(self) -> bool:
        return self.index is not None and self.metadata is not None

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        similarity_threshold: float = 0.0,
        filter_domain: Optional[str] = None,
        filter_kb_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for similar documents using embedding vector.

        Args:
            query_embedding     : Query embedding vector (1D numpy array)
            k                   : Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            filter_domain       : Opsional — filter hasil hanya dari domain ini
            filter_kb_id        : Opsional — filter hasil hanya dari kb_id ini

        Returns:
            List of document dicts sorted by similarity descending
        """
        if not self.is_available():
            logger.warning("⚠️ Vector store not available")
            return []

        if self.index is None or self.metadata is None:
            logger.warning("⚠️ Vector store components are not initialized")
            return []

        try:
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            query_embedding = query_embedding.astype(np.float32)

            # Ambil lebih banyak kandidat kalau ada filter aktif
            # supaya setelah difilter masih tersisa k hasil
            fetch_k = k * 5 if (filter_domain or filter_kb_id) else k
            fetch_k = min(fetch_k, self.index.ntotal)

            distances, indices = self.index.search(query_embedding, fetch_k)
            use_normalized_l2 = False
            if self.metadata:
                first_meta = self.metadata[0].get("metadata", {}) if isinstance(self.metadata[0], dict) else {}
                use_normalized_l2 = bool(first_meta.get("embedding_normalized", False))

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                doc_metadata = self.metadata[idx]

                # # ── Filter by domain ──
                # if filter_domain:
                #     doc_domain = doc_metadata.get("domain", "")
                #     if doc_domain != filter_domain:
                #         continue

                # # ── Filter by kb_id ──
                # if filter_kb_id:
                #     doc_kb_id = doc_metadata.get("kb_id", "")
                #     if doc_kb_id != filter_kb_id:
                #         continue

                if use_normalized_l2:
                    similarity = max(-1.0, min(1.0, 1.0 - (float(distance) / 2.0)))
                else:
                    # Legacy fallback (non-normalized embeddings)
                    similarity = 1.0 / (1.0 + float(distance))

                if similarity >= similarity_threshold:
                    results.append({
                        "index":    int(idx),
                        "document": doc_metadata.get("document", ""),
                        "intent":   doc_metadata.get("intent", ""),
                        "pattern":  doc_metadata.get("pattern", ""),
                        "response": doc_metadata.get("response", ""),
                        "response_type": doc_metadata.get("response_type", ""),
                        "is_master": doc_metadata.get("is_master", False),
                        "domain":   doc_metadata.get("domain", ""),
                        "kb_id":    doc_metadata.get("kb_id", ""),
                        "similarity": float(similarity),
                        "metadata": doc_metadata.get("metadata", {})
                    })

                    if len(results) >= k:
                        break

            logger.info(
                f"🔍 Search: filter_domain={filter_domain}, filter_kb_id={filter_kb_id} "
                f"→ {len(results)} hasil"
            )
            return sorted(results, key=lambda x: x["similarity"], reverse=True)[:k]

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_stats(self) -> Dict:
        if not self.is_available():
            return {"status": "not_available"}

        if self.index is None:
            return {"status": "not_available"}

        return {
            "status": "available",
            "total_documents": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__
        }
