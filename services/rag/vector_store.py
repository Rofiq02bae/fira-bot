"""
Vector Store implementation using FAISS for semantic search.
Supports loading embeddings and retrieving similar documents.
"""

import logging
import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic document retrieval.
    Uses pre-computed embeddings from BERT model.
    """

    def __init__(self, index_path: str, metadata_path: str):
        """
        Initialize FAISS vector store.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file (documents + metadata)
        """
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
        """Check if vector store is loaded and ready."""
        return self.index is not None and self.metadata is not None

    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict]:
        """
        Search for similar documents using embedding vector.

        Args:
            query_embedding: Query embedding vector (1D numpy array)
            k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of (document, score, metadata) dicts sorted by score
        """
        if not self.is_available():
            logger.warning("⚠️ Vector store not available")
            return []

        try:
            # Ensure query is 2D array
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Convert to float32 for FAISS compatibility
            query_embedding = query_embedding.astype(np.float32)

            # Search in FAISS index
            distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue

                # FAISS returns L2 distance, convert to similarity (0-1)
                similarity = 1.0 / (1.0 + distance)

                if similarity >= similarity_threshold:
                    doc_metadata = self.metadata[idx]
                    results.append({
                        "index": int(idx),
                        "document": doc_metadata.get("document", ""),
                        "intent": doc_metadata.get("intent", ""),
                        "pattern": doc_metadata.get("pattern", ""),
                        "response": doc_metadata.get("response", ""),
                        "similarity": float(similarity),
                        "metadata": doc_metadata.get("metadata", {})
                    })

            return sorted(results, key=lambda x: x["similarity"], reverse=True)[:k]

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get vector store statistics."""
        if not self.is_available():
            return {"status": "not_available"}

        return {
            "status": "available",
            "total_documents": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__
        }
