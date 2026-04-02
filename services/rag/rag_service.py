"""
RAG Service for document retrieval and augmented response generation.
v2: tambah parameter filter_domain dan filter_kb_id di retrieve_documents
"""

import logging
import os
import asyncio
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Retrieves relevant documents from FAISS and generates responses via LLM.
    """

    def __init__(self, vector_store, embedding_model=None, llm_client=None):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self._available = vector_store.is_available() if vector_store else False

        self.user_intent = None
        self.user_response = None
        logger.info(f"🔍 RAG Service initialized: {'Available' if self._available else 'Unavailable'}")

    def is_available(self) -> bool:
        return self._available

    async def retrieve_documents(
        self,
        query_text: str,
        k: int = 5,
        similarity_threshold: float = 0.3,
        filter_domain: Optional[str] = None,
        filter_kb_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query_text          : User query text
            k                   : Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            filter_domain       : Opsional — batasi hasil ke domain tertentu
            filter_kb_id        : Opsional — batasi hasil ke kb_id tertentu

        Returns:
            List of retrieved documents with metadata
        """
        if not self._available:
            logger.warning("⚠️ RAG service not available")
            return []

        try:
            query_embedding = await self._generate_embedding(query_text)
            if query_embedding is None:
                logger.warning("⚠️ Failed to generate embedding")
                return []

            results = self.vector_store.search(
                query_embedding,
                k=k,
                similarity_threshold=similarity_threshold,
                filter_domain=filter_domain,
                filter_kb_id=filter_kb_id
            )

            logger.info(
                f"📚 Retrieved {len(results)} documents "
                f"(domain={filter_domain}, kb_id={filter_kb_id}) "
                f"for: '{query_text}'"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        try:
            if not self.embedding_model:
                logger.warning("⚠️ No embedding model available")
                return None

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._bert_encode,
                text
            )
            if embedding is not None:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (embedding / norm).astype(np.float32)
            return embedding

        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            return None

    def _bert_encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text using BERT model (sync)."""
        try:
            embedding_model = self.embedding_model
            if embedding_model is None:
                return None

            if hasattr(embedding_model, "encode"):
                embedding = embedding_model.encode(text, convert_to_numpy=True)
                if isinstance(embedding, np.ndarray) and embedding.ndim > 1:
                    embedding = embedding.squeeze(0)
                return embedding.astype(np.float32)

            if hasattr(embedding_model, "bert_model") and hasattr(embedding_model, "bert_tokenizer"):
                import torch
                tokenizer = embedding_model.bert_tokenizer
                model = embedding_model.bert_model

                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    if hasattr(model, 'bert'):
                        outputs = model.bert(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    elif hasattr(model, 'base_model'):
                        outputs = model.base_model(**inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                        else:
                            embedding = outputs.pooler_output.squeeze().numpy()
                    else:
                        outputs = model(**inputs)
                        if hasattr(outputs, 'logits'):
                            embedding = outputs.logits.squeeze().numpy()
                        else:
                            logger.warning("⚠️ Cannot extract embeddings from model")
                            return None

                return embedding.astype(np.float32)

            logger.warning("⚠️ Unsupported embedding model type")
            return None

        except Exception as e:
            logger.error(f"❌ BERT encode error: {e}")
            return None

    async def generate_response(
        self,
        query: str,
        intent: str,
        retrieved_documents: List[Dict[str, Any]],
        fallback_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate final response using LLM with retrieved context.
        """
        try:
            if not retrieved_documents:
                if self.llm_client and fallback_response:
                    response_text = await self._paraphrase_with_llm(query, fallback_response, [])
                else:
                    response_text = fallback_response or "Maaf, saya tidak memahami pertanyaan Anda."

                return {
                    "response": response_text,
                    "augmented": False,
                    "sources": [],
                    "intent": intent,
                    "method": "fallback"
                }

            if self.llm_client:
                response_text = await self._paraphrase_with_llm(
                    query,
                    retrieved_documents[0].get("response", ""),
                    retrieved_documents
                )
            else:
                response_text = retrieved_documents[0].get("response", "")

            sources = [
                {
                    "pattern":    doc.get("pattern", ""),
                    "response":   doc.get("response", ""),
                    "response_type": doc.get("response_type", ""),
                    "is_master": doc.get("is_master", False),
                    "similarity": doc.get("similarity", 0),
                    "intent":     doc.get("intent", ""),
                    "kb_id":      doc.get("kb_id", ""),
                    "domain":     doc.get("domain", "")
                }
                for doc in retrieved_documents[:3]
            ]

            return {
                "response":    response_text,
                "augmented":   True,
                "sources":     sources,
                "intent":      intent,
                "method":      "rag_llm" if self.llm_client else "rag_direct",
                "num_sources": len(sources)
            }

        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return {
                "response":  fallback_response or "Maaf, terjadi kesalahan.",
                "augmented": False,
                "sources":   [],
                "intent":    intent,
                "method":    "error"
            }

    async def _paraphrase_with_llm(
        self,
        query: str,
        base_response: str,
        context_docs: List[Dict[str, Any]]
    ) -> str:
        try:
            if not self.llm_client:
                return base_response

            context = ""
            for i, doc in enumerate(context_docs[:3], 1):
                context += f"\n{i}. {doc.get('response', '')}"

            prompt = f"""Anda adalah asisten chatbot yang membantu menjawab pertanyaan tentang layanan pemerintah.

Pertanyaan: {query}

Informasi yang tersedia:
{context if context else base_response}

Instruksi:
- Jawab pertanyaan dengan bahasa yang natural dan mudah dipahami
- Hanya gunakan informasi yang tersedia di atas
- Jangan menambah informasi baru atau mengada-ada
- Jika informasi tidak cukup, katakan dengan jujur
- Gunakan bahasa Indonesia yang sopan

Jawaban:"""

            response = await self._call_llm(prompt)
            return response if response else base_response

        except Exception as e:
            logger.error(f"❌ LLM paraphrase error: {e}")
            return base_response

    async def _call_llm(self, prompt: str) -> Optional[str]:
        try:
            if not self.llm_client:
                return None

            if hasattr(self.llm_client, "chat"):
                if hasattr(self.llm_client.chat, "completions"):
                    from config.settings import DEFAULT_RAG_CONFIG
                    response = await self.llm_client.chat.completions.create(
                        model=DEFAULT_RAG_CONFIG.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=DEFAULT_RAG_CONFIG.llm_max_tokens,
                        temperature=DEFAULT_RAG_CONFIG.llm_temperature
                    )
                    return response.choices[0].message.content.strip()

            logger.warning("⚠️ OpenRouter client not properly initialized")
            return None

        except Exception as e:
            logger.error(f"❌ LLM call error: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available":                  self._available,
            "vector_store":               self.vector_store.get_stats() if self.vector_store else {},
            "embedding_model_available":  self.embedding_model is not None,
            "llm_client_available":       self.llm_client is not None
        }
