"""
RAG Service for document retrieval and augmented response generation.
Integrates FAISS vector store with LLM for paraphrasing.
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
        """
        Initialize RAG service.

        Args:
            vector_store: FAISSVectorStore instance
            embedding_model: Model for generating query embeddings
            llm_client: LLM client for paraphrasing (OpenAI, Anthropic, etc.)
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self._available = vector_store.is_available() if vector_store else False

        self.user_intent = None
        self.user_response = None   
        logger.info(f"🔍 RAG Service initialized: {'Available' if self._available else 'Unavailable'}")

    def is_available(self) -> bool:
        """Check if RAG service is ready."""
        return self._available

    async def retrieve_documents(
        self,
        query_text: str,
        k: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query_text: User query text
            k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of retrieved documents with metadata
        """
        if not self._available:
            logger.warning("⚠️ RAG service not available")
            return []

        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding(query_text)
            if query_embedding is None:
                logger.warning("⚠️ Failed to generate embedding")
                return []

            # Search using FAISS
            results = self.vector_store.search(
                query_embedding,
                k=k,
                similarity_threshold=similarity_threshold
            )

            logger.info(f"📚 Retrieved {len(results)} documents for query: '{query_text}'")
            return results

        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []

    async def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using embedding model.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if failed
        """
        try:
            if not self.embedding_model:
                logger.warning("⚠️ No embedding model available")
                return None

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self._bert_encode,
                text
            )
            return embedding

        except Exception as e:
            logger.error(f"❌ Embedding generation error: {e}")
            return None

    def _bert_encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text using BERT model (sync)."""
        try:
            # Try sentence-transformers first (best for embeddings)
            if hasattr(self.embedding_model, "encode"):
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding.astype(np.float32)

            # Try transformers model - use tokenizer + model directly for embeddings
            if hasattr(self.embedding_model, "bert_model") and hasattr(self.embedding_model, "bert_tokenizer"):
                import torch
                tokenizer = self.embedding_model.bert_tokenizer
                model = self.embedding_model.bert_model
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    # Get embeddings from the base model (before classification head)
                    # For classification models, we need to access the base model
                    if hasattr(model, 'bert'):
                        # BertForSequenceClassification has a 'bert' attribute
                        outputs = model.bert(**inputs)
                        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    elif hasattr(model, 'base_model'):
                        outputs = model.base_model(**inputs)
                        if hasattr(outputs, 'last_hidden_state'):
                            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                        else:
                            embedding = outputs.pooler_output.squeeze().numpy()
                    else:
                        # Fallback: use model output directly (might be logits)
                        outputs = model(**inputs)
                        if hasattr(outputs, 'logits'):
                            # Use logits as embedding (not ideal but works for similarity)
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
        fallback_response: str = None
    ) -> Dict[str, Any]:
        """
        Generate final response using LLM with retrieved context.

        Args:
            query: User query
            intent: Detected intent
            retrieved_documents: Documents from FAISS
            fallback_response: Fallback if no documents found

        Returns:
            Generated response with metadata
        """
        try:
            if not retrieved_documents:
                # No documents found, use fallback
                if self.llm_client and fallback_response:
                    # Paraphrase fallback
                    response_text = await self._paraphrase_with_llm(
                        query,
                        fallback_response,
                        []
                    )
                else:
                    response_text = fallback_response or "Maaf, saya tidak memahami pertanyaan Anda."
                
                return {
                    "response": response_text,
                    "augmented": False,
                    "sources": [],
                    "intent": intent,
                    "method": "fallback"
                }

            # Documents found, generate with context
            if self.llm_client:
                response_text = await self._paraphrase_with_llm(
                    query,
                    retrieved_documents[0].get("response", ""),
                    retrieved_documents
                )
            else:
                # No LLM, return top document response
                response_text = retrieved_documents[0].get("response", "")

            # Extract sources
            sources = [
                {
                    "pattern": doc.get("pattern", ""),
                    "response": doc.get("response", ""),
                    "similarity": doc.get("similarity", 0),
                    "intent": doc.get("intent", "")
                }
                for doc in retrieved_documents[:3]
            ]

            return {
                "response": response_text,
                "augmented": True,
                "sources": sources,
                "intent": intent,
                "method": "rag_llm" if self.llm_client else "rag_direct",
                "num_sources": len(sources)
            }

        except Exception as e:
            logger.error(f"❌ Response generation error: {e}")
            return {
                "response": fallback_response or "Maaf, terjadi kesalahan.",
                "augmented": False,
                "sources": [],
                "intent": intent,
                "method": "error"
            }

    async def _paraphrase_with_llm(
        self,
        query: str,
        base_response: str,
        context_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Paraphrase response using LLM with retrieved context.

        Args:
            query: User query
            base_response: Base response template
            context_docs: Retrieved documents for context

        Returns:
            Paraphrased response
        """
        try:
            if not self.llm_client:
                return base_response

            # Build context from documents
            context = ""
            for i, doc in enumerate(context_docs[:3], 1):
                context += f"\n{i}. {doc.get('response', '')}"

            # Create prompt for LLM
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

            # Call LLM (OpenAI, Anthropic, etc.)
            response = await self._call_llm(prompt)
            return response if response else base_response

        except Exception as e:
            logger.error(f"❌ LLM paraphrase error: {e}")
            return base_response

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API via OpenRouter."""
        try:
            if not self.llm_client:
                return None

            # OpenRouter uses OpenAI-compatible API
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
        """Get RAG service statistics."""
        return {
            "available": self._available,
            "vector_store": self.vector_store.get_stats() if self.vector_store else {},
            "embedding_model_available": self.embedding_model is not None,
            "llm_client_available": self.llm_client is not None
        }
