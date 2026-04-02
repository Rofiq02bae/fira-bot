import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    # Read from env when available (works well in Docker), otherwise fallback.
    lstm_model_path: str = os.getenv("LSTM_MODEL_PATH", "data/lstm_models/chatbot_model.h5")
    lstm_tokenizer_path: str = os.getenv("LSTM_TOKENIZER_PATH", "data/lstm_models/tokenizer.pkl")
    lstm_label_encoder_path: str = os.getenv("LSTM_LABEL_ENCODER_PATH", "data/lstm_models/label_encoder.pkl")
    bert_model_path: str = os.getenv("BERT_MODEL_PATH", "data/bert_model")
    dataset_path: str = os.getenv("DATASET_PATH", "data/dataset/bert/dataset_training_bert.csv")

@dataclass
class ThresholdConfig:
    lstm_high: float = 0.85
    bert_high: float = 0.8  
    pattern_similarity: float = 0.4
    fusion_threshold: float = 0.7
    low_confidence_threshold: float = 0.5
    min_confidence: float = 0.4
    ambiguity_margin: float = 0.1

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    # Prefer TELEGRAM_BOT_TOKEN (matches .env / telegram bot runner)
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_TOKEN", ""))
    # telegram_webhook_url: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    # telegram_admin_ids: list = field(default_factory=lambda: [])  # Admin user IDs

@dataclass
class RAGConfig:
    # Enable/disable RAG
    enabled: bool = os.getenv("USE_RAG", "false").lower() in ("1", "true", "yes")
    
    # FAISS vector store paths
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "data/rag/faiss.index")
    faiss_metadata_path: str = os.getenv("FAISS_METADATA_PATH", "data/rag/metadata.pkl")
    
    # Retrieval parameters
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3"))
    top_k_documents: int = int(os.getenv("RAG_TOP_K", "5"))
    max_context_docs: int = int(os.getenv("RAG_MAX_CONTEXT", "3"))
    
    # LLM configuration (OpenRouter)
    llm_provider: str = os.getenv("LLM_PROVIDER", "")  # "openrouter" or ""
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "anthropic/claude-opus-4.6")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # Embedding configuration (harus konsisten antara ingest dan runtime)
    embedding_model_name: str = os.getenv("RAG_EMBEDDING_MODEL", "indobenchmark/indobert-base-p1")
    
    # RAG trigger thresholds
    rag_min_confidence: float = float(os.getenv("RAG_MIN_CONFIDENCE", "0.4"))
    use_rag_for_fallback: bool = os.getenv("USE_RAG_FOR_FALLBACK", "true").lower() in ("1", "true", "yes")

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_THRESHOLD_CONFIG = ThresholdConfig()
DEFAULT_API_CONFIG = APIConfig()
DEFAULT_RAG_CONFIG = RAGConfig()