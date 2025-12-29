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
    dataset_path: str = os.getenv("DATASET_PATH", "data/dataset/dataset_training.csv")

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

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_THRESHOLD_CONFIG = ThresholdConfig()
DEFAULT_API_CONFIG = APIConfig()