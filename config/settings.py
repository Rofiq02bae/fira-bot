import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class ModelConfig:
    lstm_model_path: str = "../data/models/chatbot_model.h5"
    lstm_tokenizer_path: str = "../data/models/tokenizer.pkl" 
    lstm_label_encoder_path: str = "../data/models/label_encoder.pkl"
    bert_model_path: str = "../data/bert_model"
    dataset_path: str = "../data/dataset/dataset_training.csv"

@dataclass
class ThresholdConfig:
    lstm_high: float = 0.8
    bert_high: float = 0.7  
    pattern_similarity: float = 0.3
    fusion_threshold: float = 0.6
    low_confidence_threshold: float = 0.3
    min_confidence: float = 0.3

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    # telegram_webhook_url: str = os.getenv("TELEGRAM_WEBHOOK_URL", "")
    # telegram_admin_ids: list = field(default_factory=lambda: [])  # Admin user IDs

# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_THRESHOLD_CONFIG = ThresholdConfig()
DEFAULT_API_CONFIG = APIConfig()