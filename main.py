import logging
import os
from typing import Optional
from config.settings import ModelConfig
from core.nlu_service import HybridNLUService

logger = logging.getLogger(__name__)

# Global service instance
_hybrid_nlu: Optional[HybridNLUService] = None

def initialize_hybrid_service(
    dataset_path: Optional[str] = None,
    lstm_model_path: Optional[str] = None,
    lstm_tokenizer_path: Optional[str] = None,
    lstm_label_encoder_path: Optional[str] = None,
    bert_model_path: Optional[str] = None
) -> HybridNLUService:
    """Initialize hybrid NLU service dengan struktur modular"""
    logger.info("🎯 Initializing Modular Hybrid NLU Service...")
    
    # Resolve paths from env if not explicitly provided
    dataset_path = dataset_path or os.getenv("DATASET_PATH", "data/dataset/dataset_training.csv")
    lstm_model_path = lstm_model_path or os.getenv("LSTM_MODEL_PATH", "data/lstm_models/chatbot_model.h5")
    lstm_tokenizer_path = lstm_tokenizer_path or os.getenv("LSTM_TOKENIZER_PATH", "data/lstm_models/tokenizer.pkl")
    lstm_label_encoder_path = lstm_label_encoder_path or os.getenv("LSTM_LABEL_ENCODER_PATH", "data/lstm_models/label_encoder.pkl")
    bert_model_path = bert_model_path or os.getenv("BERT_MODEL_PATH", "data/bert_model")

    # Create config
    config = ModelConfig(
        dataset_path=dataset_path,
        lstm_model_path=lstm_model_path,
        lstm_tokenizer_path=lstm_tokenizer_path, 
        lstm_label_encoder_path=lstm_label_encoder_path,
        bert_model_path=bert_model_path
    )
    
    try:
        # Initialize service dengan struktur modular
        service = HybridNLUService(config)
        
        logger.info("🚀 Modular Hybrid NLU Service initialized successfully!")
        return service
        
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {e}")
        raise

def get_hybrid_nlu() -> HybridNLUService:
    """Singleton pattern untuk global access"""
    global _hybrid_nlu
    if _hybrid_nlu is None:
        _hybrid_nlu = initialize_hybrid_service()
    return _hybrid_nlu

if __name__ == "__main__":
    # Test the service
    try:
        service = get_hybrid_nlu()
        status = service.get_service_status()
        print("✅ Service initialized successfully!")
        print(f"Status: {status}")
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")