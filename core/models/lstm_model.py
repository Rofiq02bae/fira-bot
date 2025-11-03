import os
import pickle
import numpy as np
import logging
from typing import Dict
from .base_model import BaseNLUModel
from ..processors.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

class LSTMModel(BaseNLUModel):
    def __init__(self, config):
        self.config = config
        # LSTM Components
        self.lstm_model = None
        self.lstm_tokenizer = None
        self.lstm_max_len = None
        self.lstm_label_encoder = None
        self.text_normalizer = TextNormalizer()  # Will be injected later
        self._load_model()

    def _load_model(self):
        """Load existing LSTM model"""  
        try:
            logger.info(f"🔄 Loading LSTM model from: {self.config.lstm_model_path}")
            
            # Check if files exist
            required_paths = [
                self.config.lstm_model_path, 
                self.config.lstm_tokenizer_path, 
                self.config.lstm_label_encoder_path
            ]
            
            if not all(os.path.exists(path) for path in required_paths):
                logger.error("❌ Some LSTM model files not found")
                return False
            
            # Import tensorflow inside function to avoid early loading
            try:
                from tensorflow.keras.models import load_model
                from tensorflow.keras.preprocessing.sequence import pad_sequences
            except ImportError as e:
                logger.error(f"❌ TensorFlow not available: {e}")
                return False
            
            logger.info("📦 Loading TensorFlow model...")
            self.lstm_model = load_model(self.config.lstm_model_path)
            
            logger.info("📦 Loading tokenizer...")
            with open(self.config.lstm_tokenizer_path, 'rb') as f:
                tokenizer_data = pickle.load(f)
                self.lstm_tokenizer = tokenizer_data['tokenizer']
                self.lstm_max_len = tokenizer_data['max_len']
                
            logger.info("📦 Loading label encoder...")
            with open(self.config.lstm_label_encoder_path, 'rb') as f:
                self.lstm_label_encoder = pickle.load(f)
                
            logger.info(f"✅ LSTM model loaded successfully!")
            logger.info(f"   - Vocabulary size: {len(self.lstm_tokenizer.word_index)}")
            logger.info(f"   - Max sequence length: {self.lstm_max_len}")
            logger.info(f"   - Number of classes: {len(self.lstm_label_encoder.classes_)}")
            return True  
            
        except Exception as e:
            logger.error(f"❌ LSTM model loading failed: {e}")
            return False
        
    def predict(self, text: str) -> Dict:
        """Predict intent menggunakan LSTM dengan text normalization"""
        try:
            if self.lstm_model is None:
                return {
                    "intent": "lstm_unavailable",
                    "confidence": 0.0,
                    "method": "lstm",
                    "status": "unavailable"
                }
            
            # Preprocess dengan text normalization if available
            if self.text_normalizer:
                processed_text = self.text_normalizer.normalize(text)
                logger.debug(f"LSTM Input: '{text}' -> Normalized: '{processed_text}'")
            else:
                processed_text = text
                logger.debug(f"LSTM Input: '{text}'")
            
            # Convert text to sequences
            sequences = self.lstm_tokenizer.texts_to_sequences([processed_text])

            try:
                from tensorflow.keras.preprocessing.sequence import pad_sequences
            except ImportError:
                logger.error("❌ TensorFlow preprocessing not available")
                return {
                    "intent": "error",
                    "confidence": 0.0,
                    "method": "lstm", 
                    "status": "error"
                }

            padded = pad_sequences(sequences, maxlen=self.lstm_max_len, padding='post')

            # Predict
            prediction = self.lstm_model.predict(padded, verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            confidence = float(prediction[predicted_idx])
            
            intent = self.lstm_label_encoder.inverse_transform([predicted_idx])[0]
            
            return {
                "intent": intent,
                "confidence": confidence,
                "method": "lstm",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "method": "lstm",
                "status": "error"
            }
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self.lstm_model is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "type": "LSTM",
            "vocab_size": len(self.lstm_tokenizer.word_index) if self.lstm_tokenizer else 0,
            "max_sequence_length": self.lstm_max_len,
            "classes_count": len(self.lstm_label_encoder.classes_) if self.lstm_label_encoder else 0,
            "model_loaded": self.lstm_model is not None
        }
    
    def set_text_normalizer(self, normalizer):
        """Set text normalizer instance"""
        self.text_normalizer = normalizer