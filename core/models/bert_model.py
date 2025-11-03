import os
import pickle
import logging
from typing import Dict
from .base_model import BaseNLUModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from ..processors.text_normalizer import TextNormalizer

logger = logging.getLogger(__name__)

class BERTModel(BaseNLUModel):
    def __init__(self, config):
        self.config = config
        # BERT Components  
        self.bert_classifier = None
        self.bert_tokenizer = None
        self.bert_model = None
        self.bert_label_encoder = None
        self.text_normalizer = TextNormalizer()  # Will be injected later
        self._load_model(config.bert_model_path)

    def _load_model(self, model_path: str):
        """Load fine-tuned BERT model"""
        try:
            logger.info(f"🔄 Loading fine-tuned BERT dari: {model_path}")

            # Check if model directory exists
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ BERT model path not found: {model_path}")
                # Try alternative paths
                alternative_paths = [
                    "bert_model_optimized",
                    "./bert_model_optimized", 
                    "bert_optimized_finetuned",
                    "./bert_optimized_finetuned"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        logger.info(f"🔄 Using alternative path: {model_path}")
                        break
                else:
                    logger.error("❌ No valid BERT model path found")
                    return False

            # Load tokenizer and model
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)

            # Create pipeline (CPU)
            self.bert_classifier = pipeline(
                "text-classification",
                model=self.bert_model,
                tokenizer=self.bert_tokenizer,
                device=-1  # CPU
            )

            # Load label encoder
            le_path = os.path.join(model_path, 'label_encoder.pkl')
            if os.path.exists(le_path):
                with open(le_path, 'rb') as f:
                    self.bert_label_encoder = pickle.load(f)
                logger.info(f"✅ BERT label encoder loaded: {len(self.bert_label_encoder.classes_)} classes")
            else:
                logger.warning(f"⚠️ label_encoder.pkl not found in {model_path}")
                # Note: We'll handle label encoder fallback in the service layer
                return False

            logger.info("✅ Fine-tuned BERT loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"❌ BERT model loading failed: {e}")
            self.bert_classifier = None
            return False
        
    def predict(self, text: str) -> Dict:
        """Predict intent menggunakan BERT dengan text normalization"""
        try:
            if not self.bert_classifier:
                return {
                    "intent": "bert_unavailable",
                    "confidence": 0.0,
                    "method": "bert",
                    "status": "unavailable"
                }
            
            # Apply text normalization if available
            if self.text_normalizer:
                processed_text = self.text_normalizer.normalize(text)
                logger.debug(f"BERT Input: '{text}' -> Normalized: '{processed_text}'")
            else:
                processed_text = text
                logger.debug(f"BERT Input: '{text}'")
            
            # Predict (limit text length for BERT)
            result = self.bert_classifier(processed_text[:512])
            predicted_label = result[0]['label']
            confidence = result[0]['score']
            
            # Convert label to original intent name
            intent = self._convert_label_to_intent(predicted_label)
            
            return {
                "intent": intent,
                "confidence": float(confidence),
                "method": "bert",
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"BERT prediction failed: {e}")
            return {
                "intent": "error", 
                "confidence": 0.0,
                "method": "bert",
                "status": "error"
            }
    
    def _convert_label_to_intent(self, predicted_label: str) -> str:
        """Convert BERT label to intent name"""
        if self.bert_label_encoder is not None:
            if predicted_label.startswith('LABEL_'):
                try:
                    label_idx = int(predicted_label.split('_')[1])
                    return self.bert_label_encoder.inverse_transform([label_idx])[0]
                except (ValueError, IndexError, Exception):
                    return predicted_label
            else:
                return predicted_label
        else:
            return predicted_label
    
    def is_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self.bert_classifier is not None
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "type": "BERT",
            "model_loaded": self.bert_classifier is not None,
            "label_encoder_loaded": self.bert_label_encoder is not None,
            "classes_count": len(self.bert_label_encoder.classes_) if self.bert_label_encoder else 0,
            "tokenizer_loaded": self.bert_tokenizer is not None
        }
    
    def set_text_normalizer(self, normalizer):
        """Set text normalizer instance"""
        self.text_normalizer = normalizer