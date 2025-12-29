import sys
import os
import logging
from typing import Dict

# Tambahkan project root ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.processors.intent_matcher import IntentMatcher
from config.settings import ThresholdConfig

# Mock Normalizer
class MockNormalizer:
    def normalize(self, text):
        return text.lower().strip()

def test_margin_logic():
    thresholds = ThresholdConfig(ambiguity_margin=0.1, min_confidence=0.4)
    matcher = IntentMatcher(MockNormalizer(), thresholds)
    
    # Mock data
    intent_mappings = {
        "akta_general": {
            "patterns": ["akta"],
            "is_master": True,
            "response": "Silahkan pilih akta mana yang anda maksud? | Akta Lahir | Akta Mati"
        },
        "akta_lahir_info": {
            "patterns": ["akta lahir"],
            "is_master": False
        },
        "akta_mati_info": {
            "patterns": ["akta mati"],
            "is_master": False
        }
    }
    
    text_no_kw = "saya mau tanya sesuatu" 
    lstm_pred = {"intent": "unknown", "confidence": 0.0, "status": "success"}
    
    bert_pred_low_margin = {
        "intent": "akta_lahir_info",
        "confidence": 0.85,
        "status": "success",
        "top_scores": [
            {"intent": "akta_lahir_info", "confidence": 0.85},
            {"intent": "akta_mati_info", "confidence": 0.82} # Margin 0.03 < 0.1
        ]
    }
    
    print("\n--- MARGIN LOGIC TEST ---")
    
    result = matcher.hybrid_predict(text_no_kw, lstm_pred, bert_pred_low_margin, intent_mappings)
    
    print(f"QUERY: '{text_no_kw}'")
    print(f"INTENT: {result['intent']}")
    print(f"METHOD: {result['method']}")
    print(f"STATUS: {result['status']}")
    
    if result['status'] == 'clarification':
        print("✅ Margin-based clarification triggered successfully!")
        print(f"   [MESSAGE]: {result.get('message')}")
    else:
        print("❌ Margin-based clarification FAILED to trigger.")

if __name__ == "__main__":
    test_margin_logic()
