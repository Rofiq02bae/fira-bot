import logging
from typing import Dict, Any
from config.settings import ThresholdConfig

logger = logging.getLogger(__name__)

class IntentService:
    def __init__(self, normalizer, thresholds: ThresholdConfig):
        self.normalizer = normalizer
        self.thresholds = thresholds
        
        
    def check_pattern_similarity(self, text: str, intent: str, intent_mappings: Dict) -> float:
        """Check similarity antara input text dengan patterns di intent"""
        if intent not in intent_mappings:
            return 0.0

        user_words = set(self.normalizer.normalize(text).split())
        max_similarity = 0.0
        
        for pattern in intent_mappings[intent]['patterns']:
            pattern_words = set(self.normalizer.normalize(str(pattern)).split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))
            
            if total_words > 0:
                similarity = common_words / total_words
                max_similarity = max(max_similarity, similarity)
        
        # Tambahkan debug logging di check_pattern_similarity
        logger.info(f"User text normalized: '{self.normalizer.normalize(text)}'")
        logger.info(f"Pattern normalized: '{self.normalizer.normalize(pattern)}'")
        
        return max_similarity
    
    def hybrid_predict(self, text: str, lstm_pred: Dict, bert_pred: Dict, intent_mappings: Dict) -> Dict:
        """Hybrid prediction dengan fallback strategy"""
        
        logger.info(f"🔍 Processing: '{text}'")
        
        # STEP 1: Cek LSTM prediction pertama
        logger.info(f"   LSTM Prediction: {lstm_pred['intent']} (conf: {lstm_pred['confidence']:.3f})")
        
        if lstm_pred['status'] == 'success':
            similarity = self.check_pattern_similarity(text, lstm_pred['intent'], intent_mappings)
            logger.info(f"   LSTM Pattern Similarity: {similarity:.3f}")
            
            if similarity >= self.thresholds.pattern_similarity:
                return {
                    **lstm_pred,
                    "method": "lstm_direct",
                    "sources": ["lstm"],
                    "pattern_similarity": similarity,
                    "fallback_reason": None
                }
            else:
                logger.info(f"   ⚠️ LSTM pattern tidak cocok, mencoba BERT...")
        
        # STEP 2: Cek BERT prediction
        logger.info(f"   BERT Prediction: {bert_pred['intent']} (conf: {bert_pred['confidence']:.3f})")
        
        if bert_pred['status'] == 'success':
            bert_similarity = self.check_pattern_similarity(text, bert_pred['intent'], intent_mappings)
            logger.info(f"   BERT Pattern Similarity: {bert_similarity:.3f}")
            
            if bert_similarity >= self.thresholds.pattern_similarity:
                return {
                    **bert_pred,
                    "method": "bert_with_pattern",
                    "sources": ["bert", "pattern"],
                    "pattern_similarity": bert_similarity,
                    "fallback_reason": "lstm_pattern_mismatch"
                }
            else:
                logger.info(f"   ⚠️ Pattern masih tidak cocok, menggunakan BERT langsung...")
                return {
                    **bert_pred,
                    "method": "bert_direct", 
                    "sources": ["bert"],
                    "pattern_similarity": bert_similarity,
                    "fallback_reason": "both_patterns_mismatch"
                }
        
        # STEP 3: Fallback terakhir
        logger.info(f"   ❌ Semua method gagal, menggunakan fallback...")
        if lstm_pred['status'] == 'success':
            return {
                **lstm_pred,
                "method": "lstm_emergency_fallback",
                "sources": ["lstm"],
                "pattern_similarity": 0.0,
                "fallback_reason": "all_methods_failed"
            }
        else:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "method": "emergency_fallback",
                "status": "success",
                "sources": [],
                "pattern_similarity": 0.0,
                "fallback_reason": "complete_failure"
            }