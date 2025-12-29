import logging
from typing import Dict, Any
from config.settings import ThresholdConfig

logger = logging.getLogger(__name__)

class IntentMatcher:
    def __init__(self, normalizer, thresholds: ThresholdConfig):
        self.normalizer = normalizer
        self.thresholds = thresholds
        
    def _check_if_user_is_specific(self, text: str, master_intent: str, intent_mappings: Dict) -> bool:
        """
        Determine if the user's input is specific enough to bypass clarification.
        Logic: If the text contains any 'modifier' words that belong to sub-intents.
        """
        normalized_text = self.normalizer.normalize(text)
        user_words = set(normalized_text.split())
        
        # Get master intent patterns keywords
        master_patterns = intent_mappings.get(master_intent, {}).get('patterns', [])
        master_keywords = set()
        for p in master_patterns:
            master_keywords.update(self.normalizer.normalize(str(p)).split())
            
        for intent, data in intent_mappings.items():
            if intent == master_intent or data.get('is_master'):
                continue
            
            patterns = data.get('patterns', [])
            for p in patterns:
                p_norm = self.normalizer.normalize(str(p))
                p_words = set(p_norm.split())
                
                # Jika pola spesifik ini mengandung keyword utama master
                if p_words.intersection(master_keywords):
                    # Dan jika input user mengandung kata-kata spesifik tersebut
                    if p_words.issubset(user_words):
                        return True
        return False

    def _parse_clarification_response(self, response_text: str) -> Dict[str, Any]:
        """Parse response, support both 'Message | Option 1' and JSON format"""
        if not response_text:
            return {"message": "Mohon perjelas pertanyaan Anda.", "options": []}
            
        import re
        import json
        
        message = ""
        options = []
        
        # Try JSON first (if dataset is converted)
        try:
            # If it's already a dict (unlikely if coming from df string, but safe)
            if isinstance(response_text, dict):
                 data = response_text
            else:
                 data = json.loads(str(response_text))
            
            if data.get('type') == 'list':
                message = data.get('title', "")
                raw_items = data.get('items', [])
                
                # Format options from JSON items
                formatted_list = []
                for i, opt in enumerate(raw_items):
                     label = str(opt).strip()
                     options.append({
                         "label": label,
                         "intent": label
                     })
                     formatted_list.append(f"{i+1}. {label}")
                
                if formatted_list:
                    full_message = f"{message}\n\n" + "\n".join(formatted_list)
                else:
                    full_message = message
                    
                return {"message": full_message, "options": options}
                
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback to Pipe splitting
        parts = [p.strip() for p in str(response_text).split('|')]
        message = parts[0]
        
        # Build options and formatted text for message
        formatted_list = []
        for i, opt in enumerate(parts[1:]):
            # Strip leading numbers like "1. ", "2.", or "1)"
            label = re.sub(r'^\d+[\.\)]\s*', '', opt).strip()
            
            if label:
                options.append({
                    "label": label,
                    "intent": label # Use label as intent for better matching in turn 2
                })
                formatted_list.append(f"{i+1}. {label}")
        
        # Merge options into message for text-only displays
        if formatted_list:
            full_message = f"{message}\n\n" + "\n".join(formatted_list)
        else:
            full_message = message
            
        return {"message": full_message, "options": options}

    def _is_related(self, sub_intent: str, master_intent: str, intent_mappings: Dict) -> bool:
        """Check if an intent is related to a master intent via keyword overlap"""
        m_patterns = intent_mappings.get(master_intent, {}).get('patterns', [])
        s_patterns = intent_mappings.get(sub_intent, {}).get('patterns', [])
        
        m_words = set()
        for p in m_patterns: 
            m_words.update(self.normalizer.normalize(str(p)).split())
        
        for p in s_patterns:
            s_words = set(self.normalizer.normalize(str(p)).split())
            if s_words.intersection(m_words):
                return True
        return False

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
        
        return max_similarity
    
    def _hybrid_fusion(self, text: str, lstm_pred: Dict, bert_pred: Dict, intent_mappings: Dict) -> Dict:
        """
        Advanced Fusion Logic (Tug-of-War & Confidence Priority)
        """
        bert_conf = bert_pred.get('confidence', 0.0)
        lstm_conf = lstm_pred.get('confidence', 0.0)
        bert_intent = bert_pred.get('intent', 'unknown')
        lstm_intent = lstm_pred.get('intent', 'unknown')
        
        logger.info(f"🧬 Fusion Analysis: BERT={bert_intent}({bert_conf:.3f}) vs LSTM={lstm_intent}({lstm_conf:.3f})")

        # Thresholds
        HIGH_CONF = 0.85
        MEDIUM_CONF = 0.60
        LOW_GAP_THRESHOLD = 0.15 # Minimal gap to determine a clear winner

        # 1. SCENARIO: Salah satu model sangat yakin (Confident Priority)
        # Jika BERT sangat yakin dan LSTM ragu
        if bert_conf >= HIGH_CONF and lstm_conf < MEDIUM_CONF:
            return {
                **bert_pred,
                "method": "fusion_bert_dominant",
                "fusion_info": "Bert high confidence override"
            }
        
        # Jika LSTM sangat yakin dan BERT ragu (jarang, tapi mungkin untuk short text)
        if lstm_conf >= HIGH_CONF and bert_conf < MEDIUM_CONF:
             return {
                **lstm_pred,
                "method": "fusion_lstm_dominant",
                "fusion_info": "LSTM high confidence override"
            }

        # 2. SCENARIO: Conflict Resolution (Tug-of-War)
        # Kedua model cukup yakin (> 0.6) tapi prediksi BEDA
        if bert_conf >= MEDIUM_CONF and lstm_conf >= MEDIUM_CONF and bert_intent != lstm_intent:
            logger.info("⚔️ Tug-of-War detected!")
            
            # Sub-scenario: Cek Pattern Similarity sebagai wasit
            bert_sim = self.check_pattern_similarity(text, bert_intent, intent_mappings)
            lstm_sim = self.check_pattern_similarity(text, lstm_intent, intent_mappings)
            
            logger.info(f"   Patterns: BERT_sim={bert_sim:.3f} vs LSTM_sim={lstm_sim:.3f}")
            
            if abs(bert_sim - lstm_sim) > 0.1:
                # Menangkan yang pattern-nya lebih mirip
                if bert_sim > lstm_sim:
                    return {**bert_pred, "method": "fusion_conflict_pattern_win_bert"}
                else:
                    return {**lstm_pred, "method": "fusion_conflict_pattern_win_lstm"}
            
            # Jika pattern similarity mirip juga, gunakan panjang teks
            # Text panjang -> BERT biasanya lebih jago konteks
            # Text pendek -> LSTM/Keyword biasanya lebih OK
            word_count = len(text.split())
            if word_count > 5:
                 return {**bert_pred, "method": "fusion_conflict_length_bert"}
            elif word_count <= 2:
                 return {**lstm_pred, "method": "fusion_conflict_length_lstm"}
                 
            # Jika masih deadlock, percaya BERT karena generally more robust
            return {**bert_pred, "method": "fusion_conflict_default_bert"}

        # 3. SCENARIO: Agreement (Easy Win)
        if bert_intent == lstm_intent and bert_conf > 0.4:
            return {
                **bert_pred,
                "confidence": max(bert_conf, lstm_conf), # Ambil confidence tertinggi
                "method": "fusion_agreement"
            }

        # 4. SCENARIO: Weighted Average (Default Fallback for uncategorized cases)
        # Beri bobot sedikit lebih ke BERT
        w_bert = 0.6
        w_lstm = 0.4
        
        # Jika text sangat pendek, boost LSTM weight
        if len(text.split()) <= 3:
            w_bert = 0.4
            w_lstm = 0.6
            
        final_conf = (bert_conf * w_bert) + (lstm_conf * w_lstm)
        
        # Tentukan winner base on raw scores (ini simplifikasi, idealnya sum proba per class)
        # Di sini kita ambil yang confidence-nya lebih tinggi sebagai base
        base_pred = bert_pred if bert_conf >= lstm_conf else lstm_pred
        
        return {
            **base_pred,
            "confidence": final_conf, # Adjusted confidence
            "method": "fusion_weighted_avg"
        }

    def hybrid_predict(self, text: str, lstm_pred: Dict, bert_pred: Dict, intent_mappings: Dict) -> Dict:
        """
        Hybrid prediction dengan 5-step flow + Ambiguity Check:
        1. Ambiguity Check (Keyword & Margin)
        2. LSTM + Pattern Match
        3. BERT + Pattern Match
        4. Advanced Fusion Logic (Replacing BERT Direct)
        5. Fallback
        """
        
        logger.info(f"🔍 Processing Hybrid Logic: '{text}'")
        
        # 1. Dataset-Driven Master Intent Check
        master_intents = [name for name, data in intent_mappings.items() if data.get('is_master')]
        
        for m_intent in master_intents:
            similarity = self.check_pattern_similarity(text, m_intent, intent_mappings)
            if similarity >= self.thresholds.pattern_similarity:
                # Cek apakah input user spesifik (misal "akta lahir" vs "akta")
                if not self._check_if_user_is_specific(text, m_intent, intent_mappings):
                    logger.info(f"⚠️ Ambiguity detected for master intent: {m_intent}")
                    # Get first response from list
                    raw_response = intent_mappings[m_intent]['responses'][0] if intent_mappings[m_intent]['responses'] else ""
                    resp_data = self._parse_clarification_response(raw_response)
                    return {
                        "intent": "clarification_needed",
                        "confidence": 1.0,
                        "method": "master_intent_check",
                        "status": "clarification",
                        "message": resp_data['message'],
                        "options": resp_data['options'],
                        "pattern_similarity": similarity
                    }

        # Helper: Check Hard Minimum
        def passes_hard_min(pred):
            return pred.get('confidence', 0) >= self.thresholds.min_confidence

        # 1.5 Margin Threshold Check (BERT scores)
        # Only check margin if we are confident enough (avoid OOD triggers)
        if bert_pred['status'] == 'success' and passes_hard_min(bert_pred):
            top_scores = bert_pred.get('top_scores', [])
            if len(top_scores) >= 2:
                margin = top_scores[0]['confidence'] - top_scores[1]['confidence']
                
                if margin < self.thresholds.ambiguity_margin:
                    # Cari kategory master yang sesuai dengan top prediction
                    for m_intent in master_intents:
                        if self._is_related(bert_pred['intent'], m_intent, intent_mappings):
                             logger.warning(f"⚠️ Low margin detected ({margin:.3f}) for related intent {bert_pred['intent']}. Triggering clarification.")
                             raw_response = intent_mappings[m_intent]['responses'][0] if intent_mappings[m_intent]['responses'] else ""
                             resp_data = self._parse_clarification_response(raw_response)
                             return {
                                "intent": "clarification_needed",
                                "confidence": bert_pred['confidence'],
                                "method": "margin_clarification",
                                "status": "clarification",
                                "message": resp_data['message'],
                                "options": resp_data['options']
                             }

        # STEP 2: Cek LSTM + Pattern Match
        if lstm_pred['status'] == 'success' and passes_hard_min(lstm_pred):
            similarity = self.check_pattern_similarity(text, lstm_pred['intent'], intent_mappings)
            if similarity >= self.thresholds.pattern_similarity:
                return {
                    **lstm_pred,
                    "method": "lstm_direct",
                    "sources": ["lstm", "pattern"],
                    "pattern_similarity": similarity,
                    "fallback_reason": None
                }

        # STEP 3: Cek BERT + Pattern Match
        if bert_pred['status'] == 'success' and passes_hard_min(bert_pred):
            bert_similarity = self.check_pattern_similarity(text, bert_pred['intent'], intent_mappings)
            if bert_similarity >= self.thresholds.pattern_similarity:
                return {
                    **bert_pred,
                    "method": "bert_with_pattern",
                    "sources": ["bert", "pattern"],
                    "pattern_similarity": bert_similarity,
                    "fallback_reason": "lstm_pattern_mismatch"
                }

        # STEP 4: Advanced Hybrid Fusion
        fusion_result = self._hybrid_fusion(text, lstm_pred, bert_pred, intent_mappings)
        
        # Add mitigation check even for fusion result
        if fusion_result['status'] == 'success':
            # Check pattern similarity for the fusion winner
            fusion_sim = self.check_pattern_similarity(text, fusion_result['intent'], intent_mappings)
            
            # If fusion returns high confidence but low similarity, be careful (OOD check)
            if fusion_result['confidence'] > 0.8 and fusion_sim < 0.15:
                 logger.warning(f"⚠️ Fusion high confidence ({fusion_result['confidence']}) but low pattern similarity ({fusion_sim}). Downgrading.")
                 # Just passing through for now, but logged. Could return fallback here if strict.
            
            # Final hard min check
            if passes_hard_min(fusion_result):
                 return {
                     **fusion_result,
                     "pattern_similarity": fusion_sim
                 }

        # STEP 4 & 5: Low Confidence / Fallback
        best_pred = bert_pred if bert_pred['confidence'] >= lstm_pred['confidence'] else lstm_pred
        
        if best_pred['status'] == 'success' and best_pred['confidence'] >= self.thresholds.min_confidence:
             return {
                **best_pred,
                "method": "low_confidence_match",
                "sources": [best_pred.get('model_type', 'unknown')],
                "pattern_similarity": self.check_pattern_similarity(text, best_pred['intent'], intent_mappings),
                "fallback_reason": "low_confidence_gateway"
            }

        # True Emergency Fallback
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "method": "emergency_fallback",
            "status": "success",
            "sources": [],
            "pattern_similarity": 0.0,
            "fallback_reason": "complete_failure"
        }
