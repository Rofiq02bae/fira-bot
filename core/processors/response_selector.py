"""
Response Selector
v3: routing response mengikuti aturan kombinasi response_type + is_master
    - static + is_master=false -> return base response terbaik (boleh di-augment RAG di nlu_service)
    - static + is_master=true  -> return response asli dataset (tanpa RAG)
    - kb + is_master=false     -> return response asli dataset (tanpa RAG)
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ResponseSelector:
    def __init__(self, normalizer, intent_mappings: Dict):
        self.normalizer = normalizer
        self.intent_mappings = intent_mappings

    def get_response(self, intent_result: Dict, user_text: str) -> str:
        """Get response berdasarkan intent result"""

        # Handle clarification status dari disambiguation logic
        if intent_result.get("status") == "clarification":
            return intent_result.get("message", "Silahkan pilih opsi berikut:")

        intent = str(intent_result["intent"]).strip()
        method_used = intent_result.get("method", "default")
        pattern_similarity = intent_result.get("pattern_similarity", 0.0)

        # ── Routing berdasarkan response_type + is_master ──
        intent_data = self.intent_mappings.get(intent, {})
        response_type = str(intent_data.get("response_type", "static")).strip().lower()
        is_master = bool(intent_data.get("is_master", False))
        responses = intent_data.get("responses", [])

        if response_type == "kb":
            return responses[0] if responses else "Informasi prosedur tidak tersedia."

        if response_type == "static" and is_master:
            return responses[0] if responses else "Informasi tidak tersedia."

        # static non-master: proses pemilihan response terbaik berbasis pattern.
        return self._get_best_response(intent, user_text, method_used, pattern_similarity)

    def _get_best_response(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> str:
        """Get response berdasarkan intent — logika existing tidak berubah"""

        intent_key = str(intent).strip()
        if intent_key not in self.intent_mappings:
            sample_keys = list(self.intent_mappings.keys())[:5]
            logger.warning(
                "⚠️ ResponseSelector: intent '%s' not found (method=%s, mappings=%d, sample=%s)",
                intent_key, method_used, len(self.intent_mappings), sample_keys,
            )
            if method_used == "bert_direct":
                return "Saya memahami pertanyaan Anda tentang informasi Bappenda Tegal. Untuk informasi lengkap, silakan hubungi Bappenda Tegal langsung."
            else:
                return "Maaf, saya belum memahami pertanyaan Anda. Bisakah Anda mengulangi pertanyaannya dengan kata-kata yang berbeda?"

        intent_data = self.intent_mappings[intent_key]
        user_words = set(self.normalizer.normalize(user_text).split())
        best_score = -1.0
        best_response = intent_data['responses'][0] if intent_data.get('responses') else "Maaf, saya belum memahami pertanyaan Anda."

        for pattern, response in zip(intent_data.get('patterns', []), intent_data.get('responses', [])):
            pattern_words = set(self.normalizer.normalize(str(pattern)).split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))

            if total_words > 0:
                similarity = common_words / total_words
                score = similarity * pattern_similarity if method_used in ["lstm_direct", "bert_with_pattern"] else similarity

                if score > best_score:
                    best_score = score
                    best_response = response

        logger.debug(f"Selected response for '{intent}': score={best_score:.3f}")
        return best_response

    def get_response_scores(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> Dict:
        """Debug method — lihat semua candidate responses dan scores-nya"""
        intent_key = str(intent).strip()
        if intent_key not in self.intent_mappings:
            return {"error": "Intent not found"}

        intent_data = self.intent_mappings[intent_key]
        user_words = set(self.normalizer.normalize(user_text).split())
        candidates = []

        for pattern, response in zip(intent_data.get('patterns', []), intent_data.get('responses', [])):
            pattern_words = set(self.normalizer.normalize(str(pattern)).split())
            common_words = len(user_words.intersection(pattern_words))
            total_words = len(user_words.union(pattern_words))

            if total_words > 0:
                similarity = common_words / total_words
                score = similarity * pattern_similarity if method_used in ["lstm_direct", "bert_with_pattern"] else similarity

                candidates.append({
                    "pattern":      pattern,
                    "response":     response,
                    "similarity":   similarity,
                    "score":        score,
                    "common_words": common_words,
                    "total_words":  total_words
                })

        return {
            "intent":            intent,
            "user_text":         user_text,
            "method_used":       method_used,
            "pattern_similarity": pattern_similarity,
            "candidates":        sorted(candidates, key=lambda x: x["score"], reverse=True)
        }
