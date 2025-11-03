import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FallbackService:
    def __init__(self, normalizer, intent_mappings: Dict):
        self.normalizer = normalizer
        self.intent_mappings = intent_mappings
        
        # Fallback responses untuk berbagai skenario
        self.fallback_responses = {
            "no_intent_matched": "Maaf, saya belum memahami pertanyaan Anda. Bisakah Anda mengulangi pertanyaannya dengan kata-kata yang berbeda?",
            "bert_direct_fallback": "Maaf, saya belum memahami pertanyaan anda. Mungkin Anda bisa mengulang pertanyaan dengan kalimat yang berbeda?",
            "low_confidence": "Saya tidak yakin dengan pertanyaan Anda. Bisa Anda memberikan lebih banyak detail?",
            "service_unavailable": "Maaf, layanan sedang mengalami gangguan. Silakan coba lagi nanti.",
            "processing_error": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda."
        }
        
        # Keywords untuk intent detection fallback
        self.keyword_mappings = {
            'bappenda': ['bappenda', 'bapenda', ' Badan Perencanaan Pembangunan Daerah Penelitian dan Pengembangan'],
            'dinsos': ['dinsos', 'dinsoss', 'dinas sosial'],
            'jam': ['jam', 'pukul', 'buka', 'tutup'],
            'alamat': ['alamat', 'lokasi', 'dimana', 'dimana alamat'],
            'ktp': ['ktp', 'kartu tanda penduduk', 'buat ktp']
        }

    def get_fallback_response(self, fallback_reason: str, user_text: str = "") -> str:
        """Get fallback response berdasarkan reason"""
        response = self.fallback_responses.get(fallback_reason, self.fallback_responses["no_intent_matched"])
        
        # Jika ada user_text, coba berikan response yang lebih kontekstual
        if user_text and fallback_reason in ["no_intent_matched", "low_confidence"]:
            contextual_response = self._get_contextual_fallback(user_text)
            if contextual_response:
                return contextual_response
        
        return response

    def _get_contextual_fallback(self, user_text: str) -> str:
        """Coba berikan fallback response yang lebih kontekstual berdasarkan keywords"""
        normalized_text = self.normalizer.normalize(user_text)
        
        # Cek keywords untuk memberikan response yang lebih spesifik
        for intent_keywords, keywords in self.keyword_mappings.items():
            for keyword in keywords:
                if keyword in normalized_text:
                    if intent_keywords == 'bappenda':
                        return "Untuk informasi lengkap tentang Bappenda Tegal, silakan hubungi langsung atau kunjungi kantor Bappenda."
                    elif intent_keywords == 'dinsos':
                        return "Untuk informasi layanan Dinas Sosial Tegal, silakan hubungi langsung kantor Dinsos Tegal."
                    elif intent_keywords == 'jam':
                        return "Untuk informasi jam operasional, silakan hubungi instansi terkait langsung."
                    elif intent_keywords == 'alamat':
                        return "Saya bisa membantu memberikan informasi alamat. Instansi mana yang Anda tanyakan?"
                    elif intent_keywords == 'ktp':
                        return "Untuk pembuatan KTP, silakan kunjungi kantor kelurahan atau Dinas Kependudukan dan Catatan Sipil setempat."
        
        return ""

    def should_use_fallback(self, confidence: float, pattern_similarity: float, thresholds: Dict) -> bool:
        """Determine apakah harus menggunakan fallback berdasarkan confidence dan similarity"""
        return (confidence <= thresholds.get('min_confidence', 0.3) or 
                pattern_similarity <= thresholds.get('min_similarity', 0.2))

    def get_emergency_fallback(self, user_text: str = "") -> Dict[str, Any]:
        """Get emergency fallback ketika semua method gagal"""
        contextual_response = self._get_contextual_fallback(user_text)
        
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "response": contextual_response if contextual_response else self.fallback_responses["no_intent_matched"],
            "method": "emergency_fallback",
            "status": "success",
            "sources": [],
            "pattern_similarity": 0.0,
            "fallback_reason": "complete_failure"
        }