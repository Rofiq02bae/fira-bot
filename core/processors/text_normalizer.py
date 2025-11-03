import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Text normalizer dengan improved handling untuk avoid data leakage"""
    
    def __init__(self, aggressive_normalization: bool = False):
        """
        Args:
            aggressive_normalization: Jika True, lakukan normalization lebih aggressive
                                     Jika False, pertahankan beberapa variasi untuk avoid data leakage
        """
        self.aggressive = aggressive_normalization
        
        # Core normalization untuk typo critical
        self.critical_corrections = {
            # Hanya fix yang benar-benar critical
            'bapenda': 'bappenda',
            'dinsoss': 'dinsos',
            'disoss': 'dinsos',
            'kt': 'ktp',
        }
        
        # Optional corrections - hanya jika aggressive mode
        self.optional_corrections = {
            'gimana': 'bagaimana',
            'gmn': 'bagaimana', 
            'bgmn': 'bagaimana',
            'pukul': 'jam',
            'pkl': 'jam',
        }
        
        # Regex patterns dengan caution
        self.regex_patterns = {
            r'bap+enda': 'bappenda',  # bapenda, bappenda -> bappenda
            r'dinsos+': 'dinsos',     # dinsos, dinsoss -> dinsos
        }
        
        # Words to preserve (jangan di-normalize untuk maintain diversity)
        self.preserve_words = {
            'dimana', 'kemana', 'kapan', 'berapa', 'bagaimana',
            'apa', 'kenapa', 'siapa', 'lokasi', 'alamat'
        }
    
    def normalize(self, text: str, preserve_variations: bool = True) -> str:
        """
        Normalize text dengan opsi untuk preserve variations
        
        Args:
            text: Input text
            preserve_variations: Jika True, pertahankan beberapa variasi untuk avoid data leakage
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_text = text
        text = text.lower().strip()
        
        # Step 1: Critical fixes only (typo yang mengubah makna)
        for wrong, correct in self.critical_corrections.items():
            text = text.replace(wrong, correct)
        
        # Step 2: Optional fixes hanya jika aggressive mode
        if self.aggressive:
            for wrong, correct in self.optional_corrections.items():
                text = text.replace(wrong, correct)
        
        # Step 3: Regex patterns dengan word boundary preservation
        for pattern, replacement in self.regex_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Step 4: Clean text - tapi preserve some variations
        if preserve_variations:
            # Jangan remove punctuation sepenuhnya, maintain some diversity
            text = re.sub(r'[^\w\s?]', ' ', text)  # Keep question marks
        else:
            text = re.sub(r'[^\w\s]', ' ', text)   # Remove all punctuation
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Log jika normalization membuat perubahan signifikan
        if original_text != text and len(original_text) - len(text) > 5:
            logger.debug(f"Normalization: '{original_text}' -> '{text}'")
        
        return text
    
    def normalize_batch(self, texts: List[str], preserve_variations: bool = True) -> List[str]:
        """Normalize batch of texts"""
        return [self.normalize(text, preserve_variations) for text in texts]
    
    def analyze_normalization_impact(self, patterns: List[str]) -> Dict:
        """Analyze how normalization affects pattern diversity"""
        original_set = set(patterns)
        normalized_set = set(self.normalize_batch(patterns))
        
        return {
            'original_unique': len(original_set),
            'normalized_unique': len(normalized_set),
            'patterns_lost': len(original_set) - len(normalized_set),
            'diversity_preserved': len(normalized_set) / len(original_set) if original_set else 1.0
        }

# Global instance dengan conservative settings
text_normalizer = TextNormalizer(aggressive_normalization=False)