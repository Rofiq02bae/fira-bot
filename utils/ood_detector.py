import re
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class OODDetector:
    """
    Out-of-Distribution Detector to filter gibberish and irrelevant queries.
    """
    def __init__(self):
        # Indonesian vowels
        self.vowels = set('aeiou')
        self.stop_words = {'dan', 'di', 'ke', 'dari', 'itu', 'yang', 'ah', 'oh', 'eh'}
        
    def is_gibberish(self, text: str) -> bool:
        """
        Check if the text is likely to be gibberish or nonsense.
        """
        if not text or not isinstance(text, str):
            return True
            
        text = text.lower().strip()
        
        # 1. Length check (too short)
        if len(text) < 3:
            return True
            
        # 2. Character repetition check (e.g., "zzzzz", "!!!!!")
        if re.search(r'(.)\1{4,}', text): # 5 or more repeated chars
            return True
            
        # 3. Vowel/Consonant ratio check
        words = text.split()
        if not words:
            return True
            
        # Only check ratio for words longer than 3 characters
        long_words = [w for w in words if len(w) > 3]
        if long_words:
            v_count = 0
            c_count = 0
            for char in "".join(long_words):
                if char.isalpha():
                    if char in self.vowels:
                        v_count += 1
                    else:
                        c_count += 1
            
            if v_count == 0 or (c_count / v_count) > 5: # Highly imbalanced
                return True
                
        # 4. Symbolic density check
        symbol_count = len(re.findall(r'[^\w\s]', text))
        if symbol_count / len(text) > 0.5: # More than 50% symbols
            return True
            
        return False

    def is_meaningful(self, text: str, normalized_text: str) -> bool:
        """
        Check if the text contains meaningful content (not just stop words).
        """
        words = normalized_text.split()
        meaningful_words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        # If no meaningful words and input is short, likely irrelevant
        if not meaningful_words and len(text) < 10:
            return False
            
        return True

    def process(self, text: str, normalized_text: str) -> Dict[str, Any]:
        """
        Complete OOD detection process.
        """
        is_gib = self.is_gibberish(text)
        is_mean = self.is_meaningful(text, normalized_text)
        
        is_ood = is_gib or not is_mean
        
        return {
            "is_ood": is_ood,
            "reason": "gibberish" if is_gib else ("lack_of_meaning" if not is_mean else None)
        }
