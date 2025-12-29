import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ood_detector import OODDetector

def test_ood():
    detector = OODDetector()
    
    test_cases = [
        "halo min",                # Meaningful
        "bjirr",                   # Gibberish (Repetition or Meaningless)
        "aksjdhfaksjdhf",          # Gibberish (Vowel/Consonant)
        "zzzzzzzzzz",              # Repetition
        "p",                       # Too short
        "kantor dimana ya pak?",   # Meaningful
        "random text 123",         # Meaningful enough
        "!!!!!",                   # Symbolic density
        "bapak bapak bappenda",    # Meaningful
    ]
    
    print("\n" + "="*50)
    print(f"{'TEXT':<30} | {'IS OOD':<7} | {'REASON':<15}")
    print("-" * 50)
    
    for text in test_cases:
        # Mocking normalized text for simple test
        normalized = text.lower().strip()
        result = detector.process(text, normalized)
        
        print(f"{text[:30]:<30} | {str(result['is_ood']):<7} | {str(result['reason']):<15}")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_ood()
