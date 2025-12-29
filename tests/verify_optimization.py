import logging
import sys
import os

# Menambahkan parent directory ke sys.path agar bisa import module dari project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_hybrid_nlu

# Setup simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_queries():
    nlu = get_hybrid_nlu()
    
    test_cases = [
        "halo fira",                # Valid / Greeter
        "bjirr",                   # Gibberish / OOD
        "aksjdhfaksjdhf",          # Gibberish / OOD
        "gimana cara daftar ktp?",     # Valid / Variation
        "zzzzzzzzzz",              # Repetition / OOD
        "p",                       # Too short / OOD
        "kapan kantor pajak buka ?"   # Valid / Location
    ]
    
    print("\n--- HYBRID NLU VERIFICATION ---")
    for text in test_cases:
        print(f"\nQUERY: '{text}'")
        result = nlu.process_query(text)
        print(f"INTENT: {result['intent']}")
        print(f"CONFIDENCE: {result['confidence']:.3f}")
        print(f"METHOD: {result['method']}")
        print(f"REASON: {result.get('fallback_reason', 'None')}")
        
if __name__ == "__main__":
    try:
        test_queries()
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()
