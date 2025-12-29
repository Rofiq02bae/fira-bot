import sys
import os
import logging

# Tambahkan project root ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_hybrid_nlu

# Setup logging
logging.basicConfig(level=logging.WARNING)

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_disambiguation():
    nlu = get_hybrid_nlu()
    
    test_cases = [
        # Ambiguous keyword-based
        {"text": "urus akta", "expected_status": "clarification", "desc": "Ambiguous Akta"},
        {"text": "surat pindah", "expected_status": "clarification", "desc": "Ambiguous Pindah"},
        
        # Specific bypass filters
        {"text": "akta lahir", "expected_status": "success", "desc": "Specific Akta Lahir"},
        {"text": "akta kematian", "expected_status": "success", "desc": "Specific Akta Mati"},
        {"text": "surat pindah masuk", "expected_status": "success", "desc": "Specific Pindah Masuk"},
        
        # Valid non-ambiguous
        {"text": "halo fira", "expected_status": "success", "desc": "Greeting"},
    ]
    
    print("\n" + "="*80)
    print(f"{'QUERY':<25} | {'EXPECTED':<15} | {'ACTUAL':<15} | {'STATUS'}")
    print("-" * 80)
    
    for case in test_cases:
        result = nlu.process_query(case['text'])
        actual_status = result.get('status', 'unknown')
        
        status_icon = "✅" if actual_status == case['expected_status'] else "❌"
        
        print(f"{case['text']:<25} | {case['expected_status']:<15} | {actual_status:<15} | {status_icon}")
        print(f"   [RESPONSE]: {result.get('response')}")
        
        if actual_status == 'clarification':
            # print(f"   [MESSAGE]: {result.get('message')}")
            print(f"   [OPTIONS]: {[opt['label'] for opt in result.get('options', [])]}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    test_disambiguation()
