import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.processors.response_selector import ResponseSelector

class MockNormalizer:
    def normalize(self, text):
        return text.lower().strip()

def test_response_selector_clarification():
    selector = ResponseSelector(MockNormalizer(), {})
    
    intent_result = {
        "status": "clarification",
        "message": "Silahkan pilih akta mana yang anda maksud?",
        "options": ["Akta Lahir", "Akta Mati"],
        "intent": "clarification_needed"
    }
    
    response = selector.get_response(intent_result, "urus akta")
    
    print(f"\nTEST: Disambiguation Response")
    print(f"EXPECTED: {intent_result['message']}")
    print(f"ACTUAL  : {response}")
    
    if response == intent_result['message']:
        print("✅ SUCCESS: Response message matches clarification message.")
    else:
        print("❌ FAILURE: Response message does not match.")

if __name__ == "__main__":
    test_response_selector_clarification()
