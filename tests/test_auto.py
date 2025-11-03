#!/usr/bin/env python3
"""
Automated Testing untuk NLU System
Test multiple patterns dengan satu command
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class NLUAutomatedTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.session = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def test_single_query(self, text: str, expected_intent: str = None) -> Dict[str, Any]:
        """Test single query dan return results"""
        try:
            start_time = time.time()
            
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            ) as response:
                
                processing_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    result = {
                        'text': text,
                        'expected_intent': expected_intent,
                        'predicted_intent': data.get('predicted_intent'),
                        'confidence': data.get('confidence', 0),
                        'method_used': data.get('method_used'),
                        'response': data.get('response'),
                        'processing_time_ms': round(processing_time, 2),
                        'success': data.get('predicted_intent') == expected_intent if expected_intent else True,
                        'status': 'success'
                    }
                    
                    # Check pattern similarity jika predicted intent tidak match expected
                    if expected_intent and data.get('predicted_intent') != expected_intent:
                        similarity_result = await self.check_pattern_similarity(text, expected_intent)
                        result['pattern_similarity'] = similarity_result
                    
                    return result
                else:
                    return {
                        'text': text,
                        'expected_intent': expected_intent,
                        'predicted_intent': 'ERROR',
                        'confidence': 0,
                        'method_used': 'error',
                        'response': f"HTTP {response.status}",
                        'processing_time_ms': round(processing_time, 2),
                        'success': False,
                        'status': f'http_error_{response.status}'
                    }
                    
        except Exception as e:
            return {
                'text': text,
                'expected_intent': expected_intent, 
                'predicted_intent': 'EXCEPTION',
                'confidence': 0,
                'method_used': 'error',
                'response': str(e),
                'processing_time_ms': 0,
                'success': False,
                'status': 'exception'
            }
    
    async def check_pattern_similarity(self, text: str, intent: str) -> float:
        """Check pattern similarity untuk debugging"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/debug-response-scores",
                params={"text": text, "intent": intent, "method": "hybrid"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('pattern_similarity', 0)
        except:
            pass
        return 0
    
    async def run_test_suite(self, test_cases: List[Dict]) -> pd.DataFrame:
        """Run semua test cases secara parallel"""
        print("🚀 Starting Automated NLU Testing...")
        print(f"📋 Total test cases: {len(test_cases)}")
        print("=" * 80)
        
        tasks = []
        for test_case in test_cases:
            task = self.test_single_query(
                test_case['text'], 
                test_case.get('expected_intent')
            )
            tasks.append(task)
        
        # Run semua tests secara concurrent
        self.results = await asyncio.gather(*tasks)
        
        # Generate report
        return self.generate_report()
    
    def generate_report(self) -> pd.DataFrame:
        """Generate comprehensive test report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        total_tests = len(df)
        successful_tests = len(df[df['success'] == True])
        accuracy = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 TEST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"✅ Successful: {successful_tests}/{total_tests} ({accuracy:.1f}%)")
        print(f"🕒 Avg Processing Time: {df['processing_time_ms'].mean():.2f}ms")
        print(f"🎯 Avg Confidence: {(df['confidence'].mean() * 100):.1f}%")
        
        # Show failed tests
        failed_tests = df[df['success'] == False]
        if not failed_tests.empty:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            print("-" * 40)
            for _, test in failed_tests.iterrows():
                print(f"Text: '{test['text']}'")
                print(f"  Expected: {test['expected_intent']}")
                print(f"  Got: {test['predicted_intent']} (conf: {test['confidence']:.3f})")
                print(f"  Method: {test['method_used']}")
                if 'pattern_similarity' in test:
                    print(f"  Pattern Similarity: {test['pattern_similarity']:.3f}")
                print()
        
        # Show method distribution
        method_counts = df['method_used'].value_counts()
        print("\n🔧 METHOD DISTRIBUTION:")
        for method, count in method_counts.items():
            print(f"  {method}: {count} tests")
        
        return df
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()

# Test Cases Definition
def get_test_suite() -> List[Dict]:
    """Define comprehensive test cases"""
    return [
        # SIMPLE GREETINGS - Critical tests!
        {"text": "halo", "expected_intent": "salam"},
        {"text": "hai", "expected_intent": "salam"},
        {"text": "hi", "expected_intent": "salam"},
        {"text": "hello", "expected_intent": "salam"},
        {"text": "selamat pagi", "expected_intent": "salam"},
        {"text": "assalamualaikum", "expected_intent": "salam"},
        
        # THANKS
        {"text": "terimakasih", "expected_intent": "terimakasih"},
        {"text": "makasih", "expected_intent": "terimakasih"},
        {"text": "terima kasih banyak", "expected_intent": "terimakasih"},
        
        # BAPPENDA
        {"text": "bappenda buka jam berapa", "expected_intent": "bappenda_info"},
        
        # DINSOS
        {"text": "jam buka dinsos", "expected_intent": "dinsos_jam_operasional"},
        
        # KTP
        {"text": "cara buat ktp", "expected_intent": "ktp_info"},
        
        # EDGE CASES
        {"text": "bapenda", "expected_intent": "bappenda_info"},  # Typo test
        {"text": "dinsoss", "expected_intent": "dinsos_info"},    # Typo test
        {"text": "apa kabar"},  # No expected intent
        {"text": "cuaca hari ini"},  # Should fallback
        {"text": "saran"},  # Should fallback
        {"text": "siapa kamu"} 
    ]

async def main():
    """Main function"""
    tester = NLUAutomatedTester()
    
    try:
        await tester.initialize()
        
        # Get test suite
        test_cases = get_test_suite()
        
        # Run tests
        results_df = await tester.run_test_suite(test_cases)
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"../logs/test_results_{timestamp}.csv", index=False)
        print(f"\n💾 Results saved to: ../logs/test_results_{timestamp}.csv")
        
        # Show quick summary
        print(f"\n🎯 QUICK SUMMARY:")
        print(f"   Greetings Accuracy: Check 'halo' vs 'hai' results")
        print(f"   Overall System Health: {len(results_df[results_df['status'] == 'success'])}/{len(results_df)} successful")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
    finally:
        await tester.close()

if __name__ == "__main__":
    asyncio.run(main())