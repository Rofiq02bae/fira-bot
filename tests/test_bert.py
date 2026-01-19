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
                f"{self.base_url}/api/chat-bert",
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
# Test Cases Definition
def get_test_suite() -> List[Dict]:
    """
    Define comprehensive test cases based on the 14 balanced intents.
    Setiap intent diwakili oleh minimal 2-3 variasi kalimat.
    """
    return [
        # 1. SALAM & TERIMAKASIH (Standard Bot Intent)
        {"text": "halo", "expected_intent": "salam"},
        {"text": "assalamualaikum", "expected_intent": "salam"},
        {"text": "terima kasih banyak", "expected_intent": "terimakasih"},
        
        # 2. POPULATION DOCUMENTS (KTP, KK, Akta, Pindah)
        {"text": "cara buat ktp baru", "expected_intent": "ktp_info"},
        {"text": "syarat rekam ktp tegal", "expected_intent": "ktp_info"},
        {"text": "cara buat kartu keluarga", "expected_intent": "kk_info"},
        {"text": "syarat tambah anggota kk", "expected_intent": "kk_info"},
        {"text": "cara buat akta kelahiran", "expected_intent": "akta_lahir_info"},
        {"text": "syarat akta lahir anak", "expected_intent": "akta_lahir_info"},
        {"text": "cara urus akta kematian", "expected_intent": "mati_info"},
        {"text": "syarat akta kematian sipandu", "expected_intent": "mati_info"},
        {"text": "cara urus surat pindah keluar", "expected_intent": "surat_pindah_luar"},
        {"text": "cabut berkas kk tegal", "expected_intent": "surat_pindah_luar"},
        
        # 3. BUSINESS & LICENSING (NIB, SLS, LKPM)
        {"text": "cara buat nib di tegal", "expected_intent": "nib"},
        {"text": "daftar oss rba online", "expected_intent": "nib"},
        {"text": "apa keuntungan punya nib", "expected_intent": "nib_info"},
        {"text": "manfaat nomor induk berusaha", "expected_intent": "nib_info"},
        {"text": "apa itu sertifikat laik sehat", "expected_intent": "sls_info"},
        {"text": "syarat sls dinkes mpp", "expected_intent": "sls_info"},
        {"text": "apa itu lkpm", "expected_intent": "lkpm_info"},
        {"text": "cara lapor lkpm di oss", "expected_intent": "lkpm_info"},

        # 4. INFRASTRUCTURE & LAND (ITR)
        {"text": "apa itu itr", "expected_intent": "itr_info"},
        {"text": "cara urus informasi tata ruang", "expected_intent": "itr_info"},

        # 5. SPECIAL SERVICES (LOAKK, SICANTIK, AK1)
        {"text": "apa itu layanan loakk", "expected_intent": "loakk_info"},
        {"text": "paket akta kk kia bayi", "expected_intent": "loakk_info"},
        {"text": "apa itu layanan sicantik", "expected_intent": "sicantik_info"},
        {"text": "syarat sicantik cerai tegal", "expected_intent": "sicantik_info"},
        {"text": "bagaimana cara buat kartu kuning", "expected_intent": "ak1"},
        {"text": "syarat bikin ak1 di tegal", "expected_intent": "ak1"},

        # 6. GENERAL INFO (Jam Buka)
        {"text": "jam berapa pelayanan buka", "expected_intent": "jam_buka_layanan"},
        {"text": "hari sabtu buka tidak", "expected_intent": "jam_buka_layanan"},
        {"text": "jadwal operasional mpp", "expected_intent": "jam_buka_layanan"},

        # 7. OUT OF TOPIC / EDGE CASES
        {"text": "bjirr", "expected_intent": "out_of_topic"},
        {"text": "makan apa hari ini", "expected_intent": "out_of_topic"},
        {"text": "siapa presiden indonesia", "expected_intent": "out_of_topic"}
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
        results_df.to_csv(f"D:\\bot\\New folder\\logs\\test_results_{timestamp}.csv", index=False)
        print(f"\n💾 Results saved to: D:\\bot\\New folder\\logs\\test_results_{timestamp}.csv")
        
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