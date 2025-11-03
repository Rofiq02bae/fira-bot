import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import os

logger = logging.getLogger(__name__)

class QueryLogger:
    def __init__(self, log_file: str = "low_confidence_queries.json", 
                 csv_file: str = "training_candidates.csv"):
        # Gunakan absolute path dari current directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.log_file = os.path.join(base_dir, "logs", log_file)
        self.csv_file = os.path.join(base_dir, "data", "training_candidates", csv_file)
        self.low_confidence_threshold = 0.3
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        
        logger.info(f"📝 QueryLogger initialized:")
        logger.info(f"   JSON: {self.log_file}")
        logger.info(f"   CSV: {self.csv_file}")

    def log_low_confidence_query(self, query_data: Dict[str, Any]):
        """Log query dengan confidence rendah untuk evaluasi"""
        # ✅ Capture pertanyaan yang tidak terdeteksi
        # ✅ Simpan untuk evaluasi dan training

        try:
            # Validasi data input
            if not isinstance(query_data, dict) or 'text' not in query_data:
                logger.error(f"❌ Invalid query_data: {query_data}")
                return
            
            confidence = query_data.get('confidence', 0)
            text = query_data.get('text', '')
            
            # Filter hanya yang confidence rendah
            if confidence >= self.low_confidence_threshold:
                logger.debug(f"✅ Confidence {confidence} above threshold, skipping")
                return
            
            logger.info(f"📝 Logging low confidence query: '{text}' (conf: {confidence:.3f})")
            
            # Tambah metadata
            enhanced_data = query_data.copy()
            enhanced_data['logged_at'] = datetime.now().isoformat()
            enhanced_data['evaluated'] = False
            enhanced_data['added_to_dataset'] = False
            
            # Load existing logs
            existing_logs = self._load_logs()
            
            # Cek duplikat (hindari log query yang sama berulang)
            if not self._is_duplicate(existing_logs, text):
                existing_logs.append(enhanced_data)
                
                # Save to JSON
                self._save_logs(existing_logs)
                
                # Append to CSV untuk easy review
                self._append_to_csv(enhanced_data)
                
                logger.info(f"✅ Successfully logged: '{text}'")
            else:
                logger.debug(f"⚠️ Duplicate query skipped: '{text}'")
            
        except Exception as e:
            logger.error(f"❌ Error logging query: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def _load_logs(self) -> List[Dict]:
        """Load existing logs dari file"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading logs: {e}")
            return []

    def _save_logs(self, logs: List[Dict]):
        """Save logs ke file"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error saving logs: {e}")

    def _is_duplicate(self, existing_logs: List[Dict], text: str) -> bool:
        """Cek apakah query sudah ada di logs"""
        try:
            normalized_text = text.lower().strip()
            for log in existing_logs:
                if log.get('text', '').lower().strip() == normalized_text:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return False

    def _append_to_csv(self, query_data: Dict):
        """Append query data ke CSV untuk easy review"""
        try:
            # Prepare data for CSV
            csv_data = {
                'text': query_data.get('text', ''),
                'predicted_intent': query_data.get('predicted_intent', 'unknown'),
                'confidence': query_data.get('confidence', 0),
                'method_used': query_data.get('method_used', 'unknown'),
                'fallback_reason': query_data.get('fallback_reason', ''),
                'logged_at': query_data.get('logged_at', ''),
                'evaluated': query_data.get('evaluated', False),
                'suggested_intent': '',  # Untuk diisi manual
                'suggested_response': '',  # Untuk diisi manual
                'notes': ''  # Untuk catatan evaluator
            }
            
            # Write to CSV
            df = pd.DataFrame([csv_data])
            
            if os.path.exists(self.csv_file):
                # Append without header
                df.to_csv(self.csv_file, mode='a', header=False, index=False, encoding='utf-8')
            else:
                # Create new file with header
                df.to_csv(self.csv_file, index=False, encoding='utf-8')
                
            logger.debug(f"✅ Appended to CSV: {csv_data['text']}")
                
        except Exception as e:
            logger.error(f"❌ Error appending to CSV: {e}")

    def get_training_candidates(self, limit: int = 50) -> pd.DataFrame:
        """Get queries yang bisa dijadikan training data"""
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                # Filter yang belum dievaluasi
                unevaluated = df[df['evaluated'] == False]
                return unevaluated.head(limit)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading training candidates: {e}")
            return pd.DataFrame()

    def mark_as_evaluated(self, text: str, suggested_intent: str = None, suggested_response: str = None):
        """Mark query sebagai sudah dievaluasi"""
        try:
            logs = self._load_logs()
            updated = False
            
            for log in logs:
                if log.get('text') == text:
                    log['evaluated'] = True
                    if suggested_intent:
                        log['suggested_intent'] = suggested_intent
                    if suggested_response:
                        log['suggested_response'] = suggested_response
                    updated = True
            
            if updated:
                self._save_logs(logs)
                self._update_csv_evaluation(text, suggested_intent, suggested_response)
                logger.info(f"✅ Marked as evaluated: '{text}'")
            else:
                logger.warning(f"⚠️ Query not found for evaluation: '{text}'")
            
        except Exception as e:
            logger.error(f"Error marking as evaluated: {e}")

    def _update_csv_evaluation(self, text: str, suggested_intent: str = None, suggested_response: str = None):
        """Update evaluation status di CSV"""
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                mask = df['text'] == text
                
                if mask.any():
                    df.loc[mask, 'evaluated'] = True
                    if suggested_intent:
                        df.loc[mask, 'suggested_intent'] = suggested_intent
                    if suggested_response:
                        df.loc[mask, 'suggested_response'] = suggested_response
                    
                    df.to_csv(self.csv_file, index=False, encoding='utf-8')
                    logger.debug(f"✅ Updated CSV for: '{text}'")
                else:
                    logger.warning(f"⚠️ Query not found in CSV: '{text}'")
                    
        except Exception as e:
            logger.error(f"Error updating CSV evaluation: {e}")

# Global instance
query_logger = QueryLogger()