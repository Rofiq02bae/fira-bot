import os
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataService:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.intent_mappings = {}
    
    def load_dataset(self, dataset_path: str = None) -> bool:
        """Load dataset dari file CSV"""
        try:
            if dataset_path is None:
                dataset_path = self.config.dataset_path
            
            logger.info(f"📂 Loading dataset from: {dataset_path}")
            
            if not os.path.exists(dataset_path):
                logger.error(f"❌ Dataset file not found: {dataset_path}")
                # Try alternative paths
                alternative_paths = [
                    "dataset/data/dataset_training_lstm.csv",
                    "./dataset/data/dataset_training_lstm.csv",
                    "dataset_training_lstm.csv",
                    "dataset/data/dataset_training_bert.csv",
                    "./dataset/data/dataset_training_bert.csv",
                    "dataset_training_bert.csv"
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        dataset_path = alt_path
                        logger.info(f"🔄 Using alternative dataset path: {dataset_path}")
                        break
                else:
                    logger.error("❌ No valid dataset path found")
                    return False
            
            self.df = pd.read_csv(dataset_path, encoding='utf-8', keep_default_na=False)
            if 'intent' in self.df.columns:
                self.df['intent'] = self.df['intent'].astype(str).str.strip()
                self.df = self.df[self.df['intent'].str.lower() != 'intent']
            logger.info(f"📊 Dataset loaded: {len(self.df)} rows")
            
            # Create intent mappings
            self._create_intent_mappings()
            
            logger.info(f"✅ Intent mappings created: {len(self.intent_mappings)} intents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            return False
    
    def _create_intent_mappings(self):
        """Create intent mappings dari dataset"""
        self.intent_mappings = {}
        
        for intent in self.df['intent'].unique():
            intent_data = self.df[self.df['intent'] == intent]
            response_type = 'static'
            if 'response_type' in intent_data.columns:
                response_type = str(intent_data['response_type'].iloc[0]).strip().lower() or 'static'

            is_master = False
            if 'is_master' in intent_data.columns:
                is_master = any(
                    str(v).strip().lower() in {"1", "true", "yes", "y"}
                    for v in intent_data['is_master'].tolist()
                )

            self.intent_mappings[intent] = {
                'response_type': response_type,
                'is_master': is_master,
                'patterns': intent_data['pattern'].tolist(),
                'responses': intent_data['response'].tolist() if 'response' in intent_data.columns else ["Response not available"]
            }
    
    def get_intent_mappings(self) -> Dict[str, Any]:
        """Get intent mappings"""
        return self.intent_mappings
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        if self.df is None:
            return {"status": "not_loaded"}
        
        return {
            "rows_count": len(self.df),
            "intents_count": len(self.df['intent'].unique()),
            "columns": self.df.columns.tolist(),
            "intents": self.df['intent'].value_counts().to_dict()
        }
    
    def get_intent_patterns(self, intent: str) -> list:
        """Get patterns untuk intent tertentu"""
        if intent in self.intent_mappings:
            return self.intent_mappings[intent].get('patterns', [])
        return []
    
    def get_intent_responses(self, intent: str) -> list:
        """Get responses untuk intent tertentu"""
        if intent in self.intent_mappings:
            return self.intent_mappings[intent].get('responses', [])
        return []
