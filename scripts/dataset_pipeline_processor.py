#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Pipeline Processor
Implements 4-step dataset processing pipeline for LSTM and BERT models
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for dataset processing"""
    input_file: str
    output_lstm: str
    output_bert: str
    validation_split: float = 0.2
    random_state: int = 42
    remove_duplicates: bool = True
    normalize_text: bool = True
    split_patterns: bool = True
    convert_responses: bool = True


class DatasetPipelineProcessor:
    """
    Main processor class implementing 4-step pipeline:
    1. Remove Duplicates
    2. Validate & Fix
    3. Split Patterns
    4. Convert Responses
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data" / "dataset"
        
        # Create output directories
        (self.data_dir / "lstm").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "bert").mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "step_1": {},
            "step_2": {},
            "step_3": {},
            "step_4": {}
        }
    
    def process_all(self) -> Dict:
        """Execute complete pipeline"""
        try:
            logger.info("Starting dataset processing pipeline...")
            
            # Step 1: Remove Duplicates
            df1 = self.step_1_remove_duplicates(self.config.input_file)
            if df1 is None:
                return {"success": False, "error": "Step 1 failed"}
            
            # Step 2: Validate & Fix
            df2 = self.step_2_validate_fix(df1)
            if df2 is None:
                return {"success": False, "error": "Step 2 failed"}
            
            # Step 3: Split Patterns
            df3 = self.step_3_split_patterns(df2)
            if df3 is None:
                return {"success": False, "error": "Step 3 failed"}
            
            # Step 4: Convert Responses & Split
            lstm_path, bert_path = self.step_4_convert_responses(df3)
            
            logger.info("✓ Pipeline completed successfully")
            return {
                "success": True,
                "lstm_output": lstm_path,
                "bert_output": bert_path,
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {"success": False, "error": str(e)}
    
    def step_1_remove_duplicates(self, input_file: str) -> Optional[pd.DataFrame]:
        """
        Step 1: Remove duplicate patterns within each row
        - Splits patterns by '|' delimiter
        - Removes duplicates (case-insensitive)
        - Cleans whitespace
        """
        try:
            logger.info("Step 1: Removing duplicate patterns...")
            
            # Read CSV
            df = pd.read_csv(input_file, encoding='utf-8')
            
            initial_rows = len(df)
            total_patterns_before = 0
            total_patterns_after = 0
            rows_with_duplicates = 0
            
            cleaned_data = []
            
            for _, row in df.iterrows():
                # Handle different column names
                intent = row.get('intent', row.iloc[0]) if hasattr(row, 'get') else row.iloc[0]
                patterns_str = row.get('patterns', row.get('pattern', row.iloc[1])) if hasattr(row, 'get') else row.iloc[1]
                response_type = row.get('response_type', row.iloc[2] if len(row) > 2 else 'static') if hasattr(row, 'get') else (row.iloc[2] if len(row) > 2 else 'static')
                response = row.get('response', row.iloc[-1] if len(row) > 2 else '') if hasattr(row, 'get') else (row.iloc[-1] if len(row) > 2 else '')
                is_master = row.get('is_master', 'false') if hasattr(row, 'get') and len(row) > 4 else 'false'
                
                # Split patterns
                patterns = str(patterns_str).split('|')
                total_patterns_before += len(patterns)
                
                # Remove duplicates
                cleaned_patterns = []
                seen = set()
                
                for pattern in patterns:
                    pattern_clean = pattern.strip()
                    if not pattern_clean:
                        continue
                    
                    pattern_lower = pattern_clean.lower()
                    if pattern_lower not in seen:
                        cleaned_patterns.append(pattern_clean)
                        seen.add(pattern_lower)
                
                total_patterns_after += len(cleaned_patterns)
                
                if len(patterns) != len(cleaned_patterns):
                    rows_with_duplicates += 1
                
                if cleaned_patterns:
                    cleaned_data.append({
                        'intent': intent,
                        'pattern': '|'.join(cleaned_patterns),
                        'response_type': response_type,
                        'is_master': is_master,
                        'response': response
                    })
            
            result_df = pd.DataFrame(cleaned_data)
            
            self.stats["step_1"] = {
                "rows_before": initial_rows,
                "rows_after": len(result_df),
                "patterns_before": total_patterns_before,
                "patterns_after": total_patterns_after,
                "duplicates_removed": total_patterns_before - total_patterns_after,
                "rows_with_duplicates": rows_with_duplicates
            }
            
            logger.info(f"  ✓ Removed {total_patterns_before - total_patterns_after} duplicate patterns")
            logger.info(f"  ✓ {rows_with_duplicates} rows had duplicates")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in step 1: {e}")
            return None
    
    def step_2_validate_fix(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Step 2: Validate data and fix issues
        - Check for null values
        - Validate column count
        - Remove rows with empty patterns
        - Normalize response types
        """
        try:
            logger.info("Step 2: Validating and fixing data...")
            
            initial_rows = len(df)
            
            # Remove rows with missing critical fields
            df = df.dropna(subset=['intent', 'pattern', 'response'])
            df = df[df['pattern'].astype(str).str.strip() != '']
            df = df[df['intent'].astype(str).str.strip() != '']
            
            # Normalize response_type
            response_types = ['static', 'dynamic', 'list']
            if 'response_type' in df.columns:
                mask = ~df['response_type'].isin(response_types)
                df.loc[mask, 'response_type'] = 'static'
            else:
                df['response_type'] = 'static'
            
            # Ensure is_master is boolean string
            if 'is_master' in df.columns:
                df['is_master'] = df['is_master'].astype(str).str.lower()
                mask = ~df['is_master'].isin(['true', 'false'])
                df.loc[mask, 'is_master'] = 'false'
            else:
                df['is_master'] = 'false'
            
            rows_removed = initial_rows - len(df)
            
            self.stats["step_2"] = {
                "rows_before": initial_rows,
                "rows_after": len(df),
                "rows_removed": rows_removed,
                "valid_intents": df['intent'].nunique(),
                "valid_response_types": df['response_type'].unique().tolist() if 'response_type' in df.columns else []
            }
            
            logger.info(f"  ✓ Removed {rows_removed} invalid rows")
            logger.info(f"  ✓ {len(df)} valid rows remaining")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in step 2: {e}")
            return None
    
    def step_3_split_patterns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Step 3: Split multiple patterns into separate rows
        - Each pattern in '|' separated list becomes a separate row
        - Preserves intent, response_type, response
        """
        try:
            logger.info("Step 3: Splitting patterns into separate rows...")
            
            initial_rows = len(df)
            new_rows = []
            
            for _, row in df.iterrows():
                patterns = [p.strip() for p in str(row['pattern']).split('|') if p.strip()]
                
                for pattern in patterns:
                    new_row = {
                        'intent': row['intent'],
                        'pattern': pattern,
                        'response': row['response']
                    }
                    if 'response_type' in row:
                        new_row['response_type'] = row['response_type']
                    new_rows.append(new_row)
            
            result_df = pd.DataFrame(new_rows)
            
            final_rows = len(result_df)
            
            self.stats["step_3"] = {
                "rows_before": initial_rows,
                "rows_after": final_rows,
                "rows_created": final_rows - initial_rows,
                "avg_patterns_per_row": final_rows / initial_rows if initial_rows > 0 else 0
            }
            
            logger.info(f"  ✓ Split into {final_rows} pattern rows")
            logger.info(f"  ✓ Average {final_rows/initial_rows:.2f} patterns per original row")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in step 3: {e}")
            return None
    
    def step_4_convert_responses(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Step 4: Convert responses to JSON format and create train/val splits
        - Converts responses to JSON format based on response_type
        - Creates train/val split: 80/20
        - Saves to LSTM and BERT datasets
        """
        try:
            logger.info("Step 4: Converting responses and splitting dataset...")
            
            if df is None or len(df) == 0:
                logger.error("Input dataframe is empty")
                return "", ""
            
            # Convert responses to JSON
            try:
                df['response_json'] = df.apply(self._convert_response_to_json, axis=1)
            except Exception as e:
                logger.error(f"Error converting responses to JSON: {e}")
                raise
            
            # Prepare data for split - ensure we have required columns
            if 'response' not in df.columns:
                logger.error("Missing 'response' column")
                return "", ""
            
            columns_to_keep = ['intent', 'pattern', 'response']
            if 'response_type' in df.columns:
                columns_to_keep.insert(2, 'response_type')
            
            try:
                df_processed = df[columns_to_keep].copy()
            except KeyError as e:
                logger.error(f"Missing column: {e}")
                return "", ""
            
            # Replace response with JSON version
            df_processed['response'] = df['response_json'].values
            
            # Create train/val split
            try:
                le = LabelEncoder()
                labels = le.fit_transform(df_processed['intent'])
                
                train_idx, val_idx = train_test_split(
                    range(len(df_processed)),
                    test_size=self.config.validation_split,
                    random_state=self.config.random_state,
                    stratify=labels
                )
            except Exception as e:
                logger.error(f"Error in train/val split: {e}")
                raise
            
            df_train = df_processed.iloc[train_idx].reset_index(drop=True)
            df_val = df_processed.iloc[val_idx].reset_index(drop=True)
            
            # Combine for final dataset (both train and val)
            df_final = pd.concat([df_train, df_val], ignore_index=True)
            
            # Create output directories
            try:
                (self.data_dir / "lstm").mkdir(parents=True, exist_ok=True)
                (self.data_dir / "bert").mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating output directories: {e}")
                raise
            
            # Save to LSTM format
            lstm_path = str(self.data_dir / "lstm" / "dataset_training_lstm.csv")
            try:
                df_final.to_csv(lstm_path, index=False, encoding='utf-8')
            except Exception as e:
                logger.error(f"Error saving LSTM dataset: {e}")
                raise
            
            # Save to BERT format
            bert_path = str(self.data_dir / "bert" / "dataset_training_bert.csv")
            try:
                df_final.to_csv(bert_path, index=False, encoding='utf-8')
            except Exception as e:
                logger.error(f"Error saving BERT dataset: {e}")
                raise
            
            # Calculate statistics
            train_intents = df_train['intent'].value_counts()
            val_intents = df_val['intent'].value_counts()
            
            self.stats["step_4"] = {
                "total_rows": len(df_final),
                "train_rows": len(df_train),
                "val_rows": len(df_val),
                "train_split_percentage": (len(df_train) / len(df_final)) * 100 if len(df_final) > 0 else 0,
                "unique_intents": df_final['intent'].nunique(),
                "response_types": df_final['response_type'].unique().tolist() if 'response_type' in df_final.columns else [],
                "train_intents": train_intents.to_dict(),
                "val_intents": val_intents.to_dict()
            }
            
            logger.info(f"  ✓ Converted {len(df_final)} responses to JSON")
            logger.info(f"  ✓ Split: {len(df_train)} train, {len(df_val)} validation")
            logger.info(f"  ✓ LSTM dataset: {lstm_path}")
            logger.info(f"  ✓ BERT dataset: {bert_path}")
            
            return lstm_path, bert_path
            
        except Exception as e:
            logger.error(f"Error in step 4: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "", ""
    
    @staticmethod
    def _convert_response_to_json(row) -> str:
        """Convert response to JSON based on response_type"""
        response_text = str(row['response']).strip()
        response_type = str(row.get('response_type', 'static')).lower()
        
        if response_type == 'list' and '|' in response_text:
            # List type: split by | and create JSON
            parts = [p.strip() for p in response_text.split('|') if p.strip()]
            if len(parts) > 1:
                return json.dumps({
                    "type": "list",
                    "title": parts[0],
                    "items": parts[1:]
                }, ensure_ascii=False)
        
        # Default: text type
        return json.dumps({
            "type": "text",
            "body": response_text
        }, ensure_ascii=False)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats


def create_sample_dataset(output_path: str):
    """Create sample dataset for testing"""
    sample_data = {
        'intent': [
            'greetings', 'greetings', 'greetings',
            'bye', 'bye',
            'help', 'help', 'help',
            'info', 'info'
        ],
        'pattern': [
            'halo|hai|hello|selamat pagi',
            'apa kabar|bagaimana kabar mu',
            'pagi|malam',
            'sampai jumpa|bye|goodbye',
            'thanks|terima kasih',
            'bantuan|help|tolong',
            'bisa apa|fitur apa',
            'gimana cara',
            'tentang|about',
            'informasi'
        ],
        'response_type': [
            'static', 'static', 'dynamic',
            'static', 'static',
            'list', 'list', 'static',
            'static', 'dynamic'
        ],
        'is_master': [
            'true', 'false', 'false',
            'true', 'false',
            'true', 'false', 'false',
            'true', 'false'
        ],
        'response': [
            'Halo! ada yang bisa saya bantu?',
            'Baik-baik saja, terima kasih sudah bertanya',
            'Selamat datang di chatbot kami',
            'Sampai jumpa lagi!',
            'Sama-sama!',
            'Saya bisa membantu dengan: Informasi|FAQ|Dukungan teknis',
            'Fitur kami: Pencarian|Filter|Rekomendasi',
            'Silahkan hubungi tim support kami',
            'Kami adalah chatbot AI yang membantu pengguna',
            'Tersedia 24/7 untuk membantu anda'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"Sample dataset created: {output_path}")


if __name__ == "__main__":
    # Example usage
    config = ProcessingConfig(
        input_file=str(Path(__file__).parent.parent / "data" / "dataset" / "data_mentah.csv"),
        output_lstm=str(Path(__file__).parent.parent / "data" / "dataset" / "lstm" / "dataset_training_lstm.csv"),
        output_bert=str(Path(__file__).parent.parent / "data" / "dataset" / "bert" / "dataset_training_bert.csv")
    )
    
    processor = DatasetPipelineProcessor(config)
    result = processor.process_all()
    
    if result["success"]:
        print("\n✓ Processing completed successfully!")
        print(f"LSTM output: {result['lstm_output']}")
        print(f"BERT output: {result['bert_output']}")
        print(f"\nStatistics: {json.dumps(result['stats'], indent=2)}")
    else:
        print(f"✗ Processing failed: {result['error']}")
