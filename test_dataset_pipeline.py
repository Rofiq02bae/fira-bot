#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to demonstrate the dataset processing pipeline.
Creates sample data and processes it through the pipeline.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add scripts directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "scripts"))

from dataset_pipeline_processor import DatasetPipelineProcessor, ProcessingConfig

def create_sample_data():
    """Create sample input CSV for testing"""
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_input.csv"
    
    # Create sample data
    sample_data = {
        'intent': [
            'greet',
            'farewell',
            'help',
            'status_check',
            'data_request',
        ],
        'patterns': [
            'hello|hi|hey|morning|good morning',
            'bye|goodbye|see you|farewell|until later',
            'help|assist|support|can you help|i need help',
            'status|how are you|whats up|how is everything',
            'data|get data|show data|dataset|information',
        ],
        'response_type': [
            'static',
            'static',
            'dynamic',
            'dynamic',
            'static',
        ],
        'is_master': [
            'true',
            'true',
            'false',
            'false',
            'true',
        ],
        'response': [
            'Hello! Welcome to the system.',
            'Goodbye! Have a great day!',
            'I\'d be happy to help you. What do you need?',
            'I\'m doing great! How can I assist you?',
            'Here is the requested data|Item 1|Item 2|Item 3|Item 4',
        ],
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(sample_file, index=False)
    
    print(f"✓ Sample input file created: {sample_file}")
    print(f"\nSample data:")
    print(df.to_string())
    
    return sample_file


def test_lstm_processing():
    """Test LSTM dataset processing"""
    print("\n" + "=" * 80)
    print("Testing LSTM Dataset Processing")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "dataset"
    
    sample_file = data_dir / "sample_input.csv"
    lstm_output = data_dir / "lstm" / "dataset_training_lstm.csv"
    
    config = ProcessingConfig(
        input_file=str(sample_file),
        output_lstm_file=str(lstm_output),
        output_bert_file=str(data_dir / "bert" / "dataset_training_bert.csv"),
        backup_dir=str(data_dir / "backups"),
        dataset_type='lstm'
    )
    
    processor = DatasetPipelineProcessor(config)
    success, message = processor.process(str(sample_file))
    
    print(f"\nProcessing Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print(f"Message: {message}")
    
    if success and lstm_output.exists():
        df = pd.read_csv(lstm_output)
        print(f"\n✓ LSTM output created successfully!")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample output (first 3 rows):")
        print(df.head(3).to_string())
        return True
    
    return False


def test_bert_processing():
    """Test BERT dataset processing"""
    print("\n" + "=" * 80)
    print("Testing BERT Dataset Processing")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "dataset"
    
    sample_file = data_dir / "sample_input.csv"
    bert_output = data_dir / "bert" / "dataset_training_bert.csv"
    
    config = ProcessingConfig(
        input_file=str(sample_file),
        output_lstm_file=str(data_dir / "lstm" / "dataset_training_lstm.csv"),
        output_bert_file=str(bert_output),
        backup_dir=str(data_dir / "backups"),
        dataset_type='bert'
    )
    
    processor = DatasetPipelineProcessor(config)
    success, message = processor.process(str(sample_file))
    
    print(f"\nProcessing Result: {'SUCCESS ✓' if success else 'FAILED ✗'}")
    print(f"Message: {message}")
    
    if success and bert_output.exists():
        df = pd.read_csv(bert_output)
        print(f"\n✓ BERT output created successfully!")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Sample output (first 3 rows):")
        print(df.head(3).to_string())
        return True
    
    return False


def compare_outputs():
    """Compare LSTM and BERT outputs"""
    print("\n" + "=" * 80)
    print("Comparing LSTM vs BERT Outputs")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data" / "dataset"
    
    lstm_file = data_dir / "lstm" / "dataset_training_lstm.csv"
    bert_file = data_dir / "bert" / "dataset_training_bert.csv"
    
    if not lstm_file.exists() or not bert_file.exists():
        print("✗ Output files not found")
        return
    
    df_lstm = pd.read_csv(lstm_file)
    df_bert = pd.read_csv(bert_file)
    
    print(f"\nLSTM Dataset:")
    print(f"  - Rows: {len(df_lstm)}")
    print(f"  - Response format: JSON")
    print(f"  - Sample: {df_lstm.iloc[0]['response'][:50]}...")
    
    print(f"\nBERT Dataset:")
    print(f"  - Rows: {len(df_bert)}")
    print(f"  - Response format: Plain text")
    print(f"  - Sample: {df_bert.iloc[0]['response'][:50]}...")
    
    print(f"\n✓ Both datasets created with same number of rows: {len(df_lstm) == len(df_bert)}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("Dataset Processing Pipeline - Test Suite")
    print("=" * 80)
    
    # Create sample data
    print("\nStep 1: Creating sample input data...")
    sample_file = create_sample_data()
    
    # Test LSTM processing
    print("\nStep 2: Processing for LSTM...")
    lstm_success = test_lstm_processing()
    
    # Test BERT processing
    print("\nStep 3: Processing for BERT...")
    bert_success = test_bert_processing()
    
    # Compare outputs
    if lstm_success and bert_success:
        print("\nStep 4: Comparing outputs...")
        compare_outputs()
    
    # Final summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"LSTM Processing: {'✓ PASSED' if lstm_success else '✗ FAILED'}")
    print(f"BERT Processing: {'✓ PASSED' if bert_success else '✗ FAILED'}")
    
    if lstm_success and bert_success:
        print("\n✓ All tests completed successfully!")
        print("\nYou can now run the GUI with:")
        print("  python3 run_dataset_gui.py")
    else:
        print("\n✗ Some tests failed. Check the output above for details.")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
