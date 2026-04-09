# Dataset Pipeline GUI - User Guide

## Overview

Complete tkinter-based GUI application for processing datasets through a 4-step pipeline:
1. **Remove Duplicates** - Remove duplicate patterns within each row
2. **Validate & Fix** - Validate data integrity and fix formatting issues
3. **Split Patterns** - Split multiple patterns into separate rows
4. **Convert & Split** - Convert responses to JSON and create train/validation split

## Features

✓ 4 interactive tabs for each processing step  
✓ Real-time logging and statistics display  
✓ Sample dataset creation for testing  
✓ Input file browser and preview  
✓ Support for custom validation split ratio  
✓ Automatic JSON response conversion  
✓ Train/validation dataset splitting with stratification  
✓ Output to LSTM and BERT formats  

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (venv)

### Setup

```bash
cd /media/aas/New\ Volume1/bot/New\ folder

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip pandas scikit-learn
```

## Usage

### Option 1: Run GUI Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run GUI
python3 run_dataset_gui.py
```

### Option 2: Run Pipeline Programmatically

```python
from scripts.dataset_pipeline_processor import DatasetPipelineProcessor, ProcessingConfig

config = ProcessingConfig(
    input_file="data/dataset/data_mentah.csv",
    output_lstm="data/dataset/lstm/dataset_training_lstm.csv",
    output_bert="data/dataset/bert/dataset_training_bert.csv"
)

processor = DatasetPipelineProcessor(config)
result = processor.process_all()

if result["success"]:
    print(f"LSTM: {result['lstm_output']}")
    print(f"BERT: {result['bert_output']}")
```

## Input Format

Expected CSV format with columns:
- `intent`: Intent/category name
- `pattern`: Patterns separated by `|` (pipe character)
- `response_type`: Type of response (static, dynamic, list)
- `is_master`: Boolean flag (true/false)
- `response`: Response text

### Example:
```
intent,pattern,response_type,is_master,response
greetings,halo|hai|hello,static,true,Halo! Ada yang bisa saya bantu?
bye,bye|goodbye,static,true,Sampai jumpa!
help,bantuan|help|tolong,list,true,Saya bisa membantu dengan: Informasi|FAQ|Support
```

## Processing Steps

### Step 1: Remove Duplicates
- Splits patterns by `|` delimiter
- Removes case-insensitive duplicates
- Cleans whitespace
- Output: Clean pattern list

### Step 2: Validate & Fix
- Removes rows with empty patterns
- Validates response_type values
- Normalizes is_master field
- Removes null values

### Step 3: Split Patterns
- Converts each pattern into separate row
- Preserves intent and response
- Multiplies dataset size based on pattern count
- Example: 10 rows with 2-3 patterns each → 22-30 rows

### Step 4: Convert & Split
- Converts responses to JSON format
- Creates 80/20 train/validation split
- Uses stratified sampling to preserve intent distribution
- Outputs to LSTM and BERT formats

## Output Files

### LSTM Dataset
Path: `data/dataset/lstm/dataset_training_lstm.csv`

Columns:
- intent
- pattern
- response_type
- response (JSON format)

### BERT Dataset
Path: `data/dataset/bert/dataset_training_bert.csv`

Columns:
- intent
- pattern
- response_type
- response (JSON format)

## JSON Response Format

### Text Response
```json
{
  "type": "text",
  "body": "Response text here"
}
```

### List Response
```json
{
  "type": "list",
  "title": "Main title",
  "items": ["item1", "item2", "item3"]
}
```

## Statistics & Logging

Each processing step provides:
- **Step 1**: Duplicate patterns removed, rows affected
- **Step 2**: Invalid rows removed, valid intents count
- **Step 3**: Pattern split statistics, average patterns per row
- **Step 4**: Train/validation split, unique intents, response types

All activities logged to "Logs & Results" tab with timestamps.

## Troubleshooting

### Module Not Found Error
```bash
# Ensure venv is activated
source venv/bin/activate

# Reinstall dependencies
pip install pandas scikit-learn
```

### File Not Found
- Check that input file path is correct
- Click "Browse" to select file from dialog
- Use "Create Sample" to generate test data

### Processing Errors
- Check logs tab for detailed error messages
- Verify input CSV format matches expected structure
- Ensure all required columns are present

## Advanced Usage

### Custom Validation Split
In GUI: Modify "Validation Split (%)" in Step 4 tab

### Programmatically:
```python
config = ProcessingConfig(
    input_file="...",
    output_lstm="...",
    output_bert="...",
    validation_split=0.25  # 75/25 split
)
```

### Disable JSON Conversion
```python
config = ProcessingConfig(
    ...
    convert_responses=False
)
```

## Performance

- Typical processing time for 1000-row dataset: < 5 seconds
- Memory usage: ~100MB for 10,000 patterns
- Bottleneck: JSON conversion and stratified split

## File Structure

```
.
├── scripts/
│   ├── dataset_pipeline_processor.py  # Core processor
│   └── dataset_gui.py                 # GUI application
├── run_dataset_gui.py                 # Launcher script
├── data/
│   └── dataset/
│       ├── data_mentah.csv           # Input data
│       ├── lstm/
│       │   └── dataset_training_lstm.csv
│       └── bert/
│           └── dataset_training_bert.csv
└── README.md
```

## API Reference

### DatasetPipelineProcessor

```python
processor = DatasetPipelineProcessor(config)

# Run complete pipeline
result = processor.process_all()

# Run individual steps
df1 = processor.step_1_remove_duplicates(input_file)
df2 = processor.step_2_validate_fix(df1)
df3 = processor.step_3_split_patterns(df2)
lstm_path, bert_path = processor.step_4_convert_responses(df3)

# Get statistics
stats = processor.get_stats()
```

### ProcessingConfig

```python
@dataclass
class ProcessingConfig:
    input_file: str                    # Input CSV path
    output_lstm: str                   # LSTM output path
    output_bert: str                   # BERT output path
    validation_split: float = 0.2      # Train/val split ratio
    random_state: int = 42             # Random seed
    remove_duplicates: bool = True
    normalize_text: bool = True
    split_patterns: bool = True
    convert_responses: bool = True
```

## License

This tool is part of the Telegram Bot Dataset Processing Pipeline project.

## Support

For issues or questions:
1. Check logs tab for error details
2. Verify input file format
3. Try with sample dataset
4. Check Python version (3.8+ required)

---

Last Updated: 2026-04-06
