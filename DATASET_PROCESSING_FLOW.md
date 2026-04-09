# Dataset Processing GUI - Complete Flow

## Overview
This application implements a dataset processing pipeline with a tkinter GUI based on Figma design mockups. It allows users to process datasets for both LSTM and BERT model training through a unified interface.

## Architecture

### Components

1. **dataset_pipeline_processor.py** - Core data processing engine
   - `ProcessingConfig`: Configuration dataclass for processing settings
   - `DatasetPipelineProcessor`: Main processor class with 5-step pipeline
   
2. **dataset_gui.py** - Tkinter GUI application
   - `DatasetProcessingGUI`: Main application class implementing Figma designs

3. **run_dataset_gui.py** - Application launcher

## Processing Pipeline

The data flows through 5 main steps:

```
Input CSV
    ↓
[Step 1] Remove Duplicate Patterns
    ↓
[Step 2] Fix CSV Format
    ↓
[Step 3] Validate CSV
    ↓
[Step 4] Split Patterns into Separate Rows
    ↓
[Step 5] Convert Response Format (LSTM) / Keep Plain Text (BERT)
    ↓
Output: dataset_training_lstm.csv or dataset_training_bert.csv
```

### Step Details

#### Step 1: Remove Duplicate Patterns
- Reads input CSV with columns: `intent, patterns, response_type, is_master, response`
- Splits patterns by `|` delimiter (multiple patterns per row)
- Removes duplicate patterns (case-insensitive)
- Logs statistics about duplicates removed

**Input format:**
```csv
intent,patterns,response_type,is_master,response
greet,hello|hi|hey,static,true,Hello there!
```

**Output after Step 1:**
```csv
intent,patterns,response_type,is_master,response
greet,hello|hi|hey,static,true,Hello there!
(duplicates removed if any)
```

#### Step 2: Fix CSV Format
- Ensures consistent formatting across all fields
- Replaces commas with pipes within pattern strings
- Normalizes all field values
- Ensures boolean values are lowercase

#### Step 3: Validate CSV
- Checks for required columns: `intent, patterns, response_type, is_master, response`
- Validates no null/empty values
- Reports bad rows and continues with warnings

#### Step 4: Split Patterns
- **CRITICAL STEP**: Converts one row per intent into multiple rows (one per pattern)
- Takes patterns separated by `|` and creates individual rows
- Each pattern gets its own row with same intent/response

**Example:**
```
Input:  greet | hello|hi|hey | static | true | Hello
Output: 
        greet | hello | static | true | Hello
        greet | hi | static | true | Hello
        greet | hey | static | true | Hello
```

This is essential for training - models need individual pattern examples, not combined patterns.

#### Step 5: Response Format Conversion
- **LSTM Only**: Converts responses to JSON format
  - Single response → `{"type": "text", "body": "..."}`
  - Multi-part (with `|`) → `{"type": "list", "title": "...", "items": [...]}`
- **BERT**: Keeps plain text format for simplicity

## GUI Screens (Based on Figma Design)

### Screen 1: Main Menu (Design 2:30 & 3:52)
```
┌─────────────────────────────────────────┐
│   Dataset Processing Pipeline           │
│                                         │
│   ┌───────────────────┐                │
│   │  dataset LSTM     │                │
│   └───────────────────┘                │
│                                         │
│   ┌───────────────────┐                │
│   │  dataset BERT     │                │
│   └───────────────────┘                │
│                                         │
│   ┌───────────────────┐                │
│   │   cek dataset     │                │
│   └───────────────────┘                │
└─────────────────────────────────────────┘
```

Features:
- Three main button options
- Clean, centered layout
- Light gray background (#d9d9d9)

### Screen 2: Dataset Form - LSTM (Design 3:206)
```
┌──────────────────────────────────────────┐
│          dataset LSTM                    │
│                                          │
│  intent / topik                          │
│  [text input field                  ]    │
│                                          │
│  pertanyaan                              │
│  [text input field                  ]    │
│                                          │
│  tipe pertanyaan                         │
│  [text input field                  ]    │
│                                          │
│  pertanyaan bercabang                    │
│  [text input field                  ]    │
│                                          │
│  jawaban                                 │
│  [large text area                   ]    │
│  [                                 ]    │
│  [                                 ]    │
│                                          │
│  [upload file]                    [OK]   │
└──────────────────────────────────────────┘
```

Features:
- Form fields for manual entry:
  - intent / topik (single line)
  - pertanyaan (single line)
  - tipe pertanyaan (single line)
  - pertanyaan bercabang (single line)
  - jawaban (multi-line text area)
- Upload file button to select input CSV
- OK button to process
- Back button to return to menu

### Screen 3: Dataset Form - BERT (Design 5:278)
```
Same as Screen 2 but titled "dataset BERT"
```

### Screen 4: Dataset Verification
```
┌──────────────────────────────────────────┐
│          Check Dataset                   │
│                                          │
│  Dataset LSTM                            │
│  File: dataset_training_lstm.csv         │
│  Total Rows: 1234                        │
│  Columns: intent, pattern, ...          │
│  File Size: 456,789 bytes               │
│                                          │
│  Intent Distribution:                    │
│    • greet: 45 samples                  │
│    • farewell: 32 samples               │
│    • help: 28 samples                   │
│    ... and 15 more intents              │
│                                          │
│  ─────────────────────────────────────   │
│                                          │
│  Dataset BERT                            │
│  File: dataset_training_bert.csv         │
│  Total Rows: 1234                        │
│  ... (same as LSTM)                     │
│                                          │
│                    [Back]                │
└──────────────────────────────────────────┘
```

Features:
- Shows statistics for both LSTM and BERT datasets
- Displays intent distribution
- Shows file size and total rows
- Lists all columns
- Scrollable for large datasets

## Usage

### Running the Application

```bash
cd /media/aas/New\ Volume1/bot/New\ folder
python run_dataset_gui.py
```

### Workflow

1. **Start Application**
   - Launch `run_dataset_gui.py`
   - See main menu with 3 options

2. **Create LSTM or BERT Dataset**
   - Click "dataset LSTM" or "dataset BERT"
   - Upload input CSV file (or can populate form fields manually)
   - Click OK
   - Application processes through pipeline:
     - Removes duplicates
     - Fixes format
     - Validates
     - Splits patterns
     - Converts format
   - Output saved to:
     - LSTM: `data/dataset/lstm/dataset_training_lstm.csv`
     - BERT: `data/dataset/bert/dataset_training_bert.csv`

3. **Verify Datasets**
   - Click "cek dataset"
   - View statistics and intent distribution
   - Confirm datasets were created correctly

## Input CSV Format

The input CSV file should have the following format:

```csv
intent,patterns,response_type,is_master,response
greeting,hello|hi|hey,static,true,Good morning!
farewell,bye|goodbye|see you,static,true,Goodbye!
help,help|assist|support,dynamic,false,How can I help?
```

**Required Columns:**
- `intent`: The intent/topic/category
- `patterns`: Multiple patterns separated by `|` 
- `response_type`: Type of response (static, dynamic, list, etc.)
- `is_master`: Boolean flag (true/false)
- `response`: Response text or JSON (can contain `|` for multi-part)

## Output CSV Formats

### LSTM Format
```csv
intent,pattern,response_type,is_master,response
greeting,hello,static,true,"{""type"": ""text"", ""body"": ""Good morning!""}"
greeting,hi,static,true,"{""type"": ""text"", ""body"": ""Good morning!""}"
greeting,hey,static,true,"{""type"": ""text"", ""body"": ""Good morning!""}"
```

### BERT Format
```csv
intent,pattern,response_type,is_master,response
greeting,hello,static,true,Good morning!
greeting,hi,static,true,Good morning!
greeting,hey,static,true,Good morning!
```

Key differences:
- LSTM: responses converted to JSON format
- BERT: responses kept as plain text
- Both: patterns split into individual rows

## Output Locations

```
data/
├── dataset/
│   ├── lstm/
│   │   └── dataset_training_lstm.csv
│   └── bert/
│       └── dataset_training_bert.csv
```

## Processing Logs

During processing, the application displays detailed logs showing:
- Number of rows processed
- Duplicates removed
- Patterns split
- Final dataset statistics
- Any warnings or errors

## Error Handling

The application handles:
- Missing input files
- Malformed CSV data
- Missing required columns
- File I/O errors
- Invalid data formats

All errors are displayed in the processing output window for debugging.

## Performance Considerations

- Supports datasets with 1000+ rows
- Processing typically completes within seconds
- Background threading prevents GUI freezing
- Scrollable output for large datasets

## Dependencies

- tkinter (built-in with Python)
- pandas
- numpy

## Troubleshooting

**Problem**: File not found error
- **Solution**: Ensure input CSV exists at specified path

**Problem**: Processing fails with encoding error
- **Solution**: Ensure input CSV is UTF-8 encoded

**Problem**: Duplicate patterns not removed
- **Solution**: Check pattern format uses `|` as separator

**Problem**: Output file not created
- **Solution**: Check that `data/dataset/lstm/` or `data/dataset/bert/` directories exist

## Future Enhancements

Possible improvements:
- Batch processing multiple files
- CSV preview before processing
- Custom delimiter options
- Dataset merging
- Export to other formats
- Pattern validation rules
- Response template system

---

**Created**: April 2026
**Implementation**: tkinter GUI + Python data processing pipeline
**Design Reference**: Figma aplikasi-dataset project
