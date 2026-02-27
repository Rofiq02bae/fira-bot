# RAG (Retrieval-Augmented Generation) Setup

## Overview

Sistem RAG menambahkan kemampuan untuk memberikan response yang lebih natural dan fleksibel menggunakan LLM, sambil tetap akurat berdasarkan knowledge base yang ada.

**Flow:** `User Query → LSTM+BERT Intent Detection → FAISS Retrieval → LLM Paraphrase → Natural Response`

## Architecture

1. **Intent Detection**: Parallel LSTM+BERT inference untuk identifikasi intent
2. **Document Retrieval**: FAISS vector search untuk menemukan response yang relevan
3. **LLM Paraphrasing**: LLM (via OpenRouter) mengubah response menjadi lebih natural tanpa menambah/mengurangi informasi
4. **Grounded Response**: Response tetap berdasarkan knowledge base, tidak hallucination

## Setup Instructions

### 1. Install Dependencies

```bash
# Aktifkan virtual environment
source venv/bin/activate

# Install RAG dependencies
pip install -r requirements.rag.txt

# Or install manually:
pip install faiss-cpu>=1.7.4 sentence-transformers>=2.2.2 openai>=1.0.0
```

### 2. Create FAISS Index

```bash
# Generate embeddings and create FAISS index from dataset
python scripts/ingest_documents_rag.py \
    --input data/dataset/dataset_training.csv \
    --output-index data/rag/faiss.index \
    --output-metadata data/rag/metadata.pkl \
    --text-column response
```

### 3. Configure RAG

Edit `.env` file atau set environment variables:

```bash
# Enable RAG
USE_RAG=true

# FAISS Configuration
FAISS_INDEX_PATH=data/rag/faiss.index
FAISS_METADATA_PATH=data/rag/metadata.pkl
SIMILARITY_THRESHOLD=0.3
TOP_K_DOCUMENTS=5

# LLM Configuration (OpenRouter)
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-...   # Your OpenRouter API key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=google/gemini-2.0-flash-exp:free  # or any OpenRouter model
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=500

# RAG Behavior
RAG_MIN_CONFIDENCE=0.4
USE_RAG_FOR_FALLBACK=true
```

### 4. Test RAG

```bash
# Start FastAPI server
python app.py

# Test via curl
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "cara daftar akun baru"}'
```

## Configuration Options

### RAGConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `false` | Enable/disable RAG system |
| `faiss_index_path` | `data/rag/faiss.index` | Path to FAISS index file |
| `faiss_metadata_path` | `data/rag/metadata.pkl` | Path to metadata pickle |
| `similarity_threshold` | `0.3` | Min similarity score (0-1) |
| `top_k_documents` | `5` | Number of documents to retrieve |
| `llm_provider` | `""` | LLM provider: `openrouter` or `` (disabled) |
| `llm_api_key` | `""` | OpenRouter API key |
| `llm_model` | `google/gemini-2.0-flash-exp:free` | OpenRouter model name |
| `llm_base_url` | `https://openrouter.ai/api/v1` | OpenRouter API base URL |
| `llm_temperature` | `0.7` | Temperature (0-1) |
| `llm_max_tokens` | `500` | Max tokens in response |
| `rag_min_confidence` | `0.4` | Min confidence to trigger RAG |
| `use_rag_for_fallback` | `true` | Use RAG for low-confidence queries |

## RAG Behavior

### When RAG is Triggered

1. **High Confidence**: Intent detected dengan confidence >= 0.4
   - Flow: Intent → FAISS search → Response selector → LLM paraphrase
   
2. **Low Confidence (Fallback)**: Intent tidak jelas (confidence < 0.4)
   - Flow: Fallback → FAISS search similar queries → LLM generate response

### LLM Prompt Strategy

LLM diberi strict instruction untuk **TIDAK menambah/mengurangi informasi**:

```
Paraphrase the following response naturally in Indonesian.
IMPORTANT: Only rephrase the existing information. 
DO NOT add new information or examples not in the original response.
```

### Response Format

```json
{
  "intent": "cara_daftar",
  "response": "Untuk membuat akun baru, kamu bisa...",  // Natural paraphrased
  "confidence": 0.85,
  "sources": [
    {
      "document": "Response asli dari dataset",
      "similarity": 0.92,
      "intent": "cara_daftar"
    }
  ],
  "rag_augmented": true
}
```

## File Structure

```
services/rag/
├── __init__.py           # Module exports
├── vector_store.py       # FAISSVectorStore class
└── rag_service.py        # RAGService class

data/rag/
├── faiss.index           # FAISS vector index (created by ingestion)
└── metadata.pkl          # Document metadata (created by ingestion)

scripts/
└── ingest_documents_rag.py  # Ingestion script
```

## Usage Examples

### Example 1: High Confidence with RAG

**Input:** "cara daftar akun"
- LSTM+BERT: 0.87 confidence → Intent: `cara_daftar`
- FAISS retrieves top 5 similar responses
- LLM paraphrases: "Untuk membuat akun baru, silakan..."
- **Output**: Natural response + sources

### Example 2: Fallback with RAG

**Input:** "gmn caranya bikin account"
- LSTM+BERT: 0.32 confidence → Fallback
- FAISS searches similar queries
- LLM generates response from retrieved docs
- **Output**: "Sepertinya kamu ingin membuat akun. Berikut caranya..."

### Example 3: RAG Disabled (LLM_PROVIDER="")

**Input:** "cara daftar akun"
- LSTM+BERT: 0.87 confidence → Intent: `cara_daftar`
- FAISS retrieves top document
- **Output**: Original response from dataset (no LLM)

## OpenRouter Configuration

### Supported Models

OpenRouter memberikan akses ke berbagai LLM melalui satu API:

**Free Models:**
- `google/gemini-2.0-flash-exp:free` - Fast, multimodal (recommended)
- `meta-llama/llama-3.2-3b-instruct:free` - Lightweight
- `mistralai/mistral-7b-instruct:free` - Multilingual

**Paid Models:**
- `openai/gpt-4-turbo` - Most capable
- `anthropic/claude-3-opus` - Best reasoning
- `google/gemini-pro-1.5` - Long context

### Get API Key

1. Sign up at [openrouter.ai](https://openrouter.ai/)
2. Go to **Keys** section
3. Create new API key
4. Copy key (format: `sk-or-v1-...`)

### Configuration Example

```bash
# .env file
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-your-key-here
LLM_MODEL=google/gemini-2.0-flash-exp:free
```

## Troubleshooting

### FAISS index not found
```bash
# Create index first
python scripts/ingest_documents_rag.py
```

### LLM API errors
```bash
# Check API key
echo $LLM_API_KEY

# Check OpenRouter status at: https://status.openrouter.ai/

# Or disable LLM (will return retrieved documents directly)
export LLM_PROVIDER=""
```

### PyTorch not found
```bash
# Install PyTorch for BERT embeddings
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Performance Notes

- **FAISS search**: ~1-5ms for 1000 documents
- **BERT embedding**: ~50-100ms per query
- **LLM API call**: ~500-2000ms (depends on provider)
- **Total latency**: ~600-2200ms with LLM, ~100-150ms without

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.rag.txt`
2. ✅ Create FAISS index: `python scripts/ingest_documents_rag.py`
3. ✅ Configure environment variables
4. ✅ Test with sample queries
5. 🔄 Monitor response quality and adjust similarity threshold
6. 🔄 Fine-tune LLM prompt if needed
