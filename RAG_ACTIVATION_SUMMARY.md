# RAG Service Activation Summary

## ✅ Status: ACTIVATED & WORKING

RAG service telah berhasil diaktifkan dan terintegrasi dengan chatbot Anda.

## What Was Done

### 1. Environment Setup
- ✅ Installed required packages:
  - `faiss-cpu>=1.7.4` - Vector similarity search
  - `sentence-transformers>=2.2.2` - Embedding models
  - `openai>=1.0.0` - OpenRouter API client
  - `python-dotenv` - Environment variable loading

### 2. FAISS Index Creation
- ✅ Generated embeddings from 259 documents using BERT model
- ✅ Created FAISS L2 index with 768-dimensional vectors
- ✅ Saved index to `data/rag/faiss.index`
- ✅ Saved metadata to `data/rag/metadata.pkl`

### 3. Code Integration
- ✅ Added `dotenv` loading to `app.py` and test scripts
- ✅ Fixed BERT embedding extraction for SequenceClassification models
- ✅ Initialized pool_manager for async processing
- ✅ Updated OpenRouter configuration and LLM client initialization

### 4. Configuration
- ✅ RAGConfig configured with environment variables
- ✅ OpenRouter API key configured
- ✅ Default model set to: `meta-llama/llama-3-8b-instruct`
- ✅ All RAG parameters in `.env` file

## How It Works

### Query Flow with RAG
```
User Query: "cara daftar akun baru"
    ↓
Preprocess & Intent Detection (LSTM + BERT)
    ↓
Intent Detected: "ak1_info" (confidence: 0.404)
    ↓
RAG Check: Confidence >= 0.4? YES → Trigger RAG
    ↓
FAISS Vector Search: Find similar documents
    ↓
LLM Paraphrase: Transform base response to natural language
    ↓
Response: "Halo! Untuk pendaftaran akun baru dalam rangka pembuatan Kartu Kuning (AK1) di Kabupaten Tegal..."
```

### Fallback with RAG
If intent confidence < 0.4 and `use_rag_for_fallback=true`:
- FAISS retrieves relevant documents
- LLM generates response from context
- No intent matching needed

## Testing Results

```
✅ NLU Service Status:
  Status: healthy
  LSTM: Loaded
  BERT: Loaded
  Dataset: 20 intents
  RAG Enabled: True
  Vector Store: 259 documents (768-dim)
  LLM Client: OpenRouter (meta-llama/llama-3-8b-instruct)

🧪 Sample Query Test:
  Input: "cara daftar akun baru"
  Intent: ak1_info (confidence: 0.404)
  RAG Augmented: Yes
  Response: Natural language paraphrased response ✓
```

## Current Configuration

**File: `.env`**
```bash
USE_RAG=true
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-v1-...
LLM_MODEL=meta-llama/llama-3-8b-instruct
FAISS_INDEX_PATH=data/rag/faiss.index
FAISS_METADATA_PATH=data/rag/metadata.pkl
RAG_MIN_CONFIDENCE=0.4
USE_RAG_FOR_FALLBACK=true
```

## Key Components

### Services
- `services/rag/vector_store.py` - FAISS index management
- `services/rag/rag_service.py` - RAG pipeline with LLM integration
- `core/nlu_service.py` - Main NLU service with RAG integration

### Files
- `data/rag/faiss.index` - FAISS vector index
- `data/rag/metadata.pkl` - Document metadata
- `scripts/ingest_documents_rag.py` - Index creation script

## How to Use

### Start API Server
```bash
python app.py
```

### Test RAG Query
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "cara daftar akun baru"}'
```

### Check Service Status
```bash
python test_rag_activation.py
```

## Response Format

```json
{
  "intent": "ak1_info",
  "confidence": 0.404,
  "response": "Halo! Untuk pendaftaran akun baru...",
  "augmented": true,
  "rag_method": "rag_llm",
  "sources": [
    {
      "document": "Original response from dataset",
      "similarity": 0.85,
      "intent": "ak1_info"
    }
  ],
  "pattern_similarity": 0.0,
  "method": "fusion_weighted_avg"
}
```

## Available Models on OpenRouter

**Free Models:**
- `meta-llama/llama-3-8b-instruct` - Good quality, free
- `mistralai/mistral-7b-instruct` - Fast, good quality, free
- `meta-llama/llama-3.2-3b-instruct` - Lightweight, free

**Paid Models (higher quality):**
- `openai/gpt-4-turbo` - Most capable
- `anthropic/claude-opus-4.6` - Best reasoning
- `google/gemini-pro-1.5` - Long context

Change model by updating `.env`:
```bash
LLM_MODEL=mistralai/mistral-7b-instruct
```

## Performance Notes

- BERT embedding: ~50-100ms per query
- FAISS search: ~1-5ms for 259 documents
- LLM API call: ~500-2000ms (depends on model)
- Total latency: ~600-2200ms with RAG enabled

## Troubleshooting

### RAG Not Activated
```bash
# Check if .env is loaded
python test_env_load.py

# Verify USE_RAG=true is set
grep USE_RAG .env
```

### Model Not Found on OpenRouter
- OpenRouter model names don't use `:free` suffix
- Check available models at: https://openrouter.ai/docs/models

### FAISS Index Not Found
```bash
# Regenerate index
python scripts/ingest_documents_rag.py
```

### LLM API Errors
- Check API key: `echo $LLM_API_KEY`
- Check OpenRouter status: https://status.openrouter.ai/
- Try different model: Update `LLM_MODEL` in `.env`

## Next Steps

1. ✅ RAG Service Activated
2. ✅ FAISS Index Created
3. ✅ LLM Integration Ready
4. 🔄 Monitor response quality
5. 🔄 Adjust similarity threshold if needed (default: 0.3)
6. 🔄 Fine-tune LLM prompt if needed
7. 🔄 Deploy to production

## Files Modified

- `app.py` - Added dotenv loading
- `config/settings.py` - OpenRouter configuration
- `core/nlu_service.py` - Pool manager and RAG integration
- `services/rag/rag_service.py` - Fixed BERT embedding extraction
- `.env` - RAG configuration with OpenRouter API
- `requirements.rag.txt` - Dependencies list

