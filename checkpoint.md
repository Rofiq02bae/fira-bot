# Checkpoint Proyek - Hybrid NLU Chatbot (LSTM + IndoBERT + RAG)

**Tanggal:** March 17, 2026  
**Status Proyek:** Development - Container Issues  
**Versi:** 3.0.0 (Modular Architecture)

## 1. Struktur Proyek

### 1.1 Arsitektur Sistem

```
[User Input]
    ↓
[Telegram Bot / HTTP Client] → [FastAPI Server (app.py)]
                              ↓
                    [HybridNLUService (core/nlu_service.py)]
                    ↙           ↘
        [LSTM Model]     [BERT Model]
            ↓               ↓
        [Intent Fusion] → [Response Selection]
            ↓
    [Optional RAG] → [Final Response]
```

### 1.2 Struktur Folder

```
/media/aas/New Volume1/bot/New folder/
├── app.py                          # FastAPI entrypoint
├── main.py                         # Service factory/initialization
├── core/
│   ├── nlu_service.py             # Main NLU orchestration
│   ├── models/                    # Model wrappers
│   └── processors/                # Text processing, matching
├── interfaces/
│   └── telegram_bot.py            # Telegram bot client
├── services/
│   └── rag/                       # RAG implementation
├── config/
│   └── settings.py                # Configuration management
├── data/
│   ├── dataset/                   # Training data
│   ├── lstm_models/               # LSTM model files
│   ├── bert_model/                # BERT model
│   └── rag/                       # FAISS index & metadata
├── scripts/                       # Utility scripts
├── tests/                         # Test suite
├── training/                      # Training notebooks/scripts
├── requirements.*.txt             # Dependencies
├── docker-compose.yml             # Full stack deployment
├── docker-compose.bot.yml         # Bot-only deployment
├── Dockerfile.api                 # API container
└── Dockerfile.bot                 # Bot container
```

## 2. Alur Kerja (Flow)

### 2.1 Runtime Flow

1. **Input Reception**: User sends message via Telegram or HTTP API
2. **Preprocessing**: Text normalization, cleaning
3. **Dual Prediction**:
   - LSTM: Fast pattern-based intent classification
   - BERT: Context-aware intent understanding
4. **Fusion Logic**: Combine predictions based on confidence scores
5. **Response Selection**: Match intent to predefined responses
6. **Optional RAG**: If enabled and confidence low, retrieve from knowledge base
7. **Response Generation**: Format and return response

### 2.2 Initialization Flow

1. **Startup Event**: FastAPI startup loads models
2. **Model Loading**:
   - LSTM: Load H5 model, tokenizer, label encoder
   - BERT: Load IndoBERT model from local path
   - RAG: Load FAISS index if enabled
3. **Service Ready**: Health check endpoint validates all components

### 2.3 Docker Flow

- **API Container**: Runs FastAPI server with model inference
- **Bot Container**: Runs Telegram bot as HTTP client to API
- **Volume Mounts**: Data, models, logs shared between host and containers

## 3. Kondisi Saat Ini

### 3.1 Status Komponen

#### ✅ Working Components
- **Project Structure**: Well-organized modular architecture
- **Code Quality**: Clean separation of concerns
- **Configuration**: Environment-based config management
- **Dependencies**: Requirements files properly separated
- **Documentation**: Comprehensive README and setup guides

#### ⚠️ Issues & Blockers

**Critical: Docker Container Startup Failure**
- **Problem**: API container fails to start with "exec: "/venv/bin/python": stat /venv/bin/python: no such file or directory"
- **Root Cause**: Dockerfile.api using venv approach incompatible with container environment
- **Impact**: Cannot run full stack deployment
- **Workaround**: Run API locally (works), bot in container (works)

**Recent Changes Made:**
- Modified Dockerfile.api to use global pip installs instead of venv
- Changed from Python 3.12 to 3.11 for better package compatibility
- Fixed faiss-cpu version specification
- Simplified container build process

#### 🔄 In Progress
- **Container Fix**: Updated Dockerfile.api to install packages globally
- **Build Status**: Last build failed (exit code 1), needs rebuild
- **Testing**: Container startup needs validation

### 3.2 Environment Configuration

**Host Environment:**
- OS: Linux (Ubuntu-based)
- Python: 3.12.3 (host), 3.11 (containers)
- Docker: Available
- Virtual Environment: .venv with packages installed

**Container Environment:**
- Base Image: python:3.11-slim
- User: appuser (non-root)
- Volumes: Data, HF cache, logs
- Health Check: curl http://localhost:8000/health (30s interval, 60s start period)

**Key Environment Variables:**
```env
# Core
DATASET_PATH=/data/dataset/dataset_training.csv
LSTM_MODEL_PATH=/data/lstm_models/chatbot_model.h5
BERT_MODEL_PATH=/data/bert_model

# RAG
USE_RAG=true
FAISS_INDEX_PATH=/data/rag/faiss.index
SIMILARITY_THRESHOLD=0.3

# LLM
LLM_PROVIDER=openrouter
LLM_MODEL=x-ai/grok-4.20-beta
```

### 3.3 Data & Models Status

**Available Data:**
- Training dataset: data/dataset/dataset_training_bert.csv
- LSTM models: H5, tokenizer.pkl, label_encoder.pkl
- BERT model: Local IndoBERT model
- RAG index: FAISS index with metadata

**Model Loading Status:**
- LSTM: Should load from H5 file
- BERT: Should load from local directory
- RAG: Conditional on USE_RAG=true

### 3.4 Deployment Options

**Option A: Local API + Container Bot (Current Workaround)**
- Run API locally: `python -m uvicorn app:app --host 0.0.0.0 --port 8000`
- Run bot: `docker compose -f docker-compose.bot.yml up -d`
- Status: ✅ Working

**Option B: Full Container Stack (Blocked)**
- Run both: `docker compose up -d`
- Status: ❌ Blocked by API container startup issue

## 4. Next Steps & Recommendations

### Immediate Actions
1. **Rebuild API Image**: `docker compose build api`
2. **Test Container Startup**: `docker compose up -d`
3. **Validate Health Check**: `curl http://localhost:8000/health`
4. **Check Logs**: `docker logs fira-bot-api`

### Medium-term Improvements
1. **Fix Container Issues**: Ensure reliable container builds
2. **Add Health Checks**: Better error handling in startup
3. **Optimize Image Size**: Current API image is ~1.86GB
4. **Add Monitoring**: Container health and performance metrics

### Long-term Goals
1. **Production Deployment**: Kubernetes/Docker Swarm setup
2. **Model Optimization**: Quantization, ONNX conversion
3. **Scalability**: Load balancing, model serving optimization
4. **CI/CD Pipeline**: Automated testing and deployment

## 5. Risk Assessment

**High Risk:**
- Container deployment blocking full-stack testing
- Model loading failures in container environment
- Dependency conflicts between local and container environments

**Medium Risk:**
- RAG integration complexity
- LLM API rate limits and costs
- Telegram bot token security

**Low Risk:**
- Local development workflow
- Code architecture and maintainability

## 6. Backup Plans

1. **Continue Local Development**: Use local API + container bot
2. **Alternative Container Strategy**: Use different base images
3. **Simplified Deployment**: API-only container without RAG
4. **Cloud Deployment**: Move to cloud-based container services

---

**Last Updated:** March 17, 2026  
**Next Review:** March 18, 2026  
**Responsible:** Development Team