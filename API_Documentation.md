# Hybrid Chatbot API Documentation v3.0.0

## 📋 Daftar Isi
1. [Panduan Umum](#panduan-umum)
2. [Model Data](#model-data)
3. [Otentikasi & Autentikasi](#otentikasi--autentikasi)
4. [Endpoints](#endpoints)
5. [Contoh Penggunaan](#contoh-penggunaan)
6. [Error Handling](#error-handling)
7. [Troubleshooting](#troubleshooting)

---

## 📘 Panduan Umum

### Informasi Dasar
- **Nama Aplikasi**: Hybrid LSTM + IndoBERT Chatbot API
- **Versi**: 3.0.0
- **Arsitektur**: Modular (LSTM + BERT Hybrid Classification)
- **Framework**: FastAPI
- **Port Default**: 8000
- **Base URL**: `http://localhost:8000`

### Teknologi
- **Backend**: FastAPI + Uvicorn
- **Models**: LSTM + IndoBERT
- **Data Processing**: Pandas + Scikit-learn
- **Dokumentasi Interaktif**: Swagger UI (`/docs`), ReDoc (`/redoc`)

### Fitur Utama
- ✅ Hybrid Intent Classification (LSTM + BERT)
- ✅ Strategi Fallback (RAG Support)
- ✅ Batch Processing
- ✅ Real-time Response Generation
- ✅ Pattern Matching & Similarity Scoring
- ✅ Debug Mode untuk Analisis Prediksi

---

## 🔧 Model Data

### UserInput
Request untuk single chat request.
```json
{
  "text": "string (required) - Input teks dari user"
}
```

**Contoh:**
```json
{
  "text": "Bagaimana cara membuat akun?"
}
```

---

### BatchInput
Request untuk batch processing multiple queries.
```json
{
  "texts": ["string", "string"] - Array of input texts (required)
}
```

**Contoh:**
```json
{
  "texts": [
    "Bagaimana cara membuat akun?",
    "Berapa biaya layanan?",
    "Konfigurasi apa saja yang tersedia?"
  ]
}
```

---

### ChatResponse
Response dari chat endpoint.
```json
{
  "original_text": "string - Input user yang asli",
  "predicted_intent": "string - Intent yang diprediksi",
  "confidence": "float (0-1) - Confidence score prediksi",
  "response": "string - Respons AI terhadap user",
  "options": "[{...}] - Optional: Additional options atau follow-up actions",
  "method_used": "string - Method yang digunakan: hybrid, lstm_only, bert_only",
  "processing_time": "float - Processing time dalam milliseconds",
  "timestamp": "string - ISO 8601 timestamp"
}
```

**Contoh:**
```json
{
  "original_text": "Bagaimana cara membuat akun?",
  "predicted_intent": "create_account",
  "confidence": 0.92,
  "response": "Untuk membuat akun, ikuti langkah-langkah berikut...",
  "options": [
    {
      "action": "send_email",
      "label": "Kirim Email Verifikasi"
    }
  ],
  "method_used": "hybrid",
  "processing_time": 125.45,
  "timestamp": "2026-04-06T10:30:00.000000"
}
```

---

### HealthResponse
Response dari health check endpoint.
```json
{
  "status": "string - Status keseluruhan service",
  "model_loaded": "boolean - Apakah LSTM model terpasang",
  "bert_available": "boolean - Apakah BERT model tersedia",
  "intents_count": "integer - Jumlah intent yang tersedia",
  "total_patterns": "integer - Total patterns dalam dataset"
}
```

**Contoh:**
```json
{
  "status": "running",
  "model_loaded": true,
  "bert_available": true,
  "intents_count": 15,
  "total_patterns": 250
}
```

---

### StatsResponse
Response dari statistics endpoint.
```json
{
  "model_info": {
    "lstm_loaded": "boolean",
    "bert_loaded": "boolean",
    "dataset_loaded": "boolean",
    "intents_count": "integer",
    "models_info": "object"
  },
  "performance": {
    "prediction_methods": ["hybrid", "lstm_only", "bert_only"],
    "architecture": "string",
    "components": ["string"]
  },
  "system_info": {
    "service_started": "boolean",
    "hybrid_mode": "boolean",
    "architecture_version": "string",
    "timestamp": "string"
  }
}
```

---

### RAGService
Response dengan konteks RAG (Retrieval-Augmented Generation).
```json
{
  "response": "string - Generated response",
  "sources": "[{...}] - Optional: Source documents/references",
  "intent": "string - Detected intent"
}
```

---

## 🔐 Otentikasi & Autentikasi

### Status Saat Ini
❌ **API ini belum mengimplementasikan autentikasi.**

Untuk production environment, tambahkan:
- JWT Token Authentication
- API Key validation
- Rate limiting per endpoint
- CORS configuration

**Rekomendasi untuk Production:**
```python
from fastapi.security import HTTPBearer
from fastapi import Security

security = HTTPBearer()

@app.get("/api/chat", security=security)
async def chat(user_input: UserInput, credentials = Security(security)):
    # Validate JWT token
    pass
```

---

## 📡 Endpoints

### 1. Root Endpoint
**GET** `/`

Menampilkan informasi dasar API.

**Response:**
```json
{
  "message": "Hybrid LSTM + IndoBERT Chatbot API (Modular v3.0.0)",
  "status": "running",
  "version": "3.0.0",
  "architecture": "modular",
  "docs": "/docs",
  "health": "/health"
}
```

**Status Code**: `200 OK`

---

### 2. Health Check
**GET** `/health`

Memeriksa status kesehatan service dan ketersediaan models.

**Response Model**: `HealthResponse`

**Response Example:**
```json
{
  "status": "running",
  "model_loaded": true,
  "bert_available": true,
  "intents_count": 15,
  "total_patterns": 250
}
```

**Status Codes**:
- `200 OK` - Service healthy
- `503 Service Unavailable` - Service not ready

---

### 3. Get Available Intents
**GET** `/intents`

Mendapatkan daftar semua intent yang tersedia dalam sistem.

**Query Parameters**: None

**Response:**
```json
{
  "available_intents": ["create_account", "payment", "support", ...],
  "count": 15,
  "intents_details": [
    {
      "intent": "create_account",
      "response_type": "direct",
      "patterns_count": 20,
      "responses_count": 5
    },
    ...
  ]
}
```

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Error retrieving intents

---

### 4. Single Chat (Hybrid)
**POST** `/api/chat`

Main endpoint untuk membuat prediksi menggunakan hybrid LSTM + BERT.

**Request Body** (UserInput):
```json
{
  "text": "Bagaimana cara membuat akun?"
}
```

**Response Model**: `ChatResponse`

**Response Example:**
```json
{
  "original_text": "Bagaimana cara membuat akun?",
  "predicted_intent": "create_account",
  "confidence": 0.92,
  "augmented": false,
  "response": "Untuk membuat akun, Anda perlu mengunjungi halaman registrasi...",
  "options": [
    {
      "action": "email_verification",
      "label": "Kirim Email Verifikasi"
    }
  ],
  "method_used": "hybrid",
  "processing_time": 125.45,
  "timestamp": "2026-04-06T10:30:00.123456"
}
```

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Prediction error

---

### 5. LSTM-Only Chat
**POST** `/api/chat-lstm`

Prediksi menggunakan LSTM saja (lebih cepat, akurasi medium).

**Request Body** (UserInput):
```json
{
  "text": "Berapa biaya layanan?"
}
```

**Response:**
```json
{
  "original_text": "Berapa biaya layanan?",
  "predicted_intent": "pricing",
  "confidence": 0.88,
  "response": "Paket kami tersedia dalam 3 tier...",
  "method_used": "lstm_only",
  "processing_time": 45.23,
  "timestamp": "2026-04-06T10:31:00.123456"
}
```

**Karakteristik**:
- ⚡ Lebih cepat (45-80ms)
- 📊 Akurasi sedang (88-92%)
- 🔧 Tidak memerlukan GPU untuk BERT

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Prediction error

---

### 6. BERT-Only Chat
**POST** `/api/chat-bert`

Prediksi menggunakan BERT saja (lebih akurat, lebih lambat).

**Request Body** (UserInput):
```json
{
  "text": "Saya ingin mengubah password saya"
}
```

**Response:**
```json
{
  "original_text": "Saya ingin mengubah password saya",
  "predicted_intent": "change_password",
  "confidence": 0.96,
  "response": "Untuk mengubah password, masuk ke pengaturan akun...",
  "method_used": "bert_only",
  "processing_time": 250.78,
  "timestamp": "2026-04-06T10:32:00.123456"
}
```

**Karakteristik**:
- 🎯 Sangat akurat (93-98%)
- ⏱️ Lebih lambat (200-400ms)
- 🔧 Memerlukan GPU untuk performance optimal

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - BERT model not available
- `500 Internal Server Error` - Prediction error

---

### 7. Batch Chat
**POST** `/api/batch-chat`

Memproses multiple queries sekaligus dengan bulk pricing/efficiency.

**Request Body** (BatchInput):
```json
{
  "texts": [
    "Bagaimana cara membuat akun?",
    "Berapa biaya layanan?",
    "Bagaimana dukungan pelanggan?"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Bagaimana cara membuat akun?",
      "predicted_intent": "create_account",
      "confidence": 0.92,
      "response": "Untuk membuat akun...",
      "method_used": "hybrid"
    },
    {
      "text": "Berapa biaya layanan?",
      "predicted_intent": "pricing",
      "confidence": 0.88,
      "response": "Paket kami tersedia dalam...",
      "method_used": "hybrid"
    },
    {
      "text": "Bagaimana dukungan pelanggan?",
      "predicted_intent": "support",
      "confidence": 0.94,
      "response": "Tim support kami siap membantu...",
      "method_used": "hybrid"
    }
  ],
  "count": 3,
  "total_processing_time": 145.67,
  "average_time_per_request": 48.56
}
```

**Query Limits**:
- Minimum: 1 query
- Maximum: 100 queries per request (recommended)
- Optimal: 10-50 queries

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Batch processing error

---

### 8. Service Statistics
**GET** `/api/stats`

Mendapatkan informasi lengkap tentang service, models, dan performance.

**Response Model**: `StatsResponse`

**Response Example:**
```json
{
  "model_info": {
    "lstm_loaded": true,
    "bert_loaded": true,
    "dataset_loaded": true,
    "intents_count": 15,
    "models_info": {
      "lstm_model_size": "45MB",
      "bert_model_size": "420MB"
    }
  },
  "performance": {
    "prediction_methods": ["hybrid", "lstm_only", "bert_only"],
    "architecture": "modular",
    "components": [
      "LSTM Model",
      "BERT Model",
      "Intent Matcher",
      "Response Selector",
      "Fallback Service"
    ]
  },
  "system_info": {
    "service_started": true,
    "hybrid_mode": true,
    "architecture_version": "3.0.0",
    "timestamp": "2026-04-06T10:35:00.123456"
  }
}
```

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready

---

### 9. Debug Prediction Details
**GET** `/api/debug-prediction`

Endpoint debug untuk melihat prediksi detail dari LSTM, BERT, dan fused model.

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Input text untuk diprediksi |

**Response:**
```json
{
  "input_text": "Bagaimana cara membuat akun?",
  "lstm_prediction": {
    "intent": "create_account",
    "confidence": 0.902
  },
  "bert_prediction": {
    "intent": "create_account",
    "confidence": 0.945,
    "status": "available"
  },
  "fused_prediction": {
    "intent": "create_account",
    "confidence": 0.92,
    "method": "hybrid",
    "response": "Untuk membuat akun..."
  },
  "models_status": {
    "lstm_loaded": true,
    "bert_loaded": true,
    "dataset_loaded": true,
    "intents_count": 15,
    "status": "running"
  }
}
```

**Gunakan untuk**:
- 🔍 Debugging prediksi yang tidak akurat
- 📊 Membandingkan LSTM vs BERT output
- 🎯 Memahami fusion/weighting strategy

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Debug error

---

### 10. Debug Response Scores
**GET** `/api/debug-response-scores`

Melihat kandidat response dan scoring untuk sebuah intent.

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `text` | string | Yes | Input text dari user |
| `intent` | string | Yes | Intent yang ingin dianalisis |
| `method` | string | No | Method: hybrid, lstm_only, bert_only (default: hybrid) |

**Response:**
```json
{
  "intent": "create_account",
  "text": "Bagaimana cara membuat akun?",
  "method": "hybrid",
  "pattern_similarity": 0.87,
  "candidates": [
    {
      "response": "Untuk membuat akun, ikuti langkah-langkah berikut...",
      "score": 0.92,
      "similarity": 0.87
    },
    {
      "response": "Kunjungi halaman registrasi kami...",
      "score": 0.85,
      "similarity": 0.79
    },
    {
      "response": "Anda bisa membuat akun melalui portal...",
      "score": 0.78,
      "similarity": 0.72
    }
  ]
}
```

**Gunakan untuk**:
- 🎯 Memahami response selection logic
- 📋 Melihat ranking candidate responses
- 🔧 Tuning similarity thresholds

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready
- `500 Internal Server Error` - Scoring error

---

### 11. Service Status
**GET** `/api/service-status`

Mendapatkan status detail service secara real-time.

**Response:**
```json
{
  "status": "running",
  "lstm_loaded": true,
  "bert_loaded": true,
  "dataset_loaded": true,
  "intents_count": 15,
  "text_normalizer": true,
  "timestamp": "2026-04-06T10:40:00.123456"
}
```

**Status Codes**:
- `200 OK` - Success
- `503 Service Unavailable` - Service not ready

---

## 💻 Contoh Penggunaan

### Python (requests library)

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Health Check
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# 2. Get Available Intents
response = requests.get(f"{BASE_URL}/intents")
print("Intents:", response.json())

# 3. Single Chat (Hybrid)
user_input = {"text": "Bagaimana cara membuat akun?"}
response = requests.post(
    f"{BASE_URL}/api/chat",
    json=user_input,
    headers={"Content-Type": "application/json"}
)
print("Chat Response:", json.dumps(response.json(), indent=2))

# 4. Batch Chat
batch_input = {
    "texts": [
        "Bagaimana cara membuat akun?",
        "Berapa biaya layanan?",
        "Bagaimana dukungan pelanggan?"
    ]
}
response = requests.post(
    f"{BASE_URL}/api/batch-chat",
    json=batch_input,
    headers={"Content-Type": "application/json"}
)
print("Batch Results:", json.dumps(response.json(), indent=2))

# 5. LSTM Only
response = requests.post(
    f"{BASE_URL}/api/chat-lstm",
    json={"text": "Berapa biaya layanan?"}
)
print("LSTM Response:", response.json())

# 6. BERT Only
response = requests.post(
    f"{BASE_URL}/api/chat-bert",
    json={"text": "Saya ingin mengubah password"}
)
print("BERT Response:", response.json())

# 7. Debug Prediction
response = requests.get(
    f"{BASE_URL}/api/debug-prediction",
    params={"text": "Bagaimana cara membuat akun?"}
)
print("Debug Prediction:", json.dumps(response.json(), indent=2))

# 8. Debug Response Scores
response = requests.get(
    f"{BASE_URL}/api/debug-response-scores",
    params={
        "text": "Bagaimana cara membuat akun?",
        "intent": "create_account",
        "method": "hybrid"
    }
)
print("Response Scores:", json.dumps(response.json(), indent=2))

# 9. Get Stats
response = requests.get(f"{BASE_URL}/api/stats")
print("Stats:", json.dumps(response.json(), indent=2))

# 10. Get Service Status
response = requests.get(f"{BASE_URL}/api/service-status")
print("Service Status:", response.json())
```

---

### cURL Commands

```bash
# 1. Health Check
curl -X GET "http://localhost:8000/health"

# 2. Get Intents
curl -X GET "http://localhost:8000/intents"

# 3. Single Chat
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"text": "Bagaimana cara membuat akun?"}'

# 4. LSTM Only
curl -X POST "http://localhost:8000/api/chat-lstm" \
  -H "Content-Type: application/json" \
  -d '{"text": "Berapa biaya layanan?"}'

# 5. BERT Only
curl -X POST "http://localhost:8000/api/chat-bert" \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya ingin mengubah password"}'

# 6. Batch Chat
curl -X POST "http://localhost:8000/api/batch-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Bagaimana cara membuat akun?",
      "Berapa biaya layanan?",
      "Bagaimana dukungan pelanggan?"
    ]
  }'

# 7. Debug Prediction
curl -X GET "http://localhost:8000/api/debug-prediction?text=Bagaimana%20cara%20membuat%20akun?"

# 8. Debug Response Scores
curl -X GET "http://localhost:8000/api/debug-response-scores?text=Bagaimana%20cara%20membuat%20akun?&intent=create_account&method=hybrid"

# 9. Get Stats
curl -X GET "http://localhost:8000/api/stats"

# 10. Service Status
curl -X GET "http://localhost:8000/api/service-status"
```

---

### JavaScript/TypeScript (fetch API)

```javascript
const BASE_URL = "http://localhost:8000";

// 1. Health Check
async function checkHealth() {
  const response = await fetch(`${BASE_URL}/health`);
  const data = await response.json();
  console.log("Health:", data);
  return data;
}

// 2. Single Chat
async function chat(text) {
  const response = await fetch(`${BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text })
  });
  const data = await response.json();
  console.log("Chat Response:", data);
  return data;
}

// 3. Batch Chat
async function batchChat(texts) {
  const response = await fetch(`${BASE_URL}/api/batch-chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ texts })
  });
  const data = await response.json();
  console.log("Batch Results:", data);
  return data;
}

// 4. Get Intents
async function getIntents() {
  const response = await fetch(`${BASE_URL}/intents`);
  const data = await response.json();
  console.log("Available Intents:", data);
  return data;
}

// 5. Debug Prediction
async function debugPrediction(text) {
  const response = await fetch(
    `${BASE_URL}/api/debug-prediction?text=${encodeURIComponent(text)}`
  );
  const data = await response.json();
  console.log("Debug:", data);
  return data;
}

// Usage examples
checkHealth();
chat("Bagaimana cara membuat akun?");
batchChat(["Pertanyaan 1?", "Pertanyaan 2?", "Pertanyaan 3?"]);
getIntents();
debugPrediction("Bagaimana cara membuat akun?");
```

---

## ⚠️ Error Handling

### Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request berhasil |
| 400 | Bad Request | Input validation gagal |
| 404 | Not Found | Endpoint tidak ditemukan |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service belum ready atau model belum loaded |

---

### Error Response Format

Successful response:
```json
{
  "original_text": "...",
  "predicted_intent": "...",
  "confidence": 0.92,
  "response": "...",
  "method_used": "hybrid",
  "processing_time": 125.45,
  "timestamp": "2026-04-06T10:30:00.000000"
}
```

Error response:
```json
{
  "detail": "Service not ready"
}
```

---

### Common Errors

#### 1. Service Not Ready (503)
**Penyebab**: Models belum ter-load saat startup

**Solusi**:
```bash
# Pastikan semua model files ada di direktori yang tepat:
data/lstm_models/chatbot_model.h5
data/lstm_models/tokenizer.pkl
data/lstm_models/label_encoder.pkl
data/bert_model/  (folder dengan model files)
data/dataset/dataset_training.csv
```

---

#### 2. BERT Model Unavailable (503)
**Penyebab**: Ketika menggunakan `/api/chat-bert` tapi BERT model tidak ter-load

**Solusi**:
- Gunakan `/api/chat` (hybrid) atau `/api/chat-lstm` sebagai fallback
- Check `/api/service-status` untuk verifikasi BERT availability

---

#### 3. Internal Server Error (500)
**Penyebab**: Unexpected error dalam prediction/processing

**Debugging**:
```bash
# 1. Check logs
tail -f logs/*.log

# 2. Gunakan debug endpoint
curl -X GET "http://localhost:8000/api/debug-prediction?text=your-text"

# 3. Verify service status
curl -X GET "http://localhost:8000/api/service-status"
```

---

## 🔧 Troubleshooting

### API tidak merespons
```bash
# 1. Verifikasi service running
curl -X GET "http://localhost:8000/"

# 2. Check health
curl -X GET "http://localhost:8000/health"

# 3. Lihat logs
tail -f logs/*.log

# 4. Restart service
pkill -f "uvicorn"
python app.py
```

---

### Models tidak ter-load
```bash
# 1. Check file existence
ls -la data/lstm_models/
ls -la data/bert_model/
ls -la data/dataset/

# 2. Check file permissions
chmod +r data/lstm_models/*
chmod +r data/bert_model/*

# 3. Restart & check logs
python app.py 2>&1 | grep -i "loading\|error\|model"
```

---

### Prediksi hasil tidak akurat
```bash
# 1. Debug dengan endpoint khusus
curl -X GET "http://localhost:8000/api/debug-prediction?text=your-text"

# 2. Lihat comparison LSTM vs BERT
# Analisis confidence scores

# 3. Check response options
curl -X GET "http://localhost:8000/api/debug-response-scores?text=query&intent=intent_name"

# 4. Verifikasi dataset
# Pastikan training data representatif
```

---

### Performance Issues
```bash
# 1. Monitor response times
# Gunakan /api/chat response -> processing_time field

# 2. Use lighter endpoint jika timely response penting
# /api/chat-lstm (45-80ms) vs /api/chat-bert (200-400ms)

# 3. Batch multiple requests
# /api/batch-chat lebih efisien untuk volume besar

# 4. Check system resources
top
nvidia-smi  # jika GPU tersedia
```

---

## 📚 Interactive Documentation

API ini menyediakan dokumentasi interaktif:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Gunakan untuk:
- 🧪 Test endpoint langsung
- 📖 Lihat response schema
- 🔍 Explore semua endpoints

---

## 🚀 Deployment Notes

### Development
```bash
# Auto-reload enabled
python app.py
# atau
uvicorn app:app --reload
```

### Production
```bash
# Disable auto-reload, set workers
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

# Atau dengan Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

### Environment Variables
Buat `.env` file:
```
DATASET_PATH=data/dataset/dataset_training.csv
LSTM_MODEL_PATH=data/lstm_models/chatbot_model.h5
LSTM_TOKENIZER_PATH=data/lstm_models/tokenizer.pkl
LSTM_LABEL_ENCODER_PATH=data/lstm_models/label_encoder.pkl
BERT_MODEL_PATH=data/bert_model/
```

---

## 📊 Performance Benchmarks

| Endpoint | Avg Response Time | Notes |
|----------|-------------------|-------|
| `/health` | ~5ms | Lightweight check |
| `/intents` | ~10ms | In-memory lookup |
| `/api/chat-lstm` | 45-80ms | LSTM only |
| `/api/chat` (hybrid) | 120-180ms | LSTM + BERT fusion |
| `/api/chat-bert` | 200-400ms | BERT only, needs GPU |
| `/api/batch-chat` (10 items) | 120-200ms | ~15-20ms per item |

---

## 📝 Changelog

### v3.0.0 (Current)
- ✅ Arsitektur modular LSTM + BERT
- ✅ Hybrid prediction dengan intelligent fusion
- ✅ Batch processing support
- ✅ Debug endpoints untuk analisis
- ✅ RAG integration ready
- ✅ Comprehensive error handling

### v2.0.0
- LSTM model only
- Basic response generation

### v1.0.0
- Initial release

---

## 📞 Support & Contribution

Untuk masalah atau pertanyaan:
1. Check `/api/debug-prediction` endpoint
2. Review logs untuk error messages
3. Konsultasi documentation
4. Buat issue di repository

---

**Last Updated**: April 6, 2026  
**Version**: 3.0.0  
**API Status**: ✅ Production Ready
