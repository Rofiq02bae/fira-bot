from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime

# Import dari struktur modular baru
from core.nlu_service import HybridNLUService
from config.settings import ModelConfig, APIConfig
from main import initialize_hybrid_service, get_hybrid_nlu

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Pydantic models (SAMA)
class UserInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class ChatResponse(BaseModel):
    original_text: str
    predicted_intent: str
    confidence: float
    response: str
    method_used: str
    processing_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    bert_available: bool
    intents_count: int
    total_patterns: int

class StatsResponse(BaseModel):
    model_info: Dict
    performance: Dict
    system_info: Dict

# FastAPI app (SAMA)
app = FastAPI(
    title="Hybrid Chatbot API",
    description="LSTM + IndoBERT Hybrid Intent Classification API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global service instance - akan diisi di startup
nlu_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize NLU service on startup - LEBIH SIMPLE"""
    global nlu_service
    try:
        logger.info("🚀 Starting Hybrid Chatbot API...")
        
        # Gunakan factory function dari main.py
        nlu_service = get_hybrid_nlu()
        
        if nlu_service is None:
            raise Exception("Failed to initialize NLU service")
            
        # Get service status untuk logging
        status = nlu_service.get_service_status()
        
        logger.info("✅ Hybrid service initialized!")
        logger.info(f"📊 LSTM Model: {status['lstm_loaded']}")
        logger.info(f"📊 BERT Model: {status['bert_loaded']}") 
        logger.info(f"📊 Intents: {status['intents_count']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        import traceback
        traceback.print_exc()
        nlu_service = None

# Routes - SEMUA ENDPOINT TETAP SAMA TAPI LEBIH CLEAN
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Hybrid LSTM + IndoBERT Chatbot API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = nlu_service.get_service_status()
    
    return HealthResponse(
        status="healthy",
        model_loaded=status["lstm_loaded"],
        bert_available=status["bert_loaded"],
        intents_count=status["intents_count"],
        total_patterns=status.get("total_patterns", 0)
    )

@app.get("/intents")
async def get_intents():
    """Get list of available intents"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # NOTE: Kita perlu tambahkan method di nlu_service untuk get intent details
    intents = nlu_service.get_available_intents()  # Method baru yang perlu dibuat
    
    return {
        "available_intents": list(intents.keys()),
        "count": len(intents),
        "intents_details": [
            {
                "intent": intent,
                "response_type": data['response_type'],
                "patterns_count": len(data['patterns']),
                "responses_count": len(data['responses'])
            }
            for intent, data in intents.items()
        ]
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(user_input: UserInput):
    """Main chat endpoint - Hybrid LSTM + BERT prediction"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # SINGLE METHOD CALL - lebih clean!
        result = nlu_service.process_query(user_input.text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return ChatResponse(
            original_text=user_input.text,
            predicted_intent=result["intent"],
            confidence=result["confidence"],
            response=result["response"],
            method_used=result["method_used"],
            processing_time=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat-lstm")
async def chat_lstm_only(user_input: UserInput):
    """Chat using LSTM only (faster)"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # Gunakan LSTM langsung
        lstm_pred = nlu_service.lstm_model.predict(user_input.text)
        
        # Get response menggunakan response selector
        response_text = nlu_service.response_selector.get_response(
            lstm_pred, user_input.text, "lstm_only"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": lstm_pred["intent"],
            "confidence": lstm_pred["confidence"],
            "response": response_text,
            "method_used": "lstm_only",
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"LSTM chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat-bert")
async def chat_bert_only(user_input: UserInput):
    """Chat using BERT only (more accurate)"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # Gunakan BERT langsung
        bert_pred = nlu_service.bert_model.predict(user_input.text)
        
        # Get response menggunakan response selector
        response_text = nlu_service.response_selector.get_response(
            bert_pred, user_input.text, "bert_only"
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": bert_pred["intent"],
            "confidence": bert_pred["confidence"],
            "response": response_text,
            "method_used": "bert_only",
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"BERT chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ENDPOINT LAINNYA (batch-chat, stats, debug) - ADAPTASI SERUPA

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics and model information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = nlu_service.get_service_status()
    
    # Model info
    model_info = {
        "lstm_loaded": status["lstm_loaded"],
        "bert_loaded": status["bert_loaded"],
        "intents_count": status["intents_count"],
        "models_info": status.get("models_info", {})
    }
    
    # Performance info
    performance = {
        "prediction_methods": ["hybrid", "lstm_only", "bert_only"],
        "supported_strategies": nlu_service.intent_matcher.thresholds
    }
    
    # System info
    system_info = {
        "service_started": True,
        "hybrid_mode": status["bert_loaded"],
        "timestamp": datetime.now().isoformat()
    }
    
    return StatsResponse(
        model_info=model_info,
        performance=performance,
        system_info=system_info
    )

# Error handlers (SAMA)
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "available_endpoints": [
            "/docs", "/health", "/api/chat", "/intents", "/api/stats"
        ]}
    )

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        reload=config.debug
    )