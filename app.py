from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from fastapi.responses import JSONResponse
import uvicorn
import logging
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Pydantic models
class UserInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class ChatResponse(BaseModel):
    original_text: str
    predicted_intent: str
    confidence: float
    response: str
    options: Optional[List[Dict[str, Any]]] = []
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

# FastAPI app
app = FastAPI(
    title="Hybrid Chatbot API",
    description="LSTM + IndoBERT Hybrid Intent Classification API",
    version="3.0.0",  # Updated version for modular architecture
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global service instance
nlu_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize NLU service on startup dengan struktur modular"""
    global nlu_service
    try:
        logger.info("🚀 Starting Hybrid Chatbot API (Modular Version)...")
        
        # Import dari struktur modular baru
        from main import get_hybrid_nlu
        from config.settings import ModelConfig

        # Use environment vars atau default values
        dataset_path = os.environ.get('DATASET_PATH', 'data/dataset/dataset_training.csv')
        lstm_model_path = os.environ.get('LSTM_MODEL_PATH', 'data/lstm_models/chatbot_model.h5')
        lstm_tokenizer_path = os.environ.get('LSTM_TOKENIZER_PATH', 'data/lstm_models/tokenizer.pkl')
        lstm_label_encoder_path = os.environ.get('LSTM_LABEL_ENCODER_PATH', 'data/lstm_models/label_encoder.pkl')
        bert_model_path = os.environ.get('BERT_MODEL_PATH', 'data/bert_model')

        logger.info("📂 Checking model files...")
        # Check if model files exist
        model_files = {
            "lstm_model": lstm_model_path,
            "tokenizer": lstm_tokenizer_path,
            "label_encoder": lstm_label_encoder_path,
            "dataset": dataset_path,
            "bert_folder": bert_model_path
        }

        for file_type, file_path in model_files.items():
            if os.path.exists(file_path):
                logger.info(f"✅ {file_type}: {file_path} - EXISTS")
            else:
                logger.warning(f"⚠️ {file_type}: {file_path} - NOT FOUND")
        
        # Initialize dengan struktur modular baru
        logger.info("🔄 Initializing modular hybrid service...")
        
        # Initialize service menggunakan factory function
        nlu_service = get_hybrid_nlu()
        
        if nlu_service is None:
            raise Exception("Failed to initialize NLU service")
        
        # Get service status untuk logging
        status = nlu_service.get_service_status()
        
        logger.info("✅ Modular hybrid service initialized!")
        logger.info(f"📊 LSTM Model: {status['lstm_loaded']}")
        logger.info(f"📊 BERT Model: {status['bert_loaded']}") 
        logger.info(f"📊 Dataset: {status['dataset_loaded']}")
        logger.info(f"📊 Intents: {status['intents_count']}")
        logger.info(f"📊 Text Normalizer: {status['text_normalizer']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        import traceback
        traceback.print_exc()
        nlu_service = None

# Routes
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Hybrid LSTM + IndoBERT Chatbot API (Modular v3.0.0)",
        "status": "running",
        "version": "3.0.0",
        "architecture": "modular",
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
        status=status["status"],
        model_loaded=status["lstm_loaded"],
        bert_available=status["bert_loaded"],
        intents_count=status["intents_count"],
        total_patterns=status["intents_count"]  # Simplified for now
    )

@app.get("/intents")
async def get_intents():
    """Get list of available intents"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        intents = nlu_service.get_available_intents()
        intents_list = list(intents.keys())
        
        return {
            "available_intents": intents_list,
            "count": len(intents_list),
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
        
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving intents")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(user_input: UserInput):
    """Main chat endpoint - Hybrid LSTM + BERT prediction"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        # Gunakan method baru yang terintegrasi
        result = nlu_service.process_query(user_input.text)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
        
        return ChatResponse(
            original_text=user_input.text,
            predicted_intent=result["intent"],
            confidence=result["confidence"],
            response=result["response"],
            options=result.get("options", []),
            method_used=result["method"],
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
        # Use LSTM only dengan method baru
        prediction = nlu_service.predict_with_lstm(user_input.text)
        
        # Untuk compatibility, gunakan method legacy untuk response
        pattern_similarity = nlu_service.check_pattern_similarity(user_input.text, prediction["intent"])
        response_text = nlu_service.get_best_response(
            prediction["intent"],
            user_input.text,
            method_used="lstm_only",
            pattern_similarity=pattern_similarity
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": prediction["intent"],
            "confidence": prediction["confidence"],
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
        # Use BERT only dengan method baru
        prediction = nlu_service.predict_with_bert(user_input.text)
        
        if prediction["status"] == "unavailable":
            raise HTTPException(status_code=503, detail="BERT model not available")
        
        # Untuk compatibility, gunakan method legacy untuk response
        pattern_similarity = nlu_service.check_pattern_similarity(user_input.text, prediction["intent"])
        response_text = nlu_service.get_best_response(
            prediction["intent"],
            user_input.text,
            method_used="bert_only",
            pattern_similarity=pattern_similarity
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "original_text": user_input.text,
            "predicted_intent": prediction["intent"],
            "confidence": prediction["confidence"],
            "response": response_text,
            "method_used": "bert_only",
            "processing_time": round(processing_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"BERT chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/batch-chat")
async def batch_chat(batch_input: BatchInput):
    """Batch prediction endpoint"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    start_time = datetime.now()
    
    try:
        results = []
        for text in batch_input.texts:
            # Gunakan method baru yang terintegrasi
            result = nlu_service.process_query(text)
            
            results.append({
                "text": text,
                "predicted_intent": result["intent"],
                "confidence": result["confidence"],
                "response": result["response"],
                "method_used": result["method"]
            })
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "predictions": results,
            "count": len(results),
            "total_processing_time": round(processing_time, 2),
            "average_time_per_request": round(processing_time / len(results), 2)
        }
        
    except Exception as e:
        logger.error(f"Batch chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get service statistics and model information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = nlu_service.get_service_status()
    
    # Model info dari service status
    model_info = {
        "lstm_loaded": status["lstm_loaded"],
        "bert_loaded": status["bert_loaded"],
        "dataset_loaded": status["dataset_loaded"],
        "intents_count": status["intents_count"],
        "models_info": status.get("models_info", {})
    }
    
    # Performance info
    performance = {
        "prediction_methods": ["hybrid", "lstm_only", "bert_only"],
        "architecture": "modular",
        "components": ["LSTM Model", "BERT Model", "Intent Matcher", "Response Selector", "Fallback Service"]
    }
    
    # System info
    system_info = {
        "service_started": True,
        "hybrid_mode": status["bert_loaded"],
        "architecture_version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    }
    
    return StatsResponse(
        model_info=model_info,
        performance=performance,
        system_info=system_info
    )

@app.get("/api/debug-prediction")
async def debug_prediction(text: str):
    """Debug endpoint to see detailed prediction information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        # Get predictions from both models menggunakan method baru
        lstm_pred = nlu_service.predict_with_lstm(text)
        bert_pred = nlu_service.predict_with_bert(text)
        
        # Get fused prediction menggunakan method baru
        fused_pred = nlu_service.process_query(text)
        
        return {
            "input_text": text,
            "lstm_prediction": lstm_pred,
            "bert_prediction": bert_pred,
            "fused_prediction": fused_pred,
            "models_status": nlu_service.get_service_status()
        }
        
    except Exception as e:
        logger.error(f"Debug prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/debug-response-scores")
async def debug_response_scores(text: str, intent: str, method: str = "hybrid"):
    """Return candidate responses and their computed similarity/scores for an intent."""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        # Determine pattern_similarity
        pattern_similarity = nlu_service.check_pattern_similarity(text, intent)
        
        # Use response selector debug method
        scores = nlu_service.response_selector.get_response_scores(
            intent, text, method_used=method, pattern_similarity=pattern_similarity
        )
        
        return {
            "intent": intent, 
            "text": text, 
            "method": method, 
            "pattern_similarity": pattern_similarity, 
            "candidates": scores
        }

    except Exception as e:
        logger.error(f"Debug response scores error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/service-status")
async def get_service_status():
    """Get detailed service status information"""
    if nlu_service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return nlu_service.get_service_status()

# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "available_endpoints": [
            "/docs", "/health", "/api/chat", "/intents", "/api/stats", "/api/service-status"
        ]}
    )

@app.exception_handler(500)
async def server_error(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "error": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True  # Auto reload selama development
    )