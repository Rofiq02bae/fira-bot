import logging
from typing import Dict, Any
from config.settings import ModelConfig, ThresholdConfig, RAGConfig
from .models.lstm_model import LSTMModel
from .models.bert_model import BERTModel
from .processors.intent_matcher import IntentMatcher
from .processors.response_selector import ResponseSelector
from .processors.text_normalizer import TextNormalizer
from utils.query_logger import QueryLogger
from utils.ood_detector import OODDetector

logger = logging.getLogger(__name__)

class HybridNLUService:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.thresholds = ThresholdConfig() # confidence setting
        
        # Initialize text normalizer
        self.text_normalizer = TextNormalizer(aggressive_normalization=False) # normalisasi text typo
        
        # Initialize data service (akan dibuat nanti)
        self.intent_mappings = {}
        
        # Initialize models
        self.lstm_model = LSTMModel(config) # setting LSTM
        self.bert_model = BERTModel(config) # setting BERT
        
        # Inject text normalizer ke models
        self.lstm_model.set_text_normalizer(self.text_normalizer) # normalisasi text ke LSTM
        self.bert_model.set_text_normalizer(self.text_normalizer) # normalisasi text ke BERT
        
        # Initialize processors
        self.intent_matcher = IntentMatcher(self.text_normalizer, self.thresholds) # model logic dan decission 
        self.response_selector = ResponseSelector(self.text_normalizer, self.intent_mappings) # error handling dan response selection
        
        # Load dataset
        self._load_dataset()

        # Initialize query logger
        self.query_logger = QueryLogger() # catat query dengan confidence rendah
        
        # Initialize OOD detector
        self.ood_detector = OODDetector()
        
        # Initialize pool manager for async processing
        self.pool_manager = None
        try:
            from core.workers.pool_manager import PoolManager
            self.pool_manager = PoolManager(self.lstm_model, self.bert_model)
            logger.info("✅ Pool manager initialized for async processing")
        except ImportError:
            logger.warning("⚠️ Pool manager not available, using sync processing")
        
        # Initialize RAG service
        self.rag_config = RAGConfig()
        self.rag_service = None
        if self.rag_config.enabled:
            self._initialize_rag_service()
        else:
            logger.info("ℹ️ RAG service disabled")
        
        logger.info("✅ HybridNLUService initialized successfully!")

    def _load_dataset(self):
        """Load dataset dan setup intent mappings"""
        try:
            import pandas as pd
            import os
            
            logger.info(f"📂 Loading dataset from: {self.config.dataset_path}")
            
            if not os.path.exists(self.config.dataset_path):
                logger.error(f"❌ Dataset file not found: {self.config.dataset_path}")
                return False
            
            df = pd.read_csv(self.config.dataset_path, encoding='utf-8')
            logger.info(f"📊 Dataset loaded: {len(df)} rows")

            # Basic validation
            if 'intent' not in df.columns:
                logger.error("❌ Dataset missing required column: intent")
                self.intent_mappings = {}
                self.response_selector.intent_mappings = {}
                return False

            # Clean intent values to avoid NaN causing empty slices (nan != nan)
            cleaned = df.dropna(subset=['intent']).copy()
            cleaned['intent'] = cleaned['intent'].astype(str).str.strip()
            cleaned = cleaned[cleaned['intent'] != '']

            # Create intent mappings (build locally, then assign atomically)
            intent_mappings: Dict[str, Any] = {}
            for intent in cleaned['intent'].unique():
                intent_key = str(intent).strip()
                if not intent_key:
                    continue

                intent_data = cleaned[cleaned['intent'] == intent_key]
                if intent_data.empty:
                    continue

                response_type = 'static'
                if 'response_type' in intent_data.columns:
                    rt = intent_data['response_type'].dropna()
                    if not rt.empty:
                        response_type = str(rt.iloc[0]).strip() or 'static'

                patterns = []
                if 'pattern' in intent_data.columns:
                    patterns = intent_data['pattern'].dropna().astype(str).tolist()

                is_master = False
                if 'is_master' in intent_data.columns:
                    # Jika salah satu row untuk intent ini adalah master, maka intent ini master
                    is_master = any(str(x).lower() == 'true' for x in intent_data['is_master'].dropna())

                responses = ["Response not available"]
                if 'response' in intent_data.columns:
                    resp = intent_data['response'].dropna().astype(str).tolist()
                    if resp:
                        responses = resp

                intent_mappings[intent_key] = {
                    'response_type': response_type,
                    'patterns': patterns,
                    'responses': responses,
                    'is_master': is_master
                }

            self.intent_mappings = intent_mappings
            # Update response selector dengan intent mappings (critical)
            self.response_selector.intent_mappings = self.intent_mappings
            
            logger.info(f"✅ Intent mappings created: {len(self.intent_mappings)} intents")
            return len(self.intent_mappings) > 0
            
        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            # Keep selector + mappings consistent to avoid false fallbacks
            self.intent_mappings = {}
            self.response_selector.intent_mappings = {}
            return False

    def _initialize_rag_service(self):
        """Initialize RAG service with vector store and LLM."""
        try:
            from services.rag import RAGService, FAISSVectorStore
            
            logger.info("🔍 Initializing RAG service...")
            
            # Initialize vector store
            vector_store = FAISSVectorStore(
                index_path=self.rag_config.faiss_index_path,
                metadata_path=self.rag_config.faiss_metadata_path
            )
            
            if not vector_store.is_available():
                logger.warning("⚠️ Vector store not available. RAG disabled.")
                self.rag_service = None
                return
            
            # Initialize LLM client if configured (OpenRouter)
            llm_client = None
            if self.rag_config.llm_provider and self.rag_config.llm_api_key:
                if self.rag_config.llm_provider == "openrouter":
                    try:
                        from openai import AsyncOpenAI
                        # OpenRouter uses OpenAI-compatible API
                        llm_client = AsyncOpenAI(
                            api_key=self.rag_config.llm_api_key,
                            base_url=self.rag_config.llm_base_url
                        )
                        logger.info(f"✅ OpenRouter LLM client initialized (model: {self.rag_config.llm_model})")
                    except ImportError:
                        logger.warning("⚠️ OpenAI library not installed. Install with: pip install openai")
                else:
                    logger.warning(f"⚠️ Unknown LLM provider: {self.rag_config.llm_provider}. Use 'openrouter'.")
            
            # Initialize RAG service with BERT model for embeddings
            self.rag_service = RAGService(
                vector_store=vector_store,
                embedding_model=self.bert_model,
                llm_client=llm_client
            )
            
            logger.info("✅ RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self.rag_service = None
            # Keep selector + mappings consistent to avoid false fallbacks
            self.intent_mappings = {}
            self.response_selector.intent_mappings = {}
            return False

    def process_query(self, text: str) -> Dict[str, Any]:
        """
        Main method untuk memproses query user (sync version without RAG)
        Returns unified result dengan intent dan response
        """
        logger.info(f"🔍 Processing query: '{text}'")
        
        try:
            # 1. Get predictions dari kedua model
            lstm_pred = self.lstm_model.predict(text)
            bert_pred = self.bert_model.predict(text)
            return self._process_with_predictions(text, lstm_pred, bert_pred)
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "method": "error",
                "status": "error",
                "sources": [],
                "pattern_similarity": 0.0,
                "fallback_reason": "processing_error"
            }

    def _process_with_predictions(self, text: str, lstm_pred: Dict[str, Any], bert_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using provided LSTM and BERT predictions."""
        logger.info(f"   LSTM: {lstm_pred['intent']} (conf: {lstm_pred['confidence']:.3f})")
        logger.info(f"   BERT: {bert_pred['intent']} (conf: {bert_pred['confidence']:.3f})")

        # 2. Hybrid intent matching
        intent_result = self.intent_matcher.hybrid_predict(
            text, lstm_pred, bert_pred, self.intent_mappings
        )

        # 3. OOD Check & Log low confidence queries
        confidence = intent_result.get('confidence', 0)
        normalized_text = self.text_normalizer.normalize(text)
        ood_result = self.ood_detector.process(text, normalized_text)
        
        if ood_result['is_ood']:
            logger.warning(f"🚫 OOD Detected: '{text}' (Reason: {ood_result['reason']})")
            # Do not log OOD/gibberish to training_candidates.csv
        elif confidence < self.thresholds.low_confidence_threshold:
            log_data = {
                'text': text,
                'predicted_intent': intent_result.get('intent', 'unknown'),
                'confidence': confidence,
                'method_used': intent_result.get('method', 'unknown'),
                'fallback_reason': intent_result.get('fallback_reason', 'unknown'),
                'lstm_prediction': lstm_pred.get('intent', 'unknown'),
                'bert_prediction': bert_pred.get('intent', 'unknown'),
                'lstm_confidence': lstm_pred.get('confidence', 0),
                'bert_confidence': bert_pred.get('confidence', 0)
            }
            self.query_logger.log_low_confidence_query(log_data)
            logger.info(f"⚠️ Logged valuable candidate: {text}")

        # 4. Get best response
        response = self.response_selector.get_response(intent_result, text)
        
        result = {
            **intent_result,
            "response": response
        }
        
        logger.info(f"✅ Final result: {result['intent']} (method: {result['method']})")
        return result

    async def process_query_async(self, text: str) -> Dict[str, Any]:
        """
        Async method untuk memproses query user dengan multiprocess + RAG.
        """
        logger.info(f"🔍 Processing query (async with RAG): '{text}'")

        try:
            # 1. Get predictions (parallel or sync)
            if self.pool_manager:
                lstm_pred, bert_pred = await self.pool_manager.predict_both_parallel(text)
            else:
                lstm_pred = self.lstm_model.predict(text)
                bert_pred = self.bert_model.predict(text)

            # 2. Process with predictions to get intent
            base_result = self._process_with_predictions(text, lstm_pred, bert_pred)
            
            # 3. Apply RAG if enabled and conditions met
            if self.rag_service and self.rag_service.is_available():
                intent = base_result.get('intent', '')
                confidence = base_result.get('confidence', 0)
                
                # Trigger RAG for valid intents or fallback scenarios
                use_rag = (
                    (confidence >= self.rag_config.rag_min_confidence and intent not in ['error', 'unknown']) or
                    (confidence < self.rag_config.rag_min_confidence and self.rag_config.use_rag_for_fallback)
                )
                
                if use_rag:
                    logger.info(f"🔍 Triggering RAG for intent: {intent} (conf: {confidence:.3f})")
                    
                    # Retrieve relevant documents
                    retrieved_docs = await self.rag_service.retrieve_documents(
                        query_text=text,
                        k=self.rag_config.top_k_documents,
                        similarity_threshold=self.rag_config.similarity_threshold
                    )
                    
                    # Generate response with LLM
                    rag_result = await self.rag_service.generate_response(
                        query=text,
                        intent=intent,
                        retrieved_documents=retrieved_docs,
                        fallback_response=base_result.get('response', '')
                    )
                    
                    # Merge RAG result with base result
                    base_result.update({
                        'response': rag_result.get('response', base_result['response']),
                        'augmented': rag_result.get('augmented', False),
                        'sources': rag_result.get('sources', []),
                        'rag_method': rag_result.get('method', 'none')
                    })
                    
                    logger.info(f"✅ RAG applied: {rag_result.get('method', 'none')}")

            return base_result

        except Exception as e:
            logger.error(f"❌ Error processing query (async): {e}")
            return {
                "intent": "error",
                "confidence": 0.0,
                "response": "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
                "method": "error",
                "status": "error",
                "sources": [],
                "pattern_similarity": 0.0,
                "fallback_reason": "processing_error"
            }

    def predict_intent_hybrid(self, text: str) -> Dict[str, Any]:
        """
        Legacy method untuk compatibility dengan code existing
        TODO: Deprecate dan ganti dengan process_query()
        """
        return self.process_query(text)

    def get_best_response(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> str:
        """
        Legacy method untuk compatibility dengan code existing
        TODO: Deprecate dan ganti dengan process_query()
        """
        return self.response_selector._get_best_response(intent, user_text, method_used, pattern_similarity)

    def get_service_status(self) -> Dict[str, Any]:
        """Get health status semua komponen"""
        lstm_status = self.lstm_model.is_available()
        bert_status = self.bert_model.is_available()
        
        return {
            "status": "healthy" if (lstm_status or bert_status) else "degraded",
            "lstm_loaded": lstm_status,
            "bert_loaded": bert_status,
            "dataset_loaded": len(self.intent_mappings) > 0,
            "intents_count": len(self.intent_mappings),
            "models_info": {
                "lstm": self.lstm_model.get_model_info(),
                "bert": self.bert_model.get_model_info()
            },
            "text_normalizer": self.text_normalizer is not None
        }

    def get_available_intents(self) -> Dict[str, Any]:
        """Get available intents dan detailsnya"""
        return self.intent_mappings

    # Legacy methods untuk compatibility
    def predict_with_lstm(self, text: str) -> Dict[str, Any]:
        """Predict menggunakan LSTM only - untuk compatibility"""
        return self.lstm_model.predict(text)

    def predict_with_bert(self, text: str) -> Dict[str, Any]:
        """Predict menggunakan BERT only - untuk compatibility"""
        return self.bert_model.predict(text)

    def check_pattern_similarity(self, text: str, intent: str) -> float:
        """Check pattern similarity - untuk compatibility"""
        return self.intent_matcher.check_pattern_similarity(text, intent, self.intent_mappings)