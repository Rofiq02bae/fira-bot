"""
NLU Service — Hybrid LSTM + IndoBERT
v3: RAG trigger berbasis kombinasi response_type + is_master
    - static + is_master=false -> RAG + LLM eksternal (jawaban natural)
    - static + is_master=true  -> response asli dataset
    - kb + is_master=false     -> response asli dataset
"""

import logging
import os
import asyncio
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
        self.thresholds = ThresholdConfig()

        self.text_normalizer = TextNormalizer(aggressive_normalization=False)
        self.intent_mappings = {}

        self.lstm_model = LSTMModel(config)
        self.bert_model = BERTModel(config)

        self.lstm_model.set_text_normalizer(self.text_normalizer)
        self.bert_model.set_text_normalizer(self.text_normalizer)

        self.intent_matcher = IntentMatcher(self.text_normalizer, self.thresholds)
        self.response_selector = ResponseSelector(self.text_normalizer, self.intent_mappings)

        self._load_dataset()

        self.query_logger = QueryLogger()
        self.ood_detector = OODDetector()

        self.pool_manager = None
        # try:
        #     from core.workers.pool_manager import PoolManager
        #     self.pool_manager = PoolManager(self.lstm_model, self.bert_model)
        #     logger.info("✅ Pool manager initialized for async processing")
        # except ImportError:
        #     logger.warning("⚠️ Pool manager not available, using sync processing")

        # RAG selalu diinisialisasi (tidak lagi bergantung flag global enabled)
        # Tapi hanya dipanggil untuk intent static non-master.
        self.rag_config = RAGConfig()
        self.rag_service = None
        self._initialize_rag_service()

        logger.info("✅ HybridNLUService initialized successfully!")

    def _load_dataset(self):
        """Load dataset dan setup intent mappings"""
        try:
            import pandas as pd

            raw_dataset_path = str(self.config.dataset_path or "").strip()
            candidate_paths = []

            if raw_dataset_path:
                for token in raw_dataset_path.replace("|", ",").split(","):
                    path = token.strip()
                    if path:
                        candidate_paths.append(path)

            candidate_paths.extend([
                "data/dataset/bert/dataset_training_bert.csv",
                "data/dataset/lstm/dataset_training_lstm.csv",
                "data/dataset/dataset_training.csv",
                "dataset_training_bert.csv",
                "dataset_training_lstm.csv"
            ])

            dataset_path = None
            seen = set()
            for path in candidate_paths:
                if path in seen:
                    continue
                seen.add(path)
                if os.path.exists(path):
                    dataset_path = path
                    break

            logger.info(f"📂 Loading dataset from: {dataset_path or raw_dataset_path}")
            if dataset_path is None:
                logger.error(f"❌ Dataset file not found: {raw_dataset_path}")
                return False

            self.config.dataset_path = dataset_path
            df = pd.read_csv(dataset_path, encoding='utf-8', keep_default_na=False)
            logger.info(f"📊 Dataset loaded: {len(df)} rows")

            if 'intent' not in df.columns:
                logger.error("❌ Dataset missing required column: intent")
                self.intent_mappings = {}
                self.response_selector.intent_mappings = {}
                return False

            cleaned = df.dropna(subset=['intent']).copy()
            cleaned['intent'] = cleaned['intent'].astype(str).str.strip()
            cleaned = cleaned[cleaned['intent'] != '']
            # Buang header ganda yang kadang terbaca sebagai data.
            cleaned = cleaned[cleaned['intent'].str.lower() != 'intent']

            def _to_bool(value: Any) -> bool:
                return str(value).strip().lower() in {"1", "true", "yes", "y"}

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
                        response_type = (str(rt.iloc[0]).strip() or 'static').lower()
                if response_type == 'response_type':
                    response_type = 'static'

                patterns = []
                if 'pattern' in intent_data.columns:
                    patterns = [
                        p for p in intent_data['pattern'].dropna().astype(str).tolist()
                        if p.strip().lower() != 'pattern'
                    ]

                is_master = False
                if 'is_master' in intent_data.columns:
                    is_master = any(_to_bool(x) for x in intent_data['is_master'].dropna())

                responses = ["Response not available"]
                if 'response' in intent_data.columns:
                    resp = [
                        r for r in intent_data['response'].dropna().astype(str).tolist()
                        if r.strip().lower() != 'response'
                    ]
                    if resp:
                        responses = resp


                intent_mappings[intent_key] = {
                    'response_type': response_type,
                    'patterns':      patterns,
                    'responses':     responses,
                    'is_master':     is_master,
                    }

            self.intent_mappings = intent_mappings
            self.response_selector.intent_mappings = self.intent_mappings

            # Ringkasan response_type
            type_counts = {}
            for v in intent_mappings.values():
                rt = v['response_type']
                type_counts[rt] = type_counts.get(rt, 0) + 1
            logger.info(f"✅ Intent mappings: {len(intent_mappings)} intents | {type_counts}")
            return len(intent_mappings) > 0

        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {e}")
            self.intent_mappings = {}
            self.response_selector.intent_mappings = {}
            return False

    def _initialize_rag_service(self):
        """Initialize RAG service — dipanggil hanya untuk intent static non-master."""
        try:
            from services.rag import RAGService, FAISSVectorStore

            logger.info("🔍 Initializing RAG service...")

            vector_store = FAISSVectorStore(
                index_path=self.rag_config.faiss_index_path,
                metadata_path=self.rag_config.faiss_metadata_path
            )

            if not vector_store.is_available():
                logger.warning("⚠️ Vector store not available. RAG disabled.")
                self.rag_service = None
                return

            llm_client = None
            if self.rag_config.llm_provider and self.rag_config.llm_api_key:
                if self.rag_config.llm_provider == "openrouter":
                    try:
                        from openai import AsyncOpenAI
                        llm_client = AsyncOpenAI(
                            api_key=self.rag_config.llm_api_key,
                            base_url=self.rag_config.llm_base_url
                        )
                        logger.info(f"✅ OpenRouter LLM client initialized (model: {self.rag_config.llm_model})")
                    except ImportError:
                        logger.warning("⚠️ OpenAI library not installed.")
                else:
                    logger.warning(f"⚠️ Unknown LLM provider: {self.rag_config.llm_provider}")

            rag_embedding_model = self.bert_model
            try:
                from sentence_transformers import SentenceTransformer
                rag_embedding_model = SentenceTransformer(self.rag_config.embedding_model_name)
                logger.info(f"✅ RAG embedding model initialized: {self.rag_config.embedding_model_name}")
            except Exception as embedding_error:
                logger.warning(
                    "⚠️ Failed to init dedicated RAG embedding model (%s): %s. Fallback to BERT classifier encoder.",
                    self.rag_config.embedding_model_name,
                    embedding_error,
                )

            self.rag_service = RAGService(
                vector_store=vector_store,
                embedding_model=rag_embedding_model,
                llm_client=llm_client
            )

            logger.info("✅ RAG service initialized — aktif hanya untuk intent static non-master")

        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {e}")
            self.rag_service = None

    def process_query(self, text: str) -> Dict[str, Any]:
        """
        Sync version — tanpa RAG (RAG hanya tersedia di async).
        """
        logger.info(f"🔍 Processing query (sync): '{text}'")

        try:
            lstm_pred = self.lstm_model.predict(text)
            bert_pred = self.bert_model.predict(text)
            return self._process_with_predictions(text, lstm_pred, bert_pred)

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return self._error_response()

    def _process_with_predictions(self, text: str, lstm_pred: Dict[str, Any], bert_pred: Dict[str, Any]) -> Dict[str, Any]:
        """Process query menggunakan prediksi LSTM dan BERT."""
        logger.info(f"   LSTM: {lstm_pred['intent']} (conf: {lstm_pred['confidence']:.3f})")
        logger.info(f"   BERT: {bert_pred['intent']} (conf: {bert_pred['confidence']:.3f})")

        intent_result = self.intent_matcher.hybrid_predict(
            text, lstm_pred, bert_pred, self.intent_mappings
        )

        confidence = intent_result.get('confidence', 0)
        normalized_text = self.text_normalizer.normalize(text)
        ood_result = self.ood_detector.process(text, normalized_text)

        if ood_result['is_ood']:
            logger.warning(f"🚫 OOD Detected: '{text}' (Reason: {ood_result['reason']})")
        elif confidence < self.thresholds.low_confidence_threshold:
            log_data = {
                'text':               text,
                'predicted_intent':   intent_result.get('intent', 'unknown'),
                'confidence':         confidence,
                'method_used':        intent_result.get('method', 'unknown'),
                'fallback_reason':    intent_result.get('fallback_reason', 'unknown'),
                'lstm_prediction':    lstm_pred.get('intent', 'unknown'),
                'bert_prediction':    bert_pred.get('intent', 'unknown'),
                'lstm_confidence':    lstm_pred.get('confidence', 0),
                'bert_confidence':    bert_pred.get('confidence', 0)
            }
            self.query_logger.log_low_confidence_query(log_data)
            logger.info(f"⚠️ Logged valuable candidate: {text}")

        response = self.response_selector.get_response(intent_result, text)

        result = {**intent_result, "response": response}
        logger.info(f"✅ Final result: {result['intent']} (method: {result['method']})")
        return result

    async def process_query_async(self, text: str) -> Dict[str, Any]:
        """
        Async version — RAG hanya untuk intent static non-master.
        """
        logger.info(f"🔍 Processing query (async): '{text}'")

        try:
            # 1. Prediksi paralel
            lstm_pred, bert_pred = await asyncio.gather(
                asyncio.to_thread(self.lstm_model.predict, text),
                asyncio.to_thread(self.bert_model.predict, text)
            )

            # 2. Proses intent + response selector
            base_result = self._process_with_predictions(text, lstm_pred, bert_pred)

            # 3. Terapkan aturan routing RAG berbasis response_type + is_master
            intent_data = self.intent_mappings.get(base_result.get("intent", ""), {})
            response_type = intent_data.get("response_type", "static")
            dataset_response = base_result.get("response", "")

            if self.rag_service and self.rag_service.is_available():

                if response_type == "kb":
                    # Prosedur panjang — return langsung, bypass RAG & LLM
                    pass  # base_result sudah benar dari response_selector

                elif response_type == "static":
                    # Skip FAISS, langsung paraphrase dataset response via LLM
                    rag_result = await self.rag_service.generate_response(
                        query=text,
                        intent=base_result.get("intent", ""),
                        retrieved_documents=[],          # ← kosong, skip FAISS
                        fallback_response=dataset_response
                    )
                    base_result.update({
                        "response":   rag_result.get("response", dataset_response),
                        "augmented":  rag_result.get("augmented", False),
                        "sources":    [],
                        "rag_method": rag_result.get("method", "none")
                    })

                else:
                    # Fallback / unknown — baru pakai FAISS
                    retrieved_docs = await self.rag_service.retrieve_documents(
                        query_text=text,
                        k=self.rag_config.top_k_documents,
                        similarity_threshold=self.rag_config.similarity_threshold
                    )
                    rag_result = await self.rag_service.generate_response(
                        query=text,
                        intent=base_result.get("intent", ""),
                        retrieved_documents=retrieved_docs,
                        fallback_response=dataset_response
                    )
                    base_result.update({
                        "response":   rag_result.get("response", dataset_response),
                        "augmented":  rag_result.get("augmented", False),
                        "sources":    rag_result.get("sources", []),
                        "rag_method": rag_result.get("method", "none")
                    })
            return base_result

        except Exception as e:
            logger.error(f"❌ Error processing query (async): {e}")
            return self._error_response()

 
    def _error_response(self) -> Dict[str, Any]:
        return {
            "intent":            "error",
            "confidence":        0.0,
            "response":          "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda.",
            "method":            "error",
            "status":            "error",
            "sources":           [],
            "pattern_similarity": 0.0,
            "fallback_reason":   "processing_error"
        }

    # ── Legacy methods — tidak diubah ──────────────────────────────────────────

    def predict_intent_hybrid(self, text: str) -> Dict[str, Any]:
        """Legacy — gunakan process_query()"""
        return self.process_query(text)

    def get_best_response(self, intent: str, user_text: str, method_used: str = "default", pattern_similarity: float = 0.0) -> str:
        """Legacy — gunakan process_query()"""
        return self.response_selector._get_best_response(intent, user_text, method_used, pattern_similarity)

    def get_service_status(self) -> Dict[str, Any]:
        lstm_status = self.lstm_model.is_available()
        bert_status = self.bert_model.is_available()

        return {
            "status":        "healthy" if (lstm_status or bert_status) else "degraded",
            "lstm_loaded":   lstm_status,
            "bert_loaded":   bert_status,
            "dataset_loaded": len(self.intent_mappings) > 0,
            "intents_count": len(self.intent_mappings),
            "rag_available": self.rag_service is not None and self.rag_service.is_available(),
            "models_info": {
                "lstm": self.lstm_model.get_model_info(),
                "bert": self.bert_model.get_model_info()
            },
            "text_normalizer": self.text_normalizer is not None
        }

    def get_available_intents(self) -> Dict[str, Any]:
        return self.intent_mappings

    def predict_with_lstm(self, text: str) -> Dict[str, Any]:
        return self.lstm_model.predict(text)

    def predict_with_bert(self, text: str) -> Dict[str, Any]:
        return self.bert_model.predict(text)

    def check_pattern_similarity(self, text: str, intent: str) -> float:
        return self.intent_matcher.check_pattern_similarity(text, intent, self.intent_mappings)
