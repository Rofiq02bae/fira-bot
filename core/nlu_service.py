"""
NLU Service — Hybrid LSTM + IndoBERT
v4: RAG (LLM paraphrase) hanya aktif saat response_type == "greetings".
    Semua response_type lain (kb, static, dll) dikembalikan langsung
    dari dataset tanpa melalui LLM/FAISS.
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

        # LLM client untuk paraphrase greetings — tidak menggunakan FAISS.
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
        """Initialize LLM client untuk paraphrase greetings — tanpa FAISS/vector store."""
        try:
            from services.rag import RAGService

            if not (self.rag_config.llm_provider and self.rag_config.llm_api_key):
                logger.warning("⚠️ LLM provider/key tidak dikonfigurasi. RAG (greetings) disabled.")
                self.rag_service = None
                return

            llm_client = None
            if self.rag_config.llm_provider == "openrouter":
                try:
                    from openai import AsyncOpenAI
                    llm_client = AsyncOpenAI(
                        api_key=self.rag_config.llm_api_key,
                        base_url=self.rag_config.llm_base_url
                    )
                    logger.info(f"✅ LLM client initialized (model: {self.rag_config.llm_model})")
                except ImportError:
                    logger.warning("⚠️ OpenAI library not installed. RAG disabled.")
                    self.rag_service = None
                    return
            else:
                logger.warning(f"⚠️ Unknown LLM provider: {self.rag_config.llm_provider}. RAG disabled.")
                self.rag_service = None
                return

            self.rag_service = RAGService(
                vector_store=None,
                embedding_model=None,
                llm_client=llm_client
            )

            logger.info("✅ RAG service (LLM only) initialized — aktif eksklusif untuk response_type=greetings")

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
        Async version — RAG (LLM paraphrase) hanya aktif untuk response_type=greetings.
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

            # 3. RAG hanya aktif untuk response_type == "greetings"
            intent_data = self.intent_mappings.get(base_result.get("intent", ""), {})
            response_type = intent_data.get("response_type", "static")
            dataset_response = base_result.get("response", "")

            if response_type == "greetings" and self.rag_service and self.rag_service.is_available():
                # Paraphrase dataset response via LLM — tanpa FAISS
                rag_result = await self.rag_service.generate_response(
                    query=text,
                    intent=base_result.get("intent", ""),
                    retrieved_documents=[],
                    fallback_response=dataset_response
                )
                base_result.update({
                    "response":   rag_result.get("response", dataset_response),
                    "augmented":  rag_result.get("augmented", False),
                    "sources":    [],
                    "rag_method": rag_result.get("method", "none")
                })

            # 4. [BETA] Log semua query
            intent_data_for_log = self.intent_mappings.get(base_result.get("intent", ""), {})
            self.query_logger.log_query({
                "text":             text,
                "intent":           base_result.get("intent", "unknown"),
                "confidence":       base_result.get("confidence", 0),
                "method":           base_result.get("method", "unknown"),
                "response_type":    intent_data_for_log.get("response_type", ""),
                "augmented":        base_result.get("augmented", False),
                "rag_method":       base_result.get("rag_method", ""),
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
