#!/usr/bin/env python3
"""Test RAG service activation."""

from dotenv import load_dotenv
load_dotenv()

from core.nlu_service import HybridNLUService
from config.settings import ModelConfig
import asyncio

def main():
    print("🔄 Initializing NLU Service with RAG...")
    service = HybridNLUService(ModelConfig())
    
    print("\n✅ NLU Service Status:")
    status = service.get_service_status()
    print(f"  Status: {status['status']}")
    print(f"  LSTM: {status['lstm_loaded']}")
    print(f"  BERT: {status['bert_loaded']}")
    print(f"  Dataset: {status['dataset_loaded']} ({status['intents_count']} intents)")
    print(f"  Pool Manager: {service.pool_manager is not None}")
    print(f"  RAG Enabled: {service.rag_config.enabled}")
    print(f"  RAG Service: {service.rag_service is not None}")
    
    if service.rag_service:
        print(f"\n✅ RAG Service Details:")
        rag_stats = service.rag_service.get_stats()
        print(f"  Available: {rag_stats['available']}")
        print(f"  Vector Store: {rag_stats['vector_store']}")
        print(f"  Embedding Model: {rag_stats['embedding_model_available']}")
        print(f"  LLM Client: {rag_stats['llm_client_available']}")
        
        # Test query
        print(f"\n🧪 Testing RAG query...")
        result = asyncio.run(service.process_query_async("cara buat ktp"))
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  RAG Augmented: {result.get('augmented', False)}")
        print(f"  Response: {result['response'][:100]}...")
        if result.get('sources'):
            print(f"  Sources: {len(result['sources'])} documents")
    else:
        print("\n❌ RAG Service not initialized!")
        print(f"  Config enabled: {service.rag_config.enabled}")
        print(f"  LLM Provider: {service.rag_config.llm_provider}")
        print(f"  LLM API Key: {'SET' if service.rag_config.llm_api_key else 'NOT SET'}")

if __name__ == "__main__":
    main()
