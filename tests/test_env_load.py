#!/usr/bin/env python3
"""Test environment variable loading."""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print("🔍 Environment Variables:")
print(f"  USE_RAG: {os.getenv('USE_RAG')}")
print(f"  LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"  LLM_API_KEY: {os.getenv('LLM_API_KEY')[:20]}..." if os.getenv('LLM_API_KEY') else "  LLM_API_KEY: NOT SET")
print(f"  LLM_MODEL: {os.getenv('LLM_MODEL')}")
print(f"  TELEGRAM_BOT_TOKEN: {'SET' if os.getenv('TELEGRAM_BOT_TOKEN') else 'NOT SET'}")

from config.settings import RAGConfig
config = RAGConfig()
print(f"\n🔧 RAGConfig:")
print(f"  enabled: {config.enabled}")
print(f"  llm_provider: {config.llm_provider}")
print(f"  llm_model: {config.llm_model}")
print(f"  llm_api_key: {'SET' if config.llm_api_key else 'NOT SET'}")
