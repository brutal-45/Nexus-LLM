#!/usr/bin/env python3
"""Quick start example for Nexus-LLM."""
from nexus_llm.backend.inference import InferenceEngine

engine = InferenceEngine()
engine.load_model('gpt2-medium')
result = engine.generate('What is machine learning?', max_new_tokens=100)
print(result['text'])
engine.unload_model()
