#!/usr/bin/env python3
"""Multi-model example."""
from nexus_llm.backend.inference import InferenceEngine

engine = InferenceEngine()
for model in ['gpt2', 'gpt2-medium']:
    engine.load_model(model)
    result = engine.generate('Hello', max_new_tokens=30)
    print(f'{model}: {result["text"][:50]}...')
    engine.unload_model()
