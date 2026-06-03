#!/usr/bin/env python3
"""Speed benchmark."""
import time
from nexus_llm.backend.inference import InferenceEngine
engine = InferenceEngine()
engine.load_model('gpt2-medium')
start = time.time()
result = engine.generate('Hello', max_new_tokens=100)
elapsed = time.time() - start
print(f'Generated {result["generated_tokens"]} tokens in {elapsed:.2f}s')
print(f'Speed: {result["generated_tokens"]/elapsed:.1f} tokens/sec')
