#!/usr/bin/env python3
"""Memory usage benchmark."""
import tracemalloc
from nexus_llm.backend.inference import InferenceEngine
tracemalloc.start()
engine = InferenceEngine()
engine.load_model('gpt2-medium')
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current/1e6:.1f}MB, Peak: {peak/1e6:.1f}MB')
