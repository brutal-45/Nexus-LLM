#!/usr/bin/env python3
"""Advanced example 6 for Nexus-LLM."""
import sys
sys.path.insert(0, "..")
from nexus_llm.backend.inference import InferenceEngine

def main():
    engine = InferenceEngine()
    engine.load_model("gpt2-medium")
    result = engine.generate("Example prompt 6", max_new_tokens=50)
    print(result["text"])
    engine.unload_model()

if __name__ == "__main__":
    main()
