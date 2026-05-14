"""
Nexus Inference Script (CLI)
================================
Usage:
    python -m nexus.scripts.infer --checkpoint checkpoints/nexus-100b --prompt "Hello, world!"
    python -m nexus.scripts.infer --checkpoint checkpoints/nexus-100b --interactive
    python -m nexus.scripts.infer --checkpoint checkpoints/nexus-100b --prompt "Explain quantum computing" --max_tokens 1024 --temperature 0.7
"""

from __future__ import annotations 
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="cuda" if __import__('torch').cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    
    from nexus.model.transformer import NexusTransformer
    from nexus.data.tokenizer import BPETokenizer
    from nexus.inference.generator import TextGenerator, GenerationConfig
    
    print(f"\nLoading model from {args.checkpoint}...")
    
    # Load model
    model = NexusTransformer.from_pretrained(args.checkpoint, device=args.device)
    
    # Load tokenizer
    tokenizer_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer()
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device=args.device)
    
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    
    if args.interactive:
        print("\n=== Nexus Interactive Mode ===")
        print("Type 'quit' or 'exit' to stop.\n")
        
        while True:
            try:
                prompt = input("You: ")
                if prompt.lower() in ("quit", "exit"):
                    break
                if not prompt.strip():
                    continue
                
                print("Assistant: ", end="", flush=True)
                for chunk in generator.stream_generate(prompt, gen_config):
                    print(chunk, end="", flush=True)
                print("\n")
            except KeyboardInterrupt:
                print("\n")
                break
    
    elif args.prompt:
        print(f"\nPrompt: {args.prompt}")
        print(f"Generating (max_tokens={args.max_tokens}, temp={args.temperature})...\n")
        
        result = generator.generate(args.prompt, gen_config)
        
        print(f"Generated ({result.num_generated_tokens} tokens):")
        print("-" * 40)
        print(result.generated_text[0])
        print("-" * 40)
    else:
        print("Error: Provide --prompt or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
