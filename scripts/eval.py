"""
Nexus Evaluation Script (CLI)
=================================
Usage:
    python -m nexus.scripts.eval --checkpoint checkpoints/nexus-100b --tasks mmlu,gsm8k
    python -m nexus.scripts.eval --checkpoint checkpoints/nexus-100b --tasks all
"""

from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated benchmark names")
    parser.add_argument("--max_tokens", type=int, default=128, help="Max generation tokens")
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    import torch
    from nexus.model.transformer import NexusTransformer
    from nexus.data.tokenizer import BPETokenizer
    from nexus.evaluation.benchmarks import BenchmarkHarness, BenchmarkConfig
    
    print(f"\nLoading model from {args.checkpoint}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NexusTransformer.from_pretrained(args.checkpoint, device=device)
    
    tokenizer_path = os.path.join(args.checkpoint, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer()
    
    # Parse tasks 
    if args.tasks == "all":
        tasks = ["mmlu", "gsm8k", "humaneval", "hellaswag", "arc"]
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
    
    config = BenchmarkConfig(
        max_new_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )
    
    harness = BenchmarkHarness(model, tokenizer, config)
    results = harness.run(tasks)
    
    BenchmarkHarness.print_results(results)


if __name__ == "__main__":
    main()
