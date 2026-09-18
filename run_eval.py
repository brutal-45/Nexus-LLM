#!/usr/bin/env python3
"""Nexus-LLM Evaluation Runner Script.

Evaluate LLM models on benchmark datasets.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Main entry point for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Eval - Evaluate LLM models on benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py --model gpt2-medium
  python run_eval.py --model gpt2-medium --benchmark mmlu
  python run_eval.py --model mistral-7b --tasks hellaswag,arc_challenge
  python run_eval.py --model gpt2-medium --output ./my_results
        """,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium"),
        help="Model name or path to evaluate (default: gpt2-medium)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark dataset to use (mmlu, hellaswag, arc, truthfulqa, etc.)",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated evaluation tasks",
    )
    parser.add_argument(
        "--output",
        default="./eval_results",
        help="Output directory for evaluation results (default: ./eval_results)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_LLM_DEVICE", "auto"),
        help="Device to use: auto/cpu/cuda/mps (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 for evaluation",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use BF16 for evaluation",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        from nexus_llm.app import NexusLLMApp
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from nexus_llm.app import NexusLLMApp

    app = NexusLLMApp(config_path=args.config)

    task_list = args.tasks.split(",") if args.tasks else None

    config = {
        "model": args.model,
        "benchmark": args.benchmark,
        "tasks": task_list,
        "output_dir": args.output,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "save_predictions": args.save_predictions,
        "use_wandb": args.wandb,
    }

    app.run_eval(config)


if __name__ == "__main__":
    main()
