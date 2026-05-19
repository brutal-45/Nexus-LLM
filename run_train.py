#!/usr/bin/env python3
"""Nexus-LLM Training Runner Script.

Start fine-tuning an LLM on a dataset.
"""

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    """Main entry point for the training runner."""
    parser = argparse.ArgumentParser(
        description="Nexus-LLM Train - Fine-tune an LLM on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_train.py --dataset ./data/train.jsonl
  python run_train.py --model gpt2-medium --dataset ./data/train.jsonl --epochs 5
  python run_train.py --model mistral-7b --dataset ./data/train.jsonl --lora-rank 16 --fp16
        """,
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("NEXUS_LLM_DEFAULT_MODEL", "gpt2-medium"),
        help="Base model name or path (default: gpt2-medium)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to training dataset (JSONL/JSON/CSV/Parquet)",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("NEXUS_LLM_TRAIN_OUTPUT_DIR", "./output"),
        help="Output directory for checkpoints (default: ./output)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_TRAIN_EPOCHS", "3")),
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_TRAIN_BATCH_SIZE", "8")),
        help="Training batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("NEXUS_LLM_TRAIN_LEARNING_RATE", "2e-5")),
        help="Learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank, 0 to disable LoRA (default: 8)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha (default: 16)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate (default: 0.05)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=int(os.environ.get("NEXUS_LLM_MAX_SEQ_LENGTH", "2048")),
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=os.environ.get("NEXUS_LLM_TRAIN_FP16", "false").lower() == "true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=os.environ.get("NEXUS_LLM_TRAIN_BF16", "false").lower() == "true",
        help="Use BF16 mixed precision",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("NEXUS_LLM_DEVICE", "auto"),
        help="Device to use: auto/cpu/cuda/mps (default: auto)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint to resume training from",
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

    config = {
        "model": args.model,
        "dataset": args.dataset,
        "output_dir": args.output,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "max_seq_length": args.max_seq_length,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "device": args.device,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "validation_split": args.validation_split,
        "seed": args.seed,
        "resume_from": args.resume_from,
    }

    app.run_train(config)


if __name__ == "__main__":
    main()
