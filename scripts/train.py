"""
Nexus Training Script (CLI)
===============================
Usage:
    python -m nexus.scripts.train --config configs/base_100b.yaml
    python -m nexus.scripts.train --config configs/base_100b.yaml --resume checkpoints/checkpoint-step-00010000

Supports:
    - Training from scratch
    - Resuming from checkpoint
    - Multi-GPU distributed training (via torchrun)
    - Custom config overrides via CLI args
"""

from __future__ import annotations
import argparse
import os
import sys
import yaml
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nexus.model.config import ModelConfig
from nexus.model.transformer import NexusTransformer
from nexus.data.tokenizer import BPETokenizer
from nexus.data.dataset import StreamingDataset
from nexus.training.trainer import Trainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="Nexus Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Override micro batch size")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    model_cfg = ModelConfig.from_yaml(args.config)
    train_cfg = cfg.get("training", {})
    
    print(f"\n{'='*60}")
    print(f"Nexus Training")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Model: {model_cfg}")
    print(f"{'='*60}\n")
    
    # Override config from CLI args
    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir
    if args.max_steps:
        train_cfg["max_steps"] = args.max_steps
    if args.lr:
        train_cfg["lr"] = args.lr
    if args.batch_size:
        train_cfg["micro_batch_size"] = args.batch_size
    if args.data_dir:
        train_cfg["data_dir"] = args.data_dir
    
    # Build model
    model = NexusTransformer(model_cfg)
    print(f"Model created: {model.num_parameters():,} parameters")
    
    # Load tokenizer
    tokenizer_path = train_cfg.get("tokenizer_path", "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"[Warning] Tokenizer not found at {tokenizer_path}, using base tokenizer")
        tokenizer = BPETokenizer()
    
    # Build dataset
    data_dir = train_cfg.get("data_dir", "/data/nexus/corpus")
    data_files = []
    if os.path.isdir(data_dir):
        for fname in os.listdir(data_dir):
            if fname.endswith((".jsonl", ".txt")):
                data_files.append(os.path.join(data_dir, fname))
    elif os.path.isfile(data_dir):
        data_files = [data_dir]
    
    if not data_files:
        print(f"[Warning] No data files found in {data_dir}")
        print("Creating synthetic dataset for demonstration...")
        # Create a synthetic dataset
        from nexus.data.dataset import PackedDataset
        synthetic_docs = [
            "The quick brown fox jumps over the lazy dog. " * 100,
            "In a world where AI transforms everything, " * 100,
        ] * 1000
        train_dataset = PackedDataset(
            synthetic_docs, tokenizer,
            seq_length=train_cfg.get("seq_length", 8192),
        )
    else:
        train_dataset = StreamingDataset(
            data_files=data_files,
            tokenizer=tokenizer,
            seq_length=train_cfg.get("seq_length", 8192),
            shuffle=True,
        )
    
    print(f"Training dataset: {len(data_files)} files")
    
    # Build training arguments
    training_args = TrainingArguments(
        output_dir=train_cfg.get("output_dir", "output"),
        learning_rate=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 0.1),
        warmup_steps=train_cfg.get("warmup_steps", 2000),
        max_steps=train_cfg.get("max_steps", 5_000_000),
        max_epochs=train_cfg.get("max_epochs", 1),
        micro_batch_size=train_cfg.get("micro_batch_size", 1),
        global_batch_size=train_cfg.get("global_batch_size", 2048),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        seq_length=train_cfg.get("seq_length", 8192),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        precision=train_cfg.get("precision", "bf16_mixed"),
        use_fsdp=train_cfg.get("fsdp", True),
        fsdp_sharding=train_cfg.get("fsdp_sharding_strategy", "FULL_SHARD"),
        save_interval=train_cfg.get("save_interval", 5000),
        eval_interval=train_cfg.get("eval_interval", 1000),
        log_interval=train_cfg.get("log_interval", 10),
        log_dir=train_cfg.get("log_dir", "logs"),
        wandb_project=train_cfg.get("wandb_project", None),
        tensorboard=train_cfg.get("tensorboard", True),
        resume_from_checkpoint=args.resume,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        model_config=model_cfg,
    )
    
    # Train
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
