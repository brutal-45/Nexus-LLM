"""
Nexus Trainer - Distributed Training Engine
==============================================
Complete training loop for the Nexus transformer model.

Supports:
    - FSDP (ZeRO-3) distributed training
    - BF16 mixed precision with loss scaling
    - Gradient accumulation for large effective batch sizes
    - Gradient checkpointing for memory efficiency
    - Gradient clipping (max norm)
    - Cosine LR schedule with warmup
    - Periodic evaluation and checkpointing
    - W&B and TensorBoard logging
    - Resume from checkpoint

Usage:
    trainer = Trainer(model, config, train_dataset, val_dataset)
    trainer.train()

This trainer follows the same design principles as HuggingFace Trainer
but is built from scratch with no external dependencies beyond PyTorch.
"""

from __future__ import annotations
import os
import time
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from .parallel import setup_distributed, cleanup_distributed, configure_fsdp, get_world_info, DistributedInfo
from .scheduler import get_scheduler
from .checkpoint import CheckpointManager
from ..model.transformer import NexusTransformer, TransformerOutput
from ..model.config import ModelConfig
from ..data.dataset import DataCollator


@dataclass
class TrainingArguments:
    """Training configuration."""
    # Data
    output_dir: str = "output"
    train_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)
    tokenizer_path: Optional[str] = None
    seq_length: int = 8192
    micro_batch_size: int = 1
    global_batch_size: int = 2048
    
    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 2000
    max_steps: int = 5_000_000
    max_epochs: int = 1
    
    # Precision
    precision: str = "bf16_mixed"  # "bf16_mixed", "fp16_mixed", "fp32"
    gradient_checkpointing: bool = True
    
    # FSDP
    use_fsdp: bool = True
    fsdp_sharding: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    
    # Checkpointing
    save_interval: int = 5000
    eval_interval: int = 1000
    max_checkpoints: int = 5
    
    # Logging
    log_interval: int = 10
    log_dir: str = "logs"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    tensorboard: bool = True
    
    # Resume
    resume_from_checkpoint: Optional[str] = None
    
    @property
    def gradient_accumulation_steps_computed(self) -> int:
        """Compute gradient accumulation steps if not explicitly set."""
        if self.gradient_accumulation_steps > 0:
            return self.gradient_accumulation_steps
        dist_info = get_world_info()
        return max(1, self.global_batch_size // (self.micro_batch_size * dist_info.world_size))


class Trainer:
    """
    Main trainer class for Nexus.
    
    Handles the complete training loop including:
        1. Model initialization and FSDP wrapping
        2. Optimizer and scheduler setup
        3. Data loading and distributed sampling
        4. Training loop with mixed precision
        5. Evaluation
        6. Checkpointing
        7. Logging (W&B, TensorBoard, console)
    """

    def __init__(
        self,
        model: NexusTransformer,
        train_dataset,
        eval_dataset=None,
        args: Optional[TrainingArguments] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        self.model = model
        self.model_config = model_config or model.config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or TrainingArguments()
        
        # Setup distributed
        self.dist_info = setup_distributed()
        
        # Device
        self.device = torch.device(
            f"cuda:{self.dist_info.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        
        # Internal state
        self.global_step = 0
        self.epoch = 0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[GradScaler] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.data_collator = DataCollator()
        
        # Metrics tracking
        self.train_loss_history: List[float] = []
        self.eval_loss_history: List[float] = []
        self.tokens_seen: int = 0

    def setup(self):
        """Initialize optimizer, scheduler, FSDP, and data loaders."""
        args = self.args
        
        # === Move model to device ===
        self.model = self.model.to(self.device)
        
        # === Enable gradient checkpointing ===
        if args.gradient_checkpointing:
            self.model.enable_gradient_checkpointing()
            if self.dist_info.is_main_process:
                print(f"[Trainer] Gradient checkpointing enabled")
        
        # === Configure FSDP ===
        if args.use_fsdp and self.dist_info.is_distributed:
            if self.dist_info.is_main_process:
                print(f"[Trainer] Wrapping model with FSDP ({args.fsdp_sharding})")
            self.model = configure_fsdp(
                self.model,
                sharding_strategy=args.fsdp_sharding,
                cpu_offload=args.fsdp_cpu_offload,
            )
        
        # === Optimizer ===
        # Separate parameter groups: no decay for biases and norms, decay for others
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm", "layernorm", "rmsnorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
        )
        
        # === LR Scheduler ===
        self.scheduler = get_scheduler(
            name=args.lr_scheduler,
            optimizer=self.optimizer,
            warmup_steps=args.warmup_steps,
            total_steps=args.max_steps,
            min_lr_ratio=getattr(args, 'min_lr_ratio', 0.1),
        )
        
        # === Mixed Precision Scaler ===
        if args.precision in ("bf16_mixed", "fp16_mixed"):
            self.scaler = GradScaler(
                enabled=(args.precision == "fp16_mixed"),  # BF16 doesn't need scaling
            )
        
        # === Data Loaders ===
        self.train_loader = self._get_dataloader(self.train_dataset, shuffle=True)
        if self.eval_dataset is not None:
            self.eval_loader = self._get_dataloader(self.eval_dataset, shuffle=False)
        
        # === Checkpoint Manager ===
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            save_interval=args.save_interval,
            max_checkpoints=args.max_checkpoints,
            rank=self.dist_info.rank,
        )
        
        # === Logging ===
        self._setup_logging()
        
        # === Resume from checkpoint ===
        if args.resume_from_checkpoint:
            self._resume_from_checkpoint(args.resume_from_checkpoint)
        
        if self.dist_info.is_main_process:
            total_params = self.model.num_parameters()
            print(f"\n{'='*60}")
            print(f"Nexus Trainer Initialized")
            print(f"{'='*60}")
            print(f"  Model: {self.model_config.name}")
            print(f"  Parameters: {total_params:,} ({total_params/1e9:.1f}B)")
            print(f"  Device: {self.device}")
            print(f"  Distributed: {self.dist_info.is_distributed} "
                  f"(rank={self.dist_info.rank}, world_size={self.dist_info.world_size})")
            print(f"  Precision: {args.precision}")
            print(f"  Batch size: {args.micro_batch_size} x {args.gradient_accumulation_steps_computed} "
                  f"(global={args.global_batch_size})")
            print(f"  Learning rate: {args.learning_rate}")
            print(f"  Max steps: {args.max_steps:,}")
            print(f"  FSDP: {args.use_fsdp} ({args.fsdp_sharding})")
            print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
            print(f"{'='*60}\n")

    def _get_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader with distributed sampler."""
        if self.dist_info.is_distributed and hasattr(dataset, '__len__'):
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.dist_info.world_size,
                rank=self.dist_info.rank,
                shuffle=shuffle,
            )
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=self.args.micro_batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            collate_fn=self.data_collator,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

    def _setup_logging(self):
        """Initialize W&B and TensorBoard logging."""
        self.wandb_logger = None
        self.tb_writer = None
        
        if self.dist_info.is_main_process:
            # W&B
            if self.args.wandb_project:
                try:
                    import wandb
                    self.wandb_logger = wandb.init(
                        project=self.args.wandb_project,
                        name=self.args.wandb_run_name,
                        config=vars(self.args),
                    )
                except ImportError:
                    print("[Trainer] wandb not installed, skipping W&B logging")
            
            # TensorBoard
            if self.args.tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    log_path = os.path.join(self.args.log_dir, "tensorboard")
                    self.tb_writer = SummaryWriter(log_dir=log_path)
                except ImportError:
                    print("[Trainer] tensorboard not installed, skipping TB logging")

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all configured backends."""
        if self.dist_info.is_main_process:
            # Console
            if step % self.args.log_interval == 0:
                log_str = f"  Step {step:>8d}"
                for k, v in metrics.items():
                    log_str += f" | {k}: {v:.4f}"
                print(log_str)
            
            # W&B
            if self.wandb_logger:
                self.wandb_logger.log(metrics, step=step)
            
            # TensorBoard
            if self.tb_writer:
                for k, v in metrics.items():
                    self.tb_writer.add_scalar(k, v, step)

    def train(self):
        """
        Main training loop.
        
        Implements the complete training cycle:
        1. For each batch:
           a. Forward pass (with mixed precision)
           b. Loss computation
           c. Backward pass (with gradient accumulation)
           d. Gradient clipping
           e. Optimizer step
           f. LR scheduler step
        2. Periodic evaluation
        3. Periodic checkpoint saving
        4. Logging
        """
        self.setup()
        args = self.args
        
        if self.dist_info.is_main_process:
            print("[Trainer] Starting training...")
        
        self.model.train()
        train_iter = iter(self.train_loader)
        
        start_time = time.time()
        
        for step in range(self.global_step, args.max_steps):
            self.global_step = step
            
            # === Get next batch ===
            try:
                batch = next(train_iter)
            except StopIteration:
                # Replenish data iterator
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1
            
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # === Forward pass with mixed precision ===
            with autocast(enabled=(args.precision != "fp32"), dtype=torch.bfloat16 if args.precision == "bf16_mixed" else torch.float16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                # Apply loss mask
                loss = outputs.loss
                if loss is not None and "loss_mask" in batch:
                    # Only count loss on non-padded positions
                    loss_mask = batch["loss_mask"].float()
                    loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps_computed
            
            # === Backward pass ===
            if args.precision == "fp16_mixed":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Track tokens seen
            if "attention_mask" in batch:
                self.tokens_seen += int(batch["attention_mask"].sum().item())
            
            # === Optimizer step (with gradient accumulation) ===
            if (step + 1) % args.gradient_accumulation_steps_computed == 0:
                # Unscale gradients
                if args.precision == "fp16_mixed":
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), args.max_grad_norm
                    )
                
                # Optimizer step
                if args.precision == "fp16_mixed":
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)
                
                # LR scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
            
            # === Logging ===
            if step % args.log_interval == 0:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else args.learning_rate
                metrics = {
                    "train/loss": loss.item() * args.gradient_accumulation_steps_computed,
                    "train/lr": lr,
                    "train/tokens_seen": self.tokens_seen,
                    "train/tokens_per_sec": self.tokens_seen / max(1, time.time() - start_time),
                }
                self._log_metrics(metrics, step)
            
            # === Evaluation ===
            if step > 0 and step % args.eval_interval == 0:
                eval_metrics = self.evaluate()
                self._log_metrics(
                    {f"eval/{k}": v for k, v in eval_metrics.items()},
                    step,
                )
            
            # === Checkpointing ===
            if self.checkpoint_manager.should_save(step):
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    step=step,
                    epoch=self.epoch,
                    metrics={"train_loss": loss.item()},
                )
        
        # === Final checkpoint ===
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            step=self.global_step,
            epoch=self.epoch,
            metrics={"train_loss": loss.item()},
        )
        
        elapsed = time.time() - start_time
        if self.dist_info.is_main_process:
            print(f"\n[Trainer] Training complete!")
            print(f"  Total steps: {self.global_step:,}")
            print(f"  Total tokens: {self.tokens_seen:,}")
            print(f"  Total time: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
            print(f"  Final train loss: {loss.item():.4f}")
        
        self._close_logging()
        return self.model

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the validation dataset.
        
        Computes perplexity and loss on the eval set.
        """
        if self.eval_dataset is None or self.eval_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        eval_ppl = 0.0
        
        for batch in self.eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            with autocast(enabled=(self.args.precision != "fp32"), dtype=torch.bfloat16 if self.args.precision == "bf16_mixed" else torch.float16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                if outputs.loss is not None:
                    loss = outputs.loss.item()
                    total_loss += loss
                    num_batches += 1
        
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        
        self.model.train()
        
        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
        
        if self.dist_info.is_main_process:
            print(f"  [Eval] loss={avg_loss:.4f}, ppl={perplexity:.2f}")
        
        return metrics

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint."""
        if self.checkpoint_manager is None:
            return
        
        state = self.checkpoint_manager.load(checkpoint_path, self.model)
        
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state and self.scaler:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        if "metadata" in state:
            self.global_step = state["metadata"].get("step", 0)
            self.epoch = state["metadata"].get("epoch", 0)
        
        print(f"[Trainer] Resumed from checkpoint at step {self.global_step}")

    def _close_logging(self):
        """Close logging resources."""
        if self.wandb_logger:
            self.wandb_logger.finish()
        if self.tb_writer:
            self.tb_writer.close()
