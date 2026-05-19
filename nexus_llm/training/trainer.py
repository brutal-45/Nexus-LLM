"""Full training loop with progress tracking, gradient accumulation, mixed precision, and gradient clipping."""

import os
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from nexus_llm.training.scheduler import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from nexus_llm.training.metrics import MetricsTracker
from nexus_llm.training.callbacks import CallbackManager
from nexus_llm.training.checkpoint import CheckpointManager
from nexus_llm.training.optimizer import build_optimizer, OptimizerConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    max_steps: int = -1
    seed: int = 42
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 500
    save_total_limit: int = 5
    output_dir: str = "./output"
    scheduler_type: str = "linear"
    logging_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16_opt_level: str = "O1"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_end: float = 0.0
    power: float = 1.0
    num_cycles: float = 0.5

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.logging_dir is not None:
            os.makedirs(self.logging_dir, exist_ok=True)


class Trainer:
    """Main training loop with full support for gradient accumulation, mixed precision, and gradient clipping."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        data_collator: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        callbacks: Optional[List] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.device = self._get_device()
        self.model.to(self.device)

        self.metrics_tracker = MetricsTracker()
        self.callback_manager = CallbackManager(callbacks or [])
        self.checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            save_total_limit=config.save_total_limit,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,
        )

        self.optimizer = optimizer or build_optimizer(
            model=self.model,
            config=OptimizerConfig(
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                adam_beta1=config.adam_beta1,
                adam_beta2=config.adam_beta2,
                adam_epsilon=config.adam_epsilon,
            ),
        )

        self.scaler = None
        if config.fp16:
            self.scaler = GradScaler(init_scale=2**14)
        elif config.bf16:
            self.scaler = GradScaler(enabled=False)

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.total_train_time = 0.0
        self._rng = torch.Generator()
        self._rng.manual_seed(config.seed)

        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _create_dataloader(self, dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self.data_collator,
            generator=self._rng if shuffle else None,
        )

    def _create_scheduler(self, num_training_steps: int):
        warmup_steps = self.config.warmup_steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, num_training_steps
            )
        elif self.config.scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer, warmup_steps, num_training_steps
            )
        else:
            return get_linear_schedule_with_warmup(
                self.optimizer, warmup_steps, num_training_steps
            )

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop."""
        self.callback_manager.on_train_begin(self.config)

        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        num_update_steps_per_epoch = math.ceil(
            len(train_loader) / self.config.gradient_accumulation_steps
        )
        num_training_steps = num_update_steps_per_epoch * self.config.num_epochs
        if self.config.max_steps > 0:
            num_training_steps = min(num_training_steps, self.config.max_steps)

        self.lr_scheduler = self._create_scheduler(num_training_steps)

        if self.config.resume_from_checkpoint:
            self._resume_from_checkpoint(self.config.resume_from_checkpoint)

        self.model.train()
        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.callback_manager.on_epoch_begin(epoch, self.config)
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(train_loader):
                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self._optimization_step()
                    self.global_step += 1

                    if self.global_step % self.config.log_interval == 0:
                        avg_loss = epoch_loss / max(num_batches, 1)
                        lr = self.lr_scheduler.get_last_lr()[0]
                        self.metrics_tracker.log(
                            step=self.global_step,
                            loss=avg_loss,
                            learning_rate=lr,
                        )
                        self.callback_manager.on_log(
                            self.global_step, {"loss": avg_loss, "lr": lr}
                        )
                        logger.info(
                            f"Epoch {epoch} | Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                        )

                    if self.config.eval_interval > 0 and self.global_step % self.config.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.callback_manager.on_evaluate(
                            self.global_step, eval_metrics
                        )
                        should_save = self.checkpoint_manager.is_best(eval_metrics)
                        if should_save:
                            self._save_checkpoint(eval_metrics)

                    if self.config.save_interval > 0 and self.global_step % self.config.save_interval == 0:
                        self._save_checkpoint({})

                    if self.callback_manager.should_stop:
                        logger.info("Training stopped by callback.")
                        break

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.callback_manager.on_epoch_end(epoch, {"loss": avg_epoch_loss})
            logger.info(
                f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}"
            )

            if self.callback_manager.should_stop:
                break

        self.total_train_time = time.time() - start_time
        self._save_checkpoint({})
        self.callback_manager.on_train_end(self.metrics_tracker.get_summary())

        return self.metrics_tracker.get_summary()

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if self.config.fp16:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss = loss / self.config.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
        elif self.config.bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
                loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
        else:
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _optimization_step(self):
        """Perform gradient clipping and optimizer step."""
        if self.config.fp16:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()

        self.lr_scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataset."""
        if self.eval_dataset is None:
            return {}

        self.model.eval()
        eval_loader = self._create_dataloader(self.eval_dataset, shuffle=False)
        total_loss = 0.0
        total_samples = 0

        for batch in eval_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if self.config.fp16:
                with autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            batch_size = batch.get("input_ids", batch.get("labels", None)).shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        metrics = {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }

        self.model.train()
        return metrics

    def _save_checkpoint(self, metrics: Dict[str, float]):
        """Save a training checkpoint."""
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()

        self.checkpoint_manager.save(state, self.global_step, metrics)

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "lr_scheduler_state_dict" in checkpoint and hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.current_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Resumed training from step {self.global_step}, epoch {self.current_epoch}")

    def predict(self, test_dataset, batch_size: Optional[int] = None) -> List[Any]:
        """Run prediction on a test dataset."""
        self.model.eval()
        bs = batch_size or self.config.batch_size
        loader = DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            collate_fn=self.data_collator,
        )
        predictions = []

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                if self.config.fp16:
                    with autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
            if hasattr(outputs, "logits"):
                preds = outputs.logits.argmax(dim=-1)
            else:
                preds = outputs
            predictions.append(preds.cpu())

        return torch.cat(predictions, dim=0) if predictions else []
