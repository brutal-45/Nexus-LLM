"""Distillation training loop for Nexus-LLM.

Provides :class:`DistillationTrainer` — a higher-level wrapper around
:class:`Distiller` that adds step-by-step logging, progress tracking,
and checkpoint saving.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.distillation.config import DistillationConfig
from nexus_llm.distillation.distiller import Distiller

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Training loop for knowledge distillation with logging and checkpoints.

    Unlike the bare :class:`Distiller`, the trainer provides:

    * Step-by-step and epoch-level logging.
    * Progress-bar-style console output.
    * Periodic checkpoint saving.
    * Final model export.

    Usage::

        trainer = DistillationTrainer(
            config=DistillationConfig(epochs=3),
            checkpoint_dir="checkpoints/distill",
        )
        student = trainer.train(teacher, student, dataset)
    """

    def __init__(
        self,
        config: Optional[DistillationConfig] = None,
        distiller: Optional[Distiller] = None,
        checkpoint_dir: str = "checkpoints/distillation",
        checkpoint_every: int = 500,
        log_every: int = 10,
    ) -> None:
        """Initialise the DistillationTrainer.

        Args:
            config: Distillation hyper-parameters.  Defaults to a
                ``DistillationConfig()`` with standard values.
            distiller: A :class:`Distiller` instance.  If *None*, a
                default one is created.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_every: Save a checkpoint every N global steps.
                Set to 0 to disable intermediate checkpoints.
            log_every: Log training metrics every N steps.
        """
        self._config = config or DistillationConfig()
        self._distiller = distiller or Distiller()
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_every = checkpoint_every
        self._log_every = log_every

        # Runtime state
        self._global_step = 0
        self._current_epoch = 0
        self._history: List[Dict[str, Any]] = []
        self._best_loss = float("inf")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Current global training step."""
        return self._global_step

    @property
    def current_epoch(self) -> int:
        """Current epoch number (1-indexed during training)."""
        return self._current_epoch

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Training history — one entry per logged step."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        teacher: Any,
        student: Any,
        dataset: List[Dict[str, Any]],
        config: Optional[DistillationConfig] = None,
    ) -> Any:
        """Run the distillation training loop.

        Args:
            teacher: The teacher model (frozen).
            student: The student model (trained in-place).
            dataset: Training data — list of dicts with ``"input_ids"``
                and ``"labels"``.
            config: Override the configuration passed at init time.

        Returns:
            The trained student model.
        """
        import torch
        import torch.nn.functional as F

        cfg = config or self._config
        self._global_step = 0
        self._history.clear()
        self._best_loss = float("inf")

        device = self._resolve_device(teacher)
        teacher = teacher.to(device)
        student = student.to(device)
        teacher.eval()

        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.1
        )

        num_batches = max(1, len(dataset) // cfg.batch_size)
        total_steps = num_batches * cfg.epochs
        start_time = time.perf_counter()

        logger.info(
            "DistillationTrainer starting — %d epochs, %d batches/epoch, "
            "%d total steps, device=%s",
            cfg.epochs,
            num_batches,
            total_steps,
            device,
        )
        self._print_header(cfg)

        for epoch in range(1, cfg.epochs + 1):
            self._current_epoch = epoch
            student.train()
            epoch_loss = 0.0
            epoch_soft_loss = 0.0
            epoch_hard_loss = 0.0

            for batch_idx in range(num_batches):
                self._global_step += 1

                # --- Fetch batch ---
                start = batch_idx * cfg.batch_size
                end = start + cfg.batch_size
                batch = dataset[start:end]

                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(s["input_ids"]) for s in batch],
                    batch_first=True,
                ).to(device)
                labels = torch.cat(
                    [torch.as_tensor(s["labels"]) for s in batch]
                ).to(device)
                attention_mask = (input_ids != 0).long()

                # --- Teacher forward ---
                with torch.no_grad():
                    teacher_logits = Distiller.compute_teacher_logits(
                        teacher,
                        {"input_ids": input_ids, "attention_mask": attention_mask},
                    )

                # --- Student forward ---
                student_outputs = student(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = (
                    student_outputs.logits
                    if hasattr(student_outputs, "logits")
                    else student_outputs
                )

                # --- Losses ---
                soft_loss = Distiller.distillation_loss(
                    student_logits, teacher_logits, cfg.temperature
                )
                hard_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = cfg.alpha * soft_loss + (1.0 - cfg.alpha) * hard_loss

                # --- Backward ---
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                optimizer.step()

                # --- Track ---
                epoch_loss += total_loss.item()
                epoch_soft_loss += soft_loss.item()
                epoch_hard_loss += hard_loss.item()

                # --- Logging ---
                if self._log_every > 0 and self._global_step % self._log_every == 0:
                    self._log_step(
                        epoch=epoch,
                        step=self._global_step,
                        total_steps=total_steps,
                        total_loss=total_loss.item(),
                        soft_loss=soft_loss.item(),
                        hard_loss=hard_loss.item(),
                        lr=optimizer.param_groups[0]["lr"],
                    )

                # --- Checkpointing ---
                if self._checkpoint_every > 0 and self._global_step % self._checkpoint_every == 0:
                    self._save_checkpoint(student, optimizer, epoch, total_loss.item())

            # --- End of epoch ---
            scheduler.step()
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_soft = epoch_soft_loss / max(num_batches, 1)
            avg_hard = epoch_hard_loss / max(num_batches, 1)

            logger.info(
                "Epoch %d complete — avg_loss=%.4f (soft=%.4f, hard=%.4f)",
                epoch,
                avg_loss,
                avg_soft,
                avg_hard,
            )

            # Track best model
            if avg_loss < self._best_loss:
                self._best_loss = avg_loss
                self._save_checkpoint(student, optimizer, epoch, avg_loss, best=True)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Distillation training complete — %.1fs, %d steps, best_loss=%.4f",
            elapsed,
            self._global_step,
            self._best_loss,
        )

        # Save final model
        final_dir = self._checkpoint_dir / "final"
        self._save_model(student, final_dir)

        student.eval()
        return student

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_step(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        total_loss: float,
        soft_loss: float,
        hard_loss: float,
        lr: float,
    ) -> None:
        """Record a training step and print a progress line."""
        elapsed = time.perf_counter()
        record = {
            "epoch": epoch,
            "step": step,
            "total_loss": round(total_loss, 6),
            "soft_loss": round(soft_loss, 6),
            "hard_loss": round(hard_loss, 6),
            "lr": lr,
            "timestamp": elapsed,
        }
        self._history.append(record)

        pct = step / max(total_steps, 1) * 100
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)

        logger.debug(
            "Step %d/%d [%s] %.1f%% — loss=%.4f (soft=%.4f hard=%.4f) lr=%.2e",
            step,
            total_steps,
            bar,
            pct,
            total_loss,
            soft_loss,
            hard_loss,
            lr,
        )

    @staticmethod
    def _print_header(cfg: DistillationConfig) -> None:
        """Print the training configuration header."""
        logger.info(
            "Configuration: temperature=%.1f, alpha=%.2f, lr=%.2e, "
            "batch_size=%d, epochs=%d",
            cfg.temperature,
            cfg.alpha,
            cfg.learning_rate,
            cfg.batch_size,
            cfg.epochs,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: Any,
        optimizer: Any,
        epoch: int,
        loss: float,
        best: bool = False,
    ) -> str:
        """Save a training checkpoint.

        Args:
            model: The student model.
            optimizer: The optimiser.
            epoch: Current epoch number.
            loss: Current loss value.
            best: Whether this is the best checkpoint so far.

        Returns:
            Path to the saved checkpoint directory.
        """
        tag = "best" if best else f"step_{self._global_step}"
        ckpt_dir = self._checkpoint_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self._save_model(model, ckpt_dir)

        # Save trainer state
        state = {
            "epoch": epoch,
            "global_step": self._global_step,
            "loss": loss,
            "best_loss": self._best_loss,
        }
        state_path = ckpt_dir / "trainer_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Checkpoint saved to %s (loss=%.4f)", ckpt_dir, loss)
        return str(ckpt_dir)

    @staticmethod
    def _save_model(model: Any, output_dir: Path) -> None:
        """Save model weights to *output_dir*."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # HuggingFace model
        if hasattr(model, "save_pretrained"):
            try:
                model.save_pretrained(str(output_dir))
                return
            except Exception as exc:
                logger.warning("save_pretrained failed: %s — falling back to torch.save", exc)

        # Standard PyTorch fallback
        import torch

        model_path = output_dir / "model.pt"
        torch.save(model.state_dict(), model_path)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(model: Any) -> str:
        """Determine the device a model lives on."""
        try:
            import torch
            return str(next(model.parameters()).device)
        except Exception:
            return "cpu"
