"""Main training loop for Nexus-LLM.

Wraps HuggingFace Trainer with project-specific configuration,
ModelManager integration, callback support, and checkpoint resumption.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from nexus_llm.core.config import Settings, TrainingSettings
from nexus_llm.core.exceptions import TrainingError
from nexus_llm.training.callbacks import TrainingCallbacks
from nexus_llm.training.dataset import DatasetLoader
from nexus_llm.training.fine_tune import FineTuner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_training_args(settings: TrainingSettings) -> Any:
    """Build a HuggingFace TrainingArguments object from our settings."""
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise TrainingError(
            "The 'transformers' package is required for training. "
            "Install it with: pip install transformers"
        ) from exc

    # Determine fp16 / bf16 based on precision and device
    import torch
    use_fp16 = torch.cuda.is_available()
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=settings.num_epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        warmup_steps=settings.warmup_steps,
        save_steps=settings.save_steps,
        eval_steps=settings.eval_steps if settings.eval_steps else None,
        logging_steps=max(1, settings.save_steps // 5),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=use_fp16 and not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",  # We handle reporting ourselves
    )

    return args


# ---------------------------------------------------------------------------
# NexusTrainer
# ---------------------------------------------------------------------------

class NexusTrainer:
    """High-level training orchestrator for Nexus-LLM.

    Ties together model management, LoRA configuration, dataset loading,
    training callbacks, and the HuggingFace Trainer.

    Usage::

        trainer = NexusTrainer(settings)
        result = trainer.train(
            model=model,
            tokenizer=tokenizer,
            data_path="data/train.jsonl",
        )
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or Settings()
        self._training_settings = self._settings.training
        self._fine_tuner = FineTuner(self._training_settings)
        self._callbacks = TrainingCallbacks(self._settings)
        self._hf_trainer: Optional[Any] = None
        self._train_dataset: Optional[Any] = None
        self._val_dataset: Optional[Any] = None

    @property
    def fine_tuner(self) -> FineTuner:
        """Access the underlying FineTuner instance."""
        return self._fine_tuner

    @property
    def hf_trainer(self) -> Optional[Any]:
        """Access the underlying HuggingFace Trainer (after train is called)."""
        return self._hf_trainer

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        model: Any,
        tokenizer: Any,
        data_path: str,
        data_format: str = "auto",
        val_split: float = 0.1,
        resume_from_checkpoint: Optional[str] = None,
        apply_lora: bool = True,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
        lora_dropout: Optional[float] = None,
        hf_split: str = "train",
        hf_subset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full fine-tuning pipeline.

        Args:
            model: A HuggingFace PreTrainedModel.
            tokenizer: The matching tokenizer.
            data_path: Path or HuggingFace dataset ID.
            data_format: "alpaca", "chat", "instruction", or "auto".
            val_split: Fraction for validation split.
            resume_from_checkpoint: Path to a checkpoint to resume from.
            apply_lora: Whether to apply LoRA before training.
            lora_r: Override LoRA rank.
            lora_alpha: Override LoRA alpha.
            lora_dropout: Override LoRA dropout.
            hf_split: HuggingFace dataset split name.
            hf_subset: HuggingFace dataset subset.

        Returns:
            Dict with training metrics and output directory.
        """
        try:
            from transformers import Trainer
        except ImportError as exc:
            raise TrainingError(
                "The 'transformers' package is required for training. "
                "Install it with: pip install transformers"
            ) from exc

        # --- Step 1: Apply LoRA if requested ---
        if apply_lora:
            logger.info("Applying LoRA adapters to model")
            self._fine_tuner.create_lora_config(
                r=lora_r, alpha=lora_alpha, dropout=lora_dropout, model=model,
            )
            model = self._fine_tuner.apply_lora(model)
            model = self._fine_tuner.prepare_for_training(model)

        # --- Step 2: Load and tokenise dataset ---
        logger.info("Loading training data from %s", data_path)
        loader = DatasetLoader(tokenizer=tokenizer)
        train_ds, val_ds = loader.load(
            source=data_path,
            format=data_format,
            val_split=val_split,
            max_seq_length=self._training_settings.max_seq_length,
            hf_split=hf_split,
            hf_subset=hf_subset,
        )
        self._train_dataset = train_ds
        self._val_dataset = val_ds

        # --- Step 3: Build training arguments ---
        training_args = _build_training_args(self._training_settings)

        # Enable evaluation if we have a validation set
        if val_ds:
            training_args.evaluation_strategy = "steps"
            training_args.eval_steps = self._training_settings.eval_steps or 500
        else:
            training_args.evaluation_strategy = "no"

        # --- Step 4: Set up callbacks ---
        callbacks = self._callbacks.build_callbacks()

        # --- Step 5: Create HuggingFace Trainer ---
        self._hf_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds or None,
            callbacks=callbacks,
        )

        # --- Step 6: Train ---
        logger.info("Starting training …")
        try:
            result = self._hf_trainer.train(
                resume_from_checkpoint=resume_from_checkpoint,
            )
        except Exception as exc:
            logger.exception("Training failed")
            raise TrainingError(f"Training failed: {exc}") from exc

        # --- Step 7: Save final model ---
        output_dir = Path(self._training_settings.output_dir)
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        if apply_lora and self._fine_tuner.is_applied:
            self._fine_tuner.save_adapter(model, str(final_dir))
        else:
            self._hf_trainer.save_model(str(final_dir))

        # Save tokenizer alongside the model
        try:
            tokenizer.save_pretrained(str(final_dir))
        except Exception:
            logger.warning("Could not save tokenizer to %s", final_dir)

        metrics = {
            "training_loss": getattr(result, "training_loss", None),
            "global_step": getattr(result, "global_step", None),
            "output_dir": str(final_dir),
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds) if val_ds else 0,
        }

        logger.info("Training complete — metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def resume(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Resume training from a checkpoint.

        Args:
            checkpoint_dir: Path to the HuggingFace checkpoint directory.

        Returns:
            Updated training metrics dict.
        """
        if self._hf_trainer is None:
            raise TrainingError(
                "No trainer instance available. Call train() first, or use "
                "the resume_from_checkpoint parameter of train()."
            )

        logger.info("Resuming training from %s", checkpoint_dir)
        try:
            result = self._hf_trainer.train(resume_from_checkpoint=checkpoint_dir)
        except Exception as exc:
            raise TrainingError(f"Failed to resume training: {exc}") from exc

        return {
            "training_loss": getattr(result, "training_loss", None),
            "global_step": getattr(result, "global_step", None),
        }

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the validation dataset.

        Returns:
            Dict of evaluation metrics.
        """
        if self._hf_trainer is None:
            raise TrainingError("No trainer instance available. Call train() first.")
        if not self._val_dataset:
            raise TrainingError("No validation dataset available.")

        logger.info("Running evaluation …")
        metrics = self._hf_trainer.evaluate()
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def merge_and_save(self, output_dir: str) -> str:
        """Merge LoRA weights and save the final model.

        Args:
            output_dir: Directory to save the merged model.

        Returns:
            Path to the saved merged model.
        """
        if self._hf_trainer is None:
            raise TrainingError("No trainer instance available. Call train() first.")

        model = self._hf_trainer.model
        model = self._fine_tuner.merge_and_unload(model)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path))

        logger.info("Merged model saved to %s", out_path)
        return str(out_path)
