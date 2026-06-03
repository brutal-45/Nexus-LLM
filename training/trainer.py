"""LLM Trainer - Core training loop and training management."""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

logger = logging.getLogger(__name__)


class LLMTrainer:
    """
    Core training manager that handles the training loop,
    checkpointing, logging, and training configuration.
    """

    def __init__(
        self,
        model_name: str = "gpt2-medium",
        output_dir: str = "./models/fine-tuned",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        logging_steps: int = 10,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.device = device

        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._training_history = []

    def setup(self) -> None:
        """Set up the model and tokenizer for training."""
        logger.info(f"Setting up training for model: {self.model_name}")

        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Training setup complete.")

    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the training loop.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional tokenized evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
            callback: Optional callback function for training progress

        Returns:
            Training results and metrics
        """
        if self._model is None or self._tokenizer is None:
            self.setup()

        # Resolve device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            save_strategy="steps",
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.save_steps if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss" if eval_dataset else None,
            greater_is_better=False,
            fp16=(device == "cuda"),
            gradient_accumulation_steps=4,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb/tensorboard by default
        )

        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Create trainer
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        start_time = time.time()

        try:
            result = self._trainer.train(
                resume_from_checkpoint=resume_from_checkpoint
            )

            training_time = time.time() - start_time

            # Save the final model
            self._trainer.save_model()
            self._tokenizer.save_pretrained(self.output_dir)

            metrics = result.metrics
            metrics["training_time"] = round(training_time, 2)

            logger.info(f"Training complete. Metrics: {metrics}")
            self._training_history.append(metrics)

            return {
                "status": "success",
                "metrics": metrics,
                "model_path": self.output_dir,
                "training_time": training_time,
            }

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def evaluate(self, eval_dataset=None) -> Dict[str, Any]:
        """Evaluate the model on a dataset."""
        if self._trainer is None:
            raise ValueError("No trainer available. Run train() first.")

        results = self._trainer.evaluate(eval_dataset=eval_dataset)
        logger.info(f"Evaluation results: {results}")
        return results

    def push_to_hub(self, repo_name: str, token: Optional[str] = None) -> None:
        """Push the trained model to HuggingFace Hub."""
        if self._trainer is None:
            raise ValueError("No trainer available. Run train() first.")

        self._trainer.push_to_hub(repo_name=repo_name, token=token)
        logger.info(f"Model pushed to HuggingFace Hub: {repo_name}")

    @property
    def training_history(self) -> list:
        """Get training history."""
        return self._training_history.copy()
