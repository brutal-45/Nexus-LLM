"""Fine-Tuner - Parameter-efficient fine-tuning with LoRA and other methods."""

import logging
from typing import Optional, Dict, Any, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
)

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Handles parameter-efficient fine-tuning (PEFT) using LoRA, QLoRA,
    and other methods. Allows fine-tuning large models on limited hardware.
    """

    def __init__(
        self,
        base_model: str = "gpt2-medium",
        output_dir: str = "./models/fine-tuned",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ):
        self.base_model = base_model
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        self._model = None
        self._tokenizer = None
        self._peft_model = None

    def setup(self) -> Dict[str, Any]:
        """
        Set up the model for fine-tuning with LoRA.

        Returns:
            Information about the PEFT configuration
        """
        logger.info(f"Setting up fine-tuning for: {self.base_model}")

        # Load base model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            padding_side="left",
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(self.base_model)

        if self.use_lora:
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
                bias="none",
            )

            # Apply LoRA
            self._peft_model = get_peft_model(self._model, lora_config)

            # Print trainable parameters
            trainable_params = sum(
                p.numel() for p in self._peft_model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self._peft_model.parameters())

            info = {
                "method": "LoRA",
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_percentage": round(100 * trainable_params / total_params, 2),
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
            }

            logger.info(
                f"LoRA applied: {info['trainable_percentage']}% trainable parameters"
            )
            return info
        else:
            # Full fine-tuning
            info = {
                "method": "Full Fine-tuning",
                "trainable_params": sum(p.numel() for p in self._model.parameters()),
                "total_params": sum(p.numel() for p in self._model.parameters()),
                "trainable_percentage": 100.0,
            }
            return info

    def fine_tune(
        self,
        train_dataset,
        eval_dataset=None,
    ) -> Dict[str, Any]:
        """
        Run the fine-tuning process.

        Args:
            train_dataset: Tokenized training dataset
            eval_dataset: Optional tokenized evaluation dataset

        Returns:
            Fine-tuning results
        """
        if self._peft_model is None and self._model is None:
            self.setup()

        model = self._peft_model if self._peft_model else self._model

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            save_strategy="steps",
            save_steps=500,
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            load_best_model_at_end=True if eval_dataset else False,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting fine-tuning...")
        try:
            result = trainer.train()

            # Save the fine-tuned model
            if self.use_lora and self._peft_model:
                self._peft_model.save_pretrained(self.output_dir)
            else:
                trainer.save_model()

            self._tokenizer.save_pretrained(self.output_dir)

            return {
                "status": "success",
                "metrics": result.metrics,
                "model_path": self.output_dir,
                "method": "LoRA" if self.use_lora else "Full Fine-tuning",
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def merge_and_save(self, output_path: Optional[str] = None) -> str:
        """
        Merge LoRA weights with the base model and save the merged model.
        This creates a standalone model that doesn't need PEFT at inference time.

        Args:
            output_path: Path to save the merged model

        Returns:
            Path to the saved merged model
        """
        if not self.use_lora or self._peft_model is None:
            raise ValueError("Merge is only available for LoRA models.")

        save_path = output_path or (self.output_dir + "_merged")

        # Merge weights
        merged_model = self._peft_model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)

        logger.info(f"Merged model saved to: {save_path}")
        return save_path

    @staticmethod
    def load_finetuned_model(
        model_path: str,
        device: str = "auto",
    ) -> Dict[str, Any]:
        """
        Load a fine-tuned model (with or without LoRA).

        Args:
            model_path: Path to the fine-tuned model
            device: Device to load the model on

        Returns:
            Dict with model and tokenizer
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        try:
            # Try loading as a PEFT model first
            config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            logger.info("Loaded as PEFT/LoRA model")
        except Exception:
            # Load as a regular model
            model = AutoModelForCausalLM.from_pretrained(model_path)
            logger.info("Loaded as regular model")

        if device != "auto":
            model = model.to(device)

        return {"model": model, "tokenizer": tokenizer}
