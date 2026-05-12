"""
Supervised Fine-Tuning (SFT)
==============================
Fine-tune the base model on instruction-response pairs to teach it
to follow instructions and engage in dialogue.

SFT Data Format:
    {"instruction": "...", "input": "...", "output": "..."}
    
    These get formatted as:
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}<eos>"
    
    Or in chat format:
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

SFT is the first step in the alignment pipeline:
    Base Model -> SFT -> DPO/RLHF -> Final Model

Key considerations:
    - Learning rate: 1e-5 to 5e-5 (much smaller than pretraining)
    - Epochs: 1-3 (more causes overfitting)
    - Batch size: 64-256 (smaller than pretraining)
    - Sequence length: Can be shorter than pretraining (2048-4096)
    - Packing: Pack multiple examples into fixed-length sequences for efficiency
"""

from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..model.transformer import NexusTransformer, TransformerOutput
from ..model.config import ModelConfig
from ..data.tokenizer import BPETokenizer
from ..training.trainer import Trainer, TrainingArguments


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""
    output_dir: str = "output/sft"
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 64
    seq_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 250
    format_template: str = "alpaca"  # "alpaca", "chat", "sharegpt"


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    FORMATS = {
        "alpaca": (
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n{output}"
        ),
        "chat": None,  # Uses messages list directly
        "simple": "{instruction}\n\n{output}",
    }

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_length: int = 4096,
        format_template: str = "alpaca",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_template = format_template
        
        # Load data
        self.examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _format_example(self, example: Dict) -> str:
        """Format an example into a prompt-response string."""
        template = self.FORMATS.get(self.format_template, self.FORMATS["alpaca"])
        
        if self.format_template == "chat" or "messages" in example:
            # Chat format
            messages = example.get("messages", [])
            parts = []
            for msg in messages:
                role = msg.get("role", "").capitalize()
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n\n".join(parts)
        
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if template:
            text = template.format(
                instruction=instruction,
                input=input_text if input_text else "N/A",
                output=output,
            )
        else:
            text = f"{instruction}\n{output}"
        
        return text

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        text = self._format_example(example)
        
        # Tokenize
        token_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        # Truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length - 1] + [self.tokenizer.special_tokens["<eos>"]]
        
        # Pad to max_length
        pad_len = self.max_length - len(token_ids)
        attention_mask = [1] * len(token_ids) + [0] * pad_len
        token_ids = token_ids + [self.tokenizer.special_tokens["<pad>"]] * pad_len
        
        # Labels = token_ids for SFT (teacher forcing)
        labels = token_ids.copy()
        # Mask padding tokens in labels (don't compute loss on padding)
        labels = [-100 if token == self.tokenizer.special_tokens["<pad>"] else token for token in labels]
        
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer.
    
    Fine-tunes the base model on instruction-following data.
    Uses the standard Trainer with SFT-specific defaults.
    """

    def __init__(
        self,
        model: NexusTransformer,
        tokenizer: BPETokenizer,
        train_data: str,
        eval_data: Optional[str] = None,
        config: Optional[SFTConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SFTConfig()
        
        # Create datasets
        self.train_dataset = SFTDataset(
            train_data, tokenizer,
            max_length=self.config.seq_length,
            format_template=self.config.format_template,
        )
        
        self.eval_dataset = None
        if eval_data:
            self.eval_dataset = SFTDataset(
                eval_data, tokenizer,
                max_length=self.config.seq_length,
                format_template=self.config.format_template,
            )

    def train(self) -> NexusTransformer:
        """
        Run SFT training.
        
        Returns:
            Fine-tuned model.
        """
        # Create training arguments
        total_steps = (len(self.train_dataset) * self.config.num_epochs) // (
            self.config.batch_size * self.config.gradient_accumulation_steps
        )
        
        args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=int(total_steps * self.config.warmup_ratio),
            max_steps=total_steps,
            micro_batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            gradient_checkpointing=self.config.gradient_checkpointing,
            precision="bf16_mixed" if self.config.fp16 else "fp32",
            log_interval=self.config.logging_steps,
            save_interval=self.config.save_steps,
            eval_interval=self.config.eval_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            wandb_project="nexus-sft",
        )
        
        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=args,
            model_config=self.model.config,
        )
        
        return trainer.train()
