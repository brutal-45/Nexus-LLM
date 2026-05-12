"""
Direct Preference Optimization (DPO)
======================================
An alternative to RLHF that directly optimizes the language model
from preference data without training a separate reward model.

DPO refines the policy by treating preference pairs as a classification
task: given a prompt, the model should assign higher likelihood to the
chosen response than the rejected response.

Loss function:
    L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                              - log π_θ(y_l|x) + log π_ref(y_l|x)))]

where:
    y_w = chosen (preferred) response
    y_l = rejected response
    π_θ = current policy (model)
    π_ref = reference policy (frozen model, usually SFT checkpoint)
    β = temperature hyperparameter (typically 0.1-0.5)

DPO Data Format:
    {"prompt": "...", "chosen": "...", "rejected": "..."}

Reference:
    - Rafailov et al., "Direct Preference Optimization: Your Language Model
      is Secretly a Reward Model" (2023)
"""

from __future__ import annotations
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..model.transformer import NexusTransformer
from ..model.config import ModelConfig
from ..data.tokenizer import BPETokenizer
from ..training.trainer import Trainer, TrainingArguments


@dataclass
class DPOConfig:
    """DPO training configuration."""
    output_dir: str = "output/dpo"
    learning_rate: float = 5e-7
    num_epochs: int = 1
    batch_size: int = 64
    seq_length: int = 4096
    beta: float = 0.1  # DPO temperature
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    reference_model_path: Optional[str] = None  # Path to SFT model


class DPODataset(Dataset):
    """Dataset for DPO training with preference pairs."""

    def __init__(
        self,
        data_path: str,
        tokenizer: BPETokenizer,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.pairs = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.pairs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        # Build full sequences: prompt + response
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"
        
        # Tokenize both
        chosen_ids = self.tokenizer.encode(chosen_text, add_bos=True, add_eos=True)
        rejected_ids = self.tokenizer.encode(rejected_text, add_bos=True, add_eos=True)
        
        # Truncate
        max_resp_len = (self.max_length - len(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))) // 2
        if len(chosen_ids) > self.max_length:
            chosen_ids = chosen_ids[:self.max_length - 1] + [self.tokenizer.special_tokens["<eos>"]]
        if len(rejected_ids) > self.max_length:
            rejected_ids = rejected_ids[:self.max_length - 1] + [self.tokenizer.special_tokens["<eos>"]]
        
        # Pad to same length (need equal length for batch processing)
        max_len = max(len(chosen_ids), len(rejected_ids))
        pad_id = self.tokenizer.special_tokens["<pad>"]
        
        def pad(ids, length):
            return ids + [pad_id] * (length - len(ids))
        
        chosen_ids = pad(chosen_ids, max_len)
        rejected_ids = pad(rejected_ids, max_len)
        
        # Masks
        chosen_mask = [1] * len(chosen_ids)
        rejected_mask = [1] * len(rejected_ids)
        
        # Labels for loss computation (only compute on response part)
        prompt_len = len(self.tokenizer.encode(prompt, add_bos=True, add_eos=False))
        chosen_labels = [-100] * prompt_len + chosen_ids[prompt_len:]
        rejected_labels = [-100] * prompt_len + rejected_ids[prompt_len:]
        
        # Fix padding in labels
        chosen_labels = [l if l != pad_id else -100 for l in chosen_labels]
        rejected_labels = [l if l != pad_id else -100 for l in rejected_labels]
        
        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_mask, dtype=torch.long),
            "chosen_labels": torch.tensor(
                pad(chosen_labels, max_len), dtype=torch.long
            ),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_mask, dtype=torch.long),
            "rejected_labels": torch.tensor(
                pad(rejected_labels, max_len), dtype=torch.long
            ),
        }


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Trains the model to prefer chosen responses over rejected responses
    using the DPO objective (no reward model needed).
    """

    def __init__(
        self,
        model: NexusTransformer,
        reference_model: NexusTransformer,
        tokenizer: BPETokenizer,
        train_data: str,
        eval_data: Optional[str] = None,
        config: Optional[DPOConfig] = None,
    ):
        self.model = model
        self.reference_model = reference_model.eval()  # Frozen reference
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()
        self.beta = self.config.beta
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # Create datasets
        self.train_dataset = DPODataset(
            train_data, tokenizer, max_length=self.config.seq_length
        )
        self.eval_dataset = None
        if eval_data:
            self.eval_dataset = DPODataset(
                eval_data, tokenizer, max_length=self.config.seq_length
            )

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        The loss encourages the model to assign higher probability to
        chosen responses relative to rejected responses, compared to
        what the reference model assigns.
        
        L = -E[log σ(β * (logps_chosen - logps_rejected - ref_logps_chosen + ref_logps_rejected))]
        """
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        logits = self.beta * (chosen_logratios - rejected_logratios)
        
        # Binary cross-entropy with sigmoid
        loss = -F.logsigmoid(logits).mean()
        
        # Additional metrics
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).mean()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).mean()
        reward_margin = chosen_rewards - rejected_rewards
        
        return loss, {
            "chosen_rewards": chosen_rewards.item(),
            "rejected_rewards": rejected_rewards.item(),
            "reward_margin": reward_margin.item(),
        }

    def _compute_log_probs(
        self,
        model: NexusTransformer,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities of the labels under the model."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)
        
        # Gather log probs for the actual labels
        # Shift labels by 1 (predict next token)
        shifted_labels = labels[..., 1:].contiguous()
        shifted_log_probs = log_probs[..., :-1, :].contiguous()
        
        # Gather the log prob for each label token
        per_token_logps = torch.gather(
            shifted_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding and prompt tokens
        mask = (shifted_labels != -100).float()
        
        # Sum log probs per sequence (only for response tokens)
        sequence_logps = (per_token_logps * mask).sum(dim=-1)
        token_count = mask.sum(dim=-1).clamp(min=1)
        
        return sequence_logps / token_count

    def train(self) -> NexusTransformer:
        """Run DPO training."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.reference_model = self.reference_model.to(device)
        self.model.train()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        
        # Data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        
        # Training loop
        total_steps = len(train_loader) * self.config.num_epochs
        step = 0
        print(f"[DPO] Starting training for {self.config.num_epochs} epochs ({total_steps} steps)")
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Compute log probs under policy (current model)
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    policy_chosen_logps = self._compute_log_probs(
                        self.model,
                        batch["chosen_input_ids"],
                        batch["chosen_attention_mask"],
                        batch["chosen_labels"],
                    )
                    policy_rejected_logps = self._compute_log_probs(
                        self.model,
                        batch["rejected_input_ids"],
                        batch["rejected_attention_mask"],
                        batch["rejected_labels"],
                    )
                
                # Compute log probs under reference (frozen model)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.config.fp16):
                        ref_chosen_logps = self._compute_log_probs(
                            self.reference_model,
                            batch["chosen_input_ids"],
                            batch["chosen_attention_mask"],
                            batch["chosen_labels"],
                        )
                        ref_rejected_logps = self._compute_log_probs(
                            self.reference_model,
                            batch["rejected_input_ids"],
                            batch["rejected_attention_mask"],
                            batch["rejected_labels"],
                        )
                
                # Compute DPO loss
                loss, metrics = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                )
                
                # Backward
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                epoch_loss += loss.item()
                step += 1
                
                if step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / step
                    print(f"  Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                          f"Reward margin: {metrics['reward_margin']:.4f} | "
                          f"Chosen reward: {metrics['chosen_rewards']:.4f}")
            
            print(f"  Epoch {epoch + 1}/{self.config.num_epochs} | Avg Loss: {epoch_loss / step:.4f}")
        
        print("[DPO] Training complete!")
        return self.model
