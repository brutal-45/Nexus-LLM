"""
Reward Model for RLHF
=======================
A reward model that scores the quality of model responses.
Used in the RLHF pipeline: Base -> SFT -> Train RM -> RLHF (PPO)

The reward model is a modified version of the base LLM that outputs
a scalar reward instead of token probabilities:
    Input:  [prompt] [response]
    Output: scalar reward score

Architecture:
    - Take the base transformer model
    - Replace the LM head with a reward head (single linear projection)
    - The final hidden state of the last token is used as the representation
    - Apply a linear layer to produce a scalar reward

Training:
    - Input: preference pairs (prompt, chosen_response, rejected_response)
    - Loss: Bradley-Terry ranking loss
        L = -E[log σ(r(x, y_w) - r(x, y_l))]
    where r(x, y) is the reward for prompt x and response y

Reference:
    - Stiennon et al., "Learning to Summarize with Human Feedback" (2020)
    - Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (2022)
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


@dataclass
class RewardModelConfig:
    """Reward model configuration."""
    output_dir: str = "output/reward_model"
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 64
    seq_length: int = 4096
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    fp16: bool = True


class RewardModel(nn.Module):
    """
    Reward Model built on top of the Nexus transformer.
    
    Replaces the language modeling head with a reward scoring head.
    Takes (prompt, response) pairs and outputs a scalar reward.
    """

    def __init__(self, base_model: NexusTransformer, freeze_base: bool = True):
        super().__init__()
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        
        # Use the transformer backbone (without lm_head)
        self.transformer = base_model
        self.transformer.lm_head = nn.Identity()  # Remove LM head
        
        # Reward head: project last hidden state to scalar
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        
        # Initialize reward head
        nn.init.normal_(self.reward_head[0].weight, std=0.02)
        nn.init.zeros_(self.reward_head[0].bias)
        nn.init.normal_(self.reward_head[2].weight, std=0.02)
        nn.init.zeros_(self.reward_head[2].bias)
        
        # Optionally freeze the base model
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward score for input sequence.
        
        Args:
            input_ids: Token IDs (batch, seq_len).
            attention_mask: Optional attention mask.
        
        Returns:
            Reward scores (batch, 1).
        """
        # Get hidden states from transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        
        # Get last non-padding token's hidden state
        if attention_mask is not None:
            # Find the last real token for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
            batch_size = input_ids.shape[0]
            last_hidden = outputs.hidden_states[-1][
                torch.arange(batch_size, device=input_ids.device),
                seq_lengths,
            ] if outputs.hidden_states else outputs.logits[
                torch.arange(batch_size, device=input_ids.device),
                seq_lengths,
            ]
        else:
            last_hidden = outputs.logits[:, -1, :] if not outputs.hidden_states else outputs.hidden_states[-1][:, -1, :]
        
        # Compute reward
        reward = self.reward_head(last_hidden)  # (batch, 1)
        return reward

    def get_reward(self, prompt: str, response: str, tokenizer) -> float:
        """Get scalar reward for a (prompt, response) pair."""
        text = f"{prompt}\n\n{response}"
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        input_ids = torch.tensor([token_ids], device=next(self.parameters()).device)
        
        with torch.no_grad():
            reward = self.forward(input_ids)
        
        return reward.item()


class PreferenceDataset(Dataset):
    """Dataset of preference pairs for reward model training."""

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
                try:
                    self.pairs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        # Build sequences
        chosen_text = f"{prompt}\n\n{chosen}"
        rejected_text = f"{prompt}\n\n{rejected}"
        
        chosen_ids = self.tokenizer.encode(chosen_text, add_bos=True, add_eos=True)[:self.max_length]
        rejected_ids = self.tokenizer.encode(rejected_text, add_bos=True, add_eos=True)[:self.max_length]
        
        # Pad
        max_len = max(len(chosen_ids), len(rejected_ids))
        pad_id = self.tokenizer.special_tokens["<pad>"]
        
        def pad(ids):
            return ids + [pad_id] * (max_len - len(ids))
        
        return {
            "chosen_ids": torch.tensor(pad(chosen_ids), dtype=torch.long),
            "chosen_mask": torch.tensor(
                [1] * len(chosen_ids) + [0] * (max_len - len(chosen_ids)),
                dtype=torch.long,
            ),
            "rejected_ids": torch.tensor(pad(rejected_ids), dtype=torch.long),
            "rejected_mask": torch.tensor(
                [1] * len(rejected_ids) + [0] * (max_len - len(rejected_ids)),
                dtype=torch.long,
            ),
        }


def train_reward_model(
    base_model: NexusTransformer,
    tokenizer: BPETokenizer,
    train_data: str,
    eval_data: Optional[str] = None,
    config: Optional[RewardModelConfig] = None,
) -> RewardModel:
    """
    Train a reward model from preference data.
    
    Uses Bradley-Terry ranking loss:
        L = -E[log σ(r(x, y_w) - r(x, y_l))]
    """
    config = config or RewardModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create reward model
    reward_model = RewardModel(base_model, freeze_base=True).to(device)
    
    # Dataset and loader
    dataset = PreferenceDataset(train_data, tokenizer, max_length=config.seq_length)
    train_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    
    # Optimizer (only reward head parameters)
    optimizer = torch.optim.AdamW(
        reward_model.reward_head.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    print(f"[RewardModel] Training for {config.num_epochs} epochs...")
    
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Compute rewards
            chosen_rewards = reward_model(
                batch["chosen_ids"], batch["chosen_mask"]
            )  # (batch, 1)
            rejected_rewards = reward_model(
                batch["rejected_ids"], batch["rejected_mask"]
            )
            
            # Bradley-Terry loss
            logits = chosen_rewards - rejected_rewards  # (batch, 1)
            loss = -F.logsigmoid(logits).mean()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                reward_model.parameters(), config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            correct += (logits.squeeze() > 0).sum().item()
            total += len(logits)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"  Epoch {epoch + 1}/{config.num_epochs} | "
              f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%}")
    
    print("[RewardModel] Training complete!")
    return reward_model
