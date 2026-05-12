"""
Token & Position Embeddings
============================
Handles input token embedding lookup and optional position embedding.

For the decoder-only transformer with RoPE:
    - Token embedding: standard learned embedding matrix (vocab_size x hidden_size)
    - Position information: handled by RoPE in attention (no separate position embedding)

The embedding layer applies scaled initialization following the LLaMA approach:
    weight ~ Normal(0, hidden_size^(-0.5))
This prevents the embeddings from having values that are too large or too small
relative to the rest of the network.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from ..model.config import ModelConfig


class Embedding(nn.Module):
    """
    Token embedding layer with optional scaling.
    
    For RoPE-based models, positional information is added inside the attention
    mechanism, so this module only handles token embeddings.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings

        # Main embedding matrix
        self.weight = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_size)
        )
        # Initialize with small variance
        nn.init.normal_(self.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Look up token embeddings.
        
        Args:
            input_ids: Integer tensor of shape (batch_size, seq_len)
                       with token indices in [0, vocab_size).
        
        Returns:
            Embedded tensor of shape (batch_size, seq_len, hidden_size).
        """
        return F.embedding(input_ids, self.weight)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"


# Need to import F after torch
import torch.nn.functional as F
