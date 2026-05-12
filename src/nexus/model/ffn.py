"""
SwiGLU Feed-Forward Network (FFN)
====================================
Implements the SwiGLU activation function used in LLaMA, Mistral, and modern LLMs.

Standard FFN:  output = W2 * activation(W1 * x + b1) + b2
SwiGLU FFN:   output = W_down * (SiLU(W_gate * x) * W_up * x)

SwiGLU uses three projections (gate, up, down) instead of two, and applies
element-wise SiLU (Sigmoid Linear Unit) gating:
    SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))

The gating mechanism allows the network to dynamically control information flow,
leading to better performance than ReLU/GELU at similar parameter counts.

For a 100B model with hidden_size=12288 and intermediate_size=49152:
    - W_gate: 12288 -> 49152  (600M params)
    - W_up:   12288 -> 49152  (600M params)
    - W_down: 49152 -> 12288  (600M params)
    - Total FFN params per layer: ~1.8B
    - 80 layers: ~144B FFN params (majority of model)

Reference:
    - Shazeer, "GLU Variants Improve Transformer" (2020)
    - https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .config import ModelConfig


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network with gating mechanism.
    
    Architecture:
        input -> gate_proj -> SiLU -> * up_proj -> down_proj -> output
    
    The gate and up projections transform the input in parallel, the gate
    output modulates the up output via element-wise multiplication (gating),
    and the down projection projects back to the model dimension.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Three projection layers (no bias for training stability)
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

        # Dropout for regularization during training
        self.dropout = nn.Dropout(
            config.hidden_dropout if hasattr(config, 'hidden_dropout') else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Gate path: apply SiLU activation
        # SiLU(x) = x * sigmoid(x)
        gate = F.silu(self.gate_proj(x))

        # Up path: linear transformation (no activation)
        up = self.up_proj(x)

        # Gating: element-wise multiplication
        hidden = gate * up

        # Down projection back to model dimension
        output = self.down_proj(hidden)

        return self.dropout(output)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )
