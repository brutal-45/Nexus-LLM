"""
RMSNorm - Root Mean Square Layer Normalization
================================================
Used in LLaMA, Mistral, and most modern LLMs instead of standard LayerNorm.

RMSNorm(x) = x * w / sqrt(mean(x^2) + eps)

Advantages over LayerNorm:
    - No mean centering (simpler computation)
    - Single learnable scale parameter (no bias needed)
    - ~7-10% faster due to fewer operations
    - Equally effective in practice for transformer models

Reference:
    - Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
    - https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    
    Normalizes input by the root mean square of each vector element,
    then applies a learnable scale (weight). No mean subtraction or bias.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Args:
            hidden_size: Dimension of the input vectors (d_model).
            eps: Small constant for numerical stability.
            device: Device for the weight parameter.
            dtype: Data type for the weight parameter.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute RMS normalization.
        
        rms(x) = sqrt(mean(x^2))
        normalized = x / rms(x)
        """
        # Compute mean of squared elements along last dimension
        # Use float32 for numerical stability
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return x_normed.type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor of shape (..., hidden_size).
        
        Returns:
            Normalized tensor of same shape.
        """
        return self._norm(x) * self.weight

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
