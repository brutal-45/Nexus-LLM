"""
Rotary Position Embeddings (RoPE)
===================================
Implements the standard RoPE encoding used in LLaMA, PaLM, and most modern LLMs.

RoPE encodes position information by rotating query and key vectors in a
2D plane. For each dimension pair (2i, 2i+1) in the head embedding:
    q'[2i]   = q[2i] * cos(m*theta_i) - q[2i+1] * sin(m*theta_i)
    q'[2i+1] = q[2i] * sin(m*theta_i) + q[2i+1] * cos(m*theta_i)

where m is the position index and theta_i = rope_theta^(-2i/d).

References:
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
    - https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    Precomputes frequency tensors for efficient application during attention.
    Supports position interpolation (YaRN/NTK-aware) for extended context.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            dim: Dimension of each attention head (head_dim).
            max_position_embeddings: Maximum sequence length supported.
            base: Base for computing inverse frequencies (theta).
            scaling_factor: Scaling factor for position interpolation.
                           >1.0 enables longer context via NTK-aware scaling.
            device: Device to precompute tensors on.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.device = device

        # Compute inverse frequencies: theta_i = base^(-2i/dim)
        # Shape: (dim // 2,)
        inv_freq = self._compute_inv_freq(dim, base, scaling_factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin up to max_position_embeddings
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _compute_inv_freq(
        self, dim: int, base: float, scaling_factor: float
    ) -> torch.Tensor:
        """
        Compute the inverse frequency tensor.
        
        With scaling_factor > 1.0 (NTK-aware scaling), we effectively reduce
        the base to allow the model to handle longer sequences:
            base_scaled = base * scaling_factor^(dim / (dim - 2))
        """
        if scaling_factor != 1.0:
            # NTK-aware scaling (from CodeLlama / YaRN)
            base_scaled = base * (
                scaling_factor ** (dim / (dim - 2))
            )
        else:
            base_scaled = base

        inv_freq = 1.0 / (
            base_scaled ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        return inv_freq

    def _set_cos_sin_cache(
        self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype
    ):
        """Precompute cosine and sine tensors for all positions up to seq_len."""
        self.max_seq_len_cached = seq_len
        
        # positions: (seq_len,)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # freqs: (seq_len, dim//2) = outer product of positions * inv_freq
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate for full dimension: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer(
            "cos_cached", emb.cos().to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(dtype), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin tensors for the given input sequence.
        
        Args:
            x: Input tensor of shape (batch, num_heads, seq_len, head_dim).
               Used to determine device and dtype.
            seq_len: Override sequence length (for KV caching).
            position_ids: Optional explicit position ids of shape (batch, seq_len).
                          If None, uses 0..seq_len-1.
        
        Returns:
            Tuple of (cos, sin) tensors each of shape (1, 1, seq_len, head_dim)
            broadcastable for element-wise rotation.
        """
        seq_len = seq_len if seq_len is not None else x.shape[2]

        if position_ids is not None:
            # Custom position ids (e.g., for packed sequences)
            freqs = torch.outer(
                position_ids.float().view(-1), self.inv_freq
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(x.dtype)
            sin = emb.sin().to(x.dtype)
        else:
            if seq_len > self.max_seq_len_cached:
                # Dynamically extend cache if needed
                self._set_cos_sin_cache(seq_len, x.device, x.dtype)
            cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
            sin = self.sin_cached[:seq_len].to(dtype=x.dtype)

        # Expand for broadcasting: (1, 1, seq_len, dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the second half of the vector dims to the front.
    
    For vector [x0, x1, x2, x3, ..., x_{d-1}] -> [x_{d/2}, ..., x_{d-1}, x0, ..., x_{d/2-1}]
    
    This enables the rotation operation:
        x' = x * cos + rotate_half(x) * sin
    which is equivalent to 2D rotation per dimension pair.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.
    
    This is the core RoPE operation:
        q' = q * cos(θ) + rotate_half(q) * sin(θ)
        k' = k * cos(θ) + rotate_half(k) * sin(θ)
    
    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        cos: Cosine tensor from RotaryEmbedding.
        sin: Sine tensor from RotaryEmbedding.
        position_ids: Optional position ids for indexing into cos/sin.
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.
    
    Returns:
        Tuple of rotated (q, k) tensors.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
