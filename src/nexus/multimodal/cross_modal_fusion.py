"""
Nexus Cross-Modal Fusion Module
================================
Implements various fusion strategies for combining multiple modalities
(vision, audio, text, video) within the Nexus multimodal framework.

Supported Fusion Strategies:
    - Cross-Attention Fusion: Query from one modality attends to another
    - Co-Attention Fusion: Bidirectional attention between two modalities
    - Concatenation Fusion: Late fusion via concatenation and projection
    - Gated Fusion: Learnable gate controls modality mixing
    - Compact Bilinear Fusion: Tucker decomposition bilinear pooling
    - Adaptive Fusion: Dynamically selects fusion strategy per input
    - MultiModal Transformer: Unified transformer with modality-aware masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import math


__all__ = [
    "ModalityEmbedding",
    "CrossAttentionFusion",
    "CoAttentionFusion",
    "ConcatenationFusion",
    "GatedFusion",
    "CompactBilinearFusion",
    "AdaptiveFusion",
    "MultiModalTransformerLayer",
    "MultiModalTransformer",
    "FusionGate",
    "ModalityDropout",
    "FusionConfig",
    "ModalityType",
    "MultiHeadCrossAttention",
    "GatedMultiHeadAttention",
    "BilinearPooling",
    "FusionLayerNorm",
]


class ModalityType:
    """Constants for modality types."""
    VISION = 0
    AUDIO = 1
    TEXT = 2
    VIDEO = 3
    NUM_MODALITIES = 4

    NAMES = ["vision", "audio", "text", "video"]

    @classmethod
    def from_name(cls, name: str) -> int:
        name = name.lower().strip()
        for i, n in enumerate(cls.NAMES):
            if n == name:
                return i
        raise ValueError(f"Unknown modality: {name}. Choose from: {cls.NAMES}")

    @classmethod
    def to_name(cls, idx: int) -> str:
        if 0 <= idx < len(cls.NAMES):
            return cls.NAMES[idx]
        raise ValueError(f"Unknown modality index: {idx}")


@dataclass
class FusionConfig:
    """Configuration for fusion modules."""
    fusion_type: str = "cross_attention"  # cross_attention, co_attention, concat, gated, compact_bilinear, adaptive
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    ffn_dim: Optional[int] = None
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_residual: bool = True
    gate_activation: str = "sigmoid"
    bilinear_dim: int = 256
    adaptive_num_strategies: int = 4
    modality_dropout: float = 0.0

    def __post_init__(self):
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.hidden_dim


class ModalityEmbedding(nn.Module):
    """Learnable modality type embeddings.

    Maps each modality (vision, audio, text, video) to a learned embedding
    vector that is added to the input features. This allows the model to
    distinguish which modality each input comes from.

    Args:
        embed_dim: Dimension of the modality embeddings.
        num_modalities: Number of supported modalities.
        dropout: Dropout probability applied to embeddings.
    """

    def __init__(self, embed_dim: int = 1024, num_modalities: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.embedding = nn.Embedding(num_modalities, embed_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        input_features: Tensor,
        modality_indices: Tensor,
    ) -> Tensor:
        """Add modality embedding to input features.

        Args:
            input_features: Input tensor of shape [..., embed_dim].
            modality_indices: Modality type indices of shape [..., 1] or [...].
                Use ModalityType constants (VISION=0, AUDIO=1, TEXT=2, VIDEO=3).

        Returns:
            Tensor of same shape as input_features with modality embedding added.
        """
        if modality_indices.dim() != input_features.dim():
            if modality_indices.dim() == input_features.dim() - 1:
                modality_indices = modality_indices.unsqueeze(-1)
            elif modality_indices.dim() == 1 and input_features.dim() >= 2:
                shape = list(input_features.shape[:-1])
                modality_indices = modality_indices.expand(*shape).unsqueeze(-1)

        modality_embed = self.embedding(modality_indices)
        return self.dropout(input_features + modality_embed)

    def get_embedding(self, modality: Union[str, int]) -> Tensor:
        """Get the embedding for a specific modality.

        Args:
            modality: Either a string name ("vision", "audio", "text", "video")
                      or an integer index.

        Returns:
            Embedding tensor of shape (embed_dim,).
        """
        if isinstance(modality, str):
            idx = ModalityType.from_name(modality)
        else:
            idx = modality
        return self.embedding.weight[idx]


class FusionLayerNorm(nn.Module):
    """Layer normalization used in fusion modules.

    Supports both standard LayerNorm and RMSNorm (without mean subtraction).
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_rms: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.use_rms = use_rms
        if use_rms:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_rms:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return self.weight * (x / rms)
        return self.norm(x)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention mechanism for fusion.

    Query comes from modality A, Key and Value come from modality B.

    Args:
        query_dim: Dimension of query inputs.
        kv_dim: Dimension of key/value inputs.
        num_heads: Number of attention heads.
        head_dim: Dimension per head (computed if None).
        dropout: Attention dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        query_dim: int = 1024,
        kv_dim: int = 1024,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or query_dim // num_heads
        self.inner_dim = num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute cross-attention.

        Args:
            query: Query tensor of shape (batch, q_len, query_dim).
            key: Key tensor of shape (batch, kv_len, kv_dim).
            value: Value tensor of shape (batch, kv_len, kv_dim).
            attention_mask: Optional mask of shape (batch, q_len, kv_len).
                True/1 = attend, False/0 = ignore.

        Returns:
            Tuple of (output tensor, attention weights).
        """
        batch_size = query.size(0)
        q_len = query.size(1)
        kv_len = key.size(1)

        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, kv_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, kv_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(1), float("-inf")
                )
            elif attention_mask.dim() == 2:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, q_len, self.inner_dim)
        output = self.out_proj(output)

        return output, attn_weights


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between two modalities.

    Query comes from the primary modality, Key and Value come from the
    secondary modality. The output has the same shape as the primary input.

    Args:
        primary_dim: Dimension of primary (query) modality features.
        secondary_dim: Dimension of secondary (key/value) modality features.
        hidden_dim: Internal attention dimension.
        num_heads: Number of attention heads.
        num_layers: Number of cross-attention layers.
        dropout: Dropout probability.
        use_residual: Whether to use residual connections.
        use_layer_norm: Whether to use layer normalization.
    """

    def __init__(
        self,
        primary_dim: int = 1024,
        secondary_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or primary_dim
        self.primary_dim = primary_dim
        self.secondary_dim = secondary_dim
        self.num_layers = num_layers
        self.use_residual = use_residual

        self.input_proj = nn.Linear(primary_dim, hidden_dim) if primary_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, primary_dim) if primary_dim != hidden_dim else nn.Identity()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": MultiHeadCrossAttention(
                    query_dim=hidden_dim, kv_dim=secondary_dim,
                    num_heads=num_heads, dropout=dropout
                ),
                "norm1": FusionLayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                "norm2": FusionLayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, 4 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer["cross_attn"].q_proj.weight)
            nn.init.xavier_uniform_(layer["cross_attn"].k_proj.weight)
            nn.init.xavier_uniform_(layer["cross_attn"].v_proj.weight)
            nn.init.xavier_uniform_(layer["cross_attn"].out_proj.weight)

    def forward(
        self,
        primary: Tensor,
        secondary: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Fuse two modalities via cross-attention.

        Args:
            primary: Primary modality features (batch, seq_len, primary_dim).
            secondary: Secondary modality features (batch, seq_len, secondary_dim).
            attention_mask: Optional mask for secondary modality.

        Returns:
            Fused features of shape (batch, seq_len, primary_dim).
        """
        x = self.input_proj(primary)

        for layer in self.layers:
            residual = x
            x_norm = layer["norm1"](x)
            cross_out, _ = layer["cross_attn"](x_norm, secondary, secondary, attention_mask)
            if self.use_residual:
                x = residual + cross_out
            else:
                x = cross_out

            residual = x
            x_norm = layer["norm2"](x)
            ffn_out = layer["ffn"](x_norm)
            if self.use_residual:
                x = residual + ffn_out
            else:
                x = ffn_out

        return self.output_proj(x)


class CoAttentionFusion(nn.Module):
    """Bidirectional co-attention fusion between two modalities.

    Both modalities attend to each other simultaneously. The outputs for
    both modalities are enhanced with information from the other.

    Args:
        dim_a: Dimension of modality A features.
        dim_b: Dimension of modality B features.
        hidden_dim: Internal attention dimension.
        num_heads: Number of attention heads.
        num_layers: Number of co-attention layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_a: int = 1024,
        dim_b: int = 1024,
        hidden_dim: Optional[int] = None,
        num_heads: int = 8,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim_a
        self.dim_a = dim_a
        self.dim_b = dim_b
        self.hidden_dim = hidden_dim

        self.proj_a = nn.Linear(dim_a, hidden_dim) if dim_a != hidden_dim else nn.Identity()
        self.proj_b = nn.Linear(dim_b, hidden_dim) if dim_b != hidden_dim else nn.Identity()

        self.a_to_b_attn = nn.ModuleList([
            MultiHeadCrossAttention(hidden_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.b_to_a_attn = nn.ModuleList([
            MultiHeadCrossAttention(hidden_dim, hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_a = nn.ModuleList([
            FusionLayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norm_b = nn.ModuleList([
            FusionLayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.gate_a = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.gate_b = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.out_proj_a = nn.Linear(hidden_dim, dim_a) if dim_a != hidden_dim else nn.Identity()
        self.out_proj_b = nn.Linear(hidden_dim, dim_b) if dim_b != hidden_dim else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for attn in self.a_to_b_attn:
            nn.init.xavier_uniform_(attn.q_proj.weight)
            nn.init.xavier_uniform_(attn.k_proj.weight)
        for attn in self.b_to_a_attn:
            nn.init.xavier_uniform_(attn.q_proj.weight)
            nn.init.xavier_uniform_(attn.k_proj.weight)

    def forward(
        self,
        modality_a: Tensor,
        modality_b: Tensor,
        mask_a: Optional[Tensor] = None,
        mask_b: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Bidirectional co-attention fusion.

        Args:
            modality_a: Features from modality A (batch, seq_a, dim_a).
            modality_b: Features from modality B (batch, seq_b, dim_b).
            mask_a: Optional attention mask for modality A.
            mask_b: Optional attention mask for modality B.

        Returns:
            Tuple of (fused_a, fused_b) with original dimensions.
        """
        h_a = self.proj_a(modality_a)
        h_b = self.proj_b(modality_b)

        for i in range(len(self.a_to_b_attn)):
            h_a_norm = self.norm_a[i](h_a)
            h_b_norm = self.norm_b[i](h_b)

            a_attended_b, _ = self.a_to_b_attn[i](h_a_norm, h_b_norm, h_b_norm, mask_b)
            b_attended_a, _ = self.b_to_a_attn[i](h_b_norm, h_a_norm, h_a_norm, mask_a)

            gate_a = self.gate_a(torch.cat([h_a, a_attended_b], dim=-1))
            gate_b = self.gate_b(torch.cat([h_b, b_attended_a], dim=-1))

            h_a = self.dropout(gate_a * a_attended_b + (1 - gate_a) * h_a)
            h_b = self.dropout(gate_b * b_attended_a + (1 - gate_b) * h_b)

        return self.out_proj_a(h_a), self.out_proj_b(h_b)


class ConcatenationFusion(nn.Module):
    """Late fusion via concatenation and MLP projection.

    Concatenates features from multiple modalities and projects through
    an MLP to produce the fused output.

    Args:
        dims: List of feature dimensions for each modality.
        output_dim: Output fusion dimension.
        hidden_dim: Hidden dimension of the MLP.
        num_layers: Number of MLP layers.
        dropout: Dropout probability.
        activation: Activation function name.
    """

    def __init__(
        self,
        dims: List[int] = None,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        dims = dims or [1024, 1024]
        self.input_dim = sum(dims)
        hidden_dim = hidden_dim or 2 * self.input_dim

        act_fn = self._get_activation(activation)

        layers = [nn.Linear(self.input_dim, hidden_dim)]
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)
        self.output_norm = FusionLayerNorm(output_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "silu": nn.SiLU(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(name.lower(), nn.GELU())

    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        *modalities: Tensor,
        weights: Optional[List[float]] = None,
    ) -> Tensor:
        """Fuse modalities via concatenation.

        Args:
            *modalities: Variable number of modality tensors, each of shape
                         (batch, seq_len, dim_i).
            weights: Optional per-modality weights for weighted concatenation.

        Returns:
            Fused tensor of shape (batch, seq_len, output_dim).
        """
        if weights is not None:
            modalities = [m * w for m, w in zip(modalities, weights)]

        min_len = min(m.size(1) for m in modalities)
        modalities = [m[:, :min_len, :] for m in modalities]

        concat = torch.cat(modalities, dim=-1)
        fused = self.mlp(concat)
        return self.output_norm(fused)


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (GMU).

    Learns a gate to control the mixing ratio between two modalities:
        gate = sigmoid(W_g * [A; B] + b_g)
        output = gate * A + (1 - gate) * B

    Args:
        dim_a: Dimension of modality A.
        dim_b: Dimension of modality B.
        hidden_dim: Hidden dimension for gate computation.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_a: int = 1024,
        dim_b: int = 1024,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (dim_a + dim_b) // 2
        self.dim_a = dim_a
        self.dim_b = dim_b

        self.proj_a = nn.Linear(dim_a, hidden_dim) if dim_a != hidden_dim else nn.Identity()
        self.proj_b = nn.Linear(dim_b, hidden_dim) if dim_b != hidden_dim else nn.Identity()

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(hidden_dim, dim_a)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = FusionLayerNorm(dim_a)
        self._init_weights()

    def _init_weights(self):
        for module in self.gate_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        modality_a: Tensor,
        modality_b: Tensor,
    ) -> Tensor:
        """Gated fusion of two modalities.

        Args:
            modality_a: Features from modality A (batch, seq, dim_a).
            modality_b: Features from modality B (batch, seq, dim_b).

        Returns:
            Fused features of shape (batch, seq, dim_a).
        """
        h_a = self.proj_a(modality_a)
        h_b = self.proj_b(modality_b)

        min_len = min(h_a.size(1), h_b.size(1))
        h_a, h_b = h_a[:, :min_len, :], h_b[:, :min_len, :]

        combined = torch.cat([h_a, h_b], dim=-1)
        gate = self.gate_network(combined)

        fused = gate * h_a + (1 - gate) * h_b
        fused = self.dropout(fused)
        output = self.output_proj(fused)

        return self.layer_norm(output + modality_a[:, :min_len, :])


class BilinearPooling(nn.Module):
    """Efficient bilinear pooling using tensor decomposition.

    Implements compact bilinear pooling via Tucker decomposition,
    which reduces the O(d^2) complexity of full bilinear pooling
    to O(d * k) where k is the sketch dimension.

    Args:
        dim_a: Dimension of first input.
        dim_b: Dimension of second input.
        output_dim: Output dimension after pooling.
        sketch_dim: Internal sketch dimension for Tucker decomposition.
    """

    def __init__(self, dim_a: int = 1024, dim_b: int = 1024, output_dim: int = 1024, sketch_dim: int = 256):
        super().__init__()
        self.sketch_dim = sketch_dim
        self.sketch_a = nn.Linear(dim_a, sketch_dim, bias=False)
        self.sketch_b = nn.Linear(dim_b, sketch_dim, bias=False)
        self.output_proj = nn.Linear(sketch_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.sketch_a.weight, std=0.01)
        nn.init.normal_(self.sketch_b.weight, std=0.01)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(self, input_a: Tensor, input_b: Tensor) -> Tensor:
        """Compute compact bilinear pooling.

        Args:
            input_a: First input tensor (batch, ..., dim_a).
            input_b: Second input tensor (batch, ..., dim_b).

        Returns:
            Bilinear pooling result (batch, ..., output_dim).
        """
        sketch_a = torch.tanh(self.sketch_a(input_a))
        sketch_b = torch.tanh(self.sketch_b(input_b))

        hadamard = sketch_a * sketch_b
        sum_result = hadamard.sum(dim=-1) if hadamard.dim() > 2 else hadamard

        if hadamard.dim() > 2:
            sum_result = hadamard.mean(dim=-2)

        return self.output_proj(hadamard)


class CompactBilinearFusion(nn.Module):
    """Compact Bilinear Fusion using Tucker decomposition.

    Efficiently combines two modality features using low-rank
    bilinear pooling, significantly reducing computational cost
    compared to full bilinear pooling while maintaining expressive power.

    Args:
        dim_a: Dimension of modality A features.
        dim_b: Dimension of modality B features.
        hidden_dim: Hidden dimension for projections.
        output_dim: Output fusion dimension.
        sketch_dim: Internal sketch dimension for Tucker decomposition.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_a: int = 1024,
        dim_b: int = 1024,
        hidden_dim: Optional[int] = None,
        output_dim: int = 1024,
        sketch_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim_a
        self.dim_a = dim_a
        self.dim_b = dim_b

        self.proj_a = nn.Linear(dim_a, hidden_dim) if dim_a != hidden_dim else nn.Identity()
        self.proj_b = nn.Linear(dim_b, hidden_dim) if dim_b != hidden_dim else nn.Identity()

        self.bilinear = BilinearPooling(hidden_dim, hidden_dim, output_dim, sketch_dim)

        self.layer_norm = FusionLayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.proj_a, nn.Linear):
            nn.init.xavier_uniform_(self.proj_a.weight)
        if isinstance(self.proj_b, nn.Linear):
            nn.init.xavier_uniform_(self.proj_b.weight)

    def forward(self, modality_a: Tensor, modality_b: Tensor) -> Tensor:
        """Compact bilinear fusion.

        Args:
            modality_a: Features from modality A (batch, seq, dim_a).
            modality_b: Features from modality B (batch, seq, dim_b).

        Returns:
            Fused tensor of shape (batch, seq, output_dim).
        """
        h_a = self.proj_a(modality_a)
        h_b = self.proj_b(modality_b)

        min_len = min(h_a.size(1), h_b.size(1))
        h_a, h_b = h_a[:, :min_len, :], h_b[:, :min_len, :]

        fused = self.bilinear(h_a, h_b)
        fused = self.dropout(fused)
        return self.layer_norm(fused)


class AdaptiveFusion(nn.Module):
    """Adaptive fusion that dynamically selects strategy per input.

    Uses a small gating network to learn which fusion strategy works best
    for each input, combining multiple fusion mechanisms.

    Args:
        dim_a: Dimension of modality A.
        dim_b: Dimension of modality B.
        output_dim: Output dimension.
        num_strategies: Number of fusion strategies to combine.
        hidden_dim: Hidden dimension for gating network.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        dim_a: int = 1024,
        dim_b: int = 1024,
        output_dim: int = 1024,
        num_strategies: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (dim_a + dim_b)
        self.num_strategies = num_strategies

        self.gate_network = nn.Sequential(
            nn.Linear(dim_a + dim_b, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_strategies),
            nn.Softmax(dim=-1),
        )

        self.strategies = nn.ModuleList([
            CrossAttentionFusion(dim_a, dim_b, output_dim, num_heads=8,
                                 num_layers=1, dropout=dropout),
            ConcatenationFusion([dim_a, dim_b], output_dim, hidden_dim=output_dim,
                               num_layers=2, dropout=dropout),
            GatedFusion(dim_a, dim_b, hidden_dim=output_dim, dropout=dropout),
            CompactBilinearFusion(dim_a, dim_b, output_dim=output_dim,
                                 sketch_dim=128, dropout=dropout),
        ])

        if num_strategies > 4:
            extra = num_strategies - 4
            for i in range(extra):
                self.strategies.append(
                    CrossAttentionFusion(dim_a, dim_b, output_dim,
                                        num_heads=8, num_layers=1, dropout=dropout)
                )

        self.output_proj = nn.Linear(output_dim, output_dim)
        self.layer_norm = FusionLayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modality_a: Tensor,
        modality_b: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Adaptively fuse two modalities.

        Args:
            modality_a: Features from modality A.
            modality_b: Features from modality B.
            mask: Optional attention mask.

        Returns:
            Fused tensor.
        """
        min_len = min(modality_a.size(1), modality_b.size(1))
        m_a = modality_a[:, :min_len, :]
        m_b = modality_b[:, :min_len, :]

        pool_a = m_a.mean(dim=1)
        pool_b = m_b.mean(dim=1)
        pooled = torch.cat([pool_a, pool_b], dim=-1)
        strategy_weights = self.gate_network(pooled)

        fused_outputs = []
        for i in range(min(self.num_strategies, len(self.strategies))):
            if isinstance(self.strategies[i], CrossAttentionFusion):
                out = self.strategies[i](m_a, m_b, mask)
            elif isinstance(self.strategies[i], (GatedFusion, CompactBilinearFusion)):
                out = self.strategies[i](m_a, m_b)
            elif isinstance(self.strategies[i], ConcatenationFusion):
                out = self.strategies[i](m_a, m_b)
            else:
                out = self.strategies[i](m_a, m_b, mask)
            fused_outputs.append(out)

        stacked = torch.stack(fused_outputs, dim=0)
        weights = strategy_weights.unsqueeze(-1).unsqueeze(-1)
        combined = (stacked * weights).sum(dim=0)

        output = self.dropout(self.output_proj(combined))
        return self.layer_norm(output)


class FusionGate(nn.Module):
    """Scalar gating mechanism for modality selection.

    Computes a learned scalar gate value that controls the contribution
    of each modality to the final fused representation.

    Args:
        input_dim: Input feature dimension.
        num_modalities: Number of modalities to gate.
    """

    def __init__(self, input_dim: int = 1024, num_modalities: int = 2):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * num_modalities, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, *modalities: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gates and weighted sum.

        Args:
            *modalities: Modality tensors, each (batch, seq, dim).

        Returns:
            Tuple of (fused_output, gate_weights).
        """
        min_len = min(m.size(1) for m in modalities)
        modalities = [m[:, :min_len, :] for m in modalities]

        pool = torch.cat([m.mean(dim=1) for m in modalities], dim=-1)
        gates = self.gate_network(pool)

        stacked = torch.stack(modalities, dim=-1)
        gate_weights = gates.unsqueeze(1).unsqueeze(-1)
        fused = (stacked * gate_weights).sum(dim=-1)

        return fused, gates


class ModalityDropout(nn.Module):
    """Randomly drops entire modalities during training.

    Forces the model to not rely on any single modality by randomly
    zeroing out entire modality inputs with a given probability.

    Args:
        num_modalities: Number of modalities.
        dropout_rate: Probability of dropping each modality independently.
    """

    def __init__(self, num_modalities: int = 4, dropout_rate: float = 0.1):
        super().__init__()
        self.num_modalities = num_modalities
        self.dropout_rate = dropout_rate

    def forward(self, *modalities: Tensor) -> List[Tensor]:
        """Apply modality dropout.

        During training, each modality is zeroed out independently with
        probability dropout_rate. During evaluation, all modalities pass through.

        Args:
            *modalities: Modality tensors.

        Returns:
            List of (possibly dropped) modality tensors.
        """
        if not self.training or self.dropout_rate <= 0:
            return list(modalities)

        result = []
        for modality in modalities:
            if torch.rand(1).item() < self.dropout_rate:
                result.append(torch.zeros_like(modality))
            else:
                result.append(modality)
        return result


class GatedMultiHeadAttention(nn.Module):
    """Gated multi-head self-attention with modality-aware gating.

    Each attention head has its own gate that controls how much it attends
    to each modality. This allows the model to specialize heads for different
    modalities.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        num_modalities: Number of modalities.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        num_modalities: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.modality_gate = nn.Sequential(
            nn.Linear(embed_dim, num_heads * num_modalities),
            nn.Softmax(dim=-1),
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: Tensor,
        modality_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Gated multi-head attention.

        Args:
            x: Input tensor (batch, seq, embed_dim).
            modality_ids: Modality indices (batch, seq).
            attention_mask: Optional attention mask.

        Returns:
            Output tensor (batch, seq, embed_dim).
        """
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if modality_ids is not None:
            gate_weights = self.modality_gate(x.mean(dim=1, keepdim=False))
            gate_weights = gate_weights.view(B, self.num_heads, self.num_modalities)

            modality_mask = torch.zeros(B, S, S, device=x.device)
            for m in range(self.num_modalities):
                m_mask = (modality_ids == m).float()
                modality_mask = torch.bmm(m_mask.unsqueeze(2), m_mask.unsqueeze(1))
                for h in range(self.num_heads):
                    g = gate_weights[:, h, m].unsqueeze(1).unsqueeze(2)
                    modality_mask[:, h] += g * modality_mask[:, h] * 0.5
            attn = attn + modality_mask.unsqueeze(1)

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.proj_dropout(self.out_proj(out))

        return out


class MultiModalTransformerLayer(nn.Module):
    """Unified transformer layer for mixed-modality inputs.

    Handles sequences containing tokens from multiple modalities with
    modality-aware attention masking. Supports both self-attention and
    cross-attention within a single layer.

    Args:
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward network dimension.
        dropout: Dropout probability.
        num_modalities: Number of modalities for gating.
        use_flash_attn: Whether to use flash attention.
        activation: Feed-forward activation function.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_modalities: int = 4,
        use_flash_attn: bool = False,
        activation: str = "gelu",
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * hidden_dim
        self.hidden_dim = hidden_dim

        self.norm1 = FusionLayerNorm(hidden_dim)
        self.norm2 = FusionLayerNorm(hidden_dim)
        self.norm3 = FusionLayerNorm(hidden_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.gated_attn = GatedMultiHeadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_modalities=num_modalities,
            dropout=dropout,
        )

        self.use_gated_attention = num_modalities > 1
        self.attn_dropout = nn.Dropout(dropout)

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        modality_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of multimodal transformer layer.

        Args:
            x: Input tensor (batch, seq, hidden_dim).
            attention_mask: Optional attention mask (batch, seq, seq).
            modality_ids: Optional modality indices (batch, seq).

        Returns:
            Output tensor (batch, seq, hidden_dim).
        """
        residual = x

        if self.use_gated_attention and modality_ids is not None:
            x_norm = self.norm1(x)
            attn_out = self.gated_attn(x_norm, modality_ids, attention_mask)
        else:
            x_norm = self.norm1(x)
            if attention_mask is not None:
                attn_mask = ~attention_mask
            else:
                attn_mask = None
            attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)

        x = residual + self.dropout1(attn_out)

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_out)

        return x


class MultiModalTransformer(nn.Module):
    """Stack of multimodal transformer layers.

    Processes mixed-modality input sequences through multiple transformer
    layers with modality-aware attention and gating.

    Args:
        hidden_dim: Model hidden dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        ffn_dim: Feed-forward network dimension.
        dropout: Dropout probability.
        num_modalities: Number of modalities.
        max_seq_len: Maximum sequence length (for position embeddings).
        use_flash_attn: Whether to use flash attention.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_modalities: int = 4,
        max_seq_len: int = 4096,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_modalities = num_modalities

        self.modality_embedding = ModalityEmbedding(
            embed_dim=hidden_dim,
            num_modalities=num_modalities,
            dropout=dropout,
        )

        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MultiModalTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                num_modalities=num_modalities,
                use_flash_attn=use_flash_attn,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = FusionLayerNorm(hidden_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(
        self,
        input_embeds: Tensor,
        modality_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through multimodal transformer.

        Args:
            input_embeds: Input embeddings (batch, seq, hidden_dim).
            modality_ids: Modality type for each token (batch, seq).
            attention_mask: Optional mask (batch, seq).

        Returns:
            Output tensor (batch, seq, hidden_dim).
        """
        B, S, D = input_embeds.shape

        positions = torch.arange(S, device=input_embeds.device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_embedding(positions)

        x = self.modality_embedding(input_embeds, modality_ids) + pos_embed
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask, modality_ids)

        return self.final_norm(x)
