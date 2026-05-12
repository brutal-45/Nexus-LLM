"""
Vision Encoder Module
=====================

Production-grade vision encoder implementations for the Nexus LLM multimodal framework.
Provides multiple encoder architectures including ViT, SigLIP, ConvNeXt, and
EfficientViT, along with supporting components like patch embeddings, normalization,
image augmentation, resolution adaptation, and preprocessing.

All encoders support:
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for memory efficiency
- Flash attention (when available)
- Dynamic resolution handling
- Comprehensive output structures

Architecture implementations follow the original papers closely with engineering
optimizations for production deployment.

References:
    - ViT: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
    - SigLIP: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023)
    - ConvNeXt: "A ConvNet for the 2020s" (Liu et al., 2022)
    - EfficientViT: "EfficientViT: Multi-Scale Linear Attention" (Liu et al., 2023)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    # Primary encoder classes
    "PatchEmbedding2D",
    "ViTEncoder",
    "SigLIPEncoder",
    "ConvNeXtBlock",
    "ConvNeXtEncoder",
    "EfficientViTBlock",
    "ImageAugmentation",
    "ResolutionAdaptor",
    "ImagePreprocessor",
    # Supporting data structures
    "VisionEncoderOutput",
    "MultiScaleFeatures",
    "VisionEncoderConfig",
    # Supporting layers
    "DropPath",
    "LayerNorm2D",
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "FeedForward",
    "LayerScale",
    "ConvNeXtDownsample",
    "PatchifyStem",
    "EfficientViTAttention",
    "EfficientViTMLP",
    # Utility functions
    "window_partition",
    "window_reverse",
    "build_2d_sincos_position_embedding",
    "trunc_normal_init",
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class VisionEncoderConfig:
    """Configuration for vision encoder architectures.

    Attributes:
        image_size: Input image resolution (square).
        patch_size: Patch size for patch embedding.
        in_channels: Number of input image channels.
        hidden_size: Transformer hidden dimension.
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to hidden_size.
        dropout_rate: Dropout probability.
        attention_dropout_rate: Attention dropout probability.
        drop_path_rate: Stochastic depth rate (linearly increased across layers).
        use_checkpoint: Enable gradient checkpointing for memory efficiency.
        use_flash_attn: Try to use flash / SDPA attention when available.
        layer_norm_eps: Epsilon for layer normalization.
        use_cls_token: Prepend a learnable [CLS] token.
        pos_embed_type: Type of positional embedding ("learned" or "sinusoidal").
        use_pre_norm: Use pre-normalization (True) or post-normalization (False).
        initializer_range: Std-dev for truncated normal weight initialization.
        num_registers: Number of extra register tokens.
        convnext_stages: Stage depths for ConvNeXt.
        convnext_dims: Channel dims for each ConvNeXt stage.
        convnext_layer_scale_init: Initial value for ConvNeXt layer scale.
        efficient_vit_key_dim: Key dimension for EfficientViT attention.
        efficient_vit_value_dim: Value dimension for EfficientViT attention.
        efficient_vit_reduction_ratio: Reduction ratio for Q/K projection.
        efficient_vit_use_linear_attn: Use linear attention variant.
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    drop_path_rate: float = 0.0
    use_checkpoint: bool = False
    use_flash_attn: bool = False
    layer_norm_eps: float = 1e-6
    use_cls_token: bool = True
    pos_embed_type: str = "learned"
    use_pre_norm: bool = True
    initializer_range: float = 0.02
    num_registers: int = 0
    convnext_stages: Tuple[int, ...] = (3, 3, 9, 3)
    convnext_dims: Tuple[int, ...] = (96, 192, 384, 768)
    convnext_layer_scale_init: float = 1e-6
    efficient_vit_key_dim: int = 64
    efficient_vit_value_dim: int = 64
    efficient_vit_reduction_ratio: int = 4
    efficient_vit_use_linear_attn: bool = False


# =============================================================================
# Output Data Structures
# =============================================================================


@dataclass
class VisionEncoderOutput:
    """Structured output from a vision encoder.

    Attributes:
        last_hidden_state: Final hidden state of shape ``(batch, seq_len, hidden_size)``.
        hidden_states: Optional tuple of hidden states from every layer.
        attentions: Optional tuple of attention weight tensors from every layer.
        cls_token: CLS token representation of shape ``(batch, hidden_size)``.
        patch_embeddings: Patch-level embeddings ``(batch, num_patches, hidden_size)``.
        spatial_features: Reshaped spatial features ``(batch, hidden_size, H, W)``.
        pooler_output: Pooled global representation.
    """

    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    cls_token: Optional[torch.Tensor] = None
    patch_embeddings: Optional[torch.Tensor] = None
    spatial_features: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary for serialization."""
        out: Dict[str, Any] = {"last_hidden_state": self.last_hidden_state}
        if self.hidden_states is not None:
            out["hidden_states"] = list(self.hidden_states)
        if self.attentions is not None:
            out["attentions"] = list(self.attentions)
        if self.cls_token is not None:
            out["cls_token"] = self.cls_token
        if self.patch_embeddings is not None:
            out["patch_embeddings"] = self.patch_embeddings
        if self.spatial_features is not None:
            out["spatial_features"] = self.spatial_features
        if self.pooler_output is not None:
            out["pooler_output"] = self.pooler_output
        return out


@dataclass
class MultiScaleFeatures:
    """Multi-scale feature maps produced by a hierarchical backbone.

    Attributes:
        features: Dict mapping stage name to feature tensor ``(B, C, H, W)``.
        strides: Dict mapping stage name to its spatial stride relative to input.
        channels: Dict mapping stage name to its channel count.
    """

    features: Dict[str, torch.Tensor] = field(default_factory=dict)
    strides: Dict[str, int] = field(default_factory=dict)
    channels: Dict[str, int] = field(default_factory=dict)

    def get_feature(self, stage_name: str) -> torch.Tensor:
        """Return feature tensor for *stage_name* or raise ``KeyError``."""
        if stage_name not in self.features:
            raise KeyError(
                f"Stage '{stage_name}' not found. "
                f"Available: {list(self.features.keys())}"
            )
        return self.features[stage_name]

    def get_largest_feature(self) -> torch.Tensor:
        """Return the feature map with the largest spatial resolution."""
        if not self.features:
            raise ValueError("No features available.")
        name = max(
            self.features,
            key=lambda n: self.features[n].shape[-1] * self.features[n].shape[-2],
        )
        return self.features[name]

    def get_smallest_feature(self) -> torch.Tensor:
        """Return the feature map with the smallest spatial resolution."""
        if not self.features:
            raise ValueError("No features available.")
        name = min(
            self.features,
            key=lambda n: self.features[n].shape[-1] * self.features[n].shape[-2],
        )
        return self.features[name]

    def concatenate(self) -> torch.Tensor:
        """Up-sample all features to the largest resolution and concatenate."""
        if not self.features:
            raise ValueError("No features to concatenate.")
        target_h = max(f.shape[-2] for f in self.features.values())
        target_w = max(f.shape[-1] for f in self.features.values())
        resized = []
        for feat in self.features.values():
            if feat.shape[-2] != target_h or feat.shape[-1] != target_w:
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            resized.append(feat)
        return torch.cat(resized, dim=1)


# =============================================================================
# Utility Functions
# =============================================================================


def trunc_normal_init(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Initialize *tensor* with a truncated normal distribution.

    Args:
        tensor: The parameter tensor to fill in-place.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
        a: Minimum cutoff value.
        b: Maximum cutoff value.

    Returns:
        The initialized tensor (same object, modified in-place).
    """
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)
    return tensor


def build_2d_sincos_position_embedding(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
) -> torch.Tensor:
    """Build a 2-D sinusoidal-cosine position embedding table.

    Args:
        embed_dim: Embedding dimension (must be divisible by 4).
        grid_size: Height and width of the patch grid (assumed square).
        cls_token: Whether to leave the first row for a CLS token placeholder.

    Returns:
        Position embedding table of shape ``(1, num_tokens, embed_dim)``.
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos PE"
    pos_h = grid_h.flatten() / grid_size
    pos_w = grid_w.flatten() / grid_size
    pe = torch.zeros((grid_size * grid_size, embed_dim), dtype=torch.float32)
    dim_half = embed_dim // 4
    div_term = torch.exp(
        torch.arange(dim_half, dtype=torch.float32) * (-math.log(10000.0) / (embed_dim // 2))
    )
    pe[:, 0:embed_dim // 4] = torch.sin(pos_h.unsqueeze(1) * div_term.unsqueeze(0))
    pe[:, embed_dim // 4 : embed_dim // 2] = torch.cos(
        pos_h.unsqueeze(1) * div_term.unsqueeze(0)
    )
    pe[:, embed_dim // 2 : 3 * embed_dim // 4] = torch.sin(
        pos_w.unsqueeze(1) * div_term.unsqueeze(0)
    )
    pe[:, 3 * embed_dim // 4 :] = torch.cos(pos_w.unsqueeze(1) * div_term.unsqueeze(0))
    if cls_token:
        pe = torch.cat([torch.zeros(1, embed_dim), pe], dim=0)
    return pe.unsqueeze(0)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition a feature map into non-overlapping windows.

    Args:
        x: Tensor of shape ``(B, H, W, C)``.
        window_size: Size of each square window.

    Returns:
        Windows of shape ``(B * num_windows, window_size, window_size, C)``.
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b * (h // window_size) * (w // window_size), window_size, window_size, c)
    )
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse the window partition operation.

    Args:
        windows: Windows of shape ``(B * num_windows, window_size, window_size, C)``.
        window_size: Size of each square window.
        h: Original height of the feature map.
        w: Original width of the feature map.

    Returns:
        Reconstructed feature map of shape ``(B, H, W, C)``.
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(
        b, h // window_size, w // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# =============================================================================
# Supporting Layers
# =============================================================================


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularizing deep networks.

    Randomly drops entire residual branches during training, scaling the
    remaining paths to preserve the expected value of the output.

    Reference: "Deep Networks with Stochastic Depth" (Huang et al., 2016).
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class LayerScale(nn.Module):
    """Per-channel learnable scaling applied before the residual addition.

    Used in ConvNeXt and other modern architectures.

    Args:
        dim: Number of channels.
        init_value: Initial value for the scale parameter.
        inplace: Whether to apply scaling in-place (saves memory).
    """

    def __init__(self, dim: int, init_value: float = 1e-6, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            return x.mul_(self.gamma.view(1, -1, 1, 1))
        return x * self.gamma.view(1, -1, 1, 1)

    def extra_repr(self) -> str:
        return f"dim={self.gamma.shape[0]}, inplace={self.inplace}"


class LayerNorm2D(nn.Module):
    """Layer normalisation operating on the channel dimension of 2-D feature maps.

    Uses ``(B, C, H, W)`` layout and normalises over ``(C, H, W)`` per spatial
    location, matching the ConvNeXt design.

    Args:
        num_channels: Number of channels ``C``.
        eps: Small constant for numerical stability.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* of shape ``(B, C, H, W)``."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x


# =============================================================================
# Multi-Head Self-Attention
# =============================================================================


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional SDPA / flash backend.

    Args:
        hidden_size: Total hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout applied to attention weights.
        use_flash_attn: Whether to attempt using ``F.scaled_dot_product_attention``.
        bias: Include bias in QKV / output projections.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
        if self.k_proj.bias is not None:
            nn.init.zeros_(self.k_proj.bias)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run multi-head self-attention.

        Args:
            hidden_states: ``(# Import: ``Tensor``) of shape ``(B, L, D)``.
            attention_mask: Optional additive mask (broadcastable to ``(B, H, L, L)``).
            output_attentions: Whether to return attention weights.

        Returns:
            ``(attn_output, attn_weights)`` where *attn_weights* is ``None``
            when *output_attentions* is ``False``.
        """
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Try SDPA when requested and we don't need to export attention weights
        if self.use_flash_attn and not output_attentions and hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_weights: Optional[torch.Tensor] = None
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, v)
            if not output_attentions:
                attn_weights = None

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


# =============================================================================
# Feed-Forward Network
# =============================================================================


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside transformer blocks.

    Supports standard two-layer MLP with GELU activation, as well as gated
    (SwiGLU-style) variants.

    Args:
        hidden_size: Input and output dimensionality.
        intermediate_size: Dimensionality of the hidden layer.
        hidden_act: Activation function name (``"gelu"``, ``"relu"``, ``"silu"``, etc.).
        dropout: Dropout probability after the second linear layer.
        bias: Include bias in linear projections.
        gated: Use a gated activation (SwiGLU-style).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gated = gated
        self.act_name = hidden_act.lower()

        if gated:
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        else:
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        if self.gated:
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.xavier_uniform_(self.down_proj.weight)
            if self.gate_proj.bias is not None:
                nn.init.zeros_(self.gate_proj.bias)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
            if self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)
        else:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)

    def _apply_act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation function."""
        if self.act_name == "gelu":
            return F.gelu(x, approximate="tanh")
        elif self.act_name == "gelu_exact":
            return F.gelu(x, approximate="none")
        elif self.act_name == "relu":
            return F.relu(x)
        elif self.act_name == "relu6":
            return F.relu6(x)
        elif self.act_name in ("silu", "swish"):
            return F.silu(x)
        elif self.act_name == "mish":
            return x * torch.tanh(F.softplus(x))
        elif self.act_name == "tanh":
            return torch.tanh(x)
        elif self.act_name == "sigmoid":
            return torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.act_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network.

        Args:
            x: Input tensor of shape ``(B, L, D)``.

        Returns:
            Output tensor of shape ``(B, L, D)``.
        """
        if self.gated:
            gate = self._apply_act(self.gate_proj(x))
            up = self.up_proj(x)
            hidden = gate * up
            out = self.down_proj(hidden)
        else:
            hidden = self._apply_act(self.fc1(x))
            out = self.fc2(hidden)
        return self.dropout(out)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"act={self.act_name}, gated={self.gated}"
        )


# =============================================================================
# Transformer Encoder Block
# =============================================================================


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block: LN → MHA → residual → LN → FFN → residual.

    Supports pre-norm (default) and post-norm configurations, stochastic depth,
    and gradient checkpointing.

    Args:
        hidden_size: Transformer hidden dimension.
        num_heads: Number of attention heads.
        mlp_ratio: FFN intermediate dim = ``hidden_size * mlp_ratio``.
        dropout: General dropout probability.
        attention_dropout: Attention-specific dropout.
        drop_path: Stochastic depth probability for this block.
        layer_norm_eps: Epsilon for layer normalisation.
        pre_norm: Use pre-normalization.
        hidden_act: Activation function for FFN.
        use_flash_attn: Attempt flash attention.
        intermediate_size: Override intermediate size (takes precedence over mlp_ratio).
        gated: Use gated FFN.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
        pre_norm: bool = True,
        hidden_act: str = "gelu",
        use_flash_attn: bool = False,
        intermediate_size: Optional[int] = None,
        gated: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.pre_norm = pre_norm

        ff_dim = intermediate_size if intermediate_size is not None else int(hidden_size * mlp_ratio)

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            use_flash_attn=use_flash_attn,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=ff_dim,
            hidden_act=hidden_act,
            dropout=dropout,
            gated=gated,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _forward_pre_norm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_out, attn_w = self.attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = residual + self.drop_path(self.dropout1(attn_out))

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_out = self.ffn(hidden_states)
        hidden_states = residual + self.drop_path(self.dropout2(ffn_out))

        return hidden_states, attn_w

    def _forward_post_norm(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        attn_out, attn_w = self.attn(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)
        hidden_states = self.norm1(residual + self.drop_path(self.dropout1(attn_out)))

        residual = hidden_states
        ffn_out = self.ffn(hidden_states)
        hidden_states = self.norm2(residual + self.drop_path(self.dropout2(ffn_out)))

        return hidden_states, attn_w

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: ``(# Import: ``Tensor``) of shape ``(B, L, D)``.
            attention_mask: Optional additive attention mask.
            output_attentions: Return attention weights.

        Returns:
            ``(hidden_states, attn_weights)``.
        """
        if self.pre_norm:
            return self._forward_pre_norm(hidden_states, attention_mask, output_attentions)
        return self._forward_post_norm(hidden_states, attention_mask, output_attentions)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, pre_norm={self.pre_norm}"
        )


# =============================================================================
# 1. PatchEmbedding2D
# =============================================================================


class PatchEmbedding2D(nn.Module):
    """2-D patch embedding with overlap, CLS token, and position encoding.

    Converts an input image ``(B, C, H, W)`` into a sequence of patch tokens
    ``(B, L, D)`` via a :class:`nn.Conv2d` projection, optional CLS token
    prepending, and positional embedding addition.

    Supports both learned and sinusoidal (fixed) positional embeddings with
    an ``interpolate_pos_embed()`` method for adapting to variable resolutions
    at inference time.

    Args:
        image_size: Default square image resolution.
        patch_size: Size of each square patch (also the conv kernel size).
        in_channels: Number of input channels.
        embed_dim: Output embedding dimensionality.
        overlap: Use overlapping patches (stride = patch_size // 2).
        stride: Override the convolution stride (``None`` → *patch_size* or
            *patch_size // 2* when *overlap* is ``True``).
        padding: Extra padding added to the conv.
        use_cls_token: Prepend a learnable ``[CLS]`` token.
        pos_embed_type: ``"learned"`` or ``"sinusoidal"``.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        overlap: bool = False,
        stride: Optional[int] = None,
        padding: int = 0,
        use_cls_token: bool = True,
        pos_embed_type: str = "learned",
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.overlap = overlap
        self.use_cls_token = use_cls_token
        self.pos_embed_type = pos_embed_type

        if stride is not None:
            self.stride = stride
        elif overlap:
            self.stride = max(1, patch_size // 2)
        else:
            self.stride = patch_size

        self.padding = padding

        # --- projection convolution ---
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=self.stride,
            padding=padding,
            bias=True,
        )
        trunc_normal_init(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

        # --- flatten → LayerNorm ---
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # --- grid arithmetic ---
        self.grid_size = (image_size + 2 * padding - patch_size) // self.stride + 1
        self.num_patches = self.grid_size * self.grid_size

        # --- CLS token ---
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_init(self.cls_token, std=0.02)

        # --- position embeddings ---
        num_pos = self.num_patches + (1 if use_cls_token else 0)
        if pos_embed_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_pos, embed_dim))
            trunc_normal_init(self.pos_embed, std=0.02)
        elif pos_embed_type == "sinusoidal":
            pe = build_2d_sincos_position_embedding(embed_dim, self.grid_size, cls_token=use_cls_token)
            self.register_buffer("pos_embed", pe, persistent=False)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type!r}. Use 'learned' or 'sinusoidal'.")

    def interpolate_pos_embed(
        self,
        x: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Interpolate the position embeddings to match a different spatial grid.

        This is useful when processing images at a resolution different from
        the one the model was trained on.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` (used for batch / device).
            target_h: Desired grid height in number of patches.
            target_w: Desired grid width in number of patches.

        Returns:
            Position embedding tensor broadcastable to ``(B, target_h*target_w(+1), D)``.
        """
        if self.pos_embed_type == "sinusoidal":
            pe = build_2d_sincos_position_embedding(self.embed_dim, 1, cls_token=False)
            pe_grid = build_2d_sincos_position_embedding(self.embed_dim, self.grid_size, cls_token=False)
            pe_grid = pe_grid.squeeze(0)
            if self.use_cls_token:
                cls_pe = torch.zeros(1, self.embed_dim, device=x.device, dtype=x.dtype)
                pe_grid = pe_grid[1:]
        else:
            pe_grid = self.pos_embed.data
            if self.use_cls_token:
                cls_pe = pe_grid[:, 0:1, :]
                pe_grid = pe_grid[:, 1:, :]
            else:
                cls_pe = None

        # pe_grid: (1, num_patches, D) → (1, D, H, W)
        D = pe_grid.shape[-1]
        src_h = src_w = self.grid_size
        pe_2d = pe_grid.reshape(1, src_h, src_w, D).permute(0, 3, 1, 2)
        pe_2d = F.interpolate(
            pe_2d, size=(target_h, target_w), mode="bicubic", align_corners=False
        )
        pe_2d = pe_2d.permute(0, 2, 3, 1).reshape(1, target_h * target_w, D)

        if cls_pe is not None:
            pe_2d = torch.cat([cls_pe.to(device=x.device, dtype=x.dtype), pe_2d], dim=1)

        return pe_2d.to(device=x.device, dtype=x.dtype)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate: bool = False,
    ) -> Tuple[torch.Tensor, int, int]:
        """Create patch embeddings from pixel values.

        Args:
            pixel_values: Image tensor ``(B, C, H, W)``.
            interpolate: If ``True``, interpolate position embeddings when the
                input resolution differs from ``self.image_size``.

        Returns:
            ``(embeddings, grid_h, grid_w)`` where *embeddings* has shape
            ``(B, L, embed_dim)``.
        """
        B, C, H, W = pixel_values.shape

        # Conv2d projection: (B, C, H, W) → (B, D, grid_h, grid_w)
        x = self.proj(pixel_values)

        # Flatten spatial dims → (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # Layer normalisation on the token dimension
        x = self.norm(x)

        grid_h = x.shape[1] // (W // self.stride if W % self.stride == 0 else 1)
        if (W // self.stride) * (H // self.stride) == x.shape[1]:
            grid_h = H // self.stride
            grid_w = W // self.stride
        else:
            # Fallback: infer grid from conv output shape
            grid_h = int(math.sqrt(x.shape[1]))
            grid_w = x.shape[1] // grid_h if grid_h > 0 else x.shape[1]

        # Position embeddings
        need_interpolation = interpolate and (grid_h != self.grid_size or grid_w != self.grid_size)
        if need_interpolation:
            pos = self.interpolate_pos_embed(pixel_values, grid_h, grid_w)
        else:
            expected_len = grid_h * grid_w + (1 if self.use_cls_token else 0)
            pos = self.pos_embed[:, :expected_len].to(device=x.device, dtype=x.dtype)

        # CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([cls_tokens, x], dim=1)

        x = x + pos
        return x, grid_h, grid_w

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, patch_size={self.patch_size}, "
            f"embed_dim={self.embed_dim}, stride={self.stride}, "
            f"num_patches={self.num_patches}, grid_size={self.grid_size}, "
            f"cls_token={self.use_cls_token}, pos_embed={self.pos_embed_type}"
        )


# =============================================================================
# 2. ViTEncoder
# =============================================================================


class ViTEncoder(nn.Module):
    """Full Vision Transformer encoder (ViT).

    Architecture: ``PatchEmbedding2D`` → ``N × TransformerEncoderBlock`` → ``LayerNorm``.

    Each block follows the pre-norm (or post-norm) pattern:
        ``LN → MHA → residual → LN → MLP(GELU) → residual``

    Supports:
        - Pre-norm and post-norm transformer configurations.
        - Stochastic depth (linearly increasing drop-path rate across layers).
        - Gradient checkpointing for memory-efficient training.
        - Flash / SDPA attention when available.
        - Configurable hidden size, heads, layers, MLP ratio, and dropout.
        - Register tokens for improved attention patterns.
        - Variable-resolution inputs via interpolated position embeddings.

    Args:
        config: A :class:`VisionEncoderConfig` instance (or ``None`` for defaults).
    """

    def __init__(self, config: Optional[VisionEncoderConfig] = None):
        super().__init__()
        if config is None:
            config = VisionEncoderConfig()
        self.config = config

        cfg = config

        # --- Patch embedding ---
        self.patch_embed = PatchEmbedding2D(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.hidden_size,
            overlap=False,
            use_cls_token=cfg.use_cls_token,
            pos_embed_type=cfg.pos_embed_type,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size
        self.seq_length = self.num_patches + (1 if cfg.use_cls_token else 0)

        # --- Register tokens ---
        if cfg.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, cfg.num_registers, cfg.hidden_size))
            trunc_normal_init(self.register_tokens, std=cfg.initializer_range)
        else:
            self.register_tokens = None

        # --- Stochastic depth schedule ---
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg.drop_path_rate, cfg.num_layers)
        ]

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout_rate,
                attention_dropout=cfg.attention_dropout_rate,
                drop_path=dpr[i],
                layer_norm_eps=cfg.layer_norm_eps,
                pre_norm=cfg.use_pre_norm,
                hidden_act="gelu",
                use_flash_attn=cfg.use_flash_attn,
            )
            for i in range(cfg.num_layers)
        ])

        # --- Final layer norm ---
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.gradient_checkpointing = cfg.use_checkpoint

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply weight initialisation to final norm."""
        trunc_normal_init(self.final_norm.weight, std=self.config.initializer_range)
        nn.init.zeros_(self.final_norm.bias)

    def _make_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create an all-zero (unmasked) attention mask.

        Override this method in subclasses for masked attention patterns.
        """
        return torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[VisionEncoderOutput, Tuple[torch.Tensor, ...]]:
        """Forward pass through the ViT encoder.

        Args:
            pixel_values: Image tensor ``(B, C, H, W)``.
            attention_mask: Optional additive attention mask.
            interpolate_pos_encoding: Interpolate positional embeddings for
                non-standard resolutions.
            output_hidden_states: Return hidden states from all layers.
            output_attentions: Return attention weights from all layers.
            return_dict: If ``True``, return a :class:`VisionEncoderOutput`,
                else a plain tuple.

        Returns:
            A :class:`VisionEncoderOutput` or a tuple of tensors.
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Patch embedding + position encoding
        hidden_states, grid_h, grid_w = self.patch_embed(
            pixel_values, interpolate=interpolate_pos_encoding
        )

        # Register tokens
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B, -1, -1)
            hidden_states = torch.cat([hidden_states, reg], dim=1)

        seq_len = hidden_states.shape[1]
        if attention_mask is None:
            attention_mask = self._make_attention_mask(B, seq_len, device)

        all_hidden_states: Optional[List[torch.Tensor]] = [] if output_hidden_states else None
        all_attentions: Optional[List[torch.Tensor]] = [] if output_attentions else None

        if output_hidden_states:
            all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_w = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_w = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)  # type: ignore[union-attr]
            if output_attentions and attn_w is not None:
                all_attentions.append(attn_w)  # type: ignore[union-attr]

        hidden_states = self.final_norm(hidden_states)

        # Separate CLS and patch tokens
        cls_token: Optional[torch.Tensor] = None
        patch_embeddings: Optional[torch.Tensor] = None
        if self.config.use_cls_token:
            cls_token = hidden_states[:, 0]
            patches = hidden_states[:, 1:]
            if self.register_tokens is not None:
                patches = patches[:, :-self.config.num_registers]
            patch_embeddings = patches
        else:
            patches = hidden_states
            if self.register_tokens is not None:
                patches = patches[:, :-self.config.num_registers]
            patch_embeddings = patches

        # Spatial features
        spatial_features: Optional[torch.Tensor] = None
        if patch_embeddings is not None:
            n = patch_embeddings.shape[1]
            g = int(math.sqrt(n))
            if g * g == n:
                spatial_features = patch_embeddings.transpose(1, 2).reshape(B, -1, g, g)

        if not return_dict:
            return (
                hidden_states,
                tuple(all_hidden_states) if all_hidden_states else None,
                tuple(all_attentions) if all_attentions else None,
                cls_token,
                patch_embeddings,
                spatial_features,
            )

        return VisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            attentions=tuple(all_attentions) if all_attentions else None,
            cls_token=cls_token,
            patch_embeddings=patch_embeddings,
            spatial_features=spatial_features,
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return total number of trainable parameters."""
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if non_embedding and hasattr(self.patch_embed, "pos_embed"):
            if isinstance(self.patch_embed.pos_embed, nn.Parameter):
                n -= self.patch_embed.pos_embed.numel()
        return n

    def get_feature_resolution(self) -> int:
        """Return the spatial resolution (grid size) of output feature maps."""
        return self.grid_size


# =============================================================================
# 3. SigLIPEncoder
# =============================================================================


class SigLIPEncoder(nn.Module):
    """SigLIP-style vision encoder with learned-temperature sigmoid loss.

    Architecture is similar to ViT but differs in several important ways:
        - Uses a **sigmoid loss** (element-wise binary cross-entropy) instead
          of the softmax cross-entropy used in CLIP for contrastive learning.
        - Includes a learnable **temperature** (``logit_scale``) parameter.
        - Supports both **pre-norm** and **post-norm** configurations.
        - Uses global average pooling instead of a CLS token by default.

    The sigmoid loss avoids the batch-size dependence of the softmax contrastive
    loss and often yields better downstream performance at the same compute budget.

    Reference: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., 2023).

    Args:
        config: A :class:`VisionEncoderConfig` instance (or ``None`` for defaults).
        norm_type: ``"pre_norm"`` or ``"post_norm"``.
    """

    def __init__(
        self,
        config: Optional[VisionEncoderConfig] = None,
        norm_type: str = "pre_norm",
    ):
        super().__init__()
        if config is None:
            config = VisionEncoderConfig()
        self.config = config
        cfg = config
        self.norm_type = norm_type
        self.pre_norm = norm_type == "pre_norm"

        # --- Patch embedding (no CLS token by default for SigLIP) ---
        self.patch_embed = PatchEmbedding2D(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.hidden_size,
            use_cls_token=False,
            pos_embed_type=cfg.pos_embed_type,
        )
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        # --- Stochastic depth schedule ---
        dpr = [
            x.item() for x in torch.linspace(0, cfg.drop_path_rate, cfg.num_layers)
        ]

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout_rate,
                attention_dropout=cfg.attention_dropout_rate,
                drop_path=dpr[i],
                layer_norm_eps=cfg.layer_norm_eps,
                pre_norm=self.pre_norm,
                hidden_act="gelu",
                use_flash_attn=cfg.use_flash_attn,
            )
            for i in range(cfg.num_layers)
        ])

        # --- Final norm ---
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # --- Learnable logit temperature for sigmoid loss ---
        # Following SigLIP paper, initialised so that sigmoid(temperature * dot_product)
        # starts near-uniform.  logit_scale = log(exp(init)) = init ≈ ln(1/c) for c.
        self.logit_scale = nn.Parameter(torch.zeros(1) + math.log(1.0 / 0.07))
        self.logit_bias = nn.Parameter(torch.zeros(1))

        self.gradient_checkpointing = cfg.use_checkpoint

    def sigmoid_loss(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the SigLIP sigmoid loss.

        Unlike the standard CLIP softmax cross-entropy (which normalises over
        the batch), this computes an element-wise sigmoid BCE loss.

        Args:
            vision_features: Normalised vision features ``(B, D)``.
            text_features: Normalised text features ``(B, D)``.
            labels: Ground-truth matching matrix ``(B, B)`` where 1 indicates
                a matching pair. If ``None``, the diagonal is used.

        Returns:
            Scalar loss value.
        """
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * vision_features @ text_features.T + self.logit_bias

        if labels is None:
            labels = torch.eye(logits.shape[0], device=logits.device, dtype=logits.dtype)

        # Per-pair binary cross-entropy with sigmoid
        loss = -(
            labels * F.logsigmoid(logits) + (1.0 - labels) * F.logsigmoid(-logits)
        )
        return loss.mean()

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[VisionEncoderOutput, Tuple[torch.Tensor, ...]]:
        """Forward pass through the SigLIP vision encoder.

        Args:
            pixel_values: Image tensor ``(B, C, H, W)``.
            interpolate_pos_encoding: Interpolate pos-embed for variable resolution.
            output_hidden_states: Collect all hidden states.
            output_attentions: Collect all attention weights.
            return_dict: Return structured output.

        Returns:
            A :class:`VisionEncoderOutput` or tuple.
        """
        B = pixel_values.shape[0]

        hidden_states, grid_h, grid_w = self.patch_embed(
            pixel_values, interpolate=interpolate_pos_encoding
        )

        all_hidden_states: Optional[List[torch.Tensor]] = [] if output_hidden_states else None
        all_attentions: Optional[List[torch.Tensor]] = [] if output_attentions else None

        if output_hidden_states:
            all_hidden_states.append(hidden_states)  # type: ignore[union-attr]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_w = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, None, output_attentions, use_reentrant=False,
                )
            else:
                hidden_states, attn_w = block(
                    hidden_states, output_attentions=output_attentions
                )

            if output_hidden_states:
                all_hidden_states.append(hidden_states)  # type: ignore[union-attr]
            if output_attentions and attn_w is not None:
                all_attentions.append(attn_w)  # type: ignore[union-attr]

        hidden_states = self.final_norm(hidden_states)

        # Global average pooling (SigLIP style)
        pooler_output = hidden_states.mean(dim=1)

        # Spatial features
        n = hidden_states.shape[1]
        g = int(math.sqrt(n))
        spatial_features: Optional[torch.Tensor] = None
        if g * g == n:
            spatial_features = hidden_states.transpose(1, 2).reshape(B, -1, g, g)

        if not return_dict:
            return (
                hidden_states,
                tuple(all_hidden_states) if all_hidden_states else None,
                tuple(all_attentions) if all_attentions else None,
                pooler_output,
                hidden_states,
                spatial_features,
            )

        return VisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            attentions=tuple(all_attentions) if all_attentions else None,
            cls_token=pooler_output,
            patch_embeddings=hidden_states,
            spatial_features=spatial_features,
            pooler_output=pooler_output,
        )


# =============================================================================
# 4. ConvNeXtBlock
# =============================================================================


class ConvNeXtBlock(nn.Module):
    """ConvNeXt residual block.

    Architecture (following Liu et al., 2022):
        ``input → DepthwiseConv(7×7) → LayerNorm → 1×1 Conv(4× expand) → GELU
         → 1×1 Conv → LayerScale → + residual``

    Supports stochastic depth (drop path) and configurable layer-scale
    initialisation.

    Args:
        dim: Number of input and output channels.
        drop_path: Stochastic depth probability.
        layer_scale_init: Initial value for the per-channel learnable scale.
        kernel_size: Depthwise convolution kernel size (default 7).
        use_grn: Apply Global Response Normalisation (GRN) after GELU.
        act_layer: Activation function class (default ``nn.GELU``).
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        kernel_size: int = 7,
        use_grn: bool = False,
        act_layer: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dim = dim
        padding = kernel_size // 2

        # Depthwise 7×7 convolution
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            bias=True,
        )

        # Normalisation (Channel Last style applied as 2-D layer norm)
        self.norm = LayerNorm2D(dim, eps=1e-6)

        # Point-wise expansion
        expand_ratio = 4
        hidden_dim = int(dim * expand_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim, bias=True)

        # Activation
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = nn.GELU()

        # GRN (optional Global Response Normalisation)
        self.use_grn = use_grn
        if use_grn:
            self.grn = _GlobalResponseNorm(hidden_dim)
        else:
            self.grn = None

        # Point-wise reduction
        self.pwconv2 = nn.Linear(hidden_dim, dim, bias=True)

        # Layer scale
        self.layer_scale = LayerScale(dim, init_value=layer_scale_init)

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.dwconv.weight, std=0.02)
        if self.dwconv.bias is not None:
            nn.init.zeros_(self.dwconv.bias)
        nn.init.trunc_normal_(self.pwconv1.weight, std=0.02)
        nn.init.zeros_(self.pwconv1.bias)
        nn.init.trunc_normal_(self.pwconv2.weight, std=0.02)
        nn.init.zeros_(self.pwconv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map ``(B, C, H, W)``.

        Returns:
            Feature map ``(B, C, H, W)``.
        """
        shortcut = x

        x = self.dwconv(x)

        # Channel-last path: (B, C, H, W) → (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)

        if self.use_grn and self.grn is not None:
            x = self.grn(x)

        x = self.pwconv2(x)

        # Per-channel layer scale
        x = self.layer_scale(x.permute(0, 3, 1, 2))

        # Residual + stochastic depth
        x = shortcut + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class _GlobalResponseNorm(nn.Module):
    """Global Response Normalisation (GRN).

    Used in ConvNeXt-V2 for improved representation learning.  Computes
    a learnable affine combination of the input, its global variance, and
    its L2 norm across the spatial / batch dimensions.

    Args:
        dim: Channel dimension.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GRN.

        Args:
            x: Input of shape ``(B, H, W, C)`` or ``(B, C, H, W)``.

        Returns:
            Normalised tensor of same shape.
        """
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        x = self.gamma * (x * nx) + self.beta + x
        return x


# =============================================================================
# ConvNeXt Downsampling
# =============================================================================


class ConvNeXtDownsample(nn.Module):
    """Spatial down-sampling layer between ConvNeXt stages.

    Architecture: ``LayerNorm2D → Conv2d(stride=2, kernel=2) → Conv2d(1×1)``

    The first convolution halves the spatial resolution while the second
    adjusts the channel count.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNorm2D(in_channels, eps=1e-6)
        self.downsample = nn.Conv2d(
            in_channels, in_channels, kernel_size=2, stride=2, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.downsample.weight, std=0.02)
        nn.init.trunc_normal_(self.pointwise.weight, std=0.02)
        if self.pointwise.bias is not None:
            nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Down-sample *x* by factor 2 and change channel count.

        Args:
            x: ``(B, C_in, H, W)``.

        Returns:
            ``(B, C_out, H/2, W/2)``.
        """
        x = self.norm(x)
        x = self.downsample(x)
        x = self.pointwise(x)
        return x


class PatchifyStem(nn.Module):
    """ConvNeXt stem that converts an image into patch-level features.

    Uses a large-kernel convolution with ``stride=4`` to quickly reduce spatial
    resolution, followed by a point-wise 1×1 convolution for channel projection.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output channels.
        patch_size: Size of the stem convolution kernel (also the effective patch).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 96,
        patch_size: int = 4,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.norm = LayerNorm2D(out_channels, eps=1e-6)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify the input image.

        Args:
            x: ``(B, C_in, H, W)``.

        Returns:
            ``(B, C_out, H/patch_size, W/patch_size)``.
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


# =============================================================================
# 5. ConvNeXtEncoder
# =============================================================================


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt encoder with 4 hierarchical stages.

    Default configuration follows ConvNeXt-Base:
        - Stages: ``[3, 3, 9, 3]`` blocks
        - Dims:   ``[96, 192, 384, 768]`` channels
        - Downsampling between stages via :class:`ConvNeXtDownsample`
        - Stem: ``PatchifyStem`` with 4×4 conv, stride 4

    Supports stochastic depth (linearly increasing), GRN (ConvNeXt-V2), and
    gradient checkpointing.

    Args:
        config: A :class:`VisionEncoderConfig` instance (or ``None`` for defaults).
        in_channels: Number of input image channels.
        stem_patch_size: Stem convolution kernel/stride size.
        use_grn: Use Global Response Normalisation in each block.
        drop_path_rate: Maximum stochastic depth rate (linearly increased).
    """

    def __init__(
        self,
        config: Optional[VisionEncoderConfig] = None,
        in_channels: int = 3,
        stem_patch_size: int = 4,
        use_grn: bool = False,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        if config is None:
            config = VisionEncoderConfig()
        self.config = config
        cfg = config

        stages = cfg.convnext_stages
        dims = cfg.convnext_dims

        assert len(stages) == len(dims) == 4, (
            f"ConvNeXt requires exactly 4 stages, got {len(stages)} and {len(dims)}"
        )

        # Stem
        self.stem = PatchifyStem(
            in_channels=in_channels,
            out_channels=dims[0],
            patch_size=stem_patch_size,
        )

        # Build stages
        total_blocks = sum(stages)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        idx = 0
        for stage_idx in range(4):
            blocks = nn.ModuleList()
            for block_idx in range(stages[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_idx],
                        drop_path=dpr[idx],
                        layer_scale_init=cfg.convnext_layer_scale_init,
                        use_grn=use_grn,
                    )
                )
                idx += 1
            self.stages.append(blocks)

            # Downsampling between stages (not after the last)
            if stage_idx < 3:
                self.downsamples.append(
                    ConvNeXtDownsample(dims[stage_idx], dims[stage_idx + 1])
                )
            else:
                self.downsamples.append(nn.Identity())

        # Final layer norm
        self.final_norm = LayerNorm2D(dims[-1], eps=1e-6)

        self.gradient_checkpointing = cfg.use_checkpoint

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[MultiScaleFeatures, Tuple[MultiScaleFeatures, ...]]:
        """Forward pass through ConvNeXt encoder.

        Args:
            pixel_values: Image tensor ``(B, C, H, W)``.
            output_hidden_states: Return intermediate stage features.
            return_dict: Return structured :class:`MultiScaleFeatures`.

        Returns:
            :class:`MultiScaleFeatures` with per-stage outputs, or a tuple.
        """
        features = MultiScaleFeatures()
        all_features: List[MultiScaleFeatures] = []

        x = self.stem(pixel_values)
        stride = self.config.image_size // x.shape[-1]

        stage_names = ["stage1", "stage2", "stage3", "stage4"]
        strides = [stride, stride * 2, stride * 4, stride * 8]
        channels = list(self.config.convnext_dims)

        for i, (stage_blocks, downsample) in enumerate(
            zip(self.stages, self.downsamples)
        ):
            for block in stage_blocks:
                if self.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(
                        block, x, use_reentrant=False
                    )
                else:
                    x = block(x)

            features.features[stage_names[i]] = x
            features.strides[stage_names[i]] = strides[i]
            features.channels[stage_names[i]] = channels[i]

            if output_hidden_states:
                msf = MultiScaleFeatures(
                    features={stage_names[i]: x.clone()},
                    strides={stage_names[i]: strides[i]},
                    channels={stage_names[i]: channels[i]},
                )
                all_features.append(msf)

            if i < 3:
                x = downsample(x)

        x = self.final_norm(x)

        if return_dict:
            return features
        return tuple(all_features)

    def get_stage_dims(self) -> List[int]:
        """Return the channel dimension of each stage."""
        return list(self.config.convnext_dims)

    def get_num_params(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 6. EfficientViTBlock
# =============================================================================


class EfficientViTAttention(nn.Module):
    """Efficient attention for EfficientViT.

    Q and K are projected from a **reduced** dimensionality (divided by
    *reduction_ratio*), while V uses the full dimension.  This reduces the
    FLOPs of the attention computation from O(N^2 · D) to
    O(N^2 · D/r) without significant quality loss.

    Optionally uses a **linear attention** variant where the softmax is
    replaced by ``ReLU`` on Q, achieving O(N) complexity.

    Args:
        hidden_size: Full hidden dimension.
        key_dim: Dimension of key vectors.
        value_dim: Dimension of value vectors.
        num_heads: Number of attention heads.
        reduction_ratio: Ratio by which Q/K dimensionality is reduced.
        use_linear_attn: Use linear attention (ReLU kernel) instead of softmax.
        dropout: Dropout on attention weights.
    """

    def __init__(
        self,
        hidden_size: int,
        key_dim: int = 64,
        value_dim: int = 64,
        num_heads: int = 8,
        reduction_ratio: int = 4,
        use_linear_attn: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.reduction_ratio = reduction_ratio
        self.use_linear_attn = use_linear_attn
        self.head_key_dim = key_dim // num_heads
        self.head_value_dim = value_dim // num_heads

        self.scale = self.head_key_dim ** -0.5

        reduced_dim = hidden_size // reduction_ratio

        # Q/K projections (from reduced dim for efficiency)
        self.q = nn.Linear(hidden_size, num_heads * self.head_key_dim, bias=False)
        self.k = nn.Linear(hidden_size, num_heads * self.head_key_dim, bias=False)

        # V projection (full dim)
        self.v = nn.Linear(hidden_size, num_heads * self.head_value_dim, bias=False)

        # Output projection
        self.proj = nn.Linear(num_heads * self.head_value_dim, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Internal projector to reduced space (when using reduction)
        if reduction_ratio > 1:
            self.kv_reduce = nn.Linear(hidden_size, reduced_dim, bias=False)
        else:
            self.kv_reduce = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if hasattr(self.kv_reduce, "weight"):
            nn.init.xavier_uniform_(self.kv_reduce.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute efficient attention.

        Args:
            x: Input ``(B, L, D)``.
            attention_mask: Optional additive mask.
            output_attentions: Whether to return attention weights.

        Returns:
            ``(output, attn_weights)``.
        """
        B, L, _ = x.shape

        # Reduce input for K (and optionally Q)
        if self.reduction_ratio > 1:
            x_reduced = self.kv_reduce(x)
        else:
            x_reduced = x

        # Q from full, K from reduced
        q = self.q(x).view(B, L, self.num_heads, self.head_key_dim).transpose(1, 2)
        k = self.k(x_reduced).view(B, L, self.num_heads, self.head_key_dim).transpose(1, 2)
        v = self.v(x).view(B, L, self.num_heads, self.head_value_dim).transpose(1, 2)

        if self.use_linear_attn:
            # Linear attention: O(N) complexity
            # Use ReLU instead of softmax as kernel function
            q = F.relu(q)
            k = F.relu(k)

            # (B, H, L, dk) @ (B, H, dk, L) → (B, H, L, L)
            kv = torch.matmul(k.transpose(-2, -1), v)
            out = torch.matmul(q, kv)
            # Normalise by the sum of the keys per position
            k_sum = k.sum(dim=-1, keepdim=True) + 1e-6
            out = out / (k_sum.transpose(-2, -1) @ torch.ones_like(v))
            attn_weights = None
        else:
            # Standard softmax attention with reduced-key
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask
            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
            if self.training and isinstance(self.attn_dropout, nn.Dropout):
                attn_weights = self.attn_dropout(attn_weights)
            out = torch.matmul(attn_weights, v)
            if not output_attentions:
                attn_weights = None

        out = out.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_value_dim)
        out = self.proj(out)
        return out, attn_weights


class EfficientViTMLP(nn.Module):
    """MLP block for EfficientViT with optional local mixing.

    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (typically 2× in_features).
        out_features: Output dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EfficientViTBlock(nn.Module):
    """EfficientViT transformer block.

    Uses efficient attention where Q and K operate in a reduced dimension
    (controlled by ``reduction_ratio``), while V maintains the full dimension.
    An optional linear attention variant replaces softmax with ReLU for
    O(N) complexity.

    Architecture:
        ``LN → EfficientAttention → residual → LN → MLP → residual``

    Args:
        hidden_size: Hidden dimension of the block.
        key_dim: Key dimension for attention.
        value_dim: Value dimension for attention.
        num_heads: Number of attention heads.
        reduction_ratio: Ratio by which Q/K dimensions are reduced.
        mlp_ratio: FFN expansion factor.
        use_linear_attn: Use ReLU-based linear attention.
        drop_path: Stochastic depth rate.
        layer_norm_eps: Epsilon for layer normalisation.
        dropout: General dropout.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        key_dim: int = 64,
        value_dim: int = 64,
        num_heads: int = 8,
        reduction_ratio: int = 4,
        mlp_ratio: float = 2.0,
        use_linear_attn: bool = False,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_linear_attn = use_linear_attn

        self.norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attn = EfficientViTAttention(
            hidden_size=hidden_size,
            key_dim=key_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            reduction_ratio=reduction_ratio,
            use_linear_attn=use_linear_attn,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = EfficientViTMLP(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input ``(B, L, D)``.
            attention_mask: Optional attention mask.
            output_attentions: Whether to return attention weights.

        Returns:
            ``(output, attn_weights)``.
        """
        residual = x
        x = self.norm1(x)
        attn_out, attn_w = self.attn(x, attention_mask=attention_mask, output_attentions=output_attentions)
        x = residual + self.drop_path(attn_out)

        residual = x
        x = self.norm2(x)
        mlp_out = self.mlp(x)
        x = residual + self.drop_path(mlp_out)

        return x, attn_w

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"use_linear_attn={self.use_linear_attn}"
        )


# =============================================================================
# 7. ImageAugmentation
# =============================================================================


class ImageAugmentation(nn.Module):
    """Comprehensive image augmentation pipeline implemented from scratch.

    Every transform is implemented using pure PyTorch operations so that the
    pipeline is fully differentiable and GPU-accelerated.

    Supports the following augmentations (all randomly applied during training):

    - **RandomHorizontalFlip**: Mirror the image left-right.
    - **RandomResizedCrop**: Crop a random sub-region and resize.
    - **ColorJitter**: Adjust brightness, contrast, saturation, hue.
    - **RandomGrayscale**: Convert to grayscale with given probability.
    - **GaussianBlur**: Apply Gaussian blur with random kernel size.
    - **RandomSolarize**: Invert pixels above a random threshold.
    - **RandomPosterize**: Reduce bit-depth of pixel values.
    - **MixUp**: Convex combination of two images.
    - **CutMix**: Replace a random patch with a patch from another image.

    Args:
        hflip_prob: Probability of horizontal flip.
        resize_size: Target size for resize operations.
        crop_scale: ``(min, max)`` scale range for RandomResizedCrop.
        crop_ratio: ``(min, max)`` aspect ratio range for RandomResizedCrop.
        brightness: ``(min, max)`` brightness jitter range.
        contrast: ``(min, max)`` contrast jitter range.
        saturation: ``(min, max)`` saturation jitter range.
        hue: ``(min, max)`` hue jitter range.
        grayscale_prob: Probability of converting to grayscale.
        blur_prob: Probability of applying Gaussian blur.
        blur_kernel_size: Maximum kernel size for Gaussian blur.
        solarize_prob: Probability of applying solarization.
        solarize_threshold: Threshold for solarization.
        posterize_prob: Probability of applying posterization.
        posterize_bits: Number of bits to keep (1–8).
        mixup_alpha: Beta distribution parameter for MixUp (0 disables).
        cutmix_alpha: Beta distribution parameter for CutMix (0 disables).
        mixup_prob: Probability of applying MixUp instead of CutMix.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        resize_size: int = 224,
        crop_scale: Tuple[float, float] = (0.08, 1.0),
        crop_ratio: Tuple[float, float] = (0.75, 1.333),
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
        grayscale_prob: float = 0.2,
        blur_prob: float = 0.5,
        blur_kernel_size: int = 23,
        solarize_prob: float = 0.0,
        solarize_threshold: float = 0.5,
        posterize_prob: float = 0.0,
        posterize_bits: int = 4,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
    ):
        super().__init__()
        self.hflip_prob = hflip_prob
        self.resize_size = resize_size
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale_prob = grayscale_prob
        self.blur_prob = blur_prob
        self.blur_kernel_size = blur_kernel_size
        self.solarize_prob = solarize_prob
        self.solarize_threshold = solarize_threshold
        self.posterize_prob = posterize_prob
        self.posterize_bits = posterize_bits
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob

    # ------------------------------------------------------------------
    # Individual augmentation primitives
    # ------------------------------------------------------------------

    def random_horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        """Flip image horizontally with probability ``self.hflip_prob``.

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.

        Returns:
            Possibly flipped tensor.
        """
        if not self.training or self.hflip_prob <= 0.0:
            return x
        B = x.shape[0]
        mask = torch.rand(B, device=x.device) < self.hflip_prob
        x_flipped = x.flip(-1)
        x = torch.where(mask[:, None, None, None], x_flipped, x)
        return x

    def random_resized_crop(
        self,
        x: torch.Tensor,
        size: Optional[int] = None,
        scale: Optional[Tuple[float, float]] = None,
        ratio: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Randomly crop and resize images.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            size: Target output size (square). Defaults to ``self.resize_size``.
            scale: ``(min_scale, max_scale)`` for crop area.
            ratio: ``(min_ratio, max_ratio)`` for aspect ratio.

        Returns:
            Cropped-and-resized images ``(B, C, size, size)``.
        """
        if not self.training:
            target = size or self.resize_size
            return F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)

        B, C, H, W = x.shape
        target = size or self.resize_size
        sc = scale or self.crop_scale
        rt = ratio or self.crop_ratio

        out = torch.zeros(B, C, target, target, device=x.device, dtype=x.dtype)
        for i in range(B):
            area = H * W
            target_area = area * torch.empty(1, device=x.device).uniform_(sc[0], sc[1]).item()
            log_ratio = (math.log(rt[0]), math.log(rt[1]))
            aspect = math.exp(torch.empty(1, device=x.device).uniform_(log_ratio[0], log_ratio[1]).item())

            crop_w = int(round(math.sqrt(target_area * aspect)))
            crop_h = int(round(math.sqrt(target_area / aspect)))

            crop_w = min(crop_w, W)
            crop_h = min(crop_h, H)
            crop_w = max(crop_w, 1)
            crop_h = max(crop_h, 1)

            top = torch.randint(0, H - crop_h + 1, (1,)).item()
            left = torch.randint(0, W - crop_w + 1, (1,)).item()

            cropped = x[i : i + 1, :, top : top + crop_h, left : left + crop_w]
            out[i : i + 1] = F.interpolate(
                cropped, size=(target, target), mode="bilinear", align_corners=False
            )

        return out

    def color_jitter(
        self,
        x: torch.Tensor,
        brightness: Optional[float] = None,
        contrast: Optional[float] = None,
        saturation: Optional[float] = None,
        hue: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply random brightness, contrast, saturation, and hue jitter.

        All adjustments are applied independently.  Each is sampled uniformly
        from ``[1 - v, 1 + v]`` (or ``[0, v]`` for hue).

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.
            brightness: Brightness jitter magnitude.
            contrast: Contrast jitter magnitude.
            saturation: Saturation jitter magnitude.
            hue: Hue jitter magnitude (in radians, max ±0.5).

        Returns:
            Jittered image tensor.
        """
        if not self.training:
            return x
        br = brightness if brightness is not None else self.brightness
        co = contrast if contrast is not None else self.contrast
        sa = saturation if saturation is not None else self.saturation
        hu = hue if hue is not None else self.hue

        B, C, H, W = x.shape

        # Brightness
        if br > 0.0:
            factor = torch.empty(B, device=x.device).uniform_(max(0, 1 - br), 1 + br)
            x = x * factor[:, None, None, None].clamp(0.0, 2.0)

        # Contrast
        if co > 0.0:
            factor = torch.empty(B, device=x.device).uniform_(max(0, 1 - co), 1 + co)
            mean = x.mean(dim=[1, 2, 3], keepdim=True)
            x = (x - mean) * factor[:, None, None, None] + mean

        # Saturation (operates in a simplified HSV-like space on RGB)
        if sa > 0.0:
            factor = torch.empty(B, device=x.device).uniform_(max(0, 1 - sa), 1 + sa)
            gray = x.mean(dim=1, keepdim=True)
            x = gray + factor[:, None, None, None] * (x - gray)

        # Hue shift (rotate the RGB vector)
        if hu > 0.0:
            hue_factor = torch.empty(B, device=x.device).uniform_(-hu, hu).item()
            cos_a = math.cos(hue_factor * math.pi)
            sin_a = math.sin(hue_factor * math.pi)

            # Construct a simplified hue rotation matrix
            # This rotates the RGB vector around the (1,1,1) axis
            r = x[:, 0:1, :, :]
            g = x[:, 1:2, :, :]
            b = x[:, 2:3, :, :]

            # Simplified rotation approximation
            sr = r * cos_a + g * sin_a
            sg = g * cos_a - r * sin_a
            sb = b * cos_a + r * sin_a * 0.5

            x = torch.cat([sr, sg, sb], dim=1)

        x = x.clamp(0.0, 1.0)
        return x

    def random_grayscale(
        self, x: torch.Tensor, prob: Optional[float] = None
    ) -> torch.Tensor:
        """Convert images to grayscale with given probability.

        Uses the standard luminance weights ``(0.2989, 0.5870, 0.1140)``.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            prob: Probability per image. Defaults to ``self.grayscale_prob``.

        Returns:
            Possibly grayscaled tensor.
        """
        if not self.training:
            return x
        p = prob if prob is not None else self.grayscale_prob
        if p <= 0.0:
            return x
        B = x.shape[0]
        mask = torch.rand(B, device=x.device) < p
        # Luminance weights
        gray = (
            x[:, 0:1, :, :] * 0.2989
            + x[:, 1:2, :, :] * 0.5870
            + x[:, 2:3, :, :] * 0.1140
        )
        gray = gray.expand_as(x)
        x = torch.where(mask[:, None, None, None], gray, x)
        return x

    def gaussian_blur(
        self,
        x: torch.Tensor,
        kernel_size: Optional[int] = None,
        sigma: Optional[float] = None,
        prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply Gaussian blur with random kernel size.

        The blur kernel is constructed from scratch using the 2-D Gaussian
        function and applied as a separable convolution for efficiency.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            kernel_size: Maximum kernel size (must be odd). Defaults to ``self.blur_kernel_size``.
            sigma: Standard deviation. ``None`` → auto-computed from kernel size.
            prob: Probability of applying. Defaults to ``self.blur_prob``.

        Returns:
            Possibly blurred tensor.
        """
        if not self.training:
            return x
        p = prob if prob is not None else self.blur_prob
        if p <= 0.0:
            return x
        ks = kernel_size or self.blur_kernel_size

        B, C, H, W = x.shape
        # Choose a random kernel size (odd, >= 3)
        possible_sizes = [s for s in range(3, ks + 1) if s % 2 == 1]
        if not possible_sizes:
            return x
        chosen_ks = possible_sizes[torch.randint(0, len(possible_sizes), (1,)).item()]

        # Build 1-D Gaussian kernel
        if sigma is None:
            sigma = float(chosen_ks) / 6.0
        coords = torch.arange(chosen_ks, device=x.device, dtype=torch.float32) - (chosen_ks - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        # Apply as separable conv (channel-wise with groups)
        padding = chosen_ks // 2
        # Reshape for batched 1-D conv: (B, C, H*W) @ kernel
        x_flat = x.permute(0, 1, 3, 2).reshape(B * C, 1, W, H)
        # Horizontal blur
        g_h = g.view(1, 1, -1, 1).expand(B * C, 1, -1, 1)
        x_flat = F.conv2d(x_flat, g_h, padding=(padding, 0), groups=B * C)
        # Vertical blur
        g_v = g.view(1, 1, 1, -1).expand(B * C, 1, 1, -1)
        x_flat = F.conv2d(x_flat, g_v, padding=(0, padding), groups=B * C)
        x_blurred = x_flat.reshape(B, C, W, H).permute(0, 1, 3, 2)

        # Apply with probability
        mask = torch.rand(B, device=x.device) < p
        x = torch.where(mask[:, None, None, None], x_blurred, x)
        return x

    def random_solarize(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Invert pixels above a threshold.

        For each pixel, if its value > threshold, replace with (1 - value).

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.
            threshold: Solarization threshold. ``None`` → ``self.solarize_threshold``.
            prob: Probability of applying.

        Returns:
            Possibly solarized tensor.
        """
        if not self.training:
            return x
        p = prob if prob is not None else self.solarize_prob
        if p <= 0.0:
            return x
        th = threshold if threshold is not None else self.solarize_threshold

        B = x.shape[0]
        mask = torch.rand(B, device=x.device) < p
        # Solarize: invert pixels above threshold
        solarized = torch.where(x > th, 1.0 - x, x)
        x = torch.where(mask[:, None, None, None], solarized, x)
        return x

    def random_posterize(
        self,
        x: torch.Tensor,
        bits: Optional[int] = None,
        prob: Optional[float] = None,
    ) -> torch.Tensor:
        """Reduce the bit-depth of pixel values.

        Pixels are quantised to the nearest multiple of ``1 / 2^bits``.

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.
            bits: Number of bits to keep (1–8). ``None`` → ``self.posterize_bits``.
            prob: Probability of applying.

        Returns:
            Possibly posterized tensor.
        """
        if not self.training:
            return x
        p = prob if prob is not None else self.posterize_prob
        if p <= 0.0:
            return x
        b = bits if bits is not None else self.posterize_bits
        levels = 2 ** b
        factor = 1.0 / levels

        B = x.shape[0]
        mask = torch.rand(B, device=x.device) < p
        posterized = (x / factor).floor() * factor
        x = torch.where(mask[:, None, None, None], posterized, x)
        return x

    def mixup(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply MixUp augmentation: linear interpolation between two images.

        ``x_mix = lambda * x_i + (1 - lambda) * x_j`` where
        ``lambda ~ Beta(alpha, alpha)``.

        Args:
            x: ``(B, C, H, W)`` image batch.
            labels: Optional label tensor ``(B,)``.
            alpha: Beta distribution parameter. ``None`` → ``self.mixup_alpha``.

        Returns:
            ``(mixed_x, mixed_labels)``.
        """
        if not self.training:
            return x, labels
        a = alpha if alpha is not None else self.mixup_alpha
        if a <= 0.0:
            return x, labels

        B = x.shape[0]
        if B < 2:
            return x, labels

        lam = torch.distributions.Beta(a, a).sample((B,)).to(x.device)
        lam = lam.view(B, 1, 1, 1)

        # Shuffle indices
        perm = torch.randperm(B, device=x.device)
        x_shuffled = x[perm]

        x_mix = lam * x + (1.0 - lam) * x_shuffled

        mixed_labels: Optional[torch.Tensor] = None
        if labels is not None:
            mixed_labels = lam.squeeze() * labels + (1.0 - lam.squeeze()) * labels[perm]

        return x_mix, mixed_labels

    def cutmix(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply CutMix augmentation: replace a random patch with one from another image.

        Args:
            x: ``(B, C, H, W)`` image batch.
            labels: Optional label tensor ``(B,)``.
            alpha: Beta distribution parameter. ``None`` → ``self.cutmix_alpha``.

        Returns:
            ``(mixed_x, mixed_labels)``.
        """
        if not self.training:
            return x, labels
        a = alpha if alpha is not None else self.cutmix_alpha
        if a <= 0.0:
            return x, labels

        B, C, H, W = x.shape
        if B < 2:
            return x, labels

        lam = torch.distributions.Beta(a, a).sample().to(x.device)
        perm = torch.randperm(B, device=x.device)

        # Generate random bounding box
        cut_ratio = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        cut_w = min(cut_w, W)
        cut_h = min(cut_h, H)

        cx = torch.randint(0, W, (1,)).item()
        cy = torch.randint(0, H, (1,)).item()

        # Bounding box coordinates
        x1 = max(0, cx - cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        x2 = min(W, cx + cut_w // 2)
        y2 = min(H, cy + cut_h // 2)

        # Compute adjusted lambda based on actual box area
        actual_area = (x2 - x1) * (y2 - y1)
        total_area = H * W
        lam_adjusted = 1.0 - actual_area / total_area

        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]

        mixed_labels: Optional[torch.Tensor] = None
        if labels is not None:
            mixed_labels = lam_adjusted * labels + (1.0 - lam_adjusted) * labels[perm]

        return x_mix, mixed_labels

    def apply_mixup_or_cutmix(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Randomly apply either MixUp or CutMix.

        Args:
            x: ``(B, C, H, W)`` images.
            labels: Optional labels ``(B,)``.

        Returns:
            ``(augmented_x, augmented_labels)``.
        """
        if not self.training:
            return x, labels

        # Decide whether to apply anything
        if self.mixup_alpha <= 0 and self.cutmix_alpha <= 0:
            return x, labels

        r = torch.rand(1).item()
        if r < self.mixup_prob and self.mixup_alpha > 0:
            return self.mixup(x, labels)
        elif self.cutmix_alpha > 0:
            return self.cutmix(x, labels)
        return x, labels

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the full augmentation pipeline.

        During training, applies (in order):
            1. RandomResizedCrop
            2. RandomHorizontalFlip
            3. ColorJitter
            4. RandomGrayscale
            5. GaussianBlur
            6. RandomSolarize
            7. RandomPosterize
            8. MixUp / CutMix

        During evaluation, no augmentations are applied.

        Args:
            x: Image tensor ``(B, C, H, W)`` in ``[0, 1]``.
            labels: Optional label tensor ``(B,)``.

        Returns:
            ``(augmented_x, augmented_labels)``.
        """
        # Spatial augmentations
        x = self.random_resized_crop(x)
        x = self.random_horizontal_flip(x)

        # Colour augmentations
        x = self.color_jitter(x)
        x = self.random_grayscale(x)
        x = self.gaussian_blur(x)
        x = self.random_solarize(x)
        x = self.random_posterize(x)

        # Mix-level augmentations
        x, labels = self.apply_mixup_or_cutmix(x, labels)

        return x, labels


# =============================================================================
# 8. ResolutionAdaptor
# =============================================================================


class ResolutionAdaptor(nn.Module):
    """Adapt a ViT encoder for different input resolutions.

    Provides utilities to:
        - Interpolate learned position embeddings to new grid sizes.
        - Adjust attention masks for the new sequence lengths.
        - Handle CLS token placement when the resolution changes.

    This is essential for deploying ViT models on images with resolutions
    different from what they were trained on (e.g., fine-tuning on higher
    resolution for dense prediction tasks).

    Args:
        original_grid_size: Grid size the model was trained with.
        embed_dim: Embedding dimension of the position embeddings.
        use_cls_token: Whether the model uses a CLS token.
        num_registers: Number of register tokens (0 if none).
    """

    def __init__(
        self,
        original_grid_size: int,
        embed_dim: int,
        use_cls_token: bool = True,
        num_registers: int = 0,
    ):
        super().__init__()
        self.original_grid_size = original_grid_size
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.num_registers = num_registers

    def interpolate_position_embeddings(
        self,
        pos_embed: torch.Tensor,
        target_grid_size: int,
    ) -> torch.Tensor:
        """Interpolate 2-D position embeddings to a new grid size.

        Args:
            pos_embed: Position embedding tensor ``(1, num_tokens, D)``.
            target_grid_size: Desired grid size (height = width).

        Returns:
            Interpolated position embedding of shape
            ``(1, new_num_tokens, D)``.
        """
        D = pos_embed.shape[-1]

        # Separate CLS and register tokens
        offset = 1 if self.use_cls_token else 0
        reg_offset = self.num_registers

        if self.use_cls_token:
            cls_pe = pos_embed[:, :1, :]
            patch_pe = pos_embed[:, 1:]
        else:
            cls_pe = None
            patch_pe = pos_embed

        # Remove register tokens (they are at the end)
        if reg_offset > 0:
            patch_pe = patch_pe[:, :-reg_offset]

        src_g = self.original_grid_size
        src_n = src_g * src_g
        assert patch_pe.shape[1] == src_n, (
            f"Expected {src_n} patch positions, got {patch_pe.shape[1]}"
        )

        # Reshape to 2-D: (1, D, H, W)
        pe_2d = patch_pe.reshape(1, src_g, src_g, D).permute(0, 3, 1, 2)

        # Bicubic interpolation
        pe_2d = F.interpolate(
            pe_2d,
            size=(target_grid_size, target_grid_size),
            mode="bicubic",
            align_corners=False,
        )

        # Flatten back to sequence
        pe_new = pe_2d.permute(0, 2, 3, 1).reshape(1, target_grid_size * target_grid_size, D)

        # Re-attach CLS token
        if cls_pe is not None:
            pe_new = torch.cat([cls_pe, pe_new], dim=1)

        return pe_new

    def create_attention_mask(
        self,
        target_seq_len: int,
        original_seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        mask_type: str = "bidirectional",
    ) -> torch.Tensor:
        """Create an attention mask for the target sequence length.

        Args:
            target_seq_len: New sequence length.
            original_seq_len: Original sequence length (unused, for API compat).
            device: Target device.
            dtype: Data type for the mask.
            mask_type: ``"bidirectional"`` (all attend to all) or
                ``"causal"`` (lower-triangular).

        Returns:
            Attention mask tensor ``(1, 1, target_seq_len, target_seq_len)``.
        """
        if mask_type == "bidirectional":
            mask = torch.zeros(1, 1, target_seq_len, target_seq_len, device=device, dtype=dtype)
        elif mask_type == "causal":
            mask = torch.triu(
                torch.full(
                    (target_seq_len, target_seq_len),
                    float("-inf"),
                    device=device,
                    dtype=dtype,
                ),
                diagonal=1,
            )
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Unknown mask_type: {mask_type!r}")
        return mask

    def adjust_encoder(
        self,
        encoder: nn.Module,
        target_grid_size: int,
    ) -> None:
        """In-place adjust the encoder's position embeddings.

        Modifies the ``patch_embed.pos_embed`` parameter of the given encoder
        to match the target grid size.

        Args:
            encoder: A :class:`ViTEncoder` or similar with a ``patch_embed``
                attribute containing ``pos_embed``.
            target_grid_size: New grid size.
        """
        patch_embed = encoder.patch_embed  # type: ignore[attr-defined]
        if hasattr(patch_embed, "pos_embed"):
            if isinstance(patch_embed.pos_embed, nn.Parameter):
                with torch.no_grad():
                    new_pe = self.interpolate_position_embeddings(
                        patch_embed.pos_embed.data, target_grid_size
                    )
                    patch_embed.pos_embed.data.copy_(new_pe)
            else:
                # Buffer (sinusoidal) — rebuild
                new_pe = self.interpolate_position_embeddings(
                    patch_embed.pos_embed, target_grid_size
                )
                patch_embed.pos_embed = new_pe.to(device=patch_embed.pos_embed.device)

        # Update grid size
        if hasattr(patch_embed, "grid_size"):
            patch_embed.grid_size = target_grid_size
        if hasattr(patch_embed, "num_patches"):
            patch_embed.num_patches = target_grid_size * target_grid_size

    def compute_grid_size(
        self,
        image_size: int,
        patch_size: int,
        stride: Optional[int] = None,
        padding: int = 0,
    ) -> int:
        """Compute the grid size for a given image size and patch configuration.

        Args:
            image_size: Image height/width.
            patch_size: Patch (kernel) size.
            stride: Convolution stride. ``None`` → *patch_size*.
            padding: Convolution padding.

        Returns:
            Grid size (integer).
        """
        s = stride if stride is not None else patch_size
        return (image_size + 2 * padding - patch_size) // s + 1

    def forward(
        self,
        pixel_values: torch.Tensor,
        encoder: nn.Module,
        patch_size: int = 16,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, int]:
        """Adapt inputs and run the encoder at a potentially new resolution.

        Args:
            pixel_values: Image tensor ``(B, C, H, W)``.
            encoder: The ViT encoder to run.
            patch_size: Patch size of the encoder.
            **kwargs: Additional keyword arguments forwarded to ``encoder.forward()``.

        Returns:
            ``(output, target_grid_size)`` where output is a
            :class:`VisionEncoderOutput`.
        """
        B, C, H, W = pixel_values.shape
        stride = getattr(encoder.patch_embed, "stride", patch_size)  # type: ignore[attr-defined]
        padding = getattr(encoder.patch_embed, "padding", 0)  # type: ignore[attr-defined]
        target_grid = self.compute_grid_size(H, patch_size, stride, padding)

        # Interpolate position embeddings if needed
        if target_grid != self.original_grid_size:
            self.adjust_encoder(encoder, target_grid)

        output = encoder(pixel_values, **kwargs)
        return output, target_grid


# =============================================================================
# 9. ImagePreprocessor
# =============================================================================


class ImagePreprocessor(nn.Module):
    """Image preprocessing pipeline for vision encoders.

    Applies:
        1. Resize to target resolution.
        2. Center crop.
        3. Normalise with ImageNet statistics.
        4. Convert to tensor in ``[0, 1]`` range.

    All operations are implemented using PyTorch ops so the pipeline can
    run on GPU and is fully differentiable.

    Args:
        resize_size: Target size for resizing (square).
        center_crop_size: Size of the center crop (square). ``None`` → same as resize.
        mean: Per-channel mean for normalisation.
        std: Per-channel std for normalisation.
        resample: Interpolation mode (``"bilinear"`` or ``"bicubic"``).
    """

    def __init__(
        self,
        resize_size: int = 256,
        center_crop_size: Optional[int] = None,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        resample: str = "bilinear",
    ):
        super().__init__()
        self.resize_size = resize_size
        self.center_crop_size = center_crop_size if center_crop_size is not None else resize_size
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.resample = resample

    def _to_float(self, x: torch.Tensor) -> torch.Tensor:
        """Convert uint8 tensor to float ``[0, 1]``.

        Args:
            x: Tensor of any dtype.

        Returns:
            Float tensor in ``[0, 1]``.
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.dtype != torch.float32 and x.dtype != torch.float16:
            x = x.float()
        return x

    def resize(
        self,
        x: torch.Tensor,
        size: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """Resize the shorter edge to *size* while preserving aspect ratio.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            size: Target shorter-edge size. ``None`` → ``self.resize_size``.
            mode: Interpolation mode.

        Returns:
            Resized tensor.
        """
        target = size or self.resize_size
        m = mode or self.resample
        B, C, H, W = x.shape

        # Resize shorter edge to target, longer edge proportionally
        if H < W:
            new_h = target
            new_w = int(W * target / H)
        elif W < H:
            new_w = target
            new_h = int(H * target / W)
        else:
            new_h = new_w = target

        # Ensure dimensions are at least 1
        new_h = max(new_h, 1)
        new_w = max(new_w, 1)

        align_corners = m == "bicubic"
        x = F.interpolate(x, size=(new_h, new_w), mode=m, align_corners=align_corners)
        return x

    def center_crop(
        self,
        x: torch.Tensor,
        size: Optional[int] = None,
    ) -> torch.Tensor:
        """Crop the center region of the image.

        Args:
            x: ``(B, C, H, W)`` image tensor.
            size: Crop size (square). ``None`` → ``self.center_crop_size``.

        Returns:
            Cropped tensor ``(B, C, size, size)``.
        """
        target = size or self.center_crop_size
        B, C, H, W = x.shape

        # If image is smaller than crop, pad
        if H < target or W < target:
            pad_h = max(0, target - H)
            pad_w = max(0, target - W)
            # Pad with the mean pixel value
            mean_val = self.mean.mean().item()
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=mean_val)
            H, W = x.shape[2], x.shape[3]

        top = (H - target) // 2
        left = (W - target) // 2
        x = x[:, :, top : top + target, left : left + target]
        return x

    def normalize(
        self,
        x: torch.Tensor,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ) -> torch.Tensor:
        """Normalize the image using per-channel mean and std.

        Args:
            x: ``(B, C, H, W)`` image tensor in ``[0, 1]``.
            mean: Per-channel mean. ``None`` → ImageNet defaults.
            std: Per-channel std. ``None`` → ImageNet defaults.

        Returns:
            Normalized tensor.
        """
        if mean is not None:
            m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        else:
            m = self.mean.to(device=x.device, dtype=x.dtype)
        if std is not None:
            s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        else:
            s = self.std.to(device=x.device, dtype=x.dtype)
        return (x - m) / s

    def forward(
        self,
        x: torch.Tensor,
        resize_size: Optional[int] = None,
        center_crop_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Preprocess images for a vision encoder.

        Steps:
            1. Convert to float ``[0, 1]``.
            2. Resize (shorter edge).
            3. Center crop.
            4. Normalise (ImageNet stats).

        Args:
            x: Image tensor ``(B, C, H, W)``. Can be ``uint8`` or float.
            resize_size: Override target resize size.
            center_crop_size: Override target crop size.

        Returns:
            Preprocessed tensor ``(B, C, crop_size, crop_size)``.
        """
        x = self._to_float(x)
        x = self.resize(x, size=resize_size)
        x = self.center_crop(x, size=center_crop_size)
        x = self.normalize(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"resize_size={self.resize_size}, "
            f"center_crop_size={self.center_crop_size}, "
            f"resample={self.resample}"
        )


# =============================================================================
# Factory helpers
# =============================================================================


def create_vit_encoder(
    image_size: int = 224,
    patch_size: int = 16,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    use_cls_token: bool = True,
    pos_embed_type: str = "learned",
    use_flash_attn: bool = False,
    use_checkpoint: bool = False,
) -> ViTEncoder:
    """Create a ViT encoder with common presets.

    Args:
        image_size: Input image resolution.
        patch_size: Patch size.
        hidden_size: Hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout rate.
        drop_path_rate: Max stochastic depth rate.
        use_cls_token: Prepend CLS token.
        pos_embed_type: Position embedding type.
        use_flash_attn: Try flash attention.
        use_checkpoint: Gradient checkpointing.

    Returns:
        A configured :class:`ViTEncoder`.
    """
    config = VisionEncoderConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout,
        drop_path_rate=drop_path_rate,
        use_cls_token=use_cls_token,
        pos_embed_type=pos_embed_type,
        use_flash_attn=use_flash_attn,
        use_checkpoint=use_checkpoint,
    )
    return ViTEncoder(config)


def create_siglip_encoder(
    image_size: int = 224,
    patch_size: int = 14,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    pos_embed_type: str = "sinusoidal",
    norm_type: str = "pre_norm",
    use_checkpoint: bool = False,
) -> SigLIPEncoder:
    """Create a SigLIP encoder with common presets.

    Args:
        image_size: Input image resolution.
        patch_size: Patch size (SigLIP uses 14×14).
        hidden_size: Hidden dimension.
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout rate.
        drop_path_rate: Max stochastic depth rate.
        pos_embed_type: Position embedding type.
        norm_type: Pre-norm or post-norm.
        use_checkpoint: Gradient checkpointing.

    Returns:
        A configured :class:`SigLIPEncoder`.
    """
    config = VisionEncoderConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout_rate=dropout,
        drop_path_rate=drop_path_rate,
        pos_embed_type=pos_embed_type,
        use_cls_token=False,
        use_checkpoint=use_checkpoint,
    )
    return SigLIPEncoder(config=config, norm_type=norm_type)


def create_convnext_encoder(
    image_size: int = 224,
    in_channels: int = 3,
    depths: Tuple[int, ...] = (3, 3, 9, 3),
    dims: Tuple[int, ...] = (96, 192, 384, 768),
    drop_path_rate: float = 0.0,
    use_grn: bool = False,
    use_checkpoint: bool = False,
) -> ConvNeXtEncoder:
    """Create a ConvNeXt encoder with common presets.

    Args:
        image_size: Input image resolution.
        in_channels: Number of input channels.
        depths: Number of blocks per stage.
        dims: Channel dimension per stage.
        drop_path_rate: Max stochastic depth rate.
        use_grn: Use Global Response Normalisation.
        use_checkpoint: Gradient checkpointing.

    Returns:
        A configured :class:`ConvNeXtEncoder`.
    """
    config = VisionEncoderConfig(
        image_size=image_size,
        in_channels=in_channels,
        convnext_stages=depths,
        convnext_dims=dims,
        drop_path_rate=drop_path_rate,
        use_checkpoint=use_checkpoint,
    )
    return ConvNeXtEncoder(
        config=config,
        in_channels=in_channels,
        use_grn=use_grn,
        drop_path_rate=drop_path_rate,
    )


def create_efficient_vit_block(
    hidden_size: int = 256,
    key_dim: int = 64,
    value_dim: int = 64,
    num_heads: int = 8,
    reduction_ratio: int = 4,
    mlp_ratio: float = 2.0,
    use_linear_attn: bool = False,
    drop_path: float = 0.0,
) -> EfficientViTBlock:
    """Create an EfficientViT block with common presets.

    Args:
        hidden_size: Hidden dimension.
        key_dim: Key dimension for attention.
        value_dim: Value dimension for attention.
        num_heads: Number of attention heads.
        reduction_ratio: Q/K dimension reduction ratio.
        mlp_ratio: MLP expansion factor.
        use_linear_attn: Use linear attention variant.
        drop_path: Stochastic depth rate.

    Returns:
        A configured :class:`EfficientViTBlock`.
    """
    return EfficientViTBlock(
        hidden_size=hidden_size,
        key_dim=key_dim,
        value_dim=value_dim,
        num_heads=num_heads,
        reduction_ratio=reduction_ratio,
        mlp_ratio=mlp_ratio,
        use_linear_attn=use_linear_attn,
        drop_path=drop_path,
    )
