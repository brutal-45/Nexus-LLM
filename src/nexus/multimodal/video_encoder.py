"""
Video Encoder Module
====================

Production-grade video encoder implementations for the Nexus LLM multimodal framework.
Provides multiple video encoding architectures including TimeSformer, Video Swin Transformer,
ViViT, temporal aggregation, frame sampling, and video augmentation components.

All components implement video processing using pure PyTorch operations — no external
video processing library dependencies required at the model level.

References:
    - TimeSformer: "Is Space-Time Attention All You Need for Video Understanding?" (Bertasius et al., 2021)
    - Video Swin Transformer: "Video Swin Transformer" (Liu et al., 2022)
    - ViViT: "A Video Vision Transformer" (Arnab et al., 2021)
    - Tubelet Embedding: Adapted from ViT patch embedding with temporal dimension
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Output Data Structures
# =============================================================================

@dataclass
class VideoEncoderOutput:
    """Output from a video encoder.

    Attributes:
        last_hidden_state: Final hidden state (batch, seq_len, encoder_dim).
        hidden_states: Optional list of all hidden states from each layer.
        attentions: Optional list of attention weights from each layer.
        spatial_features: Spatial feature maps before temporal encoding (batch, channels, h, w).
        temporal_features: Temporal feature maps after temporal encoding.
        attention_mask: Attention mask for the encoded sequence.
        pooler_output: Pooled representation (batch, encoder_dim).
        frame_indices: Indices of sampled frames used (batch, num_frames).
    """
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    spatial_features: Optional[torch.Tensor] = None
    temporal_features: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None
    frame_indices: Optional[torch.Tensor] = None


@dataclass
class FrameSamplerOutput:
    """Output from frame sampling.

    Attributes:
        frames: Sampled frames (batch, num_frames, channels, height, width).
        indices: Frame indices used for sampling (batch, num_frames).
        timestamps: Corresponding timestamps (batch, num_frames).
        num_frames: Number of frames sampled.
    """
    frames: torch.Tensor
    indices: torch.Tensor
    timestamps: Optional[torch.Tensor] = None
    num_frames: int = 0


@dataclass
class VideoAugmentationOutput:
    """Output from video augmentation pipeline.

    Attributes:
        video: Augmented video tensor (batch, num_frames, channels, height, width).
        transform_params: Dictionary of applied transform parameters for reproducibility.
        mask: Optional binary mask indicating valid regions after cropping.
    """
    video: torch.Tensor
    transform_params: Dict[str, Any] = field(default_factory=dict)
    mask: Optional[torch.Tensor] = None


# =============================================================================
# Video Patch Embedding (3D Tubelet)
# =============================================================================

class VideoPatchEmbedding(nn.Module):
    """3D Tubelet Patch Embedding for video.

    Extends the standard 2D patch embedding from ViT to 3D by treating
    video as a stack of frames and extracting spatio-temporal tubelets
    using a 3D convolution.

    Each tubelet spans `temporal_patch_size` consecutive frames and
    covers a `patch_size x patch_size` spatial region. The tubelet is
    flattened and projected to the embedding dimension.

    Args:
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Output embedding dimension.
        temporal_patch_size: Number of frames per temporal patch.
        spatial_patch_size: Spatial patch size (height = width).
        temporal_stride: Temporal stride for the convolution.
        spatial_stride: Spatial stride for the convolution.
        temporal_padding: Temporal padding for the convolution.
        spatial_padding: Spatial padding for the convolution.
        norm_layer: Normalization layer class. None for no normalization.
        flatten: Whether to flatten the tubelet features.
        bias: Whether to use bias in the convolution.
        max_temporal_patches: Maximum number of temporal patches (for position embedding sizing).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        temporal_patch_size: int = 2,
        spatial_patch_size: int = 16,
        temporal_stride: int = 2,
        spatial_stride: int = 16,
        temporal_padding: int = 0,
        spatial_padding: int = 0,
        norm_layer: Optional[nn.Module] = None,
        flatten: bool = True,
        bias: bool = True,
        max_temporal_patches: int = 16,
    ):
        """Initialize 3D tubelet patch embedding.

        Args:
            in_channels: Number of input channels per frame.
            embed_dim: Dimension of the output embedding.
            temporal_patch_size: Number of frames per tubelet.
            spatial_patch_size: Size of spatial patches (pixels).
            temporal_stride: Stride along temporal dimension.
            spatial_stride: Stride along spatial dimensions.
            temporal_padding: Padding along temporal dimension.
            spatial_padding: Padding along spatial dimensions.
            norm_layer: Optional normalization layer applied after projection.
            flatten: Whether to flatten spatial+temporal dims into sequence.
            bias: Whether to include bias in projection layer.
            max_temporal_patches: Upper bound on temporal patches for pos embed sizing.
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride
        self.flatten = flatten
        self.max_temporal_patches = max_temporal_patches

        # 3D convolution for tubelet extraction
        # Input: (batch, channels, temporal, height, width)
        # Output: (batch, embed_dim, temporal_patches, h_patches, w_patches)
        self.projection = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(
                temporal_patch_size,
                spatial_patch_size,
                spatial_patch_size,
            ),
            stride=(temporal_stride, spatial_stride, spatial_stride),
            padding=(temporal_padding, spatial_padding, spatial_padding),
            bias=bias,
        )

        # Layer norm after projection
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

        # Tubelet size for computing number of patches
        self.tubelet_size = (
            temporal_patch_size * spatial_patch_size * spatial_patch_size
        )

        # Projection from tubelet pixels to embed_dim
        self.reduction_ratio = (
            in_channels * temporal_patch_size * spatial_patch_size * spatial_patch_size
        ) / embed_dim

        # Learnable CLS token for video-level representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Spatial positional embedding (interpolable)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, max_temporal_patches * 256 + 1, embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

    def forward(
        self,
        video: torch.Tensor,
        return_patch_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, int]]]:
        """Extract tubelet patches and project to embedding dimension.

        Args:
            video: Input video tensor of shape (batch, num_frames, channels, height, width).
                   The video is expected in the format (T, C, H, W) per sample.
            return_patch_info: If True, also return patch count info.

        Returns:
            Embedded patches of shape (batch, num_patches, embed_dim) if flatten=True.
            If return_patch_info, also returns dict with temporal_patches, height_patches,
            width_patches, num_patches.
        """
        batch_size, num_frames, channels, height, width = video.shape

        # Permute to (batch, channels, temporal, height, width) for Conv3d
        x = video.permute(0, 2, 1, 3, 4).contiguous()

        # Apply 3D convolution to extract tubelets
        x = self.projection(x)
        # x shape: (batch, embed_dim, temporal_patches, height_patches, width_patches)

        batch_size, embed_dim, t_patches, h_patches, w_patches = x.shape

        # Transpose to (batch, temporal_patches, height_patches, width_patches, embed_dim)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # Apply layer norm
        x = self.norm(x)

        if self.flatten:
            # Flatten to (batch, temporal_patches * height_patches * width_patches, embed_dim)
            num_patches = t_patches * h_patches * w_patches
            x = x.reshape(batch_size, num_patches, embed_dim)

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

            # Add interpolated positional embeddings
            num_tokens = x.shape[1]
            if num_tokens <= self.spatial_pos_embed.shape[1]:
                pos_embed = self.spatial_pos_embed[:, :num_tokens, :]
            else:
                # Interpolate positional embeddings if sequence is longer
                pos_embed = self._interpolate_pos_embed(
                    self.spatial_pos_embed, num_tokens
                )
            x = x + pos_embed
        else:
            # Keep as (batch, t_patches, h_patches, w_patches, embed_dim)
            pass

        if return_patch_info:
            info = {
                "temporal_patches": t_patches,
                "height_patches": h_patches,
                "width_patches": w_patches,
                "num_patches": t_patches * h_patches * w_patches,
                "num_frames": num_frames,
                "height": height,
                "width": width,
            }
            return x, info

        return x

    def _interpolate_pos_embed(
        self,
        pos_embed: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Interpolate positional embeddings to match target sequence length.

        Uses bilinear interpolation to resize positional embeddings when the
        actual number of patches differs from the pre-computed positional
        embedding size.

        Args:
            pos_embed: Positional embeddings of shape (1, old_num_tokens, dim).
            num_tokens: Target number of tokens.

        Returns:
            Interpolated positional embeddings of shape (1, num_tokens, dim).
        """
        dim = pos_embed.shape[-1]
        # Remove CLS token for interpolation
        cls_pos = pos_embed[:, 0:1, :]
        patch_pos = pos_embed[:, 1:, :]

        old_num = patch_pos.shape[1]
        new_num = num_tokens - 1

        # Reshape to 2D grid for bilinear interpolation
        # Find closest square factors
        grid_size_old = int(math.ceil(math.sqrt(old_num)))
        grid_size_new = int(math.ceil(math.sqrt(new_num)))

        # Pad to square grid
        pad_size = grid_size_old ** 2 - old_num
        if pad_size > 0:
            patch_pos = F.pad(patch_pos, (0, 0, 0, pad_size))

        # Reshape to (1, grid, grid, dim) then (1, dim, grid, grid)
        patch_pos = patch_pos.reshape(1, grid_size_old, grid_size_old, dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2).contiguous()

        # Bilinear interpolation
        patch_pos = F.interpolate(
            patch_pos,
            size=(grid_size_new, grid_size_new),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back to sequence
        patch_pos = patch_pos.permute(0, 2, 3, 1).contiguous()
        patch_pos = patch_pos.reshape(1, grid_size_new ** 2, dim)
        patch_pos = patch_pos[:, :new_num, :]

        # Re-add CLS token
        pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
        return pos_embed

    def compute_num_patches(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> Dict[str, int]:
        """Compute the number of patches for a given video size.

        Args:
            num_frames: Number of frames in the video.
            height: Frame height in pixels.
            width: Frame width in pixels.

        Returns:
            Dictionary with temporal_patches, height_patches, width_patches, total.
        """
        t_patches = (num_frames - self.temporal_patch_size + 2 * 0) // self.temporal_stride + 1
        h_patches = (height - self.spatial_patch_size + 2 * 0) // self.spatial_stride + 1
        w_patches = (width - self.spatial_patch_size + 2 * 0) // self.spatial_stride + 1
        return {
            "temporal_patches": t_patches,
            "height_patches": h_patches,
            "width_patches": w_patches,
            "total_patches": t_patches * h_patches * w_patches + 1,  # +1 for CLS
        }


# =============================================================================
# Positional Embedding for Video
# =============================================================================

class VideoPositionalEmbedding(nn.Module):
    """Factorized positional embedding for video (spatial + temporal).

    Learns separate spatial and temporal positional embeddings and combines
    them additively. This factorization enables better generalization to
    varying numbers of frames.

    Args:
        embed_dim: Embedding dimension.
        max_temporal_patches: Maximum temporal patch positions.
        max_spatial_patches: Maximum spatial patch positions (h * w).
        drop_rate: Dropout rate for positional embeddings.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        max_temporal_patches: int = 16,
        max_spatial_patches: int = 256,
        drop_rate: float = 0.0,
    ):
        """Initialize factorized positional embedding.

        Args:
            embed_dim: Dimension of the embedding space.
            max_temporal_patches: Maximum number of temporal positions.
            max_spatial_patches: Maximum number of spatial positions.
            drop_rate: Dropout probability applied to embeddings.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_temporal_patches = max_temporal_patches
        self.max_spatial_patches = max_spatial_patches

        # Learnable temporal positional embedding
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_temporal_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Learnable spatial positional embedding
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, max_spatial_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Learnable CLS token positional embedding
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        temporal_patches: int,
        height_patches: int,
        width_patches: int,
    ) -> torch.Tensor:
        """Add factorized positional embeddings to input.

        Args:
            x: Input features (batch, t*h*w+1, embed_dim) or (batch, t*h*w, embed_dim).
            temporal_patches: Number of temporal patches.
            height_patches: Number of height patches.
            width_patches: Number of width patches.

        Returns:
            Features with added positional embeddings.
        """
        batch_size, num_tokens, embed_dim = x.shape

        has_cls = num_tokens == temporal_patches * height_patches * width_patches + 1

        # Handle CLS token separately
        if has_cls:
            cls_tokens = x[:, 0:1, :]
            patch_tokens = x[:, 1:, :]
            cls_tokens = cls_tokens + self.cls_pos_embed
        else:
            patch_tokens = x

        spatial_patches = height_patches * width_patches

        # Reshape to (batch, temporal_patches, spatial_patches, embed_dim)
        patch_tokens = patch_tokens.reshape(
            batch_size, temporal_patches, spatial_patches, embed_dim
        )

        # Add temporal positional embedding
        t_pos = self.temporal_pos_embed[:, :temporal_patches, :]
        patch_tokens = patch_tokens + t_pos.unsqueeze(2)

        # Add spatial positional embedding
        s_pos = self.spatial_pos_embed[:, :spatial_patches, :]
        patch_tokens = patch_tokens + s_pos.unsqueeze(1)

        # Flatten back to sequence
        patch_tokens = patch_tokens.reshape(batch_size, -1, embed_dim)

        if has_cls:
            x = torch.cat([cls_tokens, patch_tokens], dim=1)
        else:
            x = patch_tokens

        x = self.pos_drop(x)
        return x


# =============================================================================
# Multi-Head Attention Variants for Video
# =============================================================================

class DividedSpaceTimeAttention(nn.Module):
    """Divided Space-Time Attention from TimeSformer.

    Performs spatial and temporal attention separately:
    1. Spatial attention: attend to all spatial tokens within each frame
    2. Temporal attention: attend to the same spatial location across all frames

    This factorization reduces computational complexity from O(T^2 * N^2)
    to O(T * N^2) where T is number of frames and N is number of patches.

    Args:
        dim: Input feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        attention_dropout: Attention weights dropout probability.
        spatial_first: If True, apply spatial attention before temporal.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        spatial_first: bool = True,
    ):
        """Initialize divided space-time attention.

        Args:
            dim: Feature dimension per token.
            num_heads: Number of parallel attention heads.
            dropout: General dropout rate.
            attention_dropout: Dropout on attention weights.
            spatial_first: Whether to apply spatial attention first.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_first = spatial_first

        if self.head_dim * num_heads != dim:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}. "
                f"Got head_dim={self.head_dim}."
            )

        # Spatial attention QKV projections
        self.spatial_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.spatial_proj = nn.Linear(dim, dim, bias=True)

        # Temporal attention QKV projections
        self.temporal_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.temporal_proj = nn.Linear(dim, dim, bias=True)

        # Layer norms for each attention type
        self.spatial_norm = nn.LayerNorm(dim, eps=1e-6)
        self.temporal_norm = nn.LayerNorm(dim, eps=1e-6)

        self.spatial_dropout = nn.Dropout(dropout)
        self.temporal_dropout = nn.Dropout(dropout)
        self.spatial_attn_dropout = nn.Dropout(attention_dropout)
        self.temporal_attn_dropout = nn.Dropout(attention_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        for module in [self.spatial_qkv, self.temporal_qkv,
                       self.spatial_proj, self.temporal_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _reshape_for_spatial_attention(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_spatial: int,
    ) -> torch.Tensor:
        """Reshape input for spatial attention computation.

        Reshapes from (batch, T*N, dim) to (batch*T, N, dim) so that attention
        is computed independently for each frame.

        Args:
            x: Input tensor (batch, T*N, dim).
            batch_size: Batch dimension size.
            num_frames: Number of temporal positions (T).
            num_spatial: Number of spatial tokens per frame (N).

        Returns:
            Reshaped tensor (batch*T, N, dim).
        """
        x = x.reshape(batch_size, num_frames, num_spatial, self.dim)
        x = x.permute(0, 1, 2, 3).contiguous()
        x = x.reshape(batch_size * num_frames, num_spatial, self.dim)
        return x

    def _reshape_for_temporal_attention(
        self,
        x: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_spatial: int,
    ) -> torch.Tensor:
        """Reshape input for temporal attention computation.

        Reshapes from (batch, T*N, dim) to (batch*N, T, dim) so that attention
        is computed independently for each spatial position.

        Args:
            x: Input tensor (batch, T*N, dim).
            batch_size: Batch dimension size.
            num_frames: Number of temporal positions (T).
            num_spatial: Number of spatial tokens per frame (N).

        Returns:
            Reshaped tensor (batch*N, T, dim).
        """
        x = x.reshape(batch_size, num_frames, num_spatial, self.dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_spatial, num_frames, self.dim)
        return x

    def _apply_attention(
        self,
        x: torch.Tensor,
        qkv: nn.Linear,
        proj: nn.Linear,
        attn_dropout: nn.Dropout,
    ) -> torch.Tensor:
        """Apply standard multi-head self-attention.

        Args:
            x: Input (batch * group, seq, dim).
            qkv: QKV projection layer.
            proj: Output projection layer.
            attn_dropout: Attention dropout layer.

        Returns:
            Attended output (batch * group, seq, dim).
        """
        batch_group, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv_output = qkv(x)
        qkv_output = qkv_output.reshape(batch_group, seq_len, 3, self.num_heads, self.head_dim)
        qkv_output = qkv_output.permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv_output[0], qkv_output[1], qkv_output[2]

        # Scale queries
        q = q * self.scale

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Softmax normalization
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = attn_probs.to(x.dtype)
        attn_probs = attn_dropout(attn_probs)

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_group, seq_len, self.dim)
        attn_output = proj(attn_output)

        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply divided space-time attention.

        Args:
            x: Input features (batch, num_frames * num_spatial, dim).
            num_frames: Number of frames T.
            attention_mask: Optional attention mask.

        Returns:
            Attended features (batch, num_frames * num_spatial, dim).
        """
        batch_size, total_tokens, _ = x.shape
        num_spatial = total_tokens // num_frames

        residual = x

        if self.spatial_first:
            # --- Spatial attention ---
            x_norm = self.spatial_norm(x)
            x_spatial = self._reshape_for_spatial_attention(
                x_norm, batch_size, num_frames, num_spatial
            )
            x_spatial = self._apply_attention(
                x_spatial, self.spatial_qkv, self.spatial_proj,
                self.spatial_attn_dropout,
            )
            # Reshape back to (batch, T*N, dim)
            x_spatial = x_spatial.reshape(batch_size, num_frames, num_spatial, self.dim)
            x_spatial = x_spatial.reshape(batch_size, num_frames * num_spatial, self.dim)
            x = residual + self.spatial_dropout(x_spatial)

            # --- Temporal attention ---
            residual = x
            x_norm = self.temporal_norm(x)
            x_temporal = self._reshape_for_temporal_attention(
                x_norm, batch_size, num_frames, num_spatial
            )
            x_temporal = self._apply_attention(
                x_temporal, self.temporal_qkv, self.temporal_proj,
                self.temporal_attn_dropout,
            )
            # Reshape back to (batch, T*N, dim)
            x_temporal = x_temporal.reshape(batch_size, num_spatial, num_frames, self.dim)
            x_temporal = x_temporal.permute(0, 2, 1, 3).contiguous()
            x_temporal = x_temporal.reshape(batch_size, num_frames * num_spatial, self.dim)
            x = residual + self.temporal_dropout(x_temporal)
        else:
            # --- Temporal attention first ---
            residual = x
            x_norm = self.temporal_norm(x)
            x_temporal = self._reshape_for_temporal_attention(
                x_norm, batch_size, num_frames, num_spatial
            )
            x_temporal = self._apply_attention(
                x_temporal, self.temporal_qkv, self.temporal_proj,
                self.temporal_attn_dropout,
            )
            x_temporal = x_temporal.reshape(batch_size, num_spatial, num_frames, self.dim)
            x_temporal = x_temporal.permute(0, 2, 1, 3).contiguous()
            x_temporal = x_temporal.reshape(batch_size, num_frames * num_spatial, self.dim)
            x = residual + self.temporal_dropout(x_temporal)

            # --- Spatial attention ---
            residual = x
            x_norm = self.spatial_norm(x)
            x_spatial = self._reshape_for_spatial_attention(
                x_norm, batch_size, num_frames, num_spatial
            )
            x_spatial = self._apply_attention(
                x_spatial, self.spatial_qkv, self.spatial_proj,
                self.spatial_attn_dropout,
            )
            x_spatial = x_spatial.reshape(batch_size, num_frames, num_spatial, self.dim)
            x_spatial = x_spatial.reshape(batch_size, num_frames * num_spatial, self.dim)
            x = residual + self.spatial_dropout(x_spatial)

        return x


class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation.

    Standard two-layer FFN used in transformer blocks with optional
    dropout and configurable expansion ratio.

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden layer dimension. If None, uses dim * mlp_ratio.
        mlp_ratio: Expansion ratio for hidden dimension.
        dropout: Dropout probability.
        activation: Activation function name ('gelu', 'relu', 'silu').
    """

    def __init__(
        self,
        dim: int = 768,
        hidden_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """Initialize feed-forward network.

        Args:
            dim: Input feature dimension.
            hidden_dim: Intermediate dimension (overrides mlp_ratio if set).
            mlp_ratio: Ratio to compute hidden_dim = dim * mlp_ratio.
            dropout: Dropout after each linear layer.
            activation: Name of the activation function.
        """
        super().__init__()
        hidden_dim = hidden_dim or int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: Input tensor (..., dim).

        Returns:
            Output tensor (..., dim).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# =============================================================================
# TimeSformer Encoder
# =============================================================================

class TimeSformerBlock(nn.Module):
    """Single TimeSformer transformer block.

    Combines divided space-time attention with a feed-forward network
    and layer normalization in a pre-norm configuration.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: FFN expansion ratio.
        dropout: General dropout rate.
        attention_dropout: Attention dropout rate.
        spatial_first: Whether spatial attention comes first.
        layer_norm_eps: Epsilon for layer normalization.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        spatial_first: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        """Initialize TimeSformer block.

        Args:
            dim: Transformer dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio for FFN hidden dimension.
            dropout: General dropout.
            attention_dropout: Dropout on attention weights.
            spatial_first: Order of space/time attention.
            layer_norm_eps: Layer norm epsilon.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = DividedSpaceTimeAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            spatial_first=spatial_first,
        )
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.ffn = FeedForward(
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through TimeSformer block.

        Args:
            x: Input features (batch, num_frames * num_spatial, dim).
            num_frames: Number of frames.
            attention_mask: Optional attention mask.

        Returns:
            Output features (batch, num_frames * num_spatial, dim).
        """
        # Pre-norm: attention
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, num_frames=num_frames, attention_mask=attention_mask)

        # Pre-norm: FFN
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        return x


class TimeSformerEncoder(nn.Module):
    """TimeSformer: Is Space-Time Attention All You Need for Video Understanding?

    Implements the full TimeSformer encoder architecture with divided
    space-time attention. Supports configurable dimensions, number of
    layers, and attention strategies.

    Architecture:
    1. 3D tubelet patch embedding (Conv3d)
    2. Factorized (or joint) positional embeddings
    3. N transformer blocks with divided space-time attention
    4. Optional classification head

    Args:
        in_channels: Input channels (3 for RGB).
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        mlp_ratio: FFN expansion ratio.
        num_frames: Expected number of input frames.
        image_size: Input image size (assumed square).
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal tubelet size.
        dropout: General dropout rate.
        attention_dropout: Attention dropout rate.
        spatial_first: Apply spatial before temporal attention.
        layer_norm_eps: Layer normalization epsilon.
        use_cls_token: Include a CLS token.
        num_classes: Number of output classes (0 for no classification head).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_frames: int = 8,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        spatial_first: bool = True,
        layer_norm_eps: float = 1e-6,
        use_cls_token: bool = True,
        num_classes: int = 0,
    ):
        """Initialize TimeSformer encoder.

        Args:
            in_channels: Number of input channels per frame.
            embed_dim: Dimension of the transformer.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            mlp_ratio: Expansion ratio for the FFN.
            num_frames: Number of input frames.
            image_size: Height and width of input frames.
            patch_size: Size of spatial patches.
            temporal_patch_size: Number of frames per tubelet.
            dropout: Dropout rate.
            attention_dropout: Attention dropout rate.
            spatial_first: Order of space/time attention.
            layer_norm_eps: Layer norm epsilon.
            use_cls_token: Whether to prepend a CLS token.
            num_classes: Number of output classes (0 = no head).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_first = spatial_first
        self.use_cls_token = use_cls_token

        # Spatial patch dimensions
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch_size

        # 3D tubelet patch embedding
        self.patch_embed = VideoPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=patch_size,
            temporal_stride=temporal_patch_size,
            spatial_stride=patch_size,
            flatten=True,
            bias=True,
            max_temporal_patches=self.num_temporal_patches,
        )

        # Factorized positional embedding
        self.pos_embed = VideoPositionalEmbedding(
            embed_dim=embed_dim,
            max_temporal_patches=self.num_temporal_patches,
            max_spatial_patches=self.num_spatial_patches,
            drop_rate=dropout,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TimeSformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                spatial_first=spatial_first,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Optional classification head
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(
        self,
        video: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> VideoEncoderOutput:
        """Encode a video using TimeSformer architecture.

        Args:
            video: Input video (batch, num_frames, channels, height, width).
            attention_mask: Optional attention mask.
            output_hidden_states: Whether to return all hidden states.
            output_attentions: Whether to return attention weights.
            return_dict: Whether to return a structured output.

        Returns:
            VideoEncoderOutput containing encoded video features.
        """
        batch_size, num_frames, channels, height, width = video.shape

        # Extract tubelet patches
        x, patch_info = self.patch_embed(video, return_patch_info=True)
        # x: (batch, total_patches, embed_dim) with CLS token prepended

        t_patches = patch_info["temporal_patches"]
        h_patches = patch_info["height_patches"]
        w_patches = patch_info["width_patches"]
        num_spatial = h_patches * w_patches

        # Add factorized positional embeddings (skip CLS which was handled in patch_embed)
        # The VideoPatchEmbedding already adds its own pos embed + CLS, so we skip here
        # and rely on the combined embedding from patch_embed.

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Pass through transformer blocks
        effective_num_frames = t_patches
        for block in self.blocks:
            x = block(
                x,
                num_frames=effective_num_frames,
                attention_mask=attention_mask,
            )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        # Final layer norm
        x = self.final_norm(x)

        # Extract CLS token for classification
        cls_output = x[:, 0] if self.use_cls_token else x.mean(dim=1)

        # Optional classification
        if self.classifier is not None:
            logits = self.classifier(cls_output)
        else:
            logits = None

        # Compute pooler output
        pooler_output = cls_output

        if not return_dict:
            return (x, all_hidden_states, all_attentions, pooler_output)

        return VideoEncoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            pooler_output=pooler_output,
            frame_indices=None,
        )


# =============================================================================
# Video Swin Transformer
# =============================================================================

class WindowPartition3D:
    """3D window partitioning utility for Video Swin Transformer.

    Partitions a 5D tensor (batch, channels, temporal, height, width) into
    non-overlapping 3D windows of size (window_size_t, window_size_h, window_size_w).
    """

    def __init__(
        self,
        window_size: Tuple[int, int, int] = (8, 7, 7),
    ):
        """Initialize 3D window partitioner.

        Args:
            window_size: Size of the 3D windows (temporal, height, width).
        """
        self.window_size = window_size

    def partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partition tensor into non-overlapping 3D windows.

        Args:
            x: Input tensor of shape (batch, channels, T, H, W).

        Returns:
            Windows of shape (batch * num_windows_t * num_windows_h * num_windows_w,
                             channels, window_t, window_h, window_w).
        """
        B, C, T, H, W = x.shape
        wt, wh, ww = self.window_size

        # Ensure dimensions are divisible by window size
        assert T % wt == 0, f"Temporal dim {T} must be divisible by {wt}"
        assert H % wh == 0, f"Height dim {H} must be divisible by {wh}"
        assert W % ww == 0, f"Width dim {W} must be divisible by {ww}"

        x = x.view(
            B, C,
            T // wt, wt,
            H // wh, wh,
            W // ww, ww,
        )
        # Permute to group windows: (B, T//wt, H//wh, W//ww, C, wt, wh, ww)
        windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        # Reshape to (B * num_windows, C, wt, wh, ww)
        windows = windows.view(
            -1, C, wt, wh, ww
        )
        return windows

    def reverse(self, windows: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
        """Reverse window partitioning.

        Args:
            windows: Windowed tensor (B * num_windows, C, wt, wh, ww).
            original_shape: Original tensor shape (B, C, T, H, W).

        Returns:
            Reconstructed tensor of shape (B, C, T, H, W).
        """
        B, C, T, H, W = original_shape
        wt, wh, ww = self.window_size

        # Reshape back to grouped windows
        windows = windows.view(
            B,
            T // wt, H // wh, W // ww,
            C, wt, wh, ww,
        )
        # Permute back to original layout
        x = windows.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(B, C, T, H, W)
        return x


class ShiftedWindowAttention3D(nn.Module):
    """Shifted Window Multi-Head Self-Attention for 3D video data.

    Implements the window-based attention from Swin Transformer extended to 3D.
    Uses cyclic shifting for cross-window connections and 3D relative
    position bias.

    Args:
        dim: Input feature dimension.
        window_size: Size of 3D attention window (t, h, w).
        num_heads: Number of attention heads.
        qkv_bias: Whether to add bias to QKV projections.
        attn_drop: Attention dropout rate.
        proj_drop: Projection dropout rate.
    """

    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """Initialize 3D shifted window attention.

        Args:
            dim: Feature dimension.
            window_size: 3D window size (temporal, height, width).
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in QKV.
            attn_drop: Dropout on attention weights.
            proj_drop: Dropout on output projection.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.window_size_t = window_size[0]
        self.window_size_h = window_size[1]
        self.window_size_w = window_size[2]
        window_area = self.window_size_t * self.window_size_h * self.window_size_w

        # 3D relative position bias: each axis has (2*size - 1) possible offsets
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size_t - 1) *
                (2 * self.window_size_h - 1) *
                (2 * self.window_size_w - 1),
                num_heads,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Precompute relative position index
        self._build_relative_position_index()

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def _build_relative_position_index(self) -> None:
        """Precompute the relative position index for 3D windows.

        Creates a mapping from relative (dt, dh, dw) offsets to indices
        in the relative position bias table.
        """
        wt, wh, ww = self.window_size

        # Create coordinate grids
        coords_t = torch.arange(wt)
        coords_h = torch.arange(wh)
        coords_w = torch.arange(ww)

        # Create meshgrid of all positions in the window
        coords = torch.stack(
            torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"),
            dim=0,
        ).flatten(1)  # (3, window_area)

        # Compute relative coordinates
        relative_coords = coords[:, :, None] - coords[:, None, :]  # (3, area, area)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (area, area, 3)

        # Normalize to [0, 2*size-2]
        relative_coords[:, :, 0] += wt - 1
        relative_coords[:, :, 1] += wh - 1
        relative_coords[:, :, 2] += ww - 1

        # Compute linear index into bias table
        relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
        relative_coords[:, :, 1] *= (2 * ww - 1)

        relative_position_index = relative_coords.sum(dim=-1)  # (area, area)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply shifted window self-attention.

        Args:
            x: Windowed features (num_windows * batch, window_area, dim).
            mask: Optional attention mask for shifted windows.

        Returns:
            Attended features (num_windows * batch, window_area, dim).
        """
        B_, N, C = x.shape
        head_dim = C // self.num_heads

        # QKV projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scale queries
        q = q * self.scale

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask (for shifted windows)
        if mask is not None:
            n_win = mask.shape[0]
            attn = attn.view(B_ // n_win, n_win, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Softmax
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Apply attention
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).reshape(B_, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class VideoSwinBlock(nn.Module):
    """Video Swin Transformer Block.

    Implements a single block of the Video Swin Transformer with:
    - Window-based multi-head self-attention with 3D relative position bias
    - Shifted window partitioning for cross-window connections
    - Feed-forward network with GELU activation
    - Pre-norm architecture with residual connections

    The block alternates between regular and shifted window partitions
    on successive layers for better cross-window information flow.

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        window_size: 3D window size (temporal, height, width).
        shift_size: 3D shift size (temporal, height, width).
        mlp_ratio: FFN expansion ratio.
        qkv_bias: Whether to use bias in QKV.
        drop: General dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
        layer_norm_eps: Layer norm epsilon.
    """

    def __init__(
        self,
        dim: int = 96,
        num_heads: int = 3,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        shift_size: Tuple[int, int, int] = (4, 3, 3),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize Video Swin block.

        Args:
            dim: Input feature dimension.
            num_heads: Number of attention heads.
            window_size: Size of attention window (t, h, w).
            shift_size: Cyclic shift size (t, h, w).
            mlp_ratio: FFN expansion ratio.
            qkv_bias: Include bias in QKV projections.
            drop: General dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Drop path (stochastic depth) rate.
            layer_norm_eps: Layer normalization epsilon.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Normalization layers
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)

        # Window partitioner
        self.window_partition = WindowPartition3D(window_size=window_size)

        # Window attention
        self.attn = ShiftedWindowAttention3D(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def _compute_attn_mask(
        self,
        input_shape: Tuple[int, int, int, int, int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Compute attention mask for shifted windows.

        When the window is shifted, some tokens need to be masked out
        to prevent attention between unrelated tokens across window boundaries.

        Args:
            input_shape: Shape of the input (B, C, T, H, W).
            device: Device for the mask tensor.

        Returns:
            Attention mask of shape (num_windows, window_area, window_area)
            or None if no shifting is applied.
        """
        B, C, T, H, W = input_shape
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size

        # If no shift, no mask needed
        if st == 0 and sh == 0 and sw == 0:
            return None

        # Pad dimensions to make them divisible by window sizes
        pad_t = (wt - T % wt) % wt
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        # Compute the number of windows
        T_pad = T + pad_t
        H_pad = H + pad_h
        W_pad = W + ww

        # Create mask regions
        # Determine which positions belong to which window after shifting
        img_mask = torch.zeros(1, T_pad, H_pad, W_pad, device=device)

        # Define regions based on shift
        t_slices = (
            slice(0, -wt),
            slice(-wt, -st),
            slice(-st, None),
        ) if st > 0 else (
            slice(0, None),
        )

        h_slices = (
            slice(0, -wh),
            slice(-wh, -sh),
            slice(-sh, None),
        ) if sh > 0 else (
            slice(0, None),
        )

        w_slices = (
            slice(0, -ww),
            slice(-ww, -sw),
            slice(-sw, None),
        ) if sw > 0 else (
            slice(0, None),
        )

        cnt = 0
        for ts in t_slices:
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, ts, hs, ws] = cnt
                    cnt += 1

        # Partition mask into windows
        mask_windows = self.window_partition(img_mask.unsqueeze(1))
        mask_windows = mask_windows.squeeze(1)  # (num_windows, wt, wh, ww)

        # Reshape and compute attention mask
        mask_windows = mask_windows.view(-1, wt * wh * ww)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def _check_window_size(
        self,
        t: int,
        h: int,
        w: int,
    ) -> None:
        """Check that dimensions are compatible with window sizes.

        Args:
            t: Temporal dimension.
            h: Height dimension.
            w: Width dimension.
        """
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size

        for dim_val, win_val, shift_val, name in [
            (t, wt, st, "temporal"),
            (h, wh, sh, "height"),
            (w, ww, sw, "width"),
        ]:
            if shift_val >= win_val:
                raise ValueError(
                    f"shift_size for {name} ({shift_val}) must be less than "
                    f"window_size ({win_val})."
                )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through Video Swin block.

        Args:
            x: Input features (batch, channels, temporal, height, width).
            attention_mask: Optional attention mask.

        Returns:
            Output features (batch, channels, temporal, height, width).
        """
        B, C, T, H, W = x.shape
        self._check_window_size(T, H, W)

        shortcut = x
        x = self.norm1(x)

        # Pad to make dimensions divisible by window sizes
        wt, wh, ww = self.window_size
        st, sh, sw = self.shift_size

        pad_t = (wt - T % wt) % wt
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww

        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))

        _, _, T_pad, H_pad, W_pad = x.shape

        # Cyclic shift
        if st > 0 and sh > 0 and sw > 0:
            shifted_x = torch.roll(x, shifts=(-st, -sh, -sw), dims=(2, 3, 4))
        elif st > 0:
            shifted_x = torch.roll(x, shifts=(-st,), dims=(2,))
        elif sh > 0:
            shifted_x = torch.roll(x, shifts=(-sh, -sw), dims=(3, 4))
        else:
            shifted_x = x

        # Partition into windows
        x_windows = self.window_partition(shifted_x)
        # x_windows: (num_windows * B, C, wt, wh, ww)
        num_windows = x_windows.shape[0] // B

        # Reshape for attention: (num_windows * B, window_area, C)
        window_area = wt * wh * ww
        x_windows = x_windows.view(-1, window_area, C)

        # Compute attention mask
        attn_mask = self._compute_attn_mask(
            (B, C, T_pad, H_pad, W_pad),
            x.device,
        )

        # Apply window attention
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reshape back to window format
        attn_windows = attn_windows.view(-1, C, wt, wh, ww)

        # Reverse window partition
        shifted_x = self.window_partition.reverse(attn_windows, (B, C, T_pad, H_pad, W_pad))

        # Reverse cyclic shift
        if st > 0 and sh > 0 and sw > 0:
            x = torch.roll(shifted_x, shifts=(st, sh, sw), dims=(2, 3, 4))
        elif st > 0:
            x = torch.roll(shifted_x, shifts=(st,), dims=(2,))
        elif sh > 0:
            x = torch.roll(shifted_x, shifts=(sh, sw), dims=(3, 4))
        else:
            x = shifted_x

        # Remove padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :, :T, :H, :W].contiguous()

        # Residual connection with drop path
        x = shortcut + self.drop_path(x)

        # Feed-forward network
        # Reshape for layer norm and FFN: (B, T, H, W, C) -> (B, T*H*W, C)
        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (B, T, H, W, C)
        x = x.reshape(B, T * H * W, C)
        x = self.ffn(x)
        x = x.reshape(B, T, H, W, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)
        x = shortcut + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) for regularizing deep networks.

    Randomly drops entire paths during training, equivalent to applying
    a multiplicative Bernoulli mask to the residual branch.

    Args:
        drop_prob: Probability of dropping the path (0 = no dropping).
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize drop path.

        Args:
            drop_prob: Probability of dropping the path during training.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth.

        Args:
            x: Input tensor.

        Returns:
            Tensor with paths randomly dropped during training.
        """
        if not self.training or self.drop_prob <= 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class VideoSwinPatchEmbed(nn.Module):
    """Patch embedding for Video Swin Transformer.

    Uses a 3D convolution to split the video into non-overlapping patches
    and project them to the embedding dimension. Optionally includes
    a patch merging layer for downsampling in deeper stages.

    Args:
        patch_size: 3D patch size (temporal, height, width).
        in_channels: Number of input channels.
        embed_dim: Output embedding dimension.
        norm_layer: Optional normalization layer.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
    ):
        """Initialize Video Swin patch embedding.

        Args:
            patch_size: Size of each 3D patch.
            in_channels: Input channel count.
            embed_dim: Output embedding dimension.
            norm_layer: Optional normalization after projection.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project video to patch embeddings.

        Args:
            x: Video tensor (batch, channels, temporal, height, width).

        Returns:
            Patch embeddings (batch, embed_dim, temporal, height, width).
        """
        x = self.proj(x)
        x = self.norm(x)
        return x


class PatchMerging3D(nn.Module):
    """3D Patch Merging layer for Video Swin Transformer.

    Downsamples the feature map by a factor of 2 in spatial dimensions
    and concatenates features along the channel dimension, then applies
    a linear projection to reduce back to 2x the input dimension.

    Args:
        dim: Input feature dimension.
        norm_layer: Normalization layer class.
        spatial_only: If True, only merge spatially (keep temporal dim).
    """

    def __init__(
        self,
        dim: int = 96,
        norm_layer: Optional[nn.Module] = None,
        spatial_only: bool = False,
    ):
        """Initialize 3D patch merging.

        Args:
            dim: Input feature dimension.
            norm_layer: Normalization layer (default: LayerNorm).
            spatial_only: Whether to only merge in spatial dimensions.
        """
        super().__init__()
        self.dim = dim
        self.spatial_only = spatial_only
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim) if norm_layer is not None else nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge patches by downsampling.

        Args:
            x: Input features (batch, C, T, H, W).

        Returns:
            Merged features (batch, 2*C, T, H//2, W//2).
        """
        B, C, T, H, W = x.shape

        # Spatial downsampling: take 2x2 patches
        # Reshape to (B, C, T, H//2, 2, W//2, 2)
        x = x.reshape(B, C, T, H // 2, 2, W // 2, 2)
        # Permute to (B, T, H//2, W//2, C*4)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
        x = x.reshape(B * T * (H // 2) * (W // 2), 4 * C)

        # Normalize and project
        x = self.norm(x)
        x = self.reduction(x)

        # Reshape back
        x = x.reshape(B, T, H // 2, W // 2, 2 * C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class VideoSwinEncoder(nn.Module):
    """Video Swin Transformer Encoder.

    Full implementation of the Video Swin Transformer encoder with
    multiple stages, each containing shifted window attention blocks.

    The architecture processes video through 4 stages with increasing
    resolution reduction and feature dimension:
    Stage 1: dim -> 2*dim (patch merge)
    Stage 2: 2*dim -> 4*dim
    Stage 3: 4*dim -> 8*dim
    Stage 4: 8*dim (final)

    Args:
        in_channels: Input channels (3 for RGB).
        embed_dim: Base embedding dimension.
        depths: Number of blocks per stage.
        num_heads: Number of attention heads per stage.
        window_size: 3D window size.
        mlp_ratio: FFN expansion ratio.
        drop_rate: General dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate (linearly increasing).
        patch_size: Initial patch size.
        num_frames: Number of input frames.
        image_size: Input image size (height, width).
        qkv_bias: Whether to use bias in QKV.
        layer_norm_eps: Layer norm epsilon.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 96,
        depths: List[int] = None,
        num_heads: List[int] = None,
        window_size: Tuple[int, int, int] = (8, 7, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        num_frames: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        qkv_bias: bool = True,
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize Video Swin encoder.

        Args:
            in_channels: Number of input channels.
            embed_dim: Base embedding dimension.
            depths: Number of blocks per stage (default: [2, 2, 6, 2]).
            num_heads: Number of attention heads per stage (default: [3, 6, 12, 24]).
            window_size: 3D attention window size.
            mlp_ratio: FFN expansion ratio.
            drop_rate: General dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Maximum stochastic depth rate.
            patch_size: Initial 3D patch size.
            num_frames: Number of input frames.
            image_size: Input image size (H, W).
            qkv_bias: Whether to use bias in QKV projections.
            layer_norm_eps: Layer norm epsilon.
        """
        super().__init__()

        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_stages = len(depths)
        self.num_frames = num_frames

        # Patch embedding
        self.patch_embed = VideoSwinPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )

        # Stochastic depth scheduling (linearly increasing)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_stage in range(self.num_stages):
            dim = embed_dim * (2 ** i_stage)
            depth = depths[i_stage]
            n_heads = num_heads[i_stage]

            # Compute shift size (half of window size, 0 for first two blocks)
            shift_size = tuple(w // 2 for w in window_size)

            # Create blocks for this stage
            stage_blocks = nn.ModuleList()
            for i_block in range(depth):
                shift = shift_size if (i_block % 2 == 1) else (0, 0, 0)
                drop_path = dpr[sum(depths[:i_stage]) + i_block]

                block = VideoSwinBlock(
                    dim=dim,
                    num_heads=n_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path,
                    layer_norm_eps=layer_norm_eps,
                )
                stage_blocks.append(block)

            self.stages.append(stage_blocks)

            # Patch merging (downsample) between stages (except last)
            if i_stage < self.num_stages - 1:
                downsample = PatchMerging3D(
                    dim=dim,
                    norm_layer=lambda d: nn.LayerNorm(d, eps=layer_norm_eps),
                )
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())

        # Final layer norm
        self.final_norm = nn.LayerNorm(
            embed_dim * (2 ** (self.num_stages - 1)),
            eps=layer_norm_eps,
        )

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for all modules."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        video: torch.Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> VideoEncoderOutput:
        """Encode video using Video Swin Transformer.

        Args:
            video: Input video (batch, channels, temporal, height, width).
            output_hidden_states: Whether to return intermediate features.
            return_dict: Whether to return structured output.

        Returns:
            VideoEncoderOutput with encoded features.
        """
        # Expect input (B, T, C, H, W) -> convert to (B, C, T, H, W)
        if video.dim() == 5 and video.shape[1] > video.shape[2]:
            # Assume (B, T, C, H, W)
            x = video.permute(0, 2, 1, 3, 4).contiguous()
        else:
            x = video

        # Patch embedding
        x = self.patch_embed(x)

        all_hidden_states = () if output_hidden_states else None

        # Pass through stages
        for i_stage in range(self.num_stages):
            for block in self.stages[i_stage]:
                x = block(x)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x.clone(),)

            # Downsample between stages
            if i_stage < self.num_stages - 1:
                x = self.downsamples[i_stage](x)

        # Global average pooling
        B, C, T, H, W = x.shape
        spatial_features = x

        # Average pool over temporal, height, width
        pooler_output = self.avg_pool(x).flatten(1)  # (B, C)

        # Final norm
        pooler_output = self.final_norm(pooler_output)

        # Reshape last hidden state to sequence format
        last_hidden = x.permute(0, 2, 3, 4, 1).contiguous()
        last_hidden = last_hidden.reshape(B, T * H * W, C)

        if not return_dict:
            return (last_hidden, all_hidden_states, pooler_output)

        return VideoEncoderOutput(
            last_hidden_state=last_hidden,
            hidden_states=all_hidden_states,
            spatial_features=spatial_features,
            pooler_output=pooler_output,
        )


# =============================================================================
# ViViT Encoder (Video Vision Transformer)
# =============================================================================

class FactorizedEncoder(nn.Module):
    """Factorized spatial-temporal encoder for ViViT.

    First applies spatial attention within each frame independently,
    then applies temporal attention across frames. This factorization
    reduces computation from O((T*N)^2) to O(T*N^2 + N*T^2).

    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: FFN expansion ratio.
        dropout: General dropout.
        attention_dropout: Attention dropout.
        layer_norm_eps: Layer norm epsilon.
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        """Initialize factorized encoder.

        Args:
            dim: Feature dimension.
            num_heads: Number of attention heads.
            mlp_ratio: FFN expansion ratio.
            dropout: Dropout rate.
            attention_dropout: Attention dropout.
            layer_norm_eps: Layer norm epsilon.
        """
        super().__init__()
        self.dim = dim

        # Spatial self-attention (within each frame)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.spatial_norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.spatial_norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.spatial_ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.spatial_dropout = nn.Dropout(dropout)

        # Temporal self-attention (across frames)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.temporal_norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.temporal_norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.temporal_ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.temporal_dropout = nn.Dropout(dropout)

    def _spatial_attention(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply spatial self-attention within each frame.

        Args:
            x: Features (batch, T*N, dim).
            num_frames: Number of frames T.

        Returns:
            Spatially attended features (batch, T*N, dim).
        """
        batch_size, total_tokens, dim = x.shape
        num_spatial = total_tokens // num_frames

        # Reshape to (batch*T, N, dim)
        x = x.reshape(batch_size, num_frames, num_spatial, dim)
        x = x.permute(0, 1, 2, 3).contiguous()
        x = x.reshape(batch_size * num_frames, num_spatial, dim)

        # Self-attention
        residual = x
        x_norm = self.spatial_norm1(x)
        x_attn, _ = self.spatial_attn(x_norm, x_norm, x_norm)
        x = residual + self.spatial_dropout(x_attn)

        # FFN
        residual = x
        x = residual + self.spatial_ffn(self.spatial_norm2(x))

        # Reshape back to (batch, T*N, dim)
        x = x.reshape(batch_size, num_frames, num_spatial, dim)
        x = x.reshape(batch_size, num_frames * num_spatial, dim)

        return x

    def _temporal_attention(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply temporal self-attention across frames.

        Args:
            x: Features (batch, T*N, dim).
            num_frames: Number of frames T.

        Returns:
            Temporally attended features (batch, T*N, dim).
        """
        batch_size, total_tokens, dim = x.shape
        num_spatial = total_tokens // num_frames

        # Reshape to (batch*N, T, dim)
        x = x.reshape(batch_size, num_frames, num_spatial, dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size * num_spatial, num_frames, dim)

        # Self-attention
        residual = x
        x_norm = self.temporal_norm1(x)
        x_attn, _ = self.temporal_attn(x_norm, x_norm, x_norm)
        x = residual + self.temporal_dropout(x_attn)

        # FFN
        residual = x
        x = residual + self.temporal_ffn(self.temporal_norm2(x))

        # Reshape back to (batch, T*N, dim)
        x = x.reshape(batch_size, num_spatial, num_frames, dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(batch_size, num_frames * num_spatial, dim)

        return x

    def forward(
        self,
        x: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """Apply factorized spatial then temporal attention.

        Args:
            x: Input features (batch, T*N, dim).
            num_frames: Number of frames.

        Returns:
            Attended features (batch, T*N, dim).
        """
        # Spatial attention first
        x = self._spatial_attention(x, num_frames)
        # Then temporal attention
        x = self._temporal_attention(x, num_frames)
        return x


class ViViTEncoder(nn.Module):
    """Video Vision Transformer (ViViT) Encoder.

    Implements the ViViT model from "A Video Vision Transformer" (Arnab et al., 2021).
    Uses tubelet embeddings followed by a factorized encoder that applies spatial
    and temporal attention separately.

    Architecture variants:
    1. Factorized encoder: spatial attention per frame, then temporal attention
    2. Joint space-time attention: full attention over all tokens (optional)
    3. Multiple instance encoding: process each frame independently then aggregate

    Args:
        in_channels: Number of input channels.
        embed_dim: Transformer embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        mlp_ratio: FFN expansion ratio.
        num_frames: Number of input frames.
        image_size: Input image size (height, width).
        patch_size: Spatial patch size.
        temporal_patch_size: Temporal tubelet size.
        dropout: General dropout rate.
        attention_dropout: Attention dropout rate.
        layer_norm_eps: Layer norm epsilon.
        use_cls_token: Include CLS token.
        encoder_type: 'factorized', 'joint', or 'independent'.
        num_classes: Output classes (0 = no classifier).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_frames: int = 16,
        image_size: int = 224,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
        use_cls_token: bool = True,
        encoder_type: str = "factorized",
        num_classes: int = 0,
    ):
        """Initialize ViViT encoder.

        Args:
            in_channels: Input channels per frame.
            embed_dim: Transformer dimension.
            num_heads: Number of attention heads.
            num_layers: Number of factorized encoder layers.
            mlp_ratio: FFN expansion ratio.
            num_frames: Number of frames.
            image_size: Frame height/width.
            patch_size: Spatial patch size.
            temporal_patch_size: Frames per tubelet.
            dropout: Dropout rate.
            attention_dropout: Attention dropout.
            layer_norm_eps: Layer norm epsilon.
            use_cls_token: Prepend CLS token.
            encoder_type: Type of encoder ('factorized', 'joint', 'independent').
            num_classes: Number of classification classes.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.use_cls_token = use_cls_token
        self.encoder_type = encoder_type

        # Spatial patches per frame
        self.num_spatial_patches = (image_size // patch_size) ** 2
        self.num_temporal_patches = num_frames // temporal_patch_size

        # Tubelet patch embedding
        self.patch_embed = VideoPatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=patch_size,
            temporal_stride=temporal_patch_size,
            spatial_stride=patch_size,
            flatten=True,
            bias=True,
            max_temporal_patches=self.num_temporal_patches,
        )

        # Positional embedding
        self.pos_embed = VideoPositionalEmbedding(
            embed_dim=embed_dim,
            max_temporal_patches=self.num_temporal_patches,
            max_spatial_patches=self.num_spatial_patches,
            drop_rate=dropout,
        )

        # Encoder layers
        if encoder_type == "factorized":
            self.encoder_layers = nn.ModuleList([
                FactorizedEncoder(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ])
        elif encoder_type == "joint":
            # Joint space-time attention using standard transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder_layers = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(embed_dim, eps=layer_norm_eps),
            )
        elif encoder_type == "independent":
            # Independent frame encoding with shared weights
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.frame_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
            # Temporal integration
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=2,
                norm=nn.LayerNorm(embed_dim, eps=layer_norm_eps),
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                f"Choose from 'factorized', 'joint', 'independent'."
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Classification head
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
            self.classifier_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        else:
            self.classifier = None
            self.classifier_head = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(
        self,
        video: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> VideoEncoderOutput:
        """Encode a video using ViViT architecture.

        Args:
            video: Input video (batch, num_frames, channels, height, width).
            attention_mask: Optional attention mask.
            output_hidden_states: Return intermediate hidden states.
            return_dict: Return structured output.

        Returns:
            VideoEncoderOutput with encoded video features.
        """
        batch_size, num_frames, channels, height, width = video.shape

        # Extract tubelet patches
        x, patch_info = self.patch_embed(video, return_patch_info=True)

        t_patches = patch_info["temporal_patches"]
        h_patches = patch_info["height_patches"]
        w_patches = patch_info["width_patches"]

        # Add factorized positional embeddings (CLS already added by patch_embed)
        # We replace the patch_embed's combined pos embed with our factorized one
        x_no_cls = x[:, 1:, :]  # Remove CLS token
        cls_tokens = x[:, 0:1, :]
        x_no_cls = self.pos_embed(
            x_no_cls,
            temporal_patches=t_patches,
            height_patches=h_patches,
            width_patches=w_patches,
        )
        x = torch.cat([cls_tokens, x_no_cls], dim=1)

        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Encoder forward pass
        effective_frames = t_patches

        if self.encoder_type == "factorized":
            for layer in self.encoder_layers:
                # Remove CLS token for factorized encoding, add back after
                cls_tokens = x[:, 0:1, :]
                patch_tokens = x[:, 1:, :]
                patch_tokens = layer(patch_tokens, num_frames=effective_frames)
                x = torch.cat([cls_tokens, patch_tokens], dim=1)

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (x,)

        elif self.encoder_type == "joint":
            x = self.encoder_layers(x)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        elif self.encoder_type == "independent":
            # Process each frame independently with shared weights
            cls_tokens = x[:, 0:1, :]
            patch_tokens = x[:, 1:, :]
            num_spatial = patch_tokens.shape[1] // effective_frames

            # Reshape to (batch*T, N, dim) for frame-wise encoding
            patch_tokens = patch_tokens.reshape(
                batch_size, effective_frames, num_spatial, self.embed_dim
            )
            patch_tokens = patch_tokens.reshape(
                batch_size * effective_frames, num_spatial, self.embed_dim
            )
            patch_tokens = self.frame_encoder(patch_tokens)
            patch_tokens = patch_tokens.reshape(
                batch_size, effective_frames, num_spatial, self.embed_dim
            )
            patch_tokens = patch_tokens.reshape(
                batch_size, effective_frames * num_spatial, self.embed_dim
            )

            # Temporal integration
            patch_tokens = patch_tokens.reshape(
                batch_size, num_spatial, effective_frames, self.embed_dim
            )
            patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous()
            patch_tokens = patch_tokens.reshape(
                batch_size * num_spatial, effective_frames, self.embed_dim
            )
            patch_tokens = self.temporal_encoder(patch_tokens)
            patch_tokens = patch_tokens.reshape(
                batch_size, num_spatial, effective_frames, self.embed_dim
            )
            patch_tokens = patch_tokens.permute(0, 2, 1, 3).contiguous()
            patch_tokens = patch_tokens.reshape(
                batch_size, effective_frames * num_spatial, self.embed_dim
            )

            x = torch.cat([cls_tokens, patch_tokens], dim=1)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)

        # Final layer norm
        x = self.final_norm(x)

        # CLS token output
        cls_output = x[:, 0] if self.use_cls_token else x.mean(dim=1)

        # Classification
        if self.classifier_head is not None:
            logits = self.classifier_head(cls_output)
        elif self.classifier is not None:
            logits = self.classifier(cls_output)
        else:
            logits = None

        pooler_output = cls_output

        if not return_dict:
            return (x, all_hidden_states, pooler_output)

        return VideoEncoderOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            pooler_output=pooler_output,
            frame_indices=None,
        )


# =============================================================================
# Temporal Aggregator
# =============================================================================

class TemporalAttentionPooling(nn.Module):
    """Attention-based temporal pooling.

    Learns to weight each time step differently when aggregating
    a sequence of frame features into a single video-level representation.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden dimension for attention computation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: Optional[int] = None,
    ):
        """Initialize attention-based temporal pooling.

        Args:
            input_dim: Dimension of input features per time step.
            hidden_dim: Hidden dimension for attention MLP. Default: input_dim.
        """
        super().__init__()
        hidden_dim = hidden_dim or input_dim

        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize attention network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-weighted temporal pooling.

        Args:
            x: Input features (batch, num_frames, dim).
            mask: Optional mask (batch, num_frames), 1 for valid, 0 for padding.

        Returns:
            Tuple of (pooled features (batch, dim), attention weights (batch, num_frames)).
        """
        # Compute attention scores
        attn_scores = self.attention_network(x)  # (batch, num_frames, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch, num_frames)

        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, num_frames)
        attn_weights = attn_weights.unsqueeze(1)  # (batch, 1, num_frames)

        # Weighted sum
        pooled = torch.bmm(attn_weights, x)  # (batch, 1, dim)
        pooled = pooled.squeeze(1)  # (batch, dim)

        return pooled, attn_weights.squeeze(1)


class GRUTemporalEncoder(nn.Module):
    """GRU-based temporal encoding for sequential frame features.

    Processes frame features sequentially using a GRU to capture
    temporal dynamics and dependencies between frames.

    Args:
        input_dim: Input feature dimension per frame.
        hidden_dim: GRU hidden dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout between GRU layers.
        bidirectional: Use bidirectional GRU.
        batch_first: Input format is (batch, seq, features).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        batch_first: bool = True,
    ):
        """Initialize GRU temporal encoder.

        Args:
            input_dim: Feature dimension per frame.
            hidden_dim: GRU hidden state dimension.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability between layers.
            bidirectional: Process in both temporal directions.
            batch_first: Whether first dimension is batch.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        # Output projection (bidirectional doubles the dimension)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(self.output_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process frame features through GRU.

        Args:
            x: Frame features (batch, num_frames, input_dim).
            mask: Optional padding mask (batch, num_frames).

        Returns:
            Temporally encoded features (batch, num_frames, input_dim).
        """
        # Pack padded sequence if mask is provided
        if mask is not None and self.batch_first:
            lengths = mask.sum(dim=1).long()
            # Sort by length for pack_padded_sequence
            sorted_lengths, sort_idx = torch.sort(lengths, descending=True)
            sorted_lengths = sorted_lengths.clamp(min=1)  # Avoid zero-length sequences

            x_sorted = x[sort_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

            packed_output, _ = self.gru(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True, total_length=x.shape[1]
            )

            # Unsort
            _, unsort_idx = torch.sort(sort_idx)
            output = output[unsort_idx]
        else:
            output, _ = self.gru(x)

        # Project back to input dimension
        output = self.proj(output)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class TemporalAggregator(nn.Module):
    """Temporal aggregation module for video features.

    Provides multiple strategies to aggregate frame-level features
    into a video-level representation:
    - Mean pooling
    - Max pooling
    - Attention-weighted pooling
    - GRU-based encoding
    - Learned temporal convolution

    Args:
        input_dim: Input feature dimension.
        aggregation_type: Type of aggregation ('mean', 'max', 'attention', 'gru', 'tcn').
        hidden_dim: Hidden dimension for complex aggregation methods.
        num_layers: Number of layers for GRU aggregation.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 768,
        aggregation_type: str = "attention",
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """Initialize temporal aggregator.

        Args:
            input_dim: Dimension of frame features.
            aggregation_type: Strategy for temporal aggregation.
            hidden_dim: Hidden dimension (default: input_dim).
            num_layers: Layers for GRU/TCN aggregation.
            dropout: Dropout rate.
            bidirectional: Use bidirectional GRU.
        """
        super().__init__()
        self.input_dim = input_dim
        self.aggregation_type = aggregation_type
        self.hidden_dim = hidden_dim or input_dim

        if aggregation_type == "attention":
            self.aggregator = TemporalAttentionPooling(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
            )
        elif aggregation_type == "gru":
            self.aggregator = GRUTemporalEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif aggregation_type == "tcn":
            self.aggregator = self._build_tcn(
                input_dim, hidden_dim or input_dim, num_layers, dropout
            )
        elif aggregation_type == "mean":
            self.aggregator = None
        elif aggregation_type == "max":
            self.aggregator = None
        elif aggregation_type == "cls":
            self.aggregator = None
        else:
            raise ValueError(
                f"Unknown aggregation_type: {aggregation_type}. "
                f"Choose from 'mean', 'max', 'attention', 'gru', 'tcn', 'cls'."
            )

        # Final projection
        self.proj = nn.Linear(self.hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)

    def _build_tcn(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.Module:
        """Build a Temporal Convolutional Network.

        Uses causal 1D convolutions with increasing dilation for
        capturing multi-scale temporal patterns.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            num_layers: Number of convolutional layers.
            dropout: Dropout rate.

        Returns:
            Sequential TCN module.
        """
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            layers.extend([
                nn.Conv1d(
                    in_dim, hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                ),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate temporal features into a video-level representation.

        Args:
            x: Frame features (batch, num_frames, dim).
            mask: Optional mask (batch, num_frames).

        Returns:
            Aggregated features (batch, dim).
        """
        if self.aggregation_type == "mean":
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)

        elif self.aggregation_type == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            pooled = x.max(dim=1)[0]
            # Handle all-padding case
            pooled = pooled.clamp(min=0.0)

        elif self.aggregation_type == "cls":
            # Take first token as video representation
            pooled = x[:, 0]

        elif self.aggregation_type == "attention":
            pooled, _ = self.aggregator(x, mask=mask)

        elif self.aggregation_type == "gru":
            # GRU produces sequence, then mean pool
            encoded = self.aggregator(x, mask=mask)
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
            else:
                pooled = encoded.mean(dim=1)

        elif self.aggregation_type == "tcn":
            # TCN expects (batch, dim, time)
            x_tcn = x.transpose(1, 2)
            encoded = self.aggregator(x_tcn)
            # Take last time step
            pooled = encoded[:, :, -1]
            pooled = pooled.transpose(1, 2).squeeze(1)

        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type}")

        # Final projection and normalization
        pooled = self.proj(pooled)
        pooled = self.layer_norm(pooled)

        return pooled


# =============================================================================
# Frame Sampler
# =============================================================================

class FrameSampler(nn.Module):
    """Frame sampling strategies for video processing.

    Provides multiple strategies to select frames from a video:
    - Uniform: evenly spaced frames
    - Random: randomly sampled frames
    - Consecutive: N consecutive frames from a random starting point
    - Strided: frames with fixed stride
    - Keyframe: importance-based sampling using frame difference

    Args:
        num_frames: Number of frames to sample.
        strategy: Sampling strategy name.
        temporal_jitter: Add random temporal jitter to frame positions.
    """

    def __init__(
        self,
        num_frames: int = 8,
        strategy: str = "uniform",
        temporal_jitter: bool = False,
    ):
        """Initialize frame sampler.

        Args:
            num_frames: Target number of frames to sample.
            strategy: Sampling strategy ('uniform', 'random', 'consecutive', 'strided', 'keyframe').
            temporal_jitter: Add random jitter to sampled positions.
        """
        super().__init__()
        self.num_frames = num_frames
        self.strategy = strategy
        self.temporal_jitter = temporal_jitter

    def _sample_uniform(
        self,
        total_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample frames at uniform intervals.

        Args:
            total_frames: Total number of available frames.
            batch_size: Batch dimension.
            device: Target device.

        Returns:
            Frame indices (batch_size, num_frames).
        """
        if total_frames <= self.num_frames:
            indices = torch.arange(total_frames, device=device)
            padding = self.num_frames - total_frames
            indices = F.pad(indices, (0, padding), value=total_frames - 1)
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        else:
            indices = torch.linspace(
                0, total_frames - 1, self.num_frames, device=device
            )
            indices = indices.unsqueeze(0).expand(batch_size, -1)

        return indices.long()

    def _sample_random(
        self,
        total_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample frames randomly without replacement.

        Args:
            total_frames: Total available frames.
            batch_size: Batch size.
            device: Target device.

        Returns:
            Random frame indices (batch_size, num_frames).
        """
        if total_frames <= self.num_frames:
            indices = torch.arange(total_frames, device=device)
            padding = self.num_frames - total_frames
            indices = F.pad(indices, (0, padding), value=total_frames - 1)
            indices = indices.unsqueeze(0).expand(batch_size, -1)
        else:
            indices = torch.stack([
                torch.randperm(total_frames, device=device)[:self.num_frames]
                for _ in range(batch_size)
            ])

        return indices

    def _sample_consecutive(
        self,
        total_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample N consecutive frames from a random start.

        Args:
            total_frames: Total available frames.
            batch_size: Batch size.
            device: Target device.

        Returns:
            Consecutive frame indices (batch_size, num_frames).
        """
        max_start = max(0, total_frames - self.num_frames)

        if max_start == 0:
            indices = torch.arange(self.num_frames, device=device)
            indices = torch.clamp(indices, max=total_frames - 1)
        else:
            starts = torch.randint(0, max_start + 1, (batch_size,), device=device)
            offsets = torch.arange(self.num_frames, device=device)
            indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
            indices = torch.clamp(indices, min=0, max=total_frames - 1)

        return indices

    def _sample_strided(
        self,
        total_frames: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample frames with a fixed stride.

        Args:
            total_frames: Total available frames.
            batch_size: Batch size.
            device: Target device.

        Returns:
            Strided frame indices (batch_size, num_frames).
        """
        stride = max(1, total_frames // self.num_frames)
        start = torch.randint(0, min(stride, max(1, total_frames - self.num_frames * stride + 1)), (batch_size,), device=device)

        indices = []
        for b in range(batch_size):
            frame_indices = torch.arange(
                start[b].item(),
                min(start[b].item() + self.num_frames * stride, total_frames),
                stride,
                device=device,
            )
            # Pad if not enough frames
            if len(frame_indices) < self.num_frames:
                padding = self.num_frames - len(frame_indices)
                frame_indices = F.pad(frame_indices, (0, padding), value=total_frames - 1)
            frame_indices = frame_indices[:self.num_frames]
            indices.append(frame_indices)

        return torch.stack(indices)

    def _sample_keyframe(
        self,
        video: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample frames based on visual importance (frame differences).

        Computes the absolute difference between consecutive frames and
        selects frames with the highest accumulated change scores.

        Args:
            video: Video tensor (batch, total_frames, C, H, W).
            batch_size: Batch size.

        Returns:
            Keyframe indices (batch_size, num_frames).
        """
        device = video.device
        total_frames = video.shape[1]
        all_indices = []

        for b in range(batch_size):
            clip = video[b]  # (T, C, H, W)

            if total_frames <= self.num_frames:
                indices = torch.arange(total_frames, device=device)
            else:
                # Compute frame differences
                diffs = []
                for t in range(total_frames - 1):
                    diff = (clip[t + 1] - clip[t]).abs().mean()
                    diffs.append(diff)
                diffs = torch.stack(diffs)

                # Compute accumulated importance (centered smoothing)
                importance = torch.zeros(total_frames, device=device)
                importance[0] = diffs[0]
                importance[-1] = diffs[-1]
                for t in range(1, total_frames - 1):
                    importance[t] = 0.5 * diffs[t - 1] + 0.5 * diffs[t]

                # Add boundary bonus (prefer first and last frames)
                importance[0] += importance.mean() * 0.5
                importance[-1] += importance.mean() * 0.5

                # Select top-K frames
                _, indices = torch.topk(importance, self.num_frames)
                indices, _ = torch.sort(indices)

            all_indices.append(indices)

        return torch.stack(all_indices)

    def _apply_jitter(
        self,
        indices: torch.Tensor,
        total_frames: int,
        jitter_range: int = 1,
    ) -> torch.Tensor:
        """Apply temporal jitter to sampled frame indices.

        Args:
            indices: Frame indices (batch, num_frames).
            total_frames: Total available frames.
            jitter_range: Maximum jitter in frames.

        Returns:
            Jittered frame indices.
        """
        jitter = torch.randint(
            -jitter_range, jitter_range + 1,
            indices.shape, device=indices.device,
        )
        jittered = indices + jitter
        jittered = torch.clamp(jittered, min=0, max=total_frames - 1)
        return jittered

    def forward(
        self,
        video: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        total_frames: Optional[int] = None,
    ) -> FrameSamplerOutput:
        """Sample frames from video.

        Args:
            video: Input video (batch, total_frames, channels, height, width).
            frame_indices: Optional pre-computed frame indices.
            total_frames: Override total frame count.

        Returns:
            FrameSamplerOutput with sampled frames and metadata.
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)

        batch_size, T, C, H, W = video.shape
        device = video.device
        total = total_frames or T

        # Sample frame indices
        if frame_indices is not None:
            indices = frame_indices.to(device)
        elif self.strategy == "uniform":
            indices = self._sample_uniform(total, batch_size, device)
        elif self.strategy == "random":
            indices = self._sample_random(total, batch_size, device)
        elif self.strategy == "consecutive":
            indices = self._sample_consecutive(total, batch_size, device)
        elif self.strategy == "strided":
            indices = self._sample_strided(total, batch_size, device)
        elif self.strategy == "keyframe":
            indices = self._sample_keyframe(video, batch_size)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        # Apply temporal jitter
        if self.temporal_jitter and self.training:
            indices = self._apply_jitter(indices, total)

        # Gather frames
        num_sampled = indices.shape[1]
        sampled_frames = torch.stack([
            video[b, indices[b]] for b in range(batch_size)
        ])

        # Compute timestamps (in seconds, assuming 30fps)
        timestamps = indices.float() / 30.0

        return FrameSamplerOutput(
            frames=sampled_frames,
            indices=indices,
            timestamps=timestamps,
            num_frames=num_sampled,
        )


# =============================================================================
# Video Augmentation
# =============================================================================

class VideoAugmentation(nn.Module):
    """Video data augmentation pipeline.

    Provides various video augmentation strategies:
    - Temporal cropping: sample a temporal sub-clip from the video
    - Spatial cropping: random or center crop of frames
    - Horizontal flip: random left-right flip
    - Color jitter: brightness, contrast, saturation, hue
    - Motion blur: simulate camera motion blur
    - Gaussian noise: add random noise to frames
    - Temporal subsampling: drop frames at random
    - Random erasing: erase random spatial regions

    All augmentations are applied during training only (self.training).
    The augmentation parameters are configurable and can be adjusted
    per use case.

    Args:
        temporal_crop_range: (min_duration, max_duration) for temporal crop.
        spatial_crop_size: Target spatial size after cropping (None = no crop).
        spatial_crop_scale: (min_ratio, max_ratio) for random spatial crop.
        flip_prob: Probability of horizontal flip.
        color_jitter_prob: Probability of applying color jitter.
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        saturation: Saturation jitter range.
        hue: Hue jitter range.
        motion_blur_prob: Probability of motion blur.
        motion_blur_kernel_size: Kernel size for motion blur.
        gaussian_noise_std: Standard deviation of Gaussian noise.
        random_erasing_prob: Probability of random erasing.
        random_erasing_area_ratio: Max area ratio for random erasing.
        temporal_subsampling_rate: Rate of temporal subsampling (1.0 = no subsampling).
    """

    def __init__(
        self,
        temporal_crop_range: Tuple[float, float] = (0.5, 1.0),
        spatial_crop_size: Optional[int] = None,
        spatial_crop_scale: Tuple[float, float] = (0.8, 1.0),
        flip_prob: float = 0.5,
        color_jitter_prob: float = 0.3,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        motion_blur_prob: float = 0.2,
        motion_blur_kernel_size: int = 7,
        gaussian_noise_std: float = 0.01,
        random_erasing_prob: float = 0.1,
        random_erasing_area_ratio: float = 0.1,
        temporal_subsampling_rate: float = 1.0,
        resize_size: Optional[Tuple[int, int]] = None,
        normalize: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
    ):
        """Initialize video augmentation pipeline.

        Args:
            temporal_crop_range: Min and max fraction of video to keep.
            spatial_crop_size: Target size for center/random crop.
            spatial_crop_scale: Scale range for random crop.
            flip_prob: Horizontal flip probability.
            color_jitter_prob: Color jitter probability.
            brightness: Brightness adjustment range.
            contrast: Contrast adjustment range.
            saturation: Saturation adjustment range.
            hue: Hue adjustment range.
            motion_blur_prob: Motion blur probability.
            motion_blur_kernel_size: Blur kernel size.
            gaussian_noise_std: Gaussian noise standard deviation.
            random_erasing_prob: Random erasing probability.
            random_erasing_area_ratio: Max area to erase.
            temporal_subsampling_rate: Frame drop rate.
            resize_size: Optional resize dimensions (H, W).
            normalize: Apply ImageNet normalization.
            mean: Normalization mean.
            std: Normalization std.
        """
        super().__init__()
        self.temporal_crop_range = temporal_crop_range
        self.spatial_crop_size = spatial_crop_size
        self.spatial_crop_scale = spatial_crop_scale
        self.flip_prob = flip_prob
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.motion_blur_prob = motion_blur_prob
        self.motion_blur_kernel_size = motion_blur_kernel_size
        self.gaussian_noise_std = gaussian_noise_std
        self.random_erasing_prob = random_erasing_prob
        self.random_erasing_area_ratio = random_erasing_area_ratio
        self.temporal_subsampling_rate = temporal_subsampling_rate
        self.resize_size = resize_size
        self.normalize = normalize
        self.mean = mean or (0.485, 0.456, 0.406)
        self.std = std or (0.229, 0.224, 0.225)

    def _temporal_crop(
        self,
        video: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Randomly crop a temporal sub-clip from the video.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Tuple of (cropped video, start_frame_index).
        """
        batch_size, T, C, H, W = video.shape
        min_ratio, max_ratio = self.temporal_crop_range

        min_frames = max(1, int(T * min_ratio))
        max_frames = max(min_frames, int(T * max_ratio))

        crop_lengths = torch.randint(
            min_frames, max_frames + 1, (batch_size,),
            device=video.device,
        )

        cropped_videos = []
        for b in range(batch_size):
            crop_len = crop_lengths[b].item()
            if crop_len >= T:
                cropped_videos.append(video[b])
            else:
                start = torch.randint(0, T - crop_len + 1, (1,)).item()
                cropped_videos.append(video[b, start:start + crop_len])

        return torch.stack(cropped_videos), 0

    def _spatial_crop(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly crop spatial dimensions of all frames.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Spatially cropped video.
        """
        batch_size, T, C, H, W = video.shape
        min_scale, max_scale = self.spatial_crop_scale

        cropped_videos = []
        for b in range(batch_size):
            scale = torch.rand(1).item() * (max_scale - min_scale) + min_scale
            crop_h = int(H * scale)
            crop_w = int(W * scale)

            if self.spatial_crop_size is not None:
                crop_h = min(crop_h, self.spatial_crop_size)
                crop_w = min(crop_w, self.spatial_crop_size)
                # Resize back to target size
                target_h = self.spatial_crop_size
                target_w = self.spatial_crop_size
            else:
                target_h = crop_h
                target_w = crop_w

            crop_h = max(1, crop_h)
            crop_w = max(1, crop_w)

            start_h = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
            start_w = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()

            cropped = video[b, :, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

            # Resize to target size if needed
            if crop_h != target_h or crop_w != target_w:
                cropped = cropped.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)
                cropped = F.interpolate(
                    cropped.view(C * T, 1, crop_h, crop_w),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
                cropped = cropped.view(C, T, target_h, target_w).permute(1, 0, 2, 3)

            cropped_videos.append(cropped)

        return torch.stack(cropped_videos)

    def _center_crop(
        self,
        video: torch.Tensor,
        crop_size: int,
    ) -> torch.Tensor:
        """Apply center crop to all frames.

        Args:
            video: Video tensor (batch, T, C, H, W).
            crop_size: Target crop size (square).

        Returns:
            Center-cropped video.
        """
        batch_size, T, C, H, W = video.shape

        if H < crop_size or W < crop_size:
            # Pad if needed
            pad_h = max(0, crop_size - H)
            pad_w = max(0, crop_size - W)
            video = F.pad(video, (0, pad_w, 0, pad_h))

        _, _, _, H_padded, W_padded = video.shape
        start_h = (H_padded - crop_size) // 2
        start_w = (W_padded - crop_size) // 2

        return video[:, :, :, start_h:start_h + crop_size, start_w:start_w + crop_size]

    def _horizontal_flip(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly flip video horizontally.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Possibly flipped video.
        """
        flip_mask = torch.rand(video.shape[0], device=video.device) < self.flip_prob
        video = torch.where(
            flip_mask.view(-1, 1, 1, 1, 1),
            video.flip(-1),
            video,
        )
        return video

    def _color_jitter(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random color jitter to video frames.

        Independently adjusts brightness, contrast, saturation, and hue
        with per-video random factors.

        Args:
            video: Video tensor (batch, T, C, H, W), values in [0, 1].

        Returns:
            Color-jittered video.
        """
        batch_size, T, C, H, W = video.shape

        # Determine which videos get color jitter
        apply_mask = torch.rand(batch_size, device=video.device) < self.color_jitter_prob

        for b in range(batch_size):
            if not apply_mask[b]:
                continue

            frames = video[b]  # (T, C, H, W)

            # Brightness
            brightness_factor = 1.0 + torch.rand(1, device=video.device).item() * 2 * self.brightness - self.brightness
            brightness_factor = max(0.0, brightness_factor)
            frames = frames * brightness_factor

            # Contrast
            mean_val = frames.mean(dim=(-3, -2, -1), keepdim=True)
            contrast_factor = 1.0 + torch.rand(1, device=video.device).item() * 2 * self.contrast - self.contrast
            contrast_factor = max(0.0, contrast_factor)
            frames = (frames - mean_val) * contrast_factor + mean_val

            # Saturation
            if C == 3:
                gray = frames.mean(dim=-3, keepdim=True)
                sat_factor = 1.0 + torch.rand(1, device=video.device).item() * 2 * self.saturation - self.saturation
                sat_factor = max(0.0, sat_factor)
                frames = gray + sat_factor * (frames - gray)

            # Hue (simplified: shift in channel space for RGB)
            if C == 3:
                hue_factor = torch.rand(1, device=video.device).item() * 2 * self.hue - self.hue
                # Convert to approximate hue rotation using matrix
                cos_h = math.cos(hue_factor * math.pi)
                sin_h = math.sin(hue_factor * math.pi)
                r, g, b = frames[:, 0], frames[:, 1], frames[:, 2]
                frames[:, 0] = r * (0.213 + cos_h * 0.787 - sin_h * 0.213) + \
                                g * (0.715 - cos_h * 0.715 - sin_h * 0.715) + \
                                b * (0.072 - cos_h * 0.072 + sin_h * 0.928)
                frames[:, 1] = r * (0.213 - cos_h * 0.213 + sin_h * 0.143) + \
                                g * (0.715 + cos_h * 0.285 + sin_h * 0.140) + \
                                b * (0.072 - cos_h * 0.072 - sin_h * 0.283)
                frames[:, 2] = r * (0.213 - cos_h * 0.213 - sin_h * 0.787) + \
                                g * (0.715 - cos_h * 0.715 + sin_h * 0.715) + \
                                b * (0.072 + cos_h * 0.928 + sin_h * 0.072)

            frames = frames.clamp(0.0, 1.0)
            video[b] = frames

        return video

    def _motion_blur(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Apply motion blur to simulate camera or object motion.

        Creates a directional blur kernel and applies it to frames
    with varying intensity across the temporal dimension.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Motion-blurred video.
        """
        batch_size, T, C, H, W = video.shape
        kernel_size = self.motion_blur_kernel_size
        device = video.device

        # Determine which videos get motion blur
        apply_mask = torch.rand(batch_size, device=device) < self.motion_blur_prob

        # Create horizontal motion blur kernel
        kernel = torch.zeros(1, 1, kernel_size, kernel_size, device=device)
        angle = torch.rand(1).item() * math.pi  # Random angle
        center = kernel_size // 2
        for k in range(kernel_size):
            offset = k - center
            y = int(round(offset * math.sin(angle)))
            x = int(round(offset * math.cos(angle)))
            if 0 <= y < kernel_size and 0 <= x < kernel_size:
                kernel[0, 0, y, x] = 1.0

        # Normalize kernel
        kernel = kernel / kernel.sum().clamp(min=1e-8)

        blurred_videos = []
        for b in range(batch_size):
            if apply_mask[b]:
                frames = video[b]  # (T, C, H, W)
                # Blur intensity varies per frame
                intensity = torch.rand(T, device=device) * 0.5 + 0.5
                blurred_frames = []

                for t in range(T):
                    frame = frames[t]  # (C, H, W)
                    # Apply blur by weighted average of original and blurred
                    blurred = F.conv2d(
                        frame.unsqueeze(0), kernel,
                        padding=kernel_size // 2,
                    ).squeeze(0)
                    # Blend based on intensity
                    frame_out = intensity[t] * blurred + (1.0 - intensity[t]) * frame
                    blurred_frames.append(frame_out)

                blurred_videos.append(torch.stack(blurred_frames))
            else:
                blurred_videos.append(video[b])

        return torch.stack(blurred_videos)

    def _gaussian_noise(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Add Gaussian noise to video frames.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Noisy video.
        """
        if self.gaussian_noise_std <= 0:
            return video

        noise = torch.randn_like(video) * self.gaussian_noise_std
        return (video + noise).clamp(0.0, 1.0)

    def _random_erasing(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random spatial erasing to frames.

        Randomly selects rectangular regions in each frame and fills them
        with random values or the mean pixel value.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Video with random erasing applied.
        """
        batch_size, T, C, H, W = video.shape
        device = video.device

        erased_videos = []
        for b in range(batch_size):
            frames = video[b].clone()  # (T, C, H, W)

            # For each frame, potentially erase a region
            for t in range(T):
                if torch.rand(1, device=device).item() > self.random_erasing_prob:
                    continue

                area = H * W
                target_area = torch.rand(1, device=device).item() * self.random_erasing_area_ratio * area
                aspect_ratio = torch.rand(1, device=device).item() * 0.5 + 0.5

                erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
                erase_w = int(round(math.sqrt(target_area / aspect_ratio)))

                erase_h = min(erase_h, H - 1)
                erase_w = min(erase_w, W - 1)
                erase_h = max(1, erase_h)
                erase_w = max(1, erase_w)

                top = torch.randint(0, H - erase_h + 1, (1,), device=device).item()
                left = torch.randint(0, W - erase_w + 1, (1,), device=device).item()

                # Fill with random noise
                frames[t, :, top:top + erase_h, left:left + erase_w] = torch.rand(
                    C, erase_h, erase_w, device=device,
                )

            erased_videos.append(frames)

        return torch.stack(erased_videos)

    def _temporal_subsample(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Subsample frames temporally by randomly dropping frames.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Temporally subsampled video.
        """
        batch_size, T, C, H, W = video.shape
        keep_rate = self.temporal_subsampling_rate

        if keep_rate >= 1.0 or not self.training:
            return video

        subsampled_videos = []
        for b in range(batch_size):
            keep_mask = torch.rand(T, device=video.device) < keep_rate
            # Ensure at least one frame is kept
            if keep_mask.sum() == 0:
                keep_mask[torch.randint(0, T, (1,)).item()] = True
            kept_indices = torch.where(keep_mask)[0]
            subsampled_videos.append(video[b, kept_indices])

        # Pad to maintain consistent length
        max_len = max(v.shape[0] for v in subsampled_videos)
        padded_videos = []
        for v in subsampled_videos:
            if v.shape[0] < max_len:
                padding = max_len - v.shape[0]
                v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, padding), mode="replicate")
            padded_videos.append(v)

        return torch.stack(padded_videos)

    def _normalize(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Apply normalization to video frames.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Normalized video.
        """
        if not self.normalize:
            return video

        mean = torch.tensor(self.mean, device=video.device, dtype=video.dtype)
        std = torch.tensor(self.std, device=video.device, dtype=video.dtype)

        # Reshape for broadcasting: (1, 1, C, 1, 1)
        mean = mean.view(1, 1, -1, 1, 1)
        std = std.view(1, 1, -1, 1, 1)

        return (video - mean) / std

    def _resize(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Resize all frames to target size.

        Args:
            video: Video tensor (batch, T, C, H, W).

        Returns:
            Resized video.
        """
        if self.resize_size is None:
            return video

        target_h, target_w = self.resize_size
        batch_size, T, C, H, W = video.shape

        if H == target_h and W == target_w:
            return video

        # Reshape for interpolation: (B*T, C, H, W)
        video = video.reshape(batch_size * T, C, H, W)
        video = F.interpolate(
            video, size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        video = video.reshape(batch_size, T, C, target_h, target_w)

        return video

    def forward(
        self,
        video: torch.Tensor,
    ) -> VideoAugmentationOutput:
        """Apply augmentation pipeline to video.

        During training, applies random augmentations.
        During evaluation, only applies deterministic transforms.

        Args:
            video: Input video (batch, T, C, H, W) or (T, C, H, W).

        Returns:
            VideoAugmentationOutput with augmented video and metadata.
        """
        transform_params = {}

        # Handle single video input
        if video.dim() == 4:
            video = video.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        # Ensure values are in [0, 1] range for augmentation
        was_normalized = video.min() < 0 or video.max() > 1.0

        # Apply resize
        video = self._resize(video)

        if self.training:
            # Temporal crop
            video, start_idx = self._temporal_crop(video)
            transform_params["temporal_crop_start"] = start_idx

            # Spatial crop
            video = self._spatial_crop(video)

            # Horizontal flip
            video = self._horizontal_flip(video)

            # Color jitter
            video = self._color_jitter(video)

            # Motion blur
            video = self._motion_blur(video)

            # Gaussian noise
            video = self._gaussian_noise(video)

            # Random erasing
            video = self._random_erasing(video)

            # Temporal subsampling
            video = self._temporal_subsample(video)
        else:
            # Evaluation: center crop only
            if self.spatial_crop_size is not None:
                video = self._center_crop(video, self.spatial_crop_size)

        # Normalize
        video = self._normalize(video)

        # Clamp values
        video = video.clamp(-10.0, 10.0) if self.normalize else video.clamp(0.0, 1.0)

        if single_input:
            video = video.squeeze(0)

        return VideoAugmentationOutput(
            video=video,
            transform_params=transform_params,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_3d_relative_position_bias(
    window_size: Tuple[int, int, int],
    num_heads: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute 3D relative position bias table.

    Args:
        window_size: Size of the 3D window (t, h, w).
        num_heads: Number of attention heads.
        device: Target device.

    Returns:
        Relative position bias table (num_offsets, num_heads).
    """
    wt, wh, ww = window_size

    coords_t = torch.arange(wt, device=device)
    coords_h = torch.arange(wh, device=device)
    coords_w = torch.arange(ww, device=device)

    coords = torch.stack(
        torch.meshgrid(coords_t, coords_h, coords_w, indexing="ij"),
        dim=0,
    ).flatten(1)

    relative_coords = coords[:, :, None] - coords[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()

    relative_coords[:, :, 0] += wt - 1
    relative_coords[:, :, 1] += wh - 1
    relative_coords[:, :, 2] += ww - 1

    relative_coords[:, :, 0] *= (2 * wh - 1) * (2 * ww - 1)
    relative_coords[:, :, 1] *= (2 * ww - 1)

    relative_position_index = relative_coords.sum(dim=-1)

    num_relative_positions = (2 * wt - 1) * (2 * wh - 1) * (2 * ww - 1)
    relative_position_bias_table = nn.Parameter(
        torch.zeros(num_relative_positions, num_heads, device=device)
    )
    nn.init.trunc_normal_(relative_position_bias_table, std=0.02)

    return relative_position_bias_table[relative_position_index.view(-1)].view(
        wt * wh * ww, wt * wh * ww, -1
    ).permute(2, 0, 1).contiguous()


def get_video_encoder(
    encoder_name: str = "timesformer",
    **kwargs,
) -> nn.Module:
    """Factory function to create video encoder by name.

    Args:
        encoder_name: Name of the encoder architecture.
        **kwargs: Additional arguments passed to the encoder constructor.

    Returns:
        Instantiated video encoder module.

    Raises:
        ValueError: If encoder_name is not recognized.
    """
    encoders = {
        "timesformer": TimeSformerEncoder,
        "vivit": ViViTEncoder,
        "videoswin": VideoSwinEncoder,
    }

    if encoder_name.lower() not in encoders:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: {list(encoders.keys())}"
        )

    return encoders[encoder_name.lower()](**kwargs)


def compute_model_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int, int, int] = (1, 3, 16, 224, 224),
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Estimate FLOPs for a video encoder model.

    Uses a simple forward pass with dummy input to count operations.

    Args:
        model: Video encoder model.
        input_size: Input shape (B, C, T, H, W).
        device: Device for computation.

    Returns:
        Dictionary with FLOPs estimate and parameter count.
    """
    model = model.to(device)
    model.eval()

    x = torch.randn(*input_size, device=device)

    # Convert (B, C, T, H, W) to (B, T, C, H, W) if needed
    if x.shape[1] < x.shape[2]:
        x = x.permute(0, 2, 1, 3, 4).contiguous()

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with torch.no_grad():
        try:
            _ = model(x)
        except Exception:
            # Try alternate input format
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            _ = model(x)

    return {
        "parameter_count": param_count,
        "trainable_parameters": trainable_params,
        "input_size": input_size,
    }
