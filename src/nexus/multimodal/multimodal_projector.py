"""
Nexus Multimodal Projector Module
===================================
Comprehensive set of multimodal projector implementations for bridging
visual, audio, and video encoders with language model backbones.

Supported Projector Architectures:
    - LinearProjector: Stacked linear layers with GELU, LayerNorm, dropout
    - MLPProjector: LLaVA-style 2-layer MLP vision-to-text projection
    - QFormerProjector: BLIP-2 style learned-query cross-attention
    - ResamplerProjector: Perceiver-resample with fixed-size output tokens
    - CAbstractor: LLaVA-OneVision C-Abstractor with stacked cross/self-attention
    - ModalityAdapter: Lightweight LayerNorm + Linear wrapper with residual
    - ProjectorFactory: Config-driven projector construction

All components are fully self-contained, use only torch / torch.nn /
torch.nn.functional / math / typing, and support mixed-precision (BF16)
and gradient checkpointing where applicable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Union,
    Sequence,
    Callable,
)

__all__ = [
    "LinearProjector",
    "MLPProjector",
    "QFormerProjector",
    "ResamplerProjector",
    "CAbstractor",
    "ModalityAdapter",
    "ProjectorFactory",
    "ProjectorConfig",
    "MultiHeadSelfAttention",
    "MultiHeadCrossAttention",
    "FeedForwardNetwork",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SinusoidalPositionalEmbedding",
    "LearnedPositionalEmbedding",
]


# ---------------------------------------------------------------------------
#  Utility building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization.

    Normalises the input by its RMS (without subtracting the mean), which
    is computationally cheaper and common in LLaMA-family models.

        x_out = x / sqrt(mean(x^2) + eps) * weight

    Args:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
        bias: Whether to include a learnable bias (default ``False``).
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalisation.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Normalised tensor of the same shape.
        """
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        output = (x_normed * self.weight).to(x.dtype)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, bias={self.bias is not None}"


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal (non-learned) positional embedding.

    Generates positional encodings using sine/cosine functions of different
    frequencies, following Vaswani et al. (2017).

    Args:
        embed_dim: Total embedding dimension (must be even).
        max_seq_len: Maximum sequence length supported.
        dropout: Dropout applied after embedding addition.
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, dim)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, x: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        """Add sinusoidal positional embeddings.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.
            offset: Starting position index (for incremental generation).

        Returns:
            Tensor of the same shape with positional encodings added.
        """
        seq_len = x.size(1)
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {offset + seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )
        return self.dropout(x + self.pe[:, offset : offset + seq_len, :])

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, max_seq_len={self.max_seq_len}"


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding table.

    Args:
        embed_dim: Embedding dimension.
        max_seq_len: Maximum number of positions.
        dropout: Dropout applied after addition.
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings.

        Args:
            x: Input tensor of shape ``(batch, seq_len, embed_dim)``.

        Returns:
            Tensor with positional embeddings added.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_embed = self.embedding(positions).unsqueeze(0)
        return self.dropout(x + pos_embed)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotary embeddings to query and key tensors for use in
    attention mechanisms.  Supports linearly-scaled RoPE for longer
    sequence lengths.

    Args:
        head_dim: Per-head dimension (must be even).
        max_seq_len: Maximum sequence length.
        base: Base for computing frequency bands (default 10000).
        scaling_factor: Linear scaling factor for long-context extrapolation.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            * scaling_factor
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key.

        Args:
            q: Query tensor of shape ``(batch, heads, seq_len, head_dim)``.
            k: Key tensor of shape ``(batch, heads, seq_len, head_dim)``.
            offset: Position offset for incremental decoding.

        Returns:
            Tuple of (rotated_q, rotated_k) with the same shapes.
        """
        seq_len = q.size(2)
        end = offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Requested positions up to {end} exceed max_seq_len={self.max_seq_len}"
            )
        cos = self.cos_cached[:, :, offset:end, :]  # (1, 1, seq, dim)
        sin = self.sin_cached[:, :, offset:end, :]

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.float().chunk(2, dim=-1)
            rotated = torch.cat([-x2, x1], dim=-1)
            return (x.float() * cos + rotated * sin).to(x.dtype)

        return _rotate(q), _rotate(k)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE, QK-norm, and flash compat.

    Args:
        embed_dim: Total model dimension.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension (defaults to ``embed_dim // num_heads``).
        dropout: Dropout on attention weights.
        bias: Include bias in projections.
        use_rope: Apply rotary positional embeddings.
        max_seq_len: Maximum sequence length for RoPE.
        use_qk_norm: Apply LayerNorm to Q and K before attention.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        use_rope: bool = False,
        max_seq_len: int = 2048,
        use_qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        if self.inner_dim != embed_dim:
            raise ValueError(
                f"inner_dim ({self.inner_dim}) != embed_dim ({embed_dim})"
            )

        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len=max_seq_len
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-head self-attention forward.

        Args:
            x: Input tensor ``(batch, seq_len, embed_dim)``.
            attention_mask: Boolean mask ``(batch, seq_len, seq_len)`` where
                ``True`` means *attend*. ``None`` means full attention.
            past_key_value: Cached ``(key, value)`` tensors for incremental
                decoding. Each is ``(batch, heads, past_len, head_dim)``.

        Returns:
            Tuple of (output, attention_weights).
            - output: ``(batch, seq_len, embed_dim)``
            - attention_weights: ``None`` (or optionally ``(batch, heads, q, k)``)
        """
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if self.use_rope:
            offset = 0 if past_key_value is None else past_key_value[0].size(2)
            q, k = self.rope(q, k, offset=offset)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf")
                )
            else:
                attn_weights = attn_weights + attention_mask.unsqueeze(1).unsqueeze(2)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.inner_dim)
        out = self.proj_dropout(self.out_proj(out))

        return out, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, use_rope={self.use_rope}"
        )


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention for attending from queries to key/value.

    Used extensively in Q-Former, Resampler, and C-Abstractor projectors.

    Args:
        query_dim: Dimension of the query input.
        kv_dim: Dimension of key/value inputs.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension (computed from query_dim if ``None``).
        dropout: Attention dropout probability.
        bias: Whether linear projections use bias.
        use_qk_norm: Apply LayerNorm to Q and K.
        use_rope: Apply RoPE to Q and K.
        max_seq_len: Max sequence length (only used when ``use_rope=True``).
    """

    def __init__(
        self,
        query_dim: int = 1024,
        kv_dim: int = 1024,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        use_qk_norm: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else query_dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        if self.inner_dim != query_dim:
            raise ValueError(
                f"inner_dim ({self.inner_dim}) != query_dim ({query_dim})"
            )

        self.q_proj = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.k_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, query_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)

        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len=max_seq_len
            )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Cross-attention forward.

        Args:
            query: Query tensor ``(batch, q_len, query_dim)``.
            key: Key tensor ``(batch, kv_len, kv_dim)``.
            value: Value tensor ``(batch, kv_len, kv_dim)``.
            attention_mask: Boolean mask ``(batch, q_len, kv_len)`` where
                ``True`` means *attend*.

        Returns:
            Tuple of (output, attention_weights).
        """
        B, Q, _ = query.shape
        KV = key.size(1)

        q = (
            self.q_proj(query)
            .view(B, Q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key)
            .view(B, KV, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(value)
            .view(B, KV, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            q, k = self.rope(q, k)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(1), float("-inf")
                )
            else:
                attn_weights = attn_weights + attention_mask.unsqueeze(1)

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.attn_dropout(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, Q, self.inner_dim)
        out = self.proj_dropout(self.out_proj(out))

        return out, attn_probs

    def extra_repr(self) -> str:
        return (
            f"query_dim={self.query_dim}, kv_dim={self.kv_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}"
        )


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network (SwiGLU / GELU variant).

    Supports:
    - Standard GELU MLP: Linear → GELU → Dropout → Linear → Dropout
    - Gated (SwiGLU) MLP:  Linear → SiLU + Linear → Gate → Linear → Dropout
    - Custom activation via ``activation`` string.

    Args:
        embed_dim: Input / output dimension.
        ffn_dim: Hidden dimension (intermediate size).  Defaults to ``4 * embed_dim``.
        dropout: Dropout probability on hidden layer and output.
        activation: Activation name — ``"gelu"``, ``"relu"``, ``"silu"``, ``"swiglu"``, ``"tanh"``.
        bias: Whether linear projections include bias.
        use_residual: If ``True``, ``forward`` adds the input as a residual.
    """

    _ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    def __init__(
        self,
        embed_dim: int = 1024,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * embed_dim
        self.activation_name = activation.lower().strip()
        self.use_residual = use_residual

        if self.activation_name == "swiglu":
            # SwiGLU: up_proj, gate_proj, down_proj
            self.up_proj = nn.Linear(embed_dim, self.ffn_dim, bias=bias)
            self.gate_proj = nn.Linear(embed_dim, self.ffn_dim, bias=bias)
            self.down_proj = nn.Linear(self.ffn_dim, embed_dim, bias=bias)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.SiLU()
        else:
            act_cls = self._ACTIVATIONS.get(self.activation_name, nn.GELU)
            self.fc1 = nn.Linear(embed_dim, self.ffn_dim, bias=bias)
            self.act = act_cls()
            self.fc2 = nn.Linear(self.ffn_dim, embed_dim, bias=bias)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        if self.activation_name == "swiglu":
            nn.init.xavier_uniform_(self.up_proj.weight)
            nn.init.xavier_uniform_(self.gate_proj.weight)
            nn.init.xavier_uniform_(self.down_proj.weight)
            if self.up_proj.bias is not None:
                nn.init.zeros_(self.up_proj.bias)
                nn.init.zeros_(self.gate_proj.bias)
                nn.init.zeros_(self.down_proj.bias)
        else:
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
                nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward.

        Args:
            x: Input tensor ``(..., embed_dim)``.

        Returns:
            Output tensor of the same shape.
        """
        residual = x

        if self.activation_name == "swiglu":
            up = self.up_proj(x)
            gate = self.act(self.gate_proj(x))
            hidden = self.dropout(up * gate)
            out = self.down_proj(hidden)
        else:
            hidden = self.fc1(x)
            hidden = self.act(hidden)
            hidden = self.dropout1(hidden)
            out = self.fc2(hidden)
            out = self.dropout2(out)

        if self.use_residual:
            out = out + residual
        return out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, ffn_dim={self.ffn_dim}, "
            f"activation={self.activation_name}"
        )


# ---------------------------------------------------------------------------
#  Activation helper (standalone function for use across projectors)
# ---------------------------------------------------------------------------

def _get_activation_fn(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of ``"gelu"``, ``"relu"``, ``"silu"``, ``"tanh"``, ``"sigmoid"``.

    Returns:
        An ``nn.Module`` activation.

    Raises:
        ValueError: If the name is not recognised.
    """
    mapping: Dict[str, Callable[[], nn.Module]] = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    key = name.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {sorted(mapping.keys())}"
        )
    return mapping[key]()


# ---------------------------------------------------------------------------
#  ProjectorConfig dataclass
# ---------------------------------------------------------------------------

class ProjectorConfig:
    """Lightweight configuration holder for projector construction.

    Mirrors the keyword arguments accepted by individual projector
    constructors so that serialisable configs can be round-tripped.

    Attributes:
        projector_type: Architecture name used by :class:`ProjectorFactory`.
        input_dim: Encoder output dimension.
        output_dim: LM input dimension.
        hidden_dim: Optional hidden / intermediate dimension.
        num_layers: Depth for stacked projector variants.
        num_heads: Attention heads for attention-based projectors.
        num_queries: Number of learned query tokens.
        dropout: Dropout probability.
        activation: Activation function name.
        use_layer_norm: Apply LayerNorm.
        use_residual: Add residual connections.
        bias: Use bias in linear projections.
        ffn_multiplier: FFN hidden dim = multiplier * hidden_dim.
        max_seq_len: Maximum visual token sequence length.
        norm_type: ``"layer_norm"`` or ``"rms_norm"``.
        pos_embed_type: ``"learned"``, ``"sinusoidal"``, or ``"none"``.
    """

    def __init__(
        self,
        projector_type: str = "mlp",
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 8,
        num_queries: int = 32,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        use_residual: bool = True,
        bias: bool = True,
        ffn_multiplier: int = 4,
        max_seq_len: int = 576,
        norm_type: str = "layer_norm",
        pos_embed_type: str = "none",
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.dropout = dropout
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.bias = bias
        self.ffn_multiplier = ffn_multiplier
        self.max_seq_len = max_seq_len
        self.norm_type = norm_type
        self.pos_embed_type = pos_embed_type

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectorConfig":
        """Deserialise from a dictionary."""
        valid = {k for k in cls.__init__.__code__.co_varnames if k != "self"}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# ===========================================================================
#  1. LinearProjector
# ===========================================================================


class LinearProjector(nn.Module):
    """Stacked linear layers with activation, normalisation and dropout.

    A simple but effective projector that maps an encoder's output space to
    the language-model's input space via a configurable number of fully-
    connected layers interleaved with GELU activations, LayerNorm and dropout.

    Architecture per layer:
        Linear(in_dim, out_dim) → GELU → Dropout

    Final layer optionally followed by LayerNorm.

    Args:
        input_dim: Dimension of the encoder features.
        output_dim: Target dimension (LM hidden size).
        num_layers: Number of linear layers (≥ 1).
        hidden_dim: Intermediate dimension. If ``None``, defaults to
            ``max(input_dim, output_dim)``. All hidden layers share this dim.
        dropout: Dropout probability applied after each activation.
        activation: Activation function name (default ``"gelu"``).
        use_layer_norm: Apply a final LayerNorm before output.
        bias: Include bias in all linear layers.
        norm_type: ``"layer_norm"`` or ``"rms_norm"`` for the final normalisation.
        residual_between: If ``True`` and num_layers is even, pair consecutive
            layers with residual connections.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_layer_norm: bool = True,
        bias: bool = True,
        norm_type: str = "layer_norm",
        residual_between: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1, got {num_layers}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim if hidden_dim is not None else max(input_dim, output_dim)
        self.dropout_p = dropout
        self.activation_name = activation.lower().strip()
        self.use_layer_norm = use_layer_norm
        self.residual_between = residual_between

        # Build layer dims: [input_dim, hidden, ..., hidden, output_dim]
        layer_dims = self._compute_layer_dims()
        self.layer_dims = layer_dims

        act_fn = _get_activation_fn(self.activation_name)

        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_d = layer_dims[i]
            out_d = layer_dims[i + 1]
            block: List[nn.Module] = [nn.Linear(in_d, out_d, bias=bias)]
            # Last layer: no activation, no dropout (just projection)
            if i < num_layers - 1:
                block.append(act_fn)
                block.append(nn.Dropout(dropout))
            layers.append(nn.Sequential(*block))

        self.linears = nn.ModuleList(layers)

        # Optional residual projection when input_dim != output_dim
        self.use_residual_proj = residual_between and (input_dim != output_dim)
        if self.use_residual_proj:
            self.residual_proj = nn.Linear(input_dim, output_dim, bias=False)

        # Final normalisation
        if use_layer_norm:
            if norm_type == "rms_norm":
                self.final_norm = RMSNorm(output_dim, eps=1e-6)
            else:
                self.final_norm = nn.LayerNorm(output_dim, eps=1e-6)
        else:
            self.final_norm = nn.Identity()

        self._init_weights()

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _compute_layer_dims(self) -> List[int]:
        """Compute per-layer input/output dimensions.

        Returns:
            List of length ``num_layers + 1`` where ``dims[i]`` is the
            input dim of layer *i* and ``dims[-1]`` equals ``output_dim``.
        """
        dims: List[int] = [self.input_dim]
        if self.num_layers == 1:
            dims.append(self.output_dim)
        else:
            for _ in range(self.num_layers - 1):
                dims.append(self.hidden_dim)
            dims.append(self.output_dim)
        return dims

    def _init_weights(self) -> None:
        """Xavier-uniform initialise all linear layers."""
        for module in self.linears:
            for child in module:
                if isinstance(child, nn.Linear):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
        if self.use_residual_proj:
            nn.init.xavier_uniform_(self.residual_proj.weight)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project encoder features to LM embedding space.

        Args:
            x: Input tensor ``(*, input_dim)`` or ``(batch, seq, input_dim)``.
            attention_mask: Optional mask (currently only used for shape
                compatibility; not applied inside linear layers).

        Returns:
            Projected tensor with last dim ``output_dim``.
        """
        residual = x if self.residual_between else None

        out = x
        for i, linear in enumerate(self.linears):
            out = linear(out)

        if residual is not None:
            if self.use_residual_proj:
                residual = self.residual_proj(residual)
            out = out + residual

        out = self.final_norm(out)
        return out

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return total number of parameters.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            Integer parameter count.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape given an input shape (excluding batch dim).

        Args:
            input_shape: Shape tuple like ``(seq_len, input_dim)`` or
                ``(input_dim,)``.

        Returns:
            Output shape as a list.
        """
        return list(input_shape[:-1]) + [self.output_dim]

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "LinearProjector",
            f"  input_dim      = {self.input_dim}",
            f"  output_dim     = {self.output_dim}",
            f"  num_layers     = {self.num_layers}",
            f"  hidden_dim     = {self.hidden_dim}",
            f"  layer_dims     = {self.layer_dims}",
            f"  dropout        = {self.dropout_p}",
            f"  activation     = {self.activation_name}",
            f"  use_layer_norm = {self.use_layer_norm}",
            f"  residual       = {self.residual_between}",
            f"  total_params   = {self.get_num_params():,}",
        ]
        return "\n".join(lines)

    def reset_parameters(self) -> None:
        """Re-initialise all weights (useful for fine-tuning resets)."""
        self._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"num_layers={self.num_layers}, hidden_dim={self.hidden_dim}"
        )


# ===========================================================================
#  2. MLPProjector
# ===========================================================================


class MLPProjector(nn.Module):
    """LLaVA-style 2-layer MLP vision-to-text projector.

    Maps visual encoder features to the language model's hidden dimension
    using a simple two-layer MLP:

        input_dim → hidden_dim (GELU) → output_dim

    This is the standard projector used in LLaVA-1.5, LLaVA-NeXT, and
    many derivative models.

    Args:
        input_dim: Visual encoder output dimension.
        output_dim: Language model hidden dimension.
        hidden_dim: Intermediate hidden size. Defaults to ``input_dim``.
        activation: Activation function (default ``"gelu"``).
        bias: Include bias in linear layers.
        dropout: Dropout probability.
        use_layer_norm: Apply LayerNorm after the last linear layer.
        use_residual: Add input residual if ``input_dim == output_dim``.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_residual: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.activation_name = activation.lower().strip()
        self.use_residual = use_residual and (input_dim == output_dim)

        act_fn = _get_activation_fn(self.activation_name)

        self.fc1 = nn.Linear(input_dim, self.hidden_dim, bias=bias)
        self.act = act_fn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.hidden_dim, output_dim, bias=bias)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim, eps=1e-6)
        else:
            self.layer_norm = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with Kaiming (fc1) and Xavier (fc2)."""
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias) if self.fc1.bias is not None else None
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias) if self.fc2.bias is not None else None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the 2-layer MLP projector.

        Args:
            x: Visual features ``(batch, seq_len, input_dim)``.
            attention_mask: Ignored (kept for API compatibility).

        Returns:
            Projected features ``(batch, seq_len, output_dim)``.
        """
        residual = x if self.use_residual else None

        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        out = self.fc2(h)

        if residual is not None:
            out = out + residual

        out = self.layer_norm(out)
        return out

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return parameter count.

        Args:
            trainable_only: Only count trainable params.

        Returns:
            Integer count.
        """
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape.

        Args:
            input_shape: e.g. ``(seq_len, input_dim)``.

        Returns:
            Output shape list.
        """
        return list(input_shape[:-1]) + [self.output_dim]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"MLPProjector\n"
            f"  input_dim    = {self.input_dim}\n"
            f"  output_dim   = {self.output_dim}\n"
            f"  hidden_dim   = {self.hidden_dim}\n"
            f"  activation   = {self.activation_name}\n"
            f"  residual     = {self.use_residual}\n"
            f"  total_params = {self.get_num_params():,}"
        )

    def reset_parameters(self) -> None:
        """Re-initialise all weights."""
        self._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}"
        )


# ===========================================================================
#  3. QFormerProjector
# ===========================================================================


class QFormerProjector(nn.Module):
    """BLIP-2 style Querying Transformer (Q-Former) projector.

    Uses a set of learnable query tokens that extract information from
    visual features via cross-attention, then refine the representations
    through self-attention layers.

    Architecture:
        1. Learned query embeddings (num_queries × hidden_dim)
        2. N cross-attention layers (queries attend to visual features)
        3. N self-attention layers (queries refine among themselves)
        4. LayerNorm
        5. Optional output projection → output_dim

    Each cross-attention layer:
        LN(x) → CrossAttn(Q=x, KV=visual) → Dropout → Residual
        LN(x) → FFN → Dropout → Residual

    Each self-attention layer:
        LN(x) → SelfAttn(x) → Dropout → Residual
        LN(x) → FFN → Dropout → Residual

    Args:
        input_dim: Visual encoder feature dimension.
        output_dim: Target LM dimension.
        hidden_dim: Internal Q-Former dimension (query dim).
        num_queries: Number of learned query tokens.
        num_cross_attn_layers: Number of cross-attention layers.
        num_self_attn_layers: Number of self-attention layers.
        num_heads: Attention heads.
        ffn_dim: FFN hidden dimension (``None`` → ``4 * hidden_dim``).
        dropout: Dropout probability.
        activation: Activation function for FFN.
        bias: Bias in linear projections.
        use_qk_norm: Apply LayerNorm to Q/K before attention.
        max_seq_len: Max visual sequence length (for pos embeds).
        pos_embed_type: ``"learned"``, ``"sinusoidal"``, or ``"none"``.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_cross_attn_layers: int = 2,
        num_self_attn_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
        use_qk_norm: bool = False,
        max_seq_len: int = 576,
        pos_embed_type: str = "none",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_cross_attn_layers = num_cross_attn_layers
        self.num_self_attn_layers = num_self_attn_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * hidden_dim
        self.dropout_p = dropout
        self.activation_name = activation.lower().strip()
        self.pos_embed_type = pos_embed_type.lower().strip()

        # Learned query tokens
        self.query_tokens = nn.Embedding(num_queries, hidden_dim)

        # Visual feature projection (if input_dim != hidden_dim)
        self.visual_proj: nn.Module = nn.Identity()
        if input_dim != hidden_dim:
            self.visual_proj = nn.Linear(input_dim, hidden_dim, bias=bias)

        # Positional embedding for visual features
        if self.pos_embed_type == "learned":
            self.visual_pos_embed = LearnedPositionalEmbedding(
                hidden_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        elif self.pos_embed_type == "sinusoidal":
            self.visual_pos_embed = SinusoidalPositionalEmbedding(
                hidden_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        else:
            self.visual_pos_embed = nn.Identity()

        # Cross-attention layers
        self.cross_attn_layers = nn.ModuleList()
        for _ in range(num_cross_attn_layers):
            self.cross_attn_layers.append(self._build_cross_attn_layer(bias, use_qk_norm))

        # Self-attention layers
        self.self_attn_layers = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(self._build_self_attn_layer(bias))

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Output projection
        self.output_proj: nn.Module = nn.Identity()
        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _build_cross_attn_layer(
        self, bias: bool, use_qk_norm: bool
    ) -> nn.ModuleDict:
        """Build a single cross-attention block."""
        cross_attn = MultiHeadCrossAttention(
            query_dim=self.hidden_dim,
            kv_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
            use_qk_norm=use_qk_norm,
        )
        ffn = FeedForwardNetwork(
            embed_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_p,
            activation=self.activation_name,
            bias=bias,
        )
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "cross_attn": cross_attn,
            "norm2": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "ffn": ffn,
        })

    def _build_self_attn_layer(self, bias: bool) -> nn.ModuleDict:
        """Build a single self-attention block."""
        self_attn = MultiHeadSelfAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
        )
        ffn = FeedForwardNetwork(
            embed_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_p,
            activation=self.activation_name,
            bias=bias,
        )
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "self_attn": self_attn,
            "norm2": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "ffn": ffn,
        })

    def _init_weights(self) -> None:
        """Initialise weights."""
        nn.init.normal_(self.query_tokens.weight, std=0.02)
        if isinstance(self.visual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.visual_proj.weight)
            if self.visual_proj.bias is not None:
                nn.init.zeros_(self.visual_proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Q-Former forward pass.

        Args:
            visual_features: Visual encoder output ``(batch, num_patches, input_dim)``.
            attention_mask: Boolean mask ``(batch, num_patches)`` where
                ``True`` = valid token. ``None`` means all tokens valid.

        Returns:
            Projected features ``(batch, num_queries, output_dim)``.
        """
        B = visual_features.size(0)

        # Project visual features to hidden_dim
        visual = self.visual_proj(visual_features)
        visual = self.visual_pos_embed(visual)

        # Build cross-attention mask (batch, 1, 1, num_patches)
        cross_attn_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            cross_attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,KV)

        # Initialise query tokens
        query_ids = torch.arange(self.num_queries, device=visual_features.device)
        queries = self.query_tokens(query_ids).unsqueeze(0).expand(B, -1, -1)

        # Cross-attention stages
        for layer in self.cross_attn_layers:
            residual = queries
            queries_norm = layer["norm1"](queries)
            cross_out, _ = layer["cross_attn"](
                queries_norm, visual, visual, attention_mask=cross_attn_mask
            )
            queries = residual + self.dropout(cross_out)

            residual = queries
            queries_norm = layer["norm2"](queries)
            ffn_out = layer["ffn"](queries_norm)
            queries = residual + self.dropout(ffn_out)

        # Self-attention stages
        for layer in self.self_attn_layers:
            residual = queries
            queries_norm = layer["norm1"](queries)
            self_out, _ = layer["self_attn"](queries_norm)
            queries = residual + self.dropout(self_out)

            residual = queries
            queries_norm = layer["norm2"](queries)
            ffn_out = layer["ffn"](queries_norm)
            queries = residual + self.dropout(ffn_out)

        queries = self.final_norm(queries)
        output = self.output_proj(queries)
        return output

    def get_query_embeddings(self) -> torch.Tensor:
        """Return the current learned query embedding matrix.

        Returns:
            Tensor ``(num_queries, hidden_dim)``.
        """
        return self.query_tokens.weight.data.clone()

    def set_query_embeddings(self, embeddings: torch.Tensor) -> None:
        """Override learned query embeddings.

        Args:
            embeddings: New embeddings ``(num_queries, hidden_dim)``.

        Raises:
            ValueError: If shape does not match.
        """
        if embeddings.shape != self.query_tokens.weight.shape:
            raise ValueError(
                f"Expected shape {self.query_tokens.weight.shape}, "
                f"got {embeddings.shape}"
            )
        self.query_tokens.weight.data.copy_(embeddings)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return parameter count."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape.

        Args:
            input_shape: ``(batch, num_patches, input_dim)``.

        Returns:
            ``(batch, num_queries, output_dim)``.
        """
        return [input_shape[0], self.num_queries, self.output_dim]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"QFormerProjector\n"
            f"  input_dim             = {self.input_dim}\n"
            f"  output_dim            = {self.output_dim}\n"
            f"  hidden_dim            = {self.hidden_dim}\n"
            f"  num_queries           = {self.num_queries}\n"
            f"  num_cross_attn_layers = {self.num_cross_attn_layers}\n"
            f"  num_self_attn_layers  = {self.num_self_attn_layers}\n"
            f"  num_heads             = {self.num_heads}\n"
            f"  ffn_dim               = {self.ffn_dim}\n"
            f"  dropout               = {self.dropout_p}\n"
            f"  pos_embed_type        = {self.pos_embed_type}\n"
            f"  total_params          = {self.get_num_params():,}"
        )

    def reset_parameters(self) -> None:
        """Re-initialise all weights."""
        self._init_weights()
        for layer in self.cross_attn_layers:
            layer["cross_attn"]._init_weights()
            layer["ffn"]._init_weights()
        for layer in self.self_attn_layers:
            layer["self_attn"]._init_weights()
            layer["ffn"]._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_queries(self, freeze: bool = True) -> None:
        """Freeze or unfreeze only the query embeddings.

        Args:
            freeze: ``True`` to freeze, ``False`` to unfreeze.
        """
        self.query_tokens.weight.requires_grad = not freeze

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, num_queries={self.num_queries}"
        )


# ===========================================================================
#  4. ResamplerProjector
# ===========================================================================


class ResamplerProjector(nn.Module):
    """Perceiver-resample style projector with learned queries.

    Compresses variable-length visual features into a fixed number of
    output tokens via iterative cross-attention from learned latent
    queries.  Each layer follows a pre-norm architecture:

        LN(queries) → CrossAttn(Q=queries, KV=visual) → Residual
        LN(queries) → FFN → Residual

    Optionally appends self-attention layers for query refinement
    and a final output projection.

    This architecture is used in models such as Idefics2, LLaMA-3-Vision,
    and Phi-3-Vision.

    Args:
        input_dim: Visual encoder feature dimension.
        output_dim: Target LM dimension.
        hidden_dim: Internal dimension for queries and attention.
        num_queries: Number of output tokens (fixed-size).
        num_layers: Number of cross-attention layers.
        num_self_attn_layers: Number of self-attention layers appended after
            cross-attention (``0`` to disable).
        num_heads: Number of attention heads.
        ffn_dim: FFN hidden dimension (``None`` → ``4 * hidden_dim``).
        dropout: Dropout probability.
        activation: FFN activation name.
        bias: Bias in linear projections.
        use_qk_norm: LayerNorm on Q/K before attention.
        max_seq_len: Max visual sequence length.
        pos_embed_type: ``"learned"``, ``"sinusoidal"``, or ``"none"``.
        kv_dim_override: Override KV dimension if visual features are
            projected to a different dim than hidden_dim.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: int = 1024,
        num_queries: int = 64,
        num_layers: int = 3,
        num_self_attn_layers: int = 0,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
        use_qk_norm: bool = False,
        max_seq_len: int = 576,
        pos_embed_type: str = "none",
        kv_dim_override: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_self_attn_layers = num_self_attn_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * hidden_dim
        self.dropout_p = dropout
        self.activation_name = activation.lower().strip()
        self.pos_embed_type = pos_embed_type.lower().strip()

        # Learned latent queries
        self.latent_queries = nn.Embedding(num_queries, hidden_dim)
        nn.init.normal_(self.latent_queries.weight, std=0.02)

        # KV dimension for cross-attention
        kv_dim = kv_dim_override if kv_dim_override is not None else hidden_dim

        # Visual input projection
        self.visual_proj: nn.Module = nn.Identity()
        if input_dim != kv_dim:
            self.visual_proj = nn.Linear(input_dim, kv_dim, bias=bias)

        # Visual position embedding
        if self.pos_embed_type == "learned":
            self.visual_pos_embed = LearnedPositionalEmbedding(
                kv_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        elif self.pos_embed_type == "sinusoidal":
            self.visual_pos_embed = SinusoidalPositionalEmbedding(
                kv_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        else:
            self.visual_pos_embed = nn.Identity()

        # Cross-attention layers
        self.cross_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.cross_layers.append(
                self._build_cross_layer(hidden_dim, kv_dim, bias, use_qk_norm)
            )

        # Optional self-attention refinement layers
        self.self_layers = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_layers.append(
                self._build_self_layer(hidden_dim, bias)
            )

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Output projection
        self.output_proj: nn.Module = nn.Identity()
        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def _build_cross_layer(
        self,
        hidden_dim: int,
        kv_dim: int,
        bias: bool,
        use_qk_norm: bool,
    ) -> nn.ModuleDict:
        """Build one cross-attention resampler layer."""
        cross_attn = MultiHeadCrossAttention(
            query_dim=hidden_dim,
            kv_dim=kv_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
            use_qk_norm=use_qk_norm,
        )
        ffn = FeedForwardNetwork(
            embed_dim=hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_p,
            activation=self.activation_name,
            bias=bias,
        )
        return nn.ModuleDict({
            "norm_q": nn.LayerNorm(hidden_dim, eps=1e-6),
            "norm_kv": nn.LayerNorm(kv_dim, eps=1e-6),
            "cross_attn": cross_attn,
            "norm_ffn": nn.LayerNorm(hidden_dim, eps=1e-6),
            "ffn": ffn,
        })

    def _build_self_layer(
        self, hidden_dim: int, bias: bool
    ) -> nn.ModuleDict:
        """Build one self-attention refinement layer."""
        self_attn = MultiHeadSelfAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
        )
        ffn = FeedForwardNetwork(
            embed_dim=hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_p,
            activation=self.activation_name,
            bias=bias,
        )
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(hidden_dim, eps=1e-6),
            "self_attn": self_attn,
            "norm2": nn.LayerNorm(hidden_dim, eps=1e-6),
            "ffn": ffn,
        })

    def _init_weights(self) -> None:
        """Initialise weights."""
        nn.init.normal_(self.latent_queries.weight, std=0.02)
        if isinstance(self.visual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.visual_proj.weight)
            if self.visual_proj.bias is not None:
                nn.init.zeros_(self.visual_proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resampler forward.

        Args:
            visual_features: ``(batch, num_patches, input_dim)``.
            attention_mask: ``(batch, num_patches)`` boolean mask.
                ``True`` = valid patch.

        Returns:
            Fixed-size output ``(batch, num_queries, output_dim)``.
        """
        B = visual_features.size(0)

        # Project visual features
        visual = self.visual_proj(visual_features)
        visual = self.visual_pos_embed(visual)

        # Normalise visual features
        # (pre-norm KV within each cross layer handles this, but also
        #  apply a global norm for stability)
        kv_norm = nn.LayerNorm(visual.size(-1), eps=1e-6, device=visual.device)
        visual = kv_norm(visual)

        # Build cross-attn mask: (B, 1, 1, num_patches)
        cross_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            cross_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Initialise latent queries
        query_ids = torch.arange(self.num_queries, device=visual_features.device)
        latents = self.latent_queries(query_ids).unsqueeze(0).expand(B, -1, -1)

        # Cross-attention resampling
        for layer in self.cross_layers:
            residual = latents
            q_norm = layer["norm_q"](latents)
            kv_normed = layer["norm_kv"](visual)
            cross_out, _ = layer["cross_attn"](
                q_norm, kv_normed, kv_normed, attention_mask=cross_mask
            )
            latents = residual + self.dropout(cross_out)

            residual = latents
            latents_norm = layer["norm_ffn"](latents)
            ffn_out = layer["ffn"](latents_norm)
            latents = residual + self.dropout(ffn_out)

        # Optional self-attention refinement
        for layer in self.self_layers:
            residual = latents
            latents_norm = layer["norm1"](latents)
            self_out, _ = layer["self_attn"](latents_norm)
            latents = residual + self.dropout(self_out)

            residual = latents
            latents_norm = layer["norm2"](latents)
            ffn_out = layer["ffn"](latents_norm)
            latents = residual + self.dropout(ffn_out)

        latents = self.final_norm(latents)
        output = self.output_proj(latents)
        return output

    def get_query_embeddings(self) -> torch.Tensor:
        """Return learned query embedding weights.

        Returns:
            Tensor ``(num_queries, hidden_dim)``.
        """
        return self.latent_queries.weight.data.clone()

    def set_query_embeddings(self, embeddings: torch.Tensor) -> None:
        """Override query embeddings.

        Args:
            embeddings: ``(num_queries, hidden_dim)``.
        """
        if embeddings.shape != self.latent_queries.weight.shape:
            raise ValueError(
                f"Expected shape {self.latent_queries.weight.shape}, "
                f"got {embeddings.shape}"
            )
        self.latent_queries.weight.data.copy_(embeddings)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Parameter count."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape.

        Args:
            input_shape: ``(batch, num_patches, input_dim)``.

        Returns:
            ``(batch, num_queries, output_dim)``.
        """
        return [input_shape[0], self.num_queries, self.output_dim]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"ResamplerProjector\n"
            f"  input_dim            = {self.input_dim}\n"
            f"  output_dim           = {self.output_dim}\n"
            f"  hidden_dim           = {self.hidden_dim}\n"
            f"  num_queries          = {self.num_queries}\n"
            f"  num_cross_layers     = {self.num_layers}\n"
            f"  num_self_attn_layers = {self.num_self_attn_layers}\n"
            f"  num_heads            = {self.num_heads}\n"
            f"  ffn_dim              = {self.ffn_dim}\n"
            f"  dropout              = {self.dropout_p}\n"
            f"  pos_embed_type       = {self.pos_embed_type}\n"
            f"  total_params         = {self.get_num_params():,}"
        )

    def reset_parameters(self) -> None:
        """Re-initialise all weights."""
        self._init_weights()
        for layer in self.cross_layers:
            layer["cross_attn"]._init_weights()
            layer["ffn"]._init_weights()
        for layer in self.self_layers:
            layer["self_attn"]._init_weights()
            layer["ffn"]._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_queries(self, freeze: bool = True) -> None:
        """Freeze or unfreeze query embeddings.

        Args:
            freeze: ``True`` to freeze.
        """
        self.latent_queries.weight.requires_grad = not freeze

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, num_queries={self.num_queries}, "
            f"num_layers={self.num_layers}"
        )


# ===========================================================================
#  5. CAbstractor
# ===========================================================================


class CAbstractor(nn.Module):
    """C-Abstractor from LLaVA-OneVision.

    Uses a stack of cross-attention layers where learned queries attend
    to visual features, followed by self-attention and FFN sub-layers,
    all with residual connections.  Unlike Q-Former, the cross-attention
    and self-attention are interleaved within each block:

        Block:
            1. CrossAttn(Q=learned_queries, KV=visual_features) + residual
            2. SelfAttn(queries) + residual
            3. FFN(queries) + residual

    Each sub-layer is preceded by LayerNorm (pre-norm).

    Args:
        input_dim: Visual encoder feature dimension.
        output_dim: Target LM dimension.
        hidden_dim: Internal abstraction dimension.
        num_queries: Number of learned abstract query tokens.
        num_layers: Number of C-Abstractor blocks.
        num_heads: Attention heads.
        ffn_dim: FFN hidden dimension (``None`` → ``4 * hidden_dim``).
        dropout: Dropout probability.
        activation: FFN activation name.
        bias: Bias in linear projections.
        use_qk_norm: Apply LayerNorm to Q/K before attention.
        max_seq_len: Max visual token sequence length.
        pos_embed_type: Positional embedding for visual features.
        gating: Whether to use gated residual connections.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        hidden_dim: int = 1024,
        num_queries: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
        use_qk_norm: bool = False,
        max_seq_len: int = 576,
        pos_embed_type: str = "none",
        gating: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * hidden_dim
        self.dropout_p = dropout
        self.activation_name = activation.lower().strip()
        self.pos_embed_type = pos_embed_type.lower().strip()
        self.gating = gating

        # Learned abstract queries
        self.query_tokens = nn.Embedding(num_queries, hidden_dim)

        # Visual projection
        self.visual_proj: nn.Module = nn.Identity()
        if input_dim != hidden_dim:
            self.visual_proj = nn.Linear(input_dim, hidden_dim, bias=bias)

        # Visual position embedding
        if self.pos_embed_type == "learned":
            self.visual_pos_embed = LearnedPositionalEmbedding(
                hidden_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        elif self.pos_embed_type == "sinusoidal":
            self.visual_pos_embed = SinusoidalPositionalEmbedding(
                hidden_dim, max_seq_len=max_seq_len, dropout=0.0
            )
        else:
            self.visual_pos_embed = nn.Identity()

        # C-Abstractor blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self._build_block(bias, use_qk_norm)
            )

        # Optional gating
        if gating:
            self.gate_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim, bias=bias),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ])

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # Output projection
        self.output_proj: nn.Module = nn.Identity()
        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _build_block(
        self, bias: bool, use_qk_norm: bool
    ) -> nn.ModuleDict:
        """Build a single C-Abstractor block.

        Structure:
            cross_attn: CrossAttn(Q=queries, KV=visual)
            self_attn:  SelfAttn(queries)
            ffn:        FeedForwardNetwork
            norm_cross, norm_self, norm_ffn: LayerNorm
        """
        cross_attn = MultiHeadCrossAttention(
            query_dim=self.hidden_dim,
            kv_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
            use_qk_norm=use_qk_norm,
        )
        self_attn = MultiHeadSelfAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_p,
            bias=bias,
        )
        ffn = FeedForwardNetwork(
            embed_dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout_p,
            activation=self.activation_name,
            bias=bias,
        )
        return nn.ModuleDict({
            "norm_cross": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "norm_visual": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "cross_attn": cross_attn,
            "norm_self": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "self_attn": self_attn,
            "norm_ffn": nn.LayerNorm(self.hidden_dim, eps=1e-6),
            "ffn": ffn,
        })

    def _init_weights(self) -> None:
        """Initialise weights."""
        nn.init.normal_(self.query_tokens.weight, std=0.02)
        if isinstance(self.visual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.visual_proj.weight)
            if self.visual_proj.bias is not None:
                nn.init.zeros_(self.visual_proj.bias)
        if isinstance(self.output_proj, nn.Linear):
            nn.init.xavier_uniform_(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
        if self.gating:
            for gate in self.gate_proj:
                for m in gate:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """C-Abstractor forward.

        Args:
            visual_features: ``(batch, num_patches, input_dim)``.
            attention_mask: ``(batch, num_patches)`` boolean mask.

        Returns:
            Abstracted features ``(batch, num_queries, output_dim)``.
        """
        B = visual_features.size(0)

        # Project visual features
        visual = self.visual_proj(visual_features)
        visual = self.visual_pos_embed(visual)

        # Cross-attention mask (B, 1, 1, KV)
        cross_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            cross_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Initialise query tokens
        query_ids = torch.arange(self.num_queries, device=visual_features.device)
        queries = self.query_tokens(query_ids).unsqueeze(0).expand(B, -1, -1)

        # Process through blocks
        for i, block in enumerate(self.blocks):
            # 1. Cross-attention: queries attend to visual features
            residual = queries
            q_norm = block["norm_cross"](queries)
            v_norm = block["norm_visual"](visual)
            cross_out, _ = block["cross_attn"](
                q_norm, v_norm, v_norm, attention_mask=cross_mask
            )
            cross_out = self.dropout(cross_out)

            if self.gating:
                gate = self.gate_proj[i](residual)
                queries = residual + gate * cross_out
            else:
                queries = residual + cross_out

            # 2. Self-attention: queries attend to each other
            residual = queries
            q_norm = block["norm_self"](queries)
            self_out, _ = block["self_attn"](q_norm)
            self_out = self.dropout(self_out)
            queries = residual + self_out

            # 3. Feed-forward
            residual = queries
            q_norm = block["norm_ffn"](queries)
            ffn_out = block["ffn"](q_norm)
            ffn_out = self.dropout(ffn_out)
            queries = residual + ffn_out

        queries = self.final_norm(queries)
        output = self.output_proj(queries)
        return output

    def get_query_embeddings(self) -> torch.Tensor:
        """Return learned query embedding weights.

        Returns:
            Tensor ``(num_queries, hidden_dim)``.
        """
        return self.query_tokens.weight.data.clone()

    def set_query_embeddings(self, embeddings: torch.Tensor) -> None:
        """Override query embeddings.

        Args:
            embeddings: ``(num_queries, hidden_dim)``.
        """
        if embeddings.shape != self.query_tokens.weight.shape:
            raise ValueError(
                f"Expected shape {self.query_tokens.weight.shape}, "
                f"got {embeddings.shape}"
            )
        self.query_tokens.weight.data.copy_(embeddings)

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Parameter count."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape.

        Args:
            input_shape: ``(batch, num_patches, input_dim)``.

        Returns:
            ``(batch, num_queries, output_dim)``.
        """
        return [input_shape[0], self.num_queries, self.output_dim]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"CAbstractor\n"
            f"  input_dim    = {self.input_dim}\n"
            f"  output_dim   = {self.output_dim}\n"
            f"  hidden_dim   = {self.hidden_dim}\n"
            f"  num_queries  = {self.num_queries}\n"
            f"  num_layers   = {self.num_layers}\n"
            f"  num_heads    = {self.num_heads}\n"
            f"  ffn_dim      = {self.ffn_dim}\n"
            f"  dropout      = {self.dropout_p}\n"
            f"  gating       = {self.gating}\n"
            f"  total_params = {self.get_num_params():,}"
        )

    def reset_parameters(self) -> None:
        """Re-initialise all weights."""
        self._init_weights()
        for block in self.blocks:
            block["cross_attn"]._init_weights()
            block["self_attn"]._init_weights()
            block["ffn"]._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def freeze_queries(self, freeze: bool = True) -> None:
        """Freeze or unfreeze query embeddings.

        Args:
            freeze: ``True`` to freeze.
        """
        self.query_tokens.weight.requires_grad = not freeze

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dim={self.hidden_dim}, num_queries={self.num_queries}, "
            f"num_layers={self.num_layers}, gating={self.gating}"
        )


# ===========================================================================
#  6. ModalityAdapter
# ===========================================================================


class ModalityAdapter(nn.Module):
    """Lightweight modality adapter with LayerNorm, linear projection,
    and optional residual connection.

    This module wraps any encoder output and projects it to the target
    dimension while normalising.  It is useful as a simple bridge between
    heterogeneous encoders (vision, audio, video) and the shared LLM
    backbone.

    Architecture:
        LayerNorm(input_dim) → Linear(input_dim → output_dim) → [residual] → Dropout

    If ``input_dim == output_dim``, the residual connection adds the input
    directly. Otherwise, a projection residual can be enabled.

    Args:
        input_dim: Input feature dimension.
        output_dim: Output feature dimension.
        dropout: Dropout probability.
        use_residual: Add a residual connection from input to output.
        residual_proj: If ``True`` and ``input_dim != output_dim``, use a
            learned projection for the residual path.
        bias: Bias in linear projection.
        norm_type: ``"layer_norm"`` or ``"rms_norm"`` for input normalisation.
        scale_init: Initial scale for the output (applied as a learnable
            scalar multiplied to the projected output).
        activation: Optional activation after linear (``"none"`` to disable).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,
        dropout: float = 0.0,
        use_residual: bool = True,
        residual_proj: bool = False,
        bias: bool = True,
        norm_type: str = "layer_norm",
        scale_init: float = 1.0,
        activation: str = "none",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.use_residual = use_residual
        self.residual_proj = residual_proj
        self.norm_type = norm_type.lower().strip()
        self.scale_init = scale_init
        self.activation_name = activation.lower().strip()

        # Input normalisation
        if self.norm_type == "rms_norm":
            self.input_norm = RMSNorm(input_dim, eps=1e-6)
        else:
            self.input_norm = nn.LayerNorm(input_dim, eps=1e-6)

        # Linear projection
        self.proj = nn.Linear(input_dim, output_dim, bias=bias)

        # Residual path
        self.residual: nn.Module = nn.Identity()
        self.has_residual = use_residual and (input_dim == output_dim)
        self.has_residual_proj = use_residual and (input_dim != output_dim) and residual_proj
        if self.has_residual_proj:
            self.residual = nn.Linear(input_dim, output_dim, bias=False)

        # Output scale
        self.output_scale = nn.Parameter(torch.full((1,), scale_init))

        # Optional output normalisation
        self.output_norm: nn.Module = nn.Identity()
        if input_dim != output_dim:
            if self.norm_type == "rms_norm":
                self.output_norm = RMSNorm(output_dim, eps=1e-6)
            else:
                self.output_norm = nn.LayerNorm(output_dim, eps=1e-6)

        # Optional activation
        self.act: nn.Module = nn.Identity()
        if self.activation_name != "none":
            self.act = _get_activation_fn(self.activation_name)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        if isinstance(self.residual, nn.Linear):
            nn.init.xavier_uniform_(self.residual.weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Adapt modality features.

        Args:
            x: Input features ``(*, input_dim)`` or ``(batch, seq, input_dim)``.
            attention_mask: Ignored (API compatibility).

        Returns:
            Adapted features with last dim ``output_dim``.
        """
        # Normalise input
        x_norm = self.input_norm(x)

        # Project
        out = self.proj(x_norm)

        # Scale
        out = out * self.output_scale

        # Activation
        out = self.act(out)

        # Dropout
        out = self.dropout(out)

        # Output normalisation
        out = self.output_norm(out)

        # Residual
        if self.has_residual or self.has_residual_proj:
            residual = self.residual(x)
            out = out + residual

        return out

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Parameter count."""
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def get_output_shape(self, input_shape: Sequence[int]) -> List[int]:
        """Compute output shape.

        Args:
            input_shape: e.g. ``(seq_len, input_dim)``.

        Returns:
            Output shape list.
        """
        return list(input_shape[:-1]) + [self.output_dim]

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"ModalityAdapter\n"
            f"  input_dim      = {self.input_dim}\n"
            f"  output_dim     = {self.output_dim}\n"
            f"  use_residual   = {self.use_residual}\n"
            f"  residual_proj  = {self.residual_proj}\n"
            f"  norm_type      = {self.norm_type}\n"
            f"  scale_init     = {self.scale_init}\n"
            f"  activation     = {self.activation_name}\n"
            f"  dropout        = {self.dropout_p}\n"
            f"  total_params   = {self.get_num_params():,}"
        )

    def reset_parameters(self) -> None:
        """Re-initialise weights (preserving output_scale)."""
        self._init_weights()

    def freeze(self) -> None:
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True

    def get_scale(self) -> float:
        """Return the current output scale value.

        Returns:
            Float scalar.
        """
        return self.output_scale.data.item()

    def set_scale(self, value: float) -> None:
        """Manually set the output scale.

        Args:
            value: New scale value.
        """
        self.output_scale.data.fill_(value)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"use_residual={self.use_residual}, norm_type={self.norm_type}"
        )


# ===========================================================================
#  7. ProjectorFactory
# ===========================================================================


class ProjectorFactory:
    """Factory for building multimodal projectors from configuration.

    Usage::

        projector = ProjectorFactory.create(
            projector_type="qformer",
            input_dim=1024,
            output_dim=4096,
            num_queries=32,
            num_heads=8,
        )
        output = projector(visual_features)

    Supported ``projector_type`` values:
        - ``"linear"``        → :class:`LinearProjector`
        - ``"mlp"``           → :class:`MLPProjector`
        - ``"qformer"``       → :class:`QFormerProjector`
        - ``"resampler"``     → :class:`ResamplerProjector`
        - ``"c_abstractor"``  → :class:`CAbstractor`
        - ``"modality_adapter"`` → :class:`ModalityAdapter`

    Alternatively, a :class:`ProjectorConfig` instance can be passed via
    ``ProjectorFactory.from_config(config)``.
    """

    # Registry mapping type string → (class, required kwargs)
    _REGISTRY: Dict[str, type] = {
        "linear": LinearProjector,
        "mlp": MLPProjector,
        "qformer": QFormerProjector,
        "resampler": ResamplerProjector,
        "c_abstractor": CAbstractor,
        "modality_adapter": ModalityAdapter,
    }

    # Aliases
    _ALIASES: Dict[str, str] = {
        "linear_projector": "linear",
        "mlp_projector": "mlp",
        "q-former": "qformer",
        "qformer_projector": "qformer",
        "blip2_qformer": "qformer",
        "perceiver_resampler": "resampler",
        "resampler_projector": "resampler",
        "c_abstractor_projector": "c_abstractor",
        "llava_abstractor": "c_abstractor",
        "adapter": "modality_adapter",
        "modality_adapter_projector": "modality_adapter",
    }

    @classmethod
    def _resolve_type(cls, projector_type: str) -> str:
        """Resolve projector type string, handling aliases and casing.

        Args:
            projector_type: Raw type string from config.

        Returns:
            Normalised key into ``_REGISTRY``.

        Raises:
            ValueError: If the type is not recognised.
        """
        key = projector_type.lower().strip().replace("-", "_").replace(" ", "_")
        if key in cls._REGISTRY:
            return key
        if key in cls._ALIASES:
            return cls._ALIASES[key]
        available = sorted(set(list(cls._REGISTRY.keys()) + list(cls._ALIASES.keys())))
        raise ValueError(
            f"Unknown projector type '{projector_type}'. "
            f"Available types: {available}"
        )

    @classmethod
    def _filter_kwargs(
        cls, projector_cls: type, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Remove kwargs not accepted by the projector constructor.

        Args:
            projector_cls: The projector class to instantiate.
            kwargs: Full keyword arguments dict.

        Returns:
            Filtered dict with only valid kwargs.
        """
        import inspect
        sig = inspect.signature(projector_cls.__init__)
        valid = {p.name for p in sig.parameters.values() if p.name != "self"}
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return filtered

    @classmethod
    def create(
        cls,
        projector_type: str,
        input_dim: int = 1024,
        output_dim: int = 4096,
        **kwargs: Any,
    ) -> nn.Module:
        """Build a projector from type string and dimensions.

        Args:
            projector_type: Architecture name (e.g. ``"mlp"``, ``"qformer"``).
            input_dim: Encoder output dimension.
            output_dim: Target LM dimension.
            **kwargs: Additional architecture-specific arguments forwarded
                to the projector constructor (e.g. ``num_heads``,
                ``num_queries``, ``hidden_dim``, ``dropout``, etc.).

        Returns:
            An initialised ``nn.Module`` projector instance.

        Raises:
            ValueError: If ``projector_type`` is unknown.

        Example::

            # Simple MLP projector
            proj = ProjectorFactory.create("mlp", 1024, 4096)

            # Q-Former with 32 queries and 4 cross-attention layers
            proj = ProjectorFactory.create(
                "qformer", 1024, 4096,
                num_queries=32,
                num_cross_attn_layers=4,
                num_self_attn_layers=4,
                num_heads=12,
                hidden_dim=768,
            )
        """
        resolved = cls._resolve_type(projector_type)
        projector_cls = cls._REGISTRY[resolved]

        # Always pass input_dim and output_dim, let _filter_kwargs handle
        # the rest.
        all_kwargs: Dict[str, Any] = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            **kwargs,
        }
        filtered = cls._filter_kwargs(projector_cls, all_kwargs)
        return projector_cls(**filtered)

    @classmethod
    def from_config(cls, config: ProjectorConfig) -> nn.Module:
        """Build a projector from a :class:`ProjectorConfig`.

        Args:
            config: Projector configuration instance.

        Returns:
            Initialised projector module.
        """
        return cls.create(
            projector_type=config.projector_type,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            **config.to_dict(),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> nn.Module:
        """Build a projector from a plain dictionary.

        The dictionary must contain at least ``projector_type``,
        ``input_dim``, and ``output_dim``.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Initialised projector module.

        Raises:
            KeyError: If required keys are missing.
        """
        required = {"projector_type", "input_dim", "output_dim"}
        missing = required - set(config_dict.keys())
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")
        return cls.create(
            projector_type=config_dict["projector_type"],
            input_dim=config_dict["input_dim"],
            output_dim=config_dict["output_dim"],
            **{k: v for k, v in config_dict.items() if k not in required},
        )

    @classmethod
    def list_types(cls) -> List[str]:
        """Return a sorted list of all registered projector type strings.

        Returns:
            List of type name strings.
        """
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """Return the alias → canonical name mapping.

        Returns:
            Dictionary mapping alias strings to canonical type names.
        """
        return dict(cls._ALIASES)

    @classmethod
    def register(
        cls,
        name: str,
        projector_cls: type,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a custom projector class.

        Args:
            name: Canonical name (e.g. ``"my_projector"``).
            projector_cls: An ``nn.Module`` subclass.
            aliases: Optional list of alias strings.

        Raises:
            TypeError: If ``projector_cls`` is not an ``nn.Module`` subclass.
        """
        if not (isinstance(projector_cls, type) and issubclass(projector_cls, nn.Module)):
            raise TypeError(
                f"projector_cls must be an nn.Module subclass, "
                f"got {type(projector_cls)}"
            )
        key = name.lower().strip().replace("-", "_").replace(" ", "_")
        cls._REGISTRY[key] = projector_cls
        if aliases:
            for alias in aliases:
                a_key = alias.lower().strip().replace("-", "_").replace(" ", "_")
                cls._ALIASES[a_key] = key


# ===========================================================================
#  Utility functions for projector inspection and testing
# ===========================================================================


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters of an ``nn.Module``.

    Args:
        module: Any PyTorch module.
        trainable_only: Only count parameters with ``requires_grad=True``.

    Returns:
        Total parameter count.
    """
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def compute_flops_linear(
    in_features: int,
    out_features: int,
    batch_size: int,
    seq_len: int,
) -> int:
    """Estimate FLOPs for a single linear layer (multiply-add).

    Each output element requires ``in_features`` multiplications and
    ``in_features - 1`` additions.  We approximate as ``2 * in_features``.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        batch_size: Batch size.
        seq_len: Sequence length.

    Returns:
        Estimated FLOPs (integer).
    """
    return 2 * in_features * out_features * batch_size * seq_len


def compute_flops_attention(
    seq_q: int,
    seq_kv: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int = 1,
) -> int:
    """Estimate FLOPs for multi-head attention.

    Accounts for Q/K/V projections, attention matrix computation, and
    output projection.

    Args:
        seq_q: Query sequence length.
        seq_kv: Key/value sequence length.
        hidden_dim: Total hidden dimension.
        num_heads: Number of attention heads.
        batch_size: Batch size.

    Returns:
        Estimated FLOPs.
    """
    head_dim = hidden_dim // num_heads
    # Q, K, V projections: 3 × (seq × hidden × hidden) = 3 × seq × hidden^2
    proj_flops = 3 * 2 * seq_kv * hidden_dim * hidden_dim * batch_size
    # Q projection for queries
    proj_flops += 2 * seq_q * hidden_dim * hidden_dim * batch_size
    # Attention scores: Q @ K^T  → (B, H, seq_q, head_dim) @ (B, H, head_dim, seq_kv)
    attn_flops = 2 * seq_q * seq_kv * head_dim * num_heads * batch_size
    # Attention @ V: (B, H, seq_q, seq_kv) @ (B, H, seq_kv, head_dim)
    out_flops = 2 * seq_q * seq_kv * head_dim * num_heads * batch_size
    # Output projection
    out_proj_flops = 2 * seq_q * hidden_dim * hidden_dim * batch_size
    return proj_flops + attn_flops + out_flops + out_proj_flops


def estimate_projector_flops(
    projector: nn.Module,
    input_shape: Sequence[int],
) -> Dict[str, int]:
    """Rough FLOP estimation for a projector given an input shape.

    This is a simplified estimate that does not account for activation
    functions, layer norms, or element-wise operations.  It covers linear
    projections and attention.

    Args:
        projector: A projector ``nn.Module``.
        input_shape: Input shape ``(batch, seq_len, input_dim)``.

    Returns:
        Dictionary with ``total_flops`` and optionally
        ``linear_flops``, ``attention_flops`` entries.
    """
    B, S, D = input_shape[0], input_shape[1], input_shape[2]
    linear_flops = 0
    attention_flops = 0

    for module in projector.modules():
        if isinstance(module, nn.Linear):
            linear_flops += compute_flops_linear(
                module.in_features, module.out_features, B, S
            )
        elif isinstance(module, (MultiHeadSelfAttention, MultiHeadCrossAttention)):
            attention_flops += compute_flops_attention(
                S, S, module.embed_dim, module.num_heads, B
            )

    total = linear_flops + attention_flops
    return {
        "total_flops": total,
        "linear_flops": linear_flops,
        "attention_flops": attention_flops,
    }


def validate_projector(
    projector: nn.Module,
    input_dim: int,
    output_dim: int,
    batch_size: int = 2,
    seq_len: int = 64,
) -> Dict[str, Any]:
    """Validate a projector by running a forward pass.

    Checks that:
        - Forward pass completes without error.
        - Output shape matches expectations.
        - Output does not contain NaN or Inf.

    Args:
        projector: The projector module to test.
        input_dim: Expected input feature dimension.
        output_dim: Expected output feature dimension.
        batch_size: Batch size for dummy input.
        seq_len: Sequence length for dummy input.

    Returns:
        Dictionary with validation results.
    """
    projector.eval()
    device = next(projector.parameters()).device

    # Check if projector uses learned queries (fixed output tokens)
    is_query_based = isinstance(
        projector, (QFormerProjector, ResamplerProjector, CAbstractor)
    )
    if is_query_based:
        expected_num_queries = projector.num_queries

    with torch.no_grad():
        dummy_input = torch.randn(batch_size, seq_len, input_dim, device=device)

        try:
            output = projector(dummy_input)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "input_shape": list(dummy_input.shape),
            }

        output_shape = list(output.shape)
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()

        if is_query_based:
            expected_shape = [batch_size, expected_num_queries, output_dim]
        else:
            expected_shape = [batch_size, seq_len, output_dim]

        shape_ok = output_shape == expected_shape

    return {
        "status": "ok",
        "input_shape": list(dummy_input.shape),
        "output_shape": output_shape,
        "expected_shape": expected_shape,
        "shape_ok": shape_ok,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "num_params": count_parameters(projector),
    }


def build_projector_grid(
    input_dim: int = 1024,
    output_dim: int = 4096,
    num_queries: int = 32,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    dropout: float = 0.0,
) -> Dict[str, nn.Module]:
    """Build one instance of every registered projector for benchmarking.

    Args:
        input_dim: Visual encoder dim.
        output_dim: LM dim.
        num_queries: Queries for attention-based projectors.
        hidden_dim: Hidden dim for attention-based projectors.
        num_heads: Attention heads.
        dropout: Dropout.

    Returns:
        Dictionary mapping type name → projector instance.
    """
    projectors: Dict[str, nn.Module] = {}

    # LinearProjector
    projectors["linear"] = LinearProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    # MLPProjector
    projectors["mlp"] = MLPProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=input_dim,
        dropout=dropout,
    )

    # QFormerProjector
    projectors["qformer"] = QFormerProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_cross_attn_layers=2,
        num_self_attn_layers=2,
        num_heads=num_heads,
        dropout=dropout,
    )

    # ResamplerProjector
    projectors["resampler"] = ResamplerProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=3,
        num_heads=num_heads,
        dropout=dropout,
    )

    # CAbstractor
    projectors["c_abstractor"] = CAbstractor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_queries=num_queries,
        num_layers=3,
        num_heads=num_heads,
        dropout=dropout,
    )

    # ModalityAdapter
    projectors["modality_adapter"] = ModalityAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout=dropout,
        use_residual=True,
    )

    return projectors


def benchmark_projectors(
    input_dim: int = 1024,
    output_dim: int = 4096,
    num_queries: int = 32,
    hidden_dim: int = 1024,
    num_heads: int = 8,
    batch_size: int = 4,
    seq_len: int = 196,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Benchmark all registered projector types.

    For each projector, measures average forward pass time and FLOPs.

    Args:
        input_dim: Input dimension.
        output_dim: Output dimension.
        num_queries: Queries for attention-based projectors.
        hidden_dim: Hidden dimension.
        num_heads: Attention heads.
        batch_size: Batch size.
        seq_len: Sequence length.
        num_warmup: Warmup iterations (excluded from timing).
        num_runs: Timed iterations.
        device: Device string (``"cpu"`` or ``"cuda"``).

    Returns:
        Dictionary mapping projector type → benchmark metrics.
    """
    import time

    projectors = build_projector_grid(
        input_dim=input_dim,
        output_dim=output_dim,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
    )

    results: Dict[str, Dict[str, Any]] = {}
    dummy_input = torch.randn(batch_size, seq_len, input_dim)

    for name, proj in projectors.items():
        proj = proj.to(device).eval()
        inp = dummy_input.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = proj(inp)

        # Timed runs
        if device == "cuda":
            torch.cuda.synchronize()

        times: List[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = proj(inp)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)

        avg_time = sum(times) / len(times)
        flops = estimate_projector_flops(proj, (batch_size, seq_len, input_dim))

        results[name] = {
            "avg_time_ms": avg_time * 1000.0,
            "num_params": count_parameters(proj),
            "estimated_flops": flops["total_flops"],
            "output_shape": list(proj(inp).shape),
        }

        # Cleanup
        proj.cpu()
        del proj
        if device == "cuda":
            torch.cuda.empty_cache()

    return results
