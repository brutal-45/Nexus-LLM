"""
Enhanced Model Configuration for Nexus v2
=============================================
Extended configuration supporting ALL architecture variants.

This module defines the v2 configuration system used by Nexus's advanced
transformer implementation.  It provides typed enums for every architectural
choice point and a single ``ModelConfigV2`` dataclass that can describe any
modern decoder-only LLM — from LLaMA-style dense models to DeepSeek-style
MoE architectures with MLA attention.

Key differences from v1 (``config.ModelConfig``):
    - Typed enums for every architectural choice (attention, FFN, positional,
      normalization, residual, precision).
    - MoE configuration fields (experts, routing, capacity).
    - MLA-specific fields (kv_lora_rank, q_lora_rank).
    - RoPE scaling options (YaRN, LongRoPE, Dynamic NTK, Position Interpolation).
    - DeepNet residual scaling (alpha, beta auto-computed).
    - Automatic parameter estimation for all architecture combinations.
    - Preset factory methods for popular open-source architectures.
    - Embedding scaling, output head options, fused cross-entropy toggle.

Usage::

    from nexus.model.config_v2 import ModelConfigV2, AttentionType, FFNType

    # Nexus 100B default
    cfg = ModelConfigV2.nexus_100b()

    # DeepSeek-V2 style MoE with MLA
    cfg = ModelConfigV2.deepseek_v2()

    # Custom architecture
    cfg = ModelConfigV2(
        attention_type=AttentionType.MLA,
        ffn_type=FFNType.MOE,
        num_experts=128,
    )
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ===================================================================
# Architecture Enums
# ===================================================================


class AttentionType(str, Enum):
    """Supported attention mechanism variants.

    Attributes:
        MHA: Multi-Head Attention — each head has its own Q, K, V projections.
            Used in GPT-2, BERT, original Transformer.
        MQA: Multi-Query Attention — single shared K, V across all heads.
            Used in PaLM, Falcon, StarCoder.
        GQA: Grouped-Query Attention — groups of Q heads share K, V heads.
            Used in LLaMA-2, Mistral, Qwen.
        MLA: Multi-Head Latent Attention (DeepSeek) — compresses KV into a
            low-rank latent representation for efficient caching.
        DIFFERENTIAL: Differential Attention — uses two attention patterns
            subtracted for noise cancellation.  Used in Differential Transformer.
    """

    MHA = "mha"
    MQA = "mqa"
    GQA = "gqa"
    MLA = "mla"
    DIFFERENTIAL = "differential"


class FFNType(str, Enum):
    """Supported feed-forward network variants.

    Attributes:
        STANDARD: Standard 2-layer FFN with activation (ReLU/GELU).
        SWIGLU: SwiGLU gated activation (LLaMA, Mistral, most modern LLMs).
        GEGLU: Gated GELU activation (PaLM, some Flan-T5 variants).
        REGLU: Gated ReLU activation (Shazeer, 2020).
        MOE: Mixture of Experts — replaces the FFN with a routed expert layer.
    """

    STANDARD = "standard"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    MOE = "moe"


class PositionalEncodingType(str, Enum):
    """Supported positional encoding strategies.

    Attributes:
        ROPE: Rotary Position Embedding (Su et al., 2021).
            Used in LLaMA, Mistral, Qwen, and most modern LLMs.
        ALIBI: Attention with Linear Biases (Press et al., 2021).
            Used in BLOOM, MPT.
        LEARNED: Learned absolute position embeddings (GPT-2, BERT).
        NONE: No positional encoding (relies on other inductive biases).
    """

    ROPE = "rope"
    ALIBI = "alibi"
    LEARNED = "learned"
    NONE = "none"


class NormalizationType(str, Enum):
    """Supported normalization strategies.

    Attributes:
        LAYERNORM: Standard Layer Normalization (Ba et al., 2016).
        RMSNORM: Root Mean Square Normalization (Zhang & Sennrich, 2019).
            Default for most modern LLMs (LLaMA, Mistral, etc.).
        DEEPNORM: DeepNorm for very deep transformers (Wang et al., 2022).
    """

    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    DEEPNORM = "deepnorm"


class ResidualType(str, Enum):
    """Supported residual connection strategies.

    Attributes:
        STANDARD: Simple ``x + f(x)``.
        PRE_NORM: ``x + f(norm(x))`` — LLaMA, Mistral, GPT-2 style.
        POST_NORM: ``norm(x + f(x))`` — original Transformer.
        PARALLEL: ``x + attn(norm(x)) + ffn(norm2(x))`` — GPT-J, PaLM style.
        DEEPNET: DeepNet initialization with depth-dependent scaling.
    """

    STANDARD = "standard"
    PRE_NORM = "pre_norm"
    POST_NORM = "post_norm"
    PARALLEL = "parallel"
    DEEPNET = "deepnet"


class RopeScalingType(str, Enum):
    """Supported RoPE context extension strategies.

    Attributes:
        NONE: No scaling (standard RoPE).
        LINEAR: Position Interpolation (Chen et al., 2023).
        DYNAMIC_NTK: Dynamic NTK-aware scaling (CodeLlama).
        YARN: YaRN scaling (Peng et al., 2023).
        LONGROPE: LongRoPE per-frequency scaling (Ding et al., 2024).
    """

    NONE = "none"
    LINEAR = "linear"
    DYNAMIC_NTK = "dynamic_ntk"
    YARN = "yarn"
    LONGROPE = "longrope"


class RoutingMethod(str, Enum):
    """Supported MoE routing strategies.

    Attributes:
        TOP_K: Standard top-k routing — each token selects k experts.
        EXPERT_CHOICE: Experts choose their preferred tokens (Zhou et al., 2022).
        HASH: Deterministic hash-based routing (no learned parameters).
    """

    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    HASH = "hash"


class PrecisionType(str, Enum):
    """Supported training/inference precision modes.

    Attributes:
        FP32: Full 32-bit floating point.
        FP16: 16-bit floating point.
        BF16: Bfloat16 (preferred for training on Ampere+ GPUs).
        MIXED_FP16: Mixed precision with FP16 compute, FP32 master weights.
        MIXED_BF16: Mixed precision with BF16 compute, FP32 master weights.
    """

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED_FP16 = "mixed_fp16"
    MIXED_BF16 = "mixed_bf16"


# ===================================================================
# Main Config Dataclass
# ===================================================================


@dataclass
class ModelConfigV2:
    """Complete configuration for all Nexus v2 architecture variants.

    This dataclass encapsulates every configurable aspect of the transformer
    model, from basic dimensions to advanced architectural choices like MoE
    routing, RoPE scaling, and residual stream management.

    Attributes:
        name: Human-readable model name.
        version: Configuration schema version.

        hidden_size: Model hidden dimension (d_model).
        intermediate_size: FFN intermediate dimension.
        num_hidden_layers: Number of transformer blocks.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads (for GQA/MQA).
        head_dim: Dimension of each attention head.
        vocab_size: Vocabulary size.
        max_position_embeddings: Maximum sequence length during training.

        attention_type: Attention mechanism variant.
        ffn_type: Feed-forward network variant.
        positional_encoding: Positional encoding strategy.
        normalization: Normalization layer type.
        residual_type: Residual connection strategy.

        attention_dropout: Dropout on attention weights.
        hidden_dropout: Dropout on hidden states (output of attention/FFN).
        use_qk_norm: Apply normalization to Q and K before attention.
        sliding_window_size: Sliding window attention size (None = full).

        kv_lora_rank: MLA KV compression rank (DeepSeek-V2).
        q_lora_rank: MLA Q compression rank (DeepSeek-V3).

        num_experts: Total number of MoE experts.
        num_experts_per_token: Experts activated per token (top-k).
        routing_method: MoE routing strategy.
        moe_load_balance_coeff: Auxiliary load balancing loss coefficient.
        moe_capacity_factor: Per-expert token capacity multiplier.
        shared_expert: Whether to include a shared expert (always active).
        moe_jitter_noise: Noise std dev added to router logits during training.

        rope_base: Base frequency for RoPE.
        rope_scaling: RoPE context extension strategy.
        rope_scaling_factor: Scaling factor for context extension.
        yarn_beta_fast: YaRN low-frequency mixing ratio.
        yarn_beta_slow: YaRN high-frequency mixing ratio.

        rms_norm_eps: Epsilon for RMSNorm / LayerNorm.
        norm_bias: Whether to use bias in normalization layers.

        residual_scale: Fixed scaling for residual connections.
        deepnet_alpha: DeepNet input scaling factor (auto-computed if None).
        deepnet_beta: DeepNet sublayer output scaling (auto-computed if None).

        tie_word_embeddings: Share embedding and output head weights.
        embedding_scale: Scale embeddings by sqrt(d_model).
        vocab_parallel: Shard vocabulary across GPUs.

        output_head_bias: Whether the output head uses bias.
        fused_cross_entropy: Fuse output projection + CE loss for memory.

        precision: Training/inference precision mode.

        initializer_range: Std dev for weight initialization.
        layer_norm_initializer_range: Std dev for normalization weight init.

        gradient_checkpointing: Enable gradient checkpointing for memory.
        use_flash_attention: Use Flash Attention when available.
    """

    # === Model Identity ===
    name: str = "Nexus-100B-v2"
    version: str = "2.0"

    # === Architecture Dimensions ===
    hidden_size: int = 12288
    intermediate_size: int = 32768  # 8/3 * 12288 for SwiGLU
    num_hidden_layers: int = 96
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 128000
    max_position_embeddings: int = 8192

    # === Architecture Type Selection ===
    attention_type: AttentionType = AttentionType.GQA
    ffn_type: FFNType = FFNType.SWIGLU
    positional_encoding: PositionalEncodingType = PositionalEncodingType.ROPE
    normalization: NormalizationType = NormalizationType.RMSNORM
    residual_type: ResidualType = ResidualType.PRE_NORM

    # === Attention Config ===
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    use_qk_norm: bool = False
    sliding_window_size: Optional[int] = None  # None = full attention

    # === MLA Config (DeepSeek-style) ===
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536

    # === MoE Config ===
    num_experts: int = 128
    num_experts_per_token: int = 2  # top_k
    routing_method: RoutingMethod = RoutingMethod.TOP_K
    moe_load_balance_coeff: float = 0.01
    moe_capacity_factor: float = 1.25
    shared_expert: bool = True
    moe_jitter_noise: float = 0.1

    # === RoPE Config ===
    rope_base: float = 10000.0
    rope_scaling: RopeScalingType = RopeScalingType.NONE
    rope_scaling_factor: float = 1.0
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # === Normalization Config ===
    rms_norm_eps: float = 1e-5
    norm_bias: bool = False  # bias in normalization

    # === Residual Config ===
    residual_scale: float = 1.0  # for scaled residual
    deepnet_alpha: Optional[float] = None  # auto-computed from layers
    deepnet_beta: Optional[float] = None

    # === Embeddings Config ===
    tie_word_embeddings: bool = False
    embedding_scale: bool = False  # sqrt(d_model) scaling
    vocab_parallel: bool = False  # shard vocab across GPUs

    # === Output Head Config ===
    output_head_bias: bool = False
    fused_cross_entropy: bool = True  # fuse output + loss for memory

    # === Precision Config ===
    precision: PrecisionType = PrecisionType.MIXED_BF16

    # === Regularization ===
    initializer_range: float = 0.02
    layer_norm_initializer_range: float = 1.0

    # === Gradient Checkpointing ===
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_moe(self) -> bool:
        """Whether the FFN layer is a Mixture of Experts."""
        return self.ffn_type == FFNType.MOE

    @property
    def is_mla(self) -> bool:
        """Whether the attention uses Multi-Head Latent Attention."""
        return self.attention_type == AttentionType.MLA

    @property
    def is_differential(self) -> bool:
        """Whether the attention uses Differential Attention."""
        return self.attention_type == AttentionType.DIFFERENTIAL

    @property
    def total_params(self) -> int:
        """Estimate total parameters based on architecture choices.

        Accounts for all architecture variants (dense, MoE, MLA,
        differential attention, embedding tying, etc.) to provide
        an accurate parameter count.
        """
        d = self.hidden_size
        h = self.head_dim
        H = self.num_attention_heads
        H_kv = self.num_key_value_heads
        L = self.num_hidden_layers
        V = self.vocab_size
        d_ff = self.intermediate_size

        params = 0

        # === Token Embeddings ===
        if self.tie_word_embeddings:
            params += V * d  # counted once (shared with output head)
        else:
            params += V * d  # embeddings
            # Output head counted below

        # === Positional Embeddings (if learned) ===
        if self.positional_encoding == PositionalEncodingType.LEARNED:
            params += self.max_position_embeddings * d

        # === Per-Layer Parameters ===
        for _ in range(L):
            # --- Attention ---
            if self.is_mla:
                # MLA: compressed KV + up-projection + Q + output
                params += d * self.kv_lora_rank  # c_kv (down)
                params += self.kv_lora_rank * H_kv * h  # kv_up_proj
                params += self.kv_lora_rank * H_kv * h  # duplicate for K,V split
                params += H * h * d  # q_proj
                params += H * h * d  # o_proj
                # RoPE positional projections
                rope_head_dim = h // 2
                params += rope_head_dim * H * rope_head_dim  # q_pe_proj
                params += rope_head_dim * H_kv * rope_head_dim  # k_pe_proj
                # KV gating
                params += d * self.kv_lora_rank  # kv_gate
            elif self.is_differential:
                # Differential: 2x Q, 2x K, 1x V projections
                # head_dim is hidden_size / (2 * num_heads) for differential
                diff_head_dim = d // (2 * H)
                params += 2 * H * diff_head_dim * d  # q_proj
                params += 2 * H * diff_head_dim * d  # k_proj
                params += H * (2 * diff_head_dim) * d  # v_proj
                params += H * (2 * diff_head_dim) * d  # o_proj
            else:
                # Standard / GQA / MQA attention
                params += H * h * d  # q_proj
                params += H_kv * h * d  # k_proj
                params += H_kv * h * d  # v_proj
                params += H * h * d  # o_proj

            # --- FFN ---
            if self.is_moe:
                # Each expert: gate + up + down (SwiGLU style)
                params_per_expert = d * d_ff * 3
                params += self.num_experts * params_per_expert
                if self.shared_expert:
                    params += params_per_expert
                # Router
                params += d * self.num_experts
            else:
                if self.ffn_type == FFNType.STANDARD:
                    params += d * d_ff * 2  # up + down
                else:
                    # SwiGLU, GeGLU, ReGLU: gate + up + down
                    params += d * d_ff * 3

            # --- Normalization ---
            params += d * 2  # input_norm + post_attention_norm

        # === Final Norm ===
        params += d

        # === Output Head ===
        if not self.tie_word_embeddings:
            params += d * V
            if self.output_head_bias:
                params += V

        return params

    @property
    def total_params_billions(self) -> float:
        """Total parameters in billions (for display)."""
        return self.total_params / 1e9

    @property
    def estimated_memory_gb(self) -> float:
        """Estimate memory in GB for BF16 precision.

        Assumes 2 bytes per parameter (BF16).  Does not include
        optimizer states, activations, or KV cache.
        """
        bytes_per_param = 2  # BF16
        return (self.total_params * bytes_per_param) / (1024 ** 3)

    @property
    def active_params_per_token(self) -> int:
        """Parameters active per token during a forward pass.

        For MoE models, only a fraction of experts are active.
        For dense models, all parameters are active.
        """
        if not self.is_moe:
            return self.total_params

        d = self.hidden_size
        d_ff = self.intermediate_size
        L = self.num_hidden_layers
        k = self.num_experts_per_token

        # Active MoE params per layer
        active_ffn_per_layer = k * d * d_ff * 3
        if self.shared_expert:
            active_ffn_per_layer += d * d_ff * 3

        # Non-FFN params (attention + norms + embeddings + output head)
        non_ffn = self.total_params
        # Subtract full MoE params across all layers
        full_moe_per_layer = self.num_experts * d * d_ff * 3
        if self.shared_expert:
            full_moe_per_layer += d * d_ff * 3
        non_ffn -= L * full_moe_per_layer

        # Add back active MoE params
        active = non_ffn + L * active_ffn_per_layer
        return active

    @property
    def active_params_ratio(self) -> float:
        """Fraction of total parameters active per token (MoE utilization)."""
        if self.total_params == 0:
            return 0.0
        return self.active_params_per_token / self.total_params

    @property
    def kv_cache_size_per_token(self) -> int:
        """KV cache memory per token (in elements, not bytes).

        Depends on attention type:
        - MHA/GQA/MQA: num_kv_heads * head_dim * 2 (K + V)
        - MLA: kv_lora_rank (compressed KV, much smaller)
        - Differential: 2 * num_heads * head_dim * 2 (K1, K2, V) + head_dim
        """
        if self.is_mla:
            return self.kv_lora_rank * 2  # K_compressed + V_compressed
        elif self.is_differential:
            # K1, K2 each with num_heads * head_dim, plus V
            diff_head_dim = self.hidden_size // (2 * self.num_attention_heads)
            return self.num_attention_heads * diff_head_dim * 2 * 2 + self.num_attention_heads * (2 * diff_head_dim)
        else:
            return self.num_key_value_heads * self.head_dim * 2

    # ------------------------------------------------------------------
    # Validation & post-init
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate and auto-compute derived configuration values."""
        # Validate head dimension divisibility
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Auto-adjust kv heads based on attention type
        if self.attention_type == AttentionType.MHA:
            self.num_key_value_heads = self.num_attention_heads
        elif self.attention_type == AttentionType.MQA:
            self.num_key_value_heads = 1
        elif self.attention_type == AttentionType.GQA:
            assert self.num_attention_heads % self.num_key_value_heads == 0, (
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads}) "
                f"for GQA"
            )
        # MLA and DIFFERENTIAL use separate config fields

        # Auto-compute intermediate_size for SwiGLU if not explicitly set
        if (
            self.ffn_type in (FFNType.SWIGLU, FFNType.GEGLU, FFNType.REGLU)
            and self.intermediate_size == 32768
            and self.hidden_size != 12288
        ):
            # Use LLaMA convention: 2/3 * 4 * d, rounded up to multiple of 256
            target = int((self.hidden_size * 8) / 3)
            self.intermediate_size = ((target + 255) // 256) * 256

        # Auto-compute DeepNet scaling factors
        if self.residual_type == ResidualType.DEEPNET:
            N = self.num_hidden_layers
            if self.deepnet_alpha is None:
                self.deepnet_alpha = (2 * N) ** 0.25
            if self.deepnet_beta is None:
                self.deepnet_beta = (8 * N) ** (-0.25)

        # Validate RoPE scaling factor
        if self.rope_scaling != RopeScalingType.NONE:
            assert self.rope_scaling_factor >= 1.0, (
                f"rope_scaling_factor ({self.rope_scaling_factor}) must be >= 1.0"
            )

        # Validate MoE config
        if self.is_moe:
            assert self.num_experts >= self.num_experts_per_token, (
                f"num_experts ({self.num_experts}) must be >= "
                f"num_experts_per_token ({self.num_experts_per_token})"
            )
            assert self.num_experts > 0, "num_experts must be > 0"
            assert self.num_experts_per_token > 0, "num_experts_per_token must be > 0"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dictionary.

        Enum values are converted to their string values for JSON/YAML
        compatibility.

        Returns:
            Dictionary with all config fields as plain Python types.
        """
        d = asdict(self)
        # Convert enum values to strings
        for key, value in d.items():
            if isinstance(value, Enum):
                d[key] = value.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModelConfigV2:
        """Create config from a dictionary.

        Automatically converts string values to the appropriate Enum types
        when they match known enum values.

        Args:
            d: Dictionary of config key-value pairs.

        Returns:
            A ModelConfigV2 instance.
        """
        # Build a copy with enum fields converted
        clean: Dict[str, Any] = {}
        enum_map = {
            "attention_type": AttentionType,
            "ffn_type": FFNType,
            "positional_encoding": PositionalEncodingType,
            "normalization": NormalizationType,
            "residual_type": ResidualType,
            "rope_scaling": RopeScalingType,
            "routing_method": RoutingMethod,
            "precision": PrecisionType,
        }

        for k, v in d.items():
            if k in enum_map and isinstance(v, str):
                try:
                    clean[k] = enum_map[k](v)
                except ValueError:
                    clean[k] = v
            elif k in cls.__dataclass_fields__:
                clean[k] = v
            # Skip unknown keys

        return cls(**clean)

    def to_yaml(self, path: str) -> None:
        """Save config as YAML file.

        Args:
            path: Output file path.
        """
        try:
            import yaml

            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
        except ImportError:
            import json

            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfigV2:
        """Load config from YAML (or JSON) file.

        Args:
            path: Input file path.

        Returns:
            A ModelConfigV2 instance.
        """
        try:
            import yaml

            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except ImportError:
            import json

            with open(path, "r") as f:
                data = json.load(f)

        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params_b = self.total_params / 1e9
        attn = self.attention_type.value.upper()
        ffn = self.ffn_type.value.upper()
        pos = self.positional_encoding.value.upper()
        norm = self.normalization.value.upper()

        parts = [
            f"ModelConfigV2(name={self.name!r}",
            f"params={params_b:.1f}B",
            f"d={self.hidden_size}",
            f"L={self.num_hidden_layers}",
            f"H={self.num_attention_heads}",
        ]

        if not self.is_moe:
            parts.append(f"attn={attn}")
            parts.append(f"ffn={ffn}")
        else:
            parts.append(f"attn={attn}")
            parts.append(f"ffn=MoE({self.num_experts}E,k={self.num_experts_per_token})")

        parts.append(f"pos={pos}")
        parts.append(f"norm={norm}")

        moe_info = ""
        if self.is_moe:
            active_ratio = self.active_params_ratio * 100
            moe_info = f", active={active_ratio:.1f}%"

        parts.append(f"mem≈{self.estimated_memory_gb:.0f}GB{moe_info})")
        return " ".join(parts)

    def summary(self) -> str:
        """Generate a detailed human-readable summary of the configuration.

        Returns:
            Multi-line string with full configuration details.
        """
        lines = [
            f"{'='*60}",
            f" Model Configuration: {self.name}",
            f" Version: {self.version}",
            f"{'='*60}",
            "",
            f" Dimensions:",
            f"   hidden_size:           {self.hidden_size:,}",
            f"   intermediate_size:     {self.intermediate_size:,}",
            f"   num_hidden_layers:     {self.num_hidden_layers}",
            f"   num_attention_heads:   {self.num_attention_heads}",
            f"   num_key_value_heads:   {self.num_key_value_heads}",
            f"   head_dim:              {self.head_dim}",
            f"   vocab_size:            {self.vocab_size:,}",
            f"   max_position_embeddings: {self.max_position_embeddings:,}",
            "",
            f" Architecture:",
            f"   attention:             {self.attention_type.value}",
            f"   ffn:                   {self.ffn_type.value}",
            f"   positional_encoding:   {self.positional_encoding.value}",
            f"   normalization:         {self.normalization.value}",
            f"   residual:              {self.residual_type.value}",
            "",
            f" Parameters:",
            f"   total:                 {self.total_params:,} ({self.total_params_billions:.2f}B)",
            f"   active/token:          {self.active_params_per_token:,} ({self.active_params_ratio*100:.1f}%)",
            f"   est. memory (BF16):    {self.estimated_memory_gb:.1f} GB",
            f"   KV cache/token:        {self.kv_cache_size_per_token:,} elements",
            "",
            f" RoPE:",
            f"   base:                  {self.rope_base}",
            f"   scaling:               {self.rope_scaling.value}",
            f"   scaling_factor:        {self.rope_scaling_factor}",
        ]

        if self.is_moe:
            lines.extend([
                "",
                f" MoE:",
                f"   num_experts:           {self.num_experts}",
                f"   experts_per_token:     {self.num_experts_per_token}",
                f"   routing_method:        {self.routing_method.value}",
                f"   load_balance_coeff:    {self.moe_load_balance_coeff}",
                f"   capacity_factor:       {self.moe_capacity_factor}",
                f"   shared_expert:         {self.shared_expert}",
            ])

        if self.is_mla:
            lines.extend([
                "",
                f" MLA:",
                f"   kv_lora_rank:          {self.kv_lora_rank}",
                f"   q_lora_rank:           {self.q_lora_rank}",
            ])

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ===================================================================
# Preset Configurations for Popular Architectures
# ===================================================================


def _register_presets() -> None:
    """Add preset class-methods to ModelConfigV2 for well-known models."""

    @classmethod
    def llama2_70b(cls: type) -> ModelConfigV2:
        """LLaMA-2 70B configuration."""
        return cls(
            name="LLaMA-2-70B",
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=4096,
            attention_type=AttentionType.GQA,
            ffn_type=FFNType.SWIGLU,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.RMSNORM,
            residual_type=ResidualType.PRE_NORM,
            rope_base=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def llama3_8b(cls: type) -> ModelConfigV2:
        """LLaMA-3 8B configuration."""
        return cls(
            name="LLaMA-3-8B",
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=128256,
            max_position_embeddings=8192,
            attention_type=AttentionType.GQA,
            ffn_type=FFNType.SWIGLU,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.RMSNORM,
            residual_type=ResidualType.PRE_NORM,
            rope_base=500000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def mixtral_8x7b(cls: type) -> ModelConfigV2:
        """Mixtral 8x7B configuration (MoE)."""
        return cls(
            name="Mixtral-8x7B",
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            attention_type=AttentionType.GQA,
            ffn_type=FFNType.MOE,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.RMSNORM,
            residual_type=ResidualType.PRE_NORM,
            num_experts=8,
            num_experts_per_token=2,
            routing_method=RoutingMethod.TOP_K,
            moe_load_balance_coeff=0.01,
            moe_capacity_factor=1.0,
            shared_expert=False,
            rope_base=1000000.0,
            rms_norm_eps=1e-5,
            sliding_window_size=4096,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def deepseek_v2(cls: type) -> ModelConfigV2:
        """DeepSeek-V2 configuration (MLA + MoE)."""
        return cls(
            name="DeepSeek-V2",
            hidden_size=5120,
            intermediate_size=1536,
            num_hidden_layers=60,
            num_attention_heads=128,
            num_key_value_heads=128,
            head_dim=128,
            vocab_size=102400,
            max_position_embeddings=128000,
            attention_type=AttentionType.MLA,
            ffn_type=FFNType.MOE,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.RMSNORM,
            residual_type=ResidualType.PRE_NORM,
            kv_lora_rank=1536,
            q_lora_rank=1536,
            num_experts=160,
            num_experts_per_token=6,
            routing_method=RoutingMethod.TOP_K,
            moe_load_balance_coeff=1.0,
            moe_capacity_factor=1.0,
            shared_expert=True,
            rope_base=10000.0,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def nexus_100b(cls: type) -> ModelConfigV2:
        """Nexus 100B — the reference dense configuration."""
        return cls(
            name="Nexus-100B",
            hidden_size=12288,
            intermediate_size=32768,
            num_hidden_layers=96,
            num_attention_heads=96,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=128000,
            max_position_embeddings=8192,
        )

    @classmethod
    def nexus_200b_moe(cls: type) -> ModelConfigV2:
        """Nexus 200B MoE — large-scale MoE with MLA attention."""
        return cls(
            name="Nexus-200B-MoE",
            hidden_size=16384,
            intermediate_size=43691,
            num_hidden_layers=128,
            num_attention_heads=128,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=256000,
            max_position_embeddings=131072,
            attention_type=AttentionType.GQA,
            ffn_type=FFNType.MOE,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.RMSNORM,
            residual_type=ResidualType.PRE_NORM,
            num_experts=256,
            num_experts_per_token=8,
            routing_method=RoutingMethod.TOP_K,
            moe_load_balance_coeff=0.01,
            moe_capacity_factor=1.25,
            shared_expert=True,
            rope_base=10000.0,
            rope_scaling=RopeScalingType.YARN,
            rope_scaling_factor=16.0,
            yarn_beta_fast=32.0,
            yarn_beta_slow=1.0,
            rms_norm_eps=1e-5,
        )

    @classmethod
    def gptj_6b(cls: type) -> ModelConfigV2:
        """GPT-J 6B configuration (parallel residual)."""
        return cls(
            name="GPT-J-6B",
            hidden_size=4096,
            intermediate_size=16384,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=16,
            head_dim=256,
            vocab_size=50400,
            max_position_embeddings=2048,
            attention_type=AttentionType.MHA,
            ffn_type=FFNType.GEGLU,
            positional_encoding=PositionalEncodingType.ROPE,
            normalization=NormalizationType.LAYERNORM,
            residual_type=ResidualType.PARALLEL,
            rope_base=10000.0,
            rms_norm_eps=1e-5,
            norm_bias=True,
            tie_word_embeddings=False,
            initializer_range=0.02,
        )

    @classmethod
    def bloom_176b(cls: type) -> ModelConfigV2:
        """BLOOM 176B configuration (ALiBi + post-norm)."""
        return cls(
            name="BLOOM-176B",
            hidden_size=14336,
            intermediate_size=57344,
            num_hidden_layers=70,
            num_attention_heads=112,
            num_key_value_heads=112,
            head_dim=128,
            vocab_size=250880,
            max_position_embeddings=2048,
            attention_type=AttentionType.MHA,
            ffn_type=FFNType.GEGLU,
            positional_encoding=PositionalEncodingType.ALIBI,
            normalization=NormalizationType.LAYERNORM,
            residual_type=ResidualType.POST_NORM,
            rms_norm_eps=1e-5,
            norm_bias=True,
            tie_word_embeddings=False,
            initializer_range=0.02,
            gradient_checkpointing=True,
        )

    # Register all presets on the class
    cls.llama2_70b = llama2_70b
    cls.llama3_8b = llama3_8b
    cls.mixtral_8x7b = mixtral_8x7b
    cls.deepseek_v2 = deepseek_v2
    cls.nexus_100b = nexus_100b
    cls.nexus_200b_moe = nexus_200b_moe
    cls.gptj_6b = gptj_6b
    cls.bloom_176b = bloom_176b


_register_presets()
