"""
Nexus v2 Transformer - Advanced Decoder-Only Architecture
=============================================================
Fully configurable transformer with all attention, FFN, normalization,
positional encoding, and residual variants.

This is the v2 transformer that uses all the new v2 components created
in the model package.  The original ``transformer.py`` is preserved
unchanged for backward compatibility.

Supported architecture combinations:

    Attention:  MHA, MQA, GQA, MLA (DeepSeek), Differential
    FFN:        Standard, SwiGLU, GeGLU, ReGLU, MoE (up to 256 experts)
    Positional: RoPE (+YaRN, LongRoPE, NTK, PI), ALiBi, Learned, None
    Norm:       LayerNorm, RMSNorm, DeepNorm
    Residual:   Standard, Pre/Post-norm, Parallel, DeepNet

Usage::

    from nexus.model.config_v2 import ModelConfigV2
    from nexus.model.transformer_v2 import NexusTransformerV2

    config = ModelConfigV2.nexus_100b()
    model = NexusTransformerV2(config)

    # Forward pass
    logits = model(input_ids).logits

    # Autoregressive generation
    output = model.generate(input_ids, max_new_tokens=128)

References:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023)
    - DeepSeek-AI, "DeepSeek-V2: A Strong, Economical, and Efficient MoE Model" (2024)
    - Poli et al., "Differential Transformer" (2024)
    - Wang et al., "DeepNorm: Improving Deep Transformer Training" (2022)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_v2 import (
    AttentionType,
    FFNType,
    ModelConfigV2,
    NormalizationType,
    PositionalEncodingType,
    ResidualType,
    RopeScalingType,
    RoutingMethod,
)

# Import v2 components (with fallbacks in case of import issues)
try:
    from .attention_v2 import create_attention, AttentionWithQKNorm
except ImportError:
    create_attention = None  # type: ignore[assignment]
    AttentionWithQKNorm = None  # type: ignore[assignment, misc]

try:
    from .ffn_v2 import create_ffn
except ImportError:
    create_ffn = None  # type: ignore[assignment]

try:
    from .normalization import get_norm
except ImportError:
    get_norm = None  # type: ignore[assignment]

try:
    from .residual import get_residual
except ImportError:
    get_residual = None  # type: ignore[assignment]

try:
    from .activations import get_activation
except ImportError:
    get_activation = None  # type: ignore[assignment]

try:
    from .output_head import (
        LMHead,
        FusedCrossEntropyLMHead,
        AdaptiveOutputHead,
    )
except ImportError:
    LMHead = None  # type: ignore[assignment, misc]
    FusedCrossEntropyLMHead = None  # type: ignore[assignment, misc]
    AdaptiveOutputHead = None  # type: ignore[assignment, misc]

try:
    from .positional import (
        RotaryPositionEmbedding,
        ALiBiPositionalEncoding,
        LearnedPositionEmbedding,
    )
except ImportError:
    RotaryPositionEmbedding = None  # type: ignore[assignment, misc]
    ALiBiPositionalEncoding = None  # type: ignore[assignment, misc]
    LearnedPositionEmbedding = None  # type: ignore[assignment, misc]

try:
    from .embeddings import Embedding
except ImportError:
    Embedding = None  # type: ignore[assignment, misc]


# ===================================================================
# Output Dataclass
# ===================================================================


@dataclass
class TransformerOutputV2:
    """Output container for the v2 transformer forward pass.

    Attributes:
        logits: Language model logits ``(batch, seq_len, vocab_size)``.
        hidden_states: All hidden states per layer (optional).
        past_key_values: Cached key-value states for autoregressive generation.
        attentions: Attention weights per layer (optional).
        loss: Scalar cross-entropy loss when labels are provided.
        aux_loss: MoE auxiliary load-balancing loss (optional).
        router_probs: MoE routing probabilities per token (optional).
    """

    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    past_key_values: Optional[Tuple] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None  # MoE aux loss
    router_probs: Optional[torch.Tensor] = None  # MoE routing probs


# ===================================================================
# Transformer Block V2
# ===================================================================


class TransformerBlockV2(nn.Module):
    """Configurable transformer block supporting all architecture variants.

    This block adapts its internal structure based on the ``residual_type``
    and other configuration flags:

    - **PRE_NORM** (LLaMA style):
        ``x' = x + attn(norm(x)); x'' = x' + ffn(norm(x'))``

    - **POST_NORM** (original Transformer):
        ``x' = norm(x + attn(x)); x'' = norm(x' + ffn(x'))``

    - **PARALLEL** (GPT-J / PaLM):
        ``x' = x + attn(norm(x)) + ffn(norm2(x))``

    - **DEEPNET**:
        ``x' = alpha * x + gamma * attn(beta * norm(x)); ...``

    The block also handles MoE auxiliary losses returned by the FFN
    layer when ``ffn_type == MOE``.

    Args:
        config: ModelConfigV2 instance with all architecture choices.
        layer_idx: Zero-based index of this layer (for DeepNet scaling).
    """

    def __init__(self, config: ModelConfigV2, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Determine bias setting (modern LLMs typically don't use bias)
        bias = config.norm_bias

        # ----------------------------------------------------------------
        # Attention
        # ----------------------------------------------------------------
        attn_kwargs = self._build_attention_kwargs(config)
        self.self_attn = create_attention(attn_kwargs)  # type: ignore[operator]

        # Wrap with QK normalization if requested
        if config.use_qk_norm and AttentionWithQKNorm is not None:
            self.self_attn = AttentionWithQKNorm(
                self.self_attn,
                norm_type="rmsnorm",
                head_dim=config.head_dim,
            )

        # ----------------------------------------------------------------
        # FFN
        # ----------------------------------------------------------------
        ffn_kwargs = self._build_ffn_kwargs(config)
        self.mlp = create_ffn(ffn_kwargs)  # type: ignore[operator]

        # ----------------------------------------------------------------
        # Normalization (two per block: pre-attention, pre-FFN)
        # ----------------------------------------------------------------
        norm_kwargs: Dict[str, Any] = {"eps": config.rms_norm_eps}
        if config.normalization == NormalizationType.DEEPNORM:
            norm_kwargs["num_layers"] = config.num_hidden_layers

        self.input_norm = get_norm(  # type: ignore[operator]
            config.normalization.value,
            config.hidden_size,
            **norm_kwargs,
        )
        self.post_attention_norm = get_norm(  # type: ignore[operator]
            config.normalization.value,
            config.hidden_size,
            **norm_kwargs,
        )

        # ----------------------------------------------------------------
        # Residual connections
        # ----------------------------------------------------------------
        # We use a simple residual here; the block's forward() handles
        # the ordering based on residual_type.
        # The get_residual factory wraps sublayer + norm, but since we
        # handle norm separately, we use a lightweight approach.
        self.use_parallel_residual = config.residual_type == ResidualType.PARALLEL
        self.use_deepnet = config.residual_type == ResidualType.DEEPNET
        self.use_post_norm = config.residual_type == ResidualType.POST_NORM

        # DeepNet scaling factors
        if self.use_deepnet:
            N = config.num_hidden_layers
            self.deepnet_alpha = (2.0 * N) ** 0.25
            self.deepnet_beta = (8.0 * N) ** (-0.25)
            self.deepnet_gamma = 1.0 / (2.0 * N) ** 0.5
        else:
            self.deepnet_alpha = 1.0
            self.deepnet_beta = 1.0
            self.deepnet_gamma = 1.0

    # ------------------------------------------------------------------
    # Config building helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_attention_kwargs(config: ModelConfigV2) -> Dict[str, Any]:
        """Build kwargs dict for create_attention from ModelConfigV2."""
        attn_type = config.attention_type.value
        d = {
            "attention_type": attn_type,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "dropout": config.attention_dropout,
            "bias": config.norm_bias,
        }

        # MLA-specific params
        if attn_type == "mla":
            d["kv_lora_rank"] = config.kv_lora_rank
            d["rope_head_dim"] = config.head_dim // 2

        return d

    @staticmethod
    def _build_ffn_kwargs(config: ModelConfigV2) -> Dict[str, Any]:
        """Build kwargs dict for create_ffn from ModelConfigV2."""
        d: Dict[str, Any] = {
            "ffn_type": config.ffn_type.value,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "bias": config.norm_bias,
            "hidden_dropout": config.hidden_dropout,
        }

        # MoE-specific params
        if config.ffn_type == FFNType.MOE:
            d["num_experts"] = config.num_experts
            d["num_experts_per_token"] = config.num_experts_per_token
            d["moe_aux_loss_coeff"] = config.moe_load_balance_coeff
            d["routing_method"] = config.routing_method.value
            d["expert_capacity_factor"] = config.moe_capacity_factor
            d["shared_expert"] = config.shared_expert
            d["jitter_noise"] = config.moe_jitter_noise
        else:
            d["hidden_act"] = "silu"  # default activation for gated FFN

        return d

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        alibi_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass through one transformer block.

        Handles all residual types, MoE auxiliary losses, QK-norm wrapping,
        and DeepNet scaling.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional causal mask.
            position_ids: Optional position indices.
            past_key_value: KV cache from previous step.
            rope_cos, rope_sin: Precomputed RoPE tensors.
            alibi_bias: Precomputed ALiBi bias tensor.
            output_attentions: Whether to return attention weights.
            use_cache: Whether to update and return KV cache.

        Returns:
            Tuple of ``(hidden_states, present_kv, attn_weights, aux_loss)``.
        """
        residual = hidden_states

        # ================================================================
        # Attention Branch
        # ================================================================
        if self.use_parallel_residual:
            # GPT-J style: both attn and ffn operate on the same input
            normed = self.input_norm(hidden_states)

            # Apply DeepNet scaling if needed
            if self.use_deepnet:
                normed = self.deepnet_beta * normed

            attn_output, attn_weights, present_kv = self.self_attn(
                normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # Scale attention output for DeepNet
            if self.use_deepnet:
                attn_output = self.deepnet_gamma * attn_output

            # Apply ALiBi bias if needed (applied separately since MLA handles
            # positional encoding internally)
            if alibi_bias is not None:
                # ALiBi is an additive bias; it's applied inside attention
                # but we store it for potential downstream use
                pass

            # ============================================================
            # FFN Branch (parallel with attention)
            # ============================================================
            ffn_normed = self.post_attention_norm(hidden_states)
            if self.use_deepnet:
                ffn_normed = self.deepnet_beta * ffn_normed

            ffn_output = self.mlp(ffn_normed)
            if isinstance(ffn_output, tuple):
                ffn_output, aux_loss = ffn_output
            else:
                aux_loss = None

            if self.use_deepnet:
                ffn_output = self.deepnet_gamma * ffn_output

            # Combine: residual + attn + ffn
            hidden_states = self.deepnet_alpha * residual + attn_output + ffn_output

        else:
            # Sequential attention → FFN
            normed = self.input_norm(hidden_states)

            # DeepNet scaling
            if self.use_deepnet:
                normed = self.deepnet_beta * normed

            attn_output, attn_weights, present_kv = self.self_attn(
                normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # Scale for DeepNet
            if self.use_deepnet:
                attn_output = self.deepnet_gamma * attn_output

            # Post-norm residual: norm(x + attn(x))
            if self.use_post_norm:
                hidden_states = self.input_norm(residual + attn_output)
            else:
                # Pre-norm residual: alpha * x + gamma * attn(norm(x))
                if self.use_deepnet:
                    hidden_states = self.deepnet_alpha * residual + attn_output
                else:
                    hidden_states = residual + attn_output

            # ============================================================
            # FFN
            # ============================================================
            residual = hidden_states
            ffn_normed = self.post_attention_norm(hidden_states)

            if self.use_deepnet:
                ffn_normed = self.deepnet_beta * ffn_normed

            ffn_output = self.mlp(ffn_normed)

            # MoE returns (output, aux_loss)
            if isinstance(ffn_output, tuple):
                ffn_output, aux_loss = ffn_output
            else:
                aux_loss = None

            if self.use_deepnet:
                ffn_output = self.deepnet_gamma * ffn_output

            if self.use_post_norm:
                hidden_states = self.post_attention_norm(residual + ffn_output)
            else:
                if self.use_deepnet:
                    hidden_states = self.deepnet_alpha * residual + ffn_output
                else:
                    hidden_states = residual + ffn_output

        return hidden_states, present_kv, attn_weights, aux_loss


# ===================================================================
# Full Transformer Model V2
# ===================================================================


class NexusTransformerV2(nn.Module):
    """Nexus v2: Fully configurable 100B+ decoder-only transformer.

    This is the main model class that composes all v2 components into a
    complete transformer language model.  It supports every combination of
    attention, FFN, positional encoding, normalization, and residual variants
    exposed by :class:`ModelConfigV2`.

    Key capabilities:
        - **Dense models**: GQA/SwiGLU (LLaMA, Mistral, Qwen, etc.)
        - **MoE models**: Up to 256 experts with shared expert
        - **MLA attention**: DeepSeek-style compressed KV caching
        - **Differential attention**: Noise-cancelling dual attention
        - **Long context**: YaRN, LongRoPE, ALiBi, Dynamic NTK
        - **Deep architectures**: DeepNet scaling for 1000+ layers
        - **Fused cross-entropy**: Memory-efficient loss computation
        - **Gradient checkpointing**: For training at 100B+ scale
        - **KV caching**: Efficient autoregressive generation
        - **Weight tying**: Share embedding/output head weights

    The v2 model is fully backward-compatible with v1 configs but supports
    many more options.

    Args:
        config: A :class:`ModelConfigV2` instance.

    Example::

        # Create from preset
        config = ModelConfigV2.nexus_100b()
        model = NexusTransformerV2(config)

        # Forward pass
        output = model(input_ids, labels=labels)
        print(output.loss)

        # Generate text
        generated = model.generate(input_ids, max_new_tokens=64)
    """

    def __init__(self, config: ModelConfigV2) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        # ================================================================
        # Token Embeddings
        # ================================================================
        if Embedding is not None:
            self.embed_tokens = Embedding(config)  # type: ignore[operator]
        else:
            # Fallback to standard nn.Embedding
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size, config.padding_idx
            )
            nn.init.normal_(
                self.embed_tokens.weight,
                std=config.initializer_range,
            )

        # ================================================================
        # Positional Encoding
        # ================================================================
        self.pos_encoding: Optional[nn.Module] = None
        self.alibi: Optional[nn.Module] = None

        if config.positional_encoding == PositionalEncodingType.ROPE:
            if RotaryPositionEmbedding is not None:
                scaling_type = "standard"
                if config.rope_scaling == RopeScalingType.LINEAR:
                    scaling_type = "linear"
                elif config.rope_scaling == RopeScalingType.DYNAMIC_NTK:
                    scaling_type = "dynamic_ntk"
                elif config.rope_scaling == RopeScalingType.YARN:
                    scaling_type = "yarn"
                elif config.rope_scaling == RopeScalingType.LONGROPE:
                    scaling_type = "longrope"

                self.pos_encoding = RotaryPositionEmbedding(
                    dim=config.head_dim,
                    max_seq_len=config.max_position_embeddings,
                    base=config.rope_base,
                    scaling_type=scaling_type,
                    scaling_factor=config.rope_scaling_factor,
                    yarn_params={
                        "beta_fast": config.yarn_beta_fast,
                        "beta_slow": config.yarn_beta_slow,
                    },
                )
        elif config.positional_encoding == PositionalEncodingType.ALIBI:
            if ALiBiPositionalEncoding is not None:
                self.alibi = ALiBiPositionalEncoding(
                    num_heads=config.num_attention_heads,
                    context_length=config.max_position_embeddings,
                )
        elif config.positional_encoding == PositionalEncodingType.LEARNED:
            if LearnedPositionEmbedding is not None:
                self.pos_encoding = LearnedPositionEmbedding(
                    max_seq_len=config.max_position_embeddings,
                    hidden_size=config.hidden_size,
                )

        # ================================================================
        # Transformer Blocks
        # ================================================================
        self.layers = nn.ModuleList([
            TransformerBlockV2(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # ================================================================
        # Final Normalization
        # ================================================================
        norm_kwargs: Dict[str, Any] = {"eps": config.rms_norm_eps}
        if config.normalization == NormalizationType.DEEPNORM:
            norm_kwargs["num_layers"] = config.num_hidden_layers

        self.norm = get_norm(  # type: ignore[operator]
            config.normalization.value,
            config.hidden_size,
            **norm_kwargs,
        )

        # ================================================================
        # Output Head
        # ================================================================
        if config.fused_cross_entropy and FusedCrossEntropyLMHead is not None:
            self.lm_head = FusedCrossEntropyLMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                bias=config.output_head_bias,
            )
            self._use_fused_head = True
        elif LMHead is not None:
            self.lm_head = LMHead(
                hidden_size=config.hidden_size,
                vocab_size=config.vocab_size,
                bias=config.output_head_bias,
            )
            self._use_fused_head = False
        else:
            # Fallback to standard nn.Linear
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=config.output_head_bias
            )
            self._use_fused_head = False

        # ================================================================
        # Weight Tying
        # ================================================================
        if config.tie_word_embeddings:
            if hasattr(self.lm_head, "tie_weights"):
                self.lm_head.tie_weights(self.embed_tokens.weight)  # type: ignore[union-attr]
            elif hasattr(self.lm_head, "weight"):
                self.lm_head.weight = self.embed_tokens.weight  # type: ignore[union-attr]

        # ================================================================
        # Initialize Weights
        # ================================================================
        self.apply(self._init_weights)

        # Gradient checkpointing flag
        self.gradient_checkpointing_enabled = config.gradient_checkpointing

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following the LLaMA initialization scheme.

        - Linear layers: Normal(0, initializer_range)
        - Normalization layers: weight = 1.0
        - Embedding layers: Normal(0, initializer_range)
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Embedding,)):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if self.padding_idx is not None and module.padding_idx == self.padding_idx:
                nn.init.constant_(module.weight[self.padding_idx], 0.0)
        # Norm layers are initialized by their own constructors

    # ------------------------------------------------------------------
    # RoPE / ALiBi helpers
    # ------------------------------------------------------------------

    def _prepare_rope(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """Prepare position information and RoPE cos/sin tensors.

        Returns:
            Tuple of ``(rope_cos, rope_sin, position_ids)``.
        """
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_len + past_key_values_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        rope_cos: Optional[torch.Tensor] = None
        rope_sin: Optional[torch.Tensor] = None

        if self.pos_encoding is not None and isinstance(
            self.pos_encoding, RotaryPositionEmbedding
        ):
            cos, sin = self.pos_encoding(
                input_ids,
                seq_len=seq_len + past_key_values_length,
                position_ids=position_ids,
            )
            # Slice to the current sequence window
            cos = cos.to(dtype=input_ids.dtype)
            sin = sin.to(dtype=input_ids.dtype)
            rope_cos = cos
            rope_sin = sin

        return rope_cos, rope_sin, position_ids

    def _prepare_alibi(
        self,
        seq_len: int,
        total_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Prepare ALiBi attention bias.

        Returns:
            ALiBi bias tensor of shape ``(1, num_heads, seq_len, total_seq_len)``
            or None if ALiBi is not used.
        """
        if self.alibi is None:
            return None
        return self.alibi.get_bias(
            total_seq_len, device=device, dtype=dtype
        )

    # ------------------------------------------------------------------
    # Attention Mask
    # ------------------------------------------------------------------

    def _create_causal_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values_length: int,
    ) -> Optional[torch.Tensor]:
        """Create 4D causal attention mask.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional 2D mask from the user.
            past_key_values_length: Length of cached KV.

        Returns:
            4D causal mask or None.
        """
        if attention_mask is None:
            return None

        bsz, seq_len = input_ids.shape
        total_seq_len = seq_len + past_key_values_length

        # Build causal mask
        causal_mask = torch.triu(
            torch.ones(
                seq_len, total_seq_len,
                device=input_ids.device, dtype=torch.bool,
            ),
            diagonal=past_key_values_length + 1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, T)

        # Combine with user-provided mask
        attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
        combined = attn_mask & (~causal_mask)
        combined = combined.to(dtype=input_ids.dtype)
        # Convert to additive mask: 0 for attend, -inf for block
        mask = (1.0 - combined) * torch.finfo(input_ids.dtype).min

        return mask

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TransformerOutputV2:
        """Full forward pass of the transformer model.

        Handles RoPE, ALiBi, causal masking, KV caching, MoE auxiliary loss,
        fused cross-entropy, and all architecture variants.

        Args:
            input_ids: Token indices ``(batch, seq_len)``.
            attention_mask: Optional mask ``(batch, seq_len)``.  1 = attend, 0 = ignore.
            position_ids: Optional position indices ``(batch, seq_len)``.
            past_key_values: Cached KV per layer for autoregressive generation.
            inputs_embeds: Optional embedded inputs (bypasses embed_tokens).
            labels: Target token indices for loss computation (shifted internally).
            use_cache: Whether to use and return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.

        Returns:
            :class:`TransformerOutputV2` with logits, loss, and optional extras.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False

        # Determine past length for KV cache
        if past_key_values is not None and len(past_key_values) > 0:
            past_len = past_key_values[0][0].shape[2]
        else:
            past_len = 0

        # Resolve input (ids → embeddings)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len = inputs_embeds.shape[:2]

        # Apply embedding scaling if configured
        if self.config.embedding_scale:
            inputs_embeds = inputs_embeds * (self.config.hidden_size ** 0.5)

        # ================================================================
        # Positional Encoding
        # ================================================================
        rope_cos, rope_sin, position_ids = self._prepare_rope(
            input_ids if input_ids is not None else inputs_embeds,
            position_ids,
            past_len,
        )

        # ALiBi bias
        alibi_bias = self._prepare_alibi(
            seq_len,
            seq_len + past_len,
            inputs_embeds.device,
            inputs_embeds.dtype,
        )

        # Add learned positional embeddings
        if (
            self.pos_encoding is not None
            and isinstance(self.pos_encoding, LearnedPositionEmbedding)
            and position_ids is not None
        ):
            inputs_embeds = inputs_embeds + self.pos_encoding(position_ids)

        # ================================================================
        # Causal Attention Mask
        # ================================================================
        attn_mask = self._create_causal_mask(
            input_ids if input_ids is not None else inputs_embeds,
            attention_mask,
            past_len,
        )

        # Combine with ALiBi
        if alibi_bias is not None:
            if attn_mask is not None:
                # Slice alibi to match query sequence length
                alibi_sliced = alibi_bias[:, :, :seq_len, :seq_len + past_len]
                attn_mask = attn_mask + alibi_sliced
            else:
                attn_mask = alibi_bias[:, :, :seq_len, :seq_len + past_len]

        # ================================================================
        # Process through all layers
        # ================================================================
        hidden_states = inputs_embeds
        all_hidden_states: Tuple[torch.Tensor, ...] = () if output_hidden_states else ()  # type: ignore[assignment]
        all_self_attns: Tuple[torch.Tensor, ...] = () if output_attentions else ()  # type: ignore[assignment]
        next_cache: Tuple = () if use_cache else ()  # type: ignore[assignment]
        total_aux_loss: Optional[torch.Tensor] = None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Get past KV for this layer
            layer_past = None
            if past_key_values is not None and idx < len(past_key_values):
                layer_past = past_key_values[idx]

            # Gradient checkpointing
            if self.gradient_checkpointing_enabled and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attn_mask,
                    position_ids,
                    layer_past,
                    rope_cos,
                    rope_sin,
                    alibi_bias,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    alibi_bias=alibi_bias,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache and layer_outputs[1] is not None:
                next_cache = next_cache + (layer_outputs[1],)

            if output_attentions and layer_outputs[2] is not None:
                all_self_attns = all_self_attns + (layer_outputs[2],)

            # Accumulate MoE auxiliary loss
            if layer_outputs[3] is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_outputs[3]
                else:
                    total_aux_loss = total_aux_loss + layer_outputs[3]

        # ================================================================
        # Final Normalization
        # ================================================================
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # ================================================================
        # Output Head
        # ================================================================
        if self._use_fused_head and labels is not None:
            # Fused cross-entropy: compute loss directly
            shift_labels = labels[:, 1:].contiguous()
            loss = self.lm_head(hidden_states[:, :-1, :].contiguous(), shift_labels)
            logits = self.lm_head.compute_logits(hidden_states)
        else:
            if self._use_fused_head:
                logits = self.lm_head.compute_logits(hidden_states)
            elif hasattr(self.lm_head, "forward"):
                logits = self.lm_head(hidden_states)
            else:
                logits = self.lm_head(hidden_states)

            # Standard cross-entropy loss
            loss = None
            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction="mean")
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                )

        # Add MoE auxiliary loss to main loss
        if loss is not None and total_aux_loss is not None:
            loss = loss + self.config.moe_load_balance_coeff * total_aux_loss

        return TransformerOutputV2(
            logits=logits,
            hidden_states=all_hidden_states if output_hidden_states else None,
            past_key_values=next_cache if use_cache else None,
            attentions=all_self_attns if output_attentions else None,
            loss=loss,
            aux_loss=total_aux_loss,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        stop_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Autoregressive text generation with KV caching.

        Supports temperature sampling, top-p (nucleus), top-k filtering,
        repetition penalty, and stop token detection.

        Args:
            input_ids: Prompt tokens ``(batch, seq_len)``.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (< 1.0 = more deterministic).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filtering threshold.
            repetition_penalty: Penalty for repeating tokens (> 1.0 = penalize).
            use_cache: Whether to use KV caching.
            stop_token_ids: List of token IDs that stop generation.
            eos_token_id: End-of-sequence token ID.

        Returns:
            Generated token sequence ``(batch, prompt_len + gen_len)``.
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass with KV cache
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_val = min(top_k, next_token_logits.shape[-1])
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k_val
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # Check stop tokens
            if stop_token_ids is not None:
                if any(t in stop_token_ids for t in next_token.view(-1).tolist()):
                    break
            if eos_token_id is not None:
                if eos_token_id in next_token.view(-1).tolist():
                    break

        return generated

    # ------------------------------------------------------------------
    # Gradient Checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to save memory during training."""
        self.gradient_checkpointing_enabled = True
        # Also enable on each block
        for module in self.modules():
            if isinstance(module, TransformerBlockV2):
                module.gradient_checkpointing_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        for module in self.modules():
            if isinstance(module, TransformerBlockV2):
                module.gradient_checkpointing_enabled = False

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count total parameters in the model.

        Args:
            trainable_only: Only count parameters with requires_grad=True.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def estimated_memory_gb(self) -> float:
        """Estimate memory in GB for BF16 precision.

        Returns:
            Estimated memory in gigabytes.
        """
        bytes_per_param = 2  # BF16
        return (self.num_parameters(trainable_only=False) * bytes_per_param) / (1024 ** 3)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str) -> None:
        """Save model checkpoint to disk.

        Saves the configuration as YAML and weights as safetensors
        (or pytorch_model.bin as fallback).

        Args:
            path: Directory to save the checkpoint into.
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        self.config.to_yaml(os.path.join(path, "config_v2.yaml"))

        # Save weights
        try:
            from safetensors.torch import save_file

            save_file(
                {k: v for k, v in self.state_dict().items()},
                os.path.join(path, "model.safetensors"),
            )
        except ImportError:
            torch.save(
                self.state_dict(),
                os.path.join(path, "pytorch_model.bin"),
            )

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "NexusTransformerV2":
        """Load model from a checkpoint directory.

        Args:
            path: Directory containing the checkpoint.
            device: Device to load the model onto.

        Returns:
            Loaded :class:`NexusTransformerV2` instance.
        """
        # Load config
        config = ModelConfigV2.from_yaml(os.path.join(path, "config_v2.yaml"))

        # Create model
        model = cls(config)

        # Load weights
        safetensors_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file

                state_dict = load_file(safetensors_path)
            except ImportError:
                state_dict = torch.load(safetensors_path, map_location="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {path}. Expected model.safetensors or pytorch_model.bin."
            )

        model.load_state_dict(state_dict, strict=False)
        return model.to(device)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        c = self.config
        parts = [
            f"name={c.name!r}",
            f"layers={c.num_hidden_layers}",
            f"hidden={c.hidden_size}",
            f"heads={c.num_attention_heads}",
            f"kv_heads={c.num_key_value_heads}",
            f"attn={c.attention_type.value}",
            f"ffn={c.ffn_type.value}",
            f"pos={c.positional_encoding.value}",
            f"norm={c.normalization.value}",
            f"residual={c.residual_type.value}",
        ]
        if c.is_moe:
            parts.append(f"experts={c.num_experts}")
        return ", ".join(parts)
