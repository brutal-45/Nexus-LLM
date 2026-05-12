"""
Nexus Model Configuration
============================
Dataclass-based config for the decoder-only transformer model.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Full model configuration for Nexus decoder-only transformer."""

    # Model identity
    name: str = "Nexus-100B"
    arch: str = "decoder_only_transformer"

    # Dimensions
    hidden_size: int = 12288
    intermediate_size: int = 49152
    num_hidden_layers: int = 80
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 8192
    vocab_size: int = 65536

    # Normalization
    rms_norm_eps: float = 1.0e-5
    hidden_act: str = "siu"
    tie_word_embeddings: bool = False

    # Regularization
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    initializer_range: float = 0.02

    # RoPE
    rope_theta: float = 10000.0
    use_rope_scaling: bool = False
    rope_scaling_factor: float = 1.0

    # MoE (optional, future support)
    use_moe: bool = False
    num_experts: int = 0
    num_experts_per_token: int = 0
    moe_aux_loss_coeff: float = 0.01

    def __post_init__(self):
        """Validate configuration after creation."""
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        self.head_dim = self.hidden_size // self.num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({self.num_key_value_heads})"
        )
        self.num_kv_groups = self.num_attention_heads // self.num_key_value_heads

    @property
    def total_params(self) -> int:
        """Estimate total model parameters."""
        head_dim = self.hidden_size // self.num_attention_heads
        num_kv_groups = self.num_attention_heads // self.num_key_value_heads

        # Per layer: attention + FFN + norms
        attn_params = (
            # Q projection
            self.hidden_size * self.num_attention_heads * head_dim +
            # K projection (GQA)
            self.hidden_size * self.num_key_value_heads * head_dim +
            # V projection (GQA)
            self.hidden_size * self.num_key_value_heads * head_dim +
            # Output projection
            self.num_attention_heads * head_dim * self.hidden_size
        )
        ffn_params = (
            # SwiGLU gate + up + down
            self.hidden_size * self.intermediate_size * 3
        )
        norm_params = self.hidden_size * 2  # input + post-attn RMSNorm
        per_layer = attn_params + ffn_params + norm_params

        total = per_layer * self.num_hidden_layers
        # Final norm
        total += self.hidden_size
        # Output head (lm_head)
        total += self.hidden_size * self.vocab_size
        # Embedding
        total += self.vocab_size * self.hidden_size

        return total

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str) -> ModelConfig:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg.get("model", cfg))

    def save_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        params_b = self.total_params / 1e9
        return (
            f"ModelConfig(name={self.name}, "
            f"layers={self.num_hidden_layers}, "
            f"hidden={self.hidden_size}, "
            f"heads={self.num_attention_heads}, "
            f"kv_heads={self.num_key_value_heads}, "
            f"params={params_b:.1f}B)"
        )
