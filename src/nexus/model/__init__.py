"""
Model Package Init
==================
Complete model package for Nexus decoder-only transformer.

v1 (preserved): Original GQA + SwiGLU + RoPE + RMSNorm transformer
v2 (new): All attention variants, FFN variants, positional encodings,
          normalizations, residual connections, MoE, Flash Attention, etc.
v3 (innovations): Parallel Attn+FFN, Shared Attention, MoD, Early Exit, Universal Transformers
"""

# === v1 Modules (preserved) ===
from .config import ModelConfig
from .transformer import NexusTransformer
from .attention import GroupedQueryAttention
from .ffn import SwiGLUFFN
from .embeddings import Embedding
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary_pos_emb, rotate_half

# === v2 Modules (new) ===
from .config_v2 import (
    ModelConfigV2,
    AttentionType, FFNType, PositionalEncodingType,
    NormalizationType, ResidualType, RopeScalingType,
    RoutingMethod, PrecisionType,
)
from .transformer_v2 import NexusTransformerV2, TransformerBlockV2, TransformerOutputV2
from .positional import (
    RotaryPositionEmbedding, ALiBiPositionalEncoding,
    LearnedPositionEmbedding, ContextLengthExtension,
    get_positional_encoding,
)
from .attention_v2 import (
    MultiHeadAttention, MultiQueryAttention, GroupedQueryAttentionV2,
    MultiHeadLatentAttention, DifferentialAttention,
    AttentionWithQKNorm, create_attention,
)
from .flash_attention import (
    FlashAttentionV2, FlashAttentionV3,
    PagedAttentionBlock, RingAttention,
)
from .sparse_attention import (
    SlidingWindowAttention, BigBirdStyleAttention,
    DilatedAttention, BlockSparseAttention,
)
from .ffn_v2 import (
    StandardFFN, SwiGLUFFNv2, GeGLUFFN, ReGLUFFN,
    ExpertRouter, Expert, MixtureOfExperts, FineGrainedMoE,
    create_ffn,
)
from .normalization import (
    LayerNorm, DeepNorm, QKNorm, SubLayerNorm,
    get_norm,
)
from .residual import (
    StandardResidual, ScaledResidual, PreNormResidual,
    PostNormResidual, DeepNetResidual, ParallelResidual,
    ResidualStream, get_residual_block,
)
from .activations import (
    ReLU, GELU, GELUTanh, SiLU, Mish, SquaredReLU,
    GatedGELU, GatedReLU, StarReLU, QuickGELU,
    LeakyReLU, HardSwish, HardSigmoid, ELU, GLU_SiLU,
    get_activation,
)
from .embeddings_v2 import (
    VocabParallelEmbedding, ScaledEmbedding,
    EmbeddingWithWeightTying, RotaryEmbeddingV2,
    ALiBiPositionalBias, CombinedEmbedding,
)
from .output_head import (
    LMHead, ParallelLMHead, FusedCrossEntropyLMHead,
    LogSoftmaxHead, ScaledDotProductScorer, AdaptiveOutputHead,
)

# === v3 Innovations ===
from .innovations import (
    ParallelAttentionFFNBlock,
    SharedAttentionTransformerBlock,
    SharedAttentionTransformer,
    MixtureOfDepthsRouter,
    MixtureOfDepthsBlock,
    MixtureOfDepthsTransformer,
    EarlyExitHead,
    EarlyExitTransformer,
    UniversalTransformerBlock,
    UniversalTransformer,
)

__all__ = [
    # v1 (preserved)
    "ModelConfig", "NexusTransformer", "GroupedQueryAttention",
    "SwiGLUFFN", "Embedding", "RMSNorm",
    "RotaryEmbedding", "apply_rotary_pos_emb", "rotate_half",
    # v2 Config
    "ModelConfigV2", "AttentionType", "FFNType", "PositionalEncodingType",
    "NormalizationType", "ResidualType", "RopeScalingType",
    "RoutingMethod", "PrecisionType",
    # v2 Transformer
    "NexusTransformerV2", "TransformerBlockV2", "TransformerOutputV2",
    # v2 Positional
    "RotaryPositionEmbedding", "ALiBiPositionalEncoding",
    "LearnedPositionEmbedding", "ContextLengthExtension", "get_positional_encoding",
    # v2 Attention
    "MultiHeadAttention", "MultiQueryAttention", "GroupedQueryAttentionV2",
    "MultiHeadLatentAttention", "DifferentialAttention",
    "AttentionWithQKNorm", "create_attention",
    # v2 Flash/Sparse Attention
    "FlashAttentionV2", "FlashAttentionV3",
    "PagedAttentionBlock", "RingAttention",
    "SlidingWindowAttention", "BigBirdStyleAttention",
    "DilatedAttention", "BlockSparseAttention",
    # v2 FFN
    "StandardFFN", "SwiGLUFFNv2", "GeGLUFFN", "ReGLUFFN",
    "ExpertRouter", "Expert", "MixtureOfExperts", "FineGrainedMoE", "create_ffn",
    # v2 Normalization
    "LayerNorm", "DeepNorm", "QKNorm", "SubLayerNorm", "get_norm",
    # v2 Residual
    "StandardResidual", "ScaledResidual", "PreNormResidual",
    "PostNormResidual", "DeepNetResidual", "ParallelResidual",
    "ResidualStream", "get_residual_block",
    # v2 Activations
    "ReLU", "GELU", "GELUTanh", "SiLU", "Mish", "SquaredReLU",
    "GatedGELU", "GatedReLU", "StarReLU", "QuickGELU",
    "LeakyReLU", "HardSwish", "HardSigmoid", "ELU", "GLU_SiLU", "get_activation",
    # v2 Embeddings
    "VocabParallelEmbedding", "ScaledEmbedding",
    "EmbeddingWithWeightTying", "RotaryEmbeddingV2",
    "ALiBiPositionalBias", "CombinedEmbedding",
    # v2 Output Head
    "LMHead", "ParallelLMHead", "FusedCrossEntropyLMHead",
    "LogSoftmaxHead", "ScaledDotProductScorer", "AdaptiveOutputHead",
    # v3 Innovations
    "ParallelAttentionFFNBlock",
    "SharedAttentionTransformerBlock", "SharedAttentionTransformer",
    "MixtureOfDepthsRouter", "MixtureOfDepthsBlock", "MixtureOfDepthsTransformer",
    "EarlyExitHead", "EarlyExitTransformer",
    "UniversalTransformerBlock", "UniversalTransformer",
]
