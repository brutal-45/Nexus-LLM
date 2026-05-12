"""
Multimodal Configuration Module
================================

Comprehensive configuration dataclasses for all multimodal components in the
Nexus LLM framework. Provides typed, validated, serializable configuration
for vision encoders, audio encoders, video encoders, cross-modal fusion,
processing pipelines, and the overall multimodal system.

Each configuration class includes:
- Typed fields with sensible defaults
- Validation logic ensuring consistent and valid parameters
- Serialization to/from dictionaries and JSON
- Deep copy and merge capabilities for configuration management
- Compatibility checks between related components
- Extensive property methods for derived quantities

Dataclasses:
    - VisionConfig: Image encoder parameters (ViT, CLIP, SigLIP, ConvNeXt)
    - AudioConfig: Audio processing parameters (Whisper, spectrogram, MFCC)
    - VideoConfig: Video encoder parameters (TimeSformer, ViViT, VideoSwin)
    - CrossModalConfig: Fusion, projection, gating, and cross-modal attention
    - MultimodalConfig: Master configuration combining all sub-configs
    - ProcessorConfig: Image, audio, video, and text processor pipelines

Usage:
    >>> from dataclasses import dataclass, field
    >>> config = MultimodalConfig()
    >>> config.vision.image_size = 512
    >>> errors = config.validate()
    >>> assert not errors, f"Config errors: {errors}"
    >>> d = config.to_dict()
    >>> config2 = MultimodalConfig.from_dict(d)
    >>> assert config.vision.image_size == config2.vision.image_size
"""

import copy
import json
import math
import os
import warnings
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch


# =============================================================================
# Enums
# =============================================================================

class NormalizationStrategy(str, Enum):
    """Strategy for normalizing multimodal inputs."""
    IMAGENET = "imagenet"
    CLIP = "clip"
    SIGLIP = "siglip"
    CUSTOM = "custom"
    NONE = "none"
    LAYER_NORM = "layer_norm"
    INSTANCE_NORM = "instance_norm"
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"


class PretrainStrategy(str, Enum):
    """Strategy for pretraining vision-language models."""
    CONTRASTIVE = "contrastive"
    GENERATIVE = "generative"
    ALIGNMENT = "alignment"
    MASKED = "masked"
    MULTI_TASK = "multi_task"
    INSTRUCTION_TUNING = "instruction_tuning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DISTILLATION = "distillation"


class FusionType(str, Enum):
    """Type of cross-modal fusion mechanism."""
    CROSS_ATTENTION = "cross_attention"
    CONCATENATION = "concatenation"
    CO_ATTENTION = "co_attention"
    COMPACT_BILINEAR = "compact_bilinear"
    GATED = "gated"
    ADAPTIVE = "adaptive"
    TRANSFORMER = "transformer"
    NONE = "none"


class GatingType(str, Enum):
    """Type of gating mechanism for modality fusion."""
    SCALAR = "scalar"
    BINARY = "binary"
    SOFT = "soft"
    SPARSE = "sparse"
    TOP_K = "top_k"
    NONE = "none"


class AttentionPattern(str, Enum):
    """Attention pattern configuration for transformer architectures."""
    FULL = "full"
    DIVIDED_SPACE_TIME = "divided_space_time"
    FACTORIZED = "factorized"
    WINDOWED = "windowed"
    LOCAL_GLOBAL = "local_global"
    LINEAR = "linear"
    FLASH = "flash"
    MEMORY_EFFICIENT = "memory_efficient"


class ModalityType(str, Enum):
    """Supported modality types."""
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    POINT_CLOUD = "point_cloud"
    DEPTH = "depth"
    THERMAL = "thermal"


class ProjectionType(str, Enum):
    """Type of projection layer for connecting encoders to LLM."""
    LINEAR = "linear"
    MLP = "mlp"
    QFORMER = "qformer"
    RESAMPLER = "resampler"
    C_ABSTRACTOR = "c_abstractor"
    CONV = "conv"
    IDENTITY = "identity"


class ActivationType(str, Enum):
    """Activation function type."""
    GELU = "gelu"
    RELU = "relu"
    SILU = "silu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    GELU_QUICK = "gelu_quick"
    MISH = "mish"
    LEAKY_RELU = "leaky_relu"


class PositionEncodingType(str, Enum):
    """Type of position encoding."""
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"
    ROTARY = "rotary"
    ALIBI = "alibi"
    RELATIVE = "relative"
    ROPE = "rope"
    NONE = "none"


class NormType(str, Enum):
    """Type of normalization layer."""
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    BATCH_NORM = "batch_norm"
    GROUP_NORM = "group_norm"
    INSTANCE_NORM = "instance_norm"
    NONE = "none"


class EncoderArchType(str, Enum):
    """Encoder architecture family."""
    VIT = "vit"
    CLIP = "clip"
    SIGLIP = "siglip"
    CONVNEXT = "convnext"
    SWIN = "swin"
    WHISPER = "whisper"
    HUBERT = "hubert"
    TIMESFORMER = "timesformer"
    VIVIT = "vivit"
    VIDEOSWIN = "videoswin"
    CUSTOM = "custom"


# =============================================================================
# Resolver Functions
# =============================================================================

def resolve_enum_value(value: Any, enum_class: type) -> Any:
    """Resolve an enum value from string or enum instance.

    Args:
        value: Either an enum instance or a string representation.
        enum_class: The enum class to resolve against.

    Returns:
        The corresponding enum value.

    Raises:
        ValueError: If the value cannot be resolved.
        TypeError: If the value type is not supported.
    """
    if isinstance(value, enum_class):
        return value
    if isinstance(value, str):
        try:
            return enum_class(value.lower())
        except ValueError:
            valid_options = [e.value for e in enum_class]
            raise ValueError(
                f"Unknown value '{value}' for {enum_class.__name__}. "
                f"Valid options: {valid_options}"
            )
    raise TypeError(
        f"Expected str or {enum_class.__name__}, got {type(value).__name__}"
    )


def resolve_fusion_type(value: Union[str, FusionType]) -> FusionType:
    """Resolve a fusion type from string or enum value."""
    return resolve_enum_value(value, FusionType)


def resolve_gating_type(value: Union[str, GatingType]) -> GatingType:
    """Resolve a gating type from string or enum value."""
    return resolve_enum_value(value, GatingType)


def resolve_normalization_strategy(value: Union[str, NormalizationStrategy]) -> NormalizationStrategy:
    """Resolve a normalization strategy from string or enum value."""
    return resolve_enum_value(value, NormalizationStrategy)


def resolve_pretrain_strategy(value: Union[str, PretrainStrategy]) -> PretrainStrategy:
    """Resolve a pretrain strategy from string or enum value."""
    return resolve_enum_value(value, PretrainStrategy)


def resolve_attention_pattern(value: Union[str, AttentionPattern]) -> AttentionPattern:
    """Resolve an attention pattern from string or enum value."""
    return resolve_enum_value(value, AttentionPattern)


def resolve_modality_type(value: Union[str, ModalityType]) -> ModalityType:
    """Resolve a modality type from string or enum value."""
    return resolve_enum_value(value, ModalityType)


def resolve_projection_type(value: Union[str, ProjectionType]) -> ProjectionType:
    """Resolve a projection type from string or enum value."""
    return resolve_enum_value(value, ProjectionType)


def resolve_activation_type(value: Union[str, ActivationType]) -> ActivationType:
    """Resolve an activation type from string or enum value."""
    return resolve_enum_value(value, ActivationType)


def resolve_position_encoding_type(value: Union[str, PositionEncodingType]) -> PositionEncodingType:
    """Resolve a position encoding type from string or enum value."""
    return resolve_enum_value(value, PositionEncodingType)


def resolve_norm_type(value: Union[str, NormType]) -> NormType:
    """Resolve a norm type from string or enum value."""
    return resolve_enum_value(value, NormType)


# =============================================================================
# Serialization Utilities
# =============================================================================

def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON/dict output.

    Handles dataclasses, enums, lists, tuples, dicts, and primitive types.

    Args:
        value: Any value to serialize.

    Returns:
        JSON-serializable representation of the value.
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        result = {}
        for f in fields(value):
            result[f.name] = _serialize_value(getattr(value, f.name))
        return result
    return str(value)


def _deserialize_value(value: Any, target_type: Any) -> Any:
    """Recursively deserialize a value from dict/JSON input.

    Handles nested dataclasses, enum values, lists, and dicts.

    Args:
        value: Raw value from JSON/dict.
        target_type: Expected type annotation.

    Returns:
        Deserialized value of the appropriate type.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        origin = getattr(target_type, "__origin__", None)
        if origin is dict:
            args = getattr(target_type, "__args__", (str, Any))
            if len(args) >= 2:
                return {
                    _deserialize_value(k, args[0]): _deserialize_value(v, args[1])
                    for k, v in value.items()
                }
        if is_dataclass(target_type) and isinstance(target_type, type):
            return target_type.from_dict(value)
        return value
    if isinstance(value, (list, tuple)):
        args = getattr(target_type, "__args__", ())
        if args:
            return [_deserialize_value(v, args[0]) for v in value]
        return list(value)
    if isinstance(value, str):
        for enum_type in [
            FusionType, GatingType, NormalizationStrategy, PretrainStrategy,
            AttentionPattern, ModalityType, ProjectionType, ActivationType,
            PositionEncodingType, NormType, EncoderArchType,
        ]:
            try:
                return enum_type(value)
            except ValueError:
                continue
    return value


def _deep_merge_dicts(
    base: Dict[str, Any],
    override_dict: Dict[str, Any],
    override: bool = True,
) -> Dict[str, Any]:
    """Deep merge two dictionaries recursively.

    Args:
        base: Base dictionary with default values.
        override_dict: Dictionary with override values.
        override: If True, override_dict values take precedence.

    Returns:
        A new merged dictionary.
    """
    result = copy.deepcopy(base)
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value, override=override)
        elif override or key not in result:
            result[key] = copy.deepcopy(value)
    return result


def _is_positive(value: Any, name: str, allow_zero: bool = False) -> Optional[str]:
    """Validate that a value is positive.

    Args:
        value: Value to check.
        name: Parameter name for error message.
        allow_zero: Whether zero is acceptable. Default: False.

    Returns:
        Error string if invalid, None if valid.
    """
    if allow_zero:
        if value < 0:
            return f"{name} must be non-negative, got {value}"
    else:
        if value <= 0:
            return f"{name} must be positive, got {value}"
    return None


def _is_probability(value: Any, name: str) -> Optional[str]:
    """Validate that a value is a valid probability [0, 1].

    Args:
        value: Value to check.
        name: Parameter name for error message.

    Returns:
        Error string if invalid, None if valid.
    """
    if not (0.0 <= value <= 1.0):
        return f"{name} must be in [0, 1], got {value}"
    return None


def _is_divisible(dividend: int, divisor: int, dividend_name: str, divisor_name: str) -> Optional[str]:
    """Validate that dividend is divisible by divisor.

    Args:
        dividend: Value to check divisibility of.
        divisor: Value to divide by.
        dividend_name: Name for error message.
        divisor_name: Name for error message.

    Returns:
        Error string if invalid, None if valid.
    """
    if dividend % divisor != 0:
        return f"{dividend_name} ({dividend}) must be divisible by {divisor_name} ({divisor})"
    return None


# =============================================================================
# Vision Configuration
# =============================================================================

@dataclass
class VisionConfig:
    """Configuration for vision encoder components.

    Controls image encoding parameters for ViT, CLIP, SigLIP, ConvNeXt,
    and other vision transformer architectures.

    Attributes:
        image_size: Input image resolution (square). Default: 224.
        patch_size: Patch size for patch embedding. Default: 16.
        num_channels: Number of input image channels. Default: 3.
        embed_dim: Dimensionality of patch embeddings. Default: 768.
        num_heads: Number of attention heads in transformer. Default: 12.
        num_layers: Number of transformer encoder layers. Default: 12.
        mlp_ratio: Ratio of MLP hidden dim to embed dim. Default: 4.0.
        dropout: Dropout probability for attention and MLP. Default: 0.0.
        use_flash_attn: Whether to use flash attention. Default: False.
        pretrain_strategy: Pretraining strategy enum. Default: "contrastive".
        encoder_type: Architecture family. Default: "vit".
        hidden_act: Activation function name. Default: "gelu".
        layer_norm_eps: Epsilon for layer normalization. Default: 1e-6.
        attention_dropout: Attention weight dropout. Default: 0.0.
        projection_dropout: Projection layer dropout. Default: 0.0.
        use_gradient_checkpointing: Enable gradient checkpointing. Default: False.
        init_std: Weight initialization std. Default: 0.02.
        use_cls_token: Include CLS token. Default: True.
        use_position_encoding: Use positional embeddings. Default: True.
        position_encoding_type: Position encoding type. Default: "learned".
        norm_type: Normalization layer type. Default: "layer_norm".
        use_pre_norm: Pre-normalization vs post-norm. Default: True.
        intermediate_size: Override MLP hidden size. Default: None (auto-computed).
        num_registers: Register tokens for attention. Default: 0.
        overlap_patch: Overlapping patches. Default: False.
        patch_stride: Patch embedding stride. Default: None (equals patch_size).
        patch_padding: Patch embedding padding. Default: 0.
        output_hidden_states: Return all hidden states. Default: False.
        output_attentions: Return attention weights. Default: False.
    """
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_flash_attn: bool = False
    pretrain_strategy: str = "contrastive"
    encoder_type: str = "vit"
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    projection_dropout: float = 0.0
    use_gradient_checkpointing: bool = False
    init_std: float = 0.02
    use_cls_token: bool = True
    use_position_encoding: bool = True
    position_encoding_type: str = "learned"
    norm_type: str = "layer_norm"
    use_pre_norm: bool = True
    intermediate_size: Optional[int] = None
    num_registers: int = 0
    overlap_patch: bool = False
    patch_stride: Optional[int] = None
    patch_padding: int = 0
    output_hidden_states: bool = False
    output_attentions: bool = False

    def __post_init__(self):
        """Post-initialization validation and computed defaults."""
        if self.patch_stride is None:
            self.patch_stride = self.patch_size
        if self.intermediate_size is None:
            self.intermediate_size = int(self.embed_dim * self.mlp_ratio)

    def validate(self) -> List[str]:
        """Validate all vision configuration parameters.

        Checks:
        - All size/dimension parameters are positive.
        - image_size is divisible by patch_size (unless overlap_patch).
        - embed_dim is divisible by num_heads.
        - Dropout values are in [0, 1].
        - init_std is non-negative.

        Returns:
            List of validation error strings. Empty list if valid.
        """
        errors = []

        err = _is_positive(self.image_size, "image_size")
        if err:
            errors.append(err)

        err = _is_positive(self.patch_size, "patch_size")
        if err:
            errors.append(err)

        if self.image_size % self.patch_size != 0 and not self.overlap_patch:
            errors.append(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size}) unless overlap_patch is True"
            )

        err = _is_positive(self.num_channels, "num_channels")
        if err:
            errors.append(err)

        err = _is_positive(self.embed_dim, "embed_dim")
        if err:
            errors.append(err)

        err = _is_positive(self.num_heads, "num_heads")
        if err:
            errors.append(err)

        err = _is_divisible(self.embed_dim, self.num_heads, "embed_dim", "num_heads")
        if err:
            errors.append(err)

        err = _is_positive(self.num_layers, "num_layers")
        if err:
            errors.append(err)

        err = _is_positive(self.mlp_ratio, "mlp_ratio")
        if err:
            errors.append(err)

        err = _is_probability(self.dropout, "dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.attention_dropout, "attention_dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.projection_dropout, "projection_dropout")
        if err:
            errors.append(err)

        err = _is_positive(self.layer_norm_eps, "layer_norm_eps")
        if err:
            errors.append(err)

        err = _is_positive(self.init_std, "init_std", allow_zero=True)
        if err:
            errors.append(err)

        if self.patch_stride is not None:
            err = _is_positive(self.patch_stride, "patch_stride")
            if err:
                errors.append(err)

        valid_pretrain = [ps.value for ps in PretrainStrategy]
        if self.pretrain_strategy.lower() not in valid_pretrain:
            errors.append(
                f"Unknown pretrain_strategy '{self.pretrain_strategy}'. "
                f"Valid: {valid_pretrain}"
            )

        valid_acts = [at.value for at in ActivationType]
        if self.hidden_act.lower() not in valid_acts:
            errors.append(
                f"Unknown hidden_act '{self.hidden_act}'. Valid: {valid_acts}"
            )

        valid_norms = [nt.value for nt in NormType]
        if self.norm_type.lower() not in valid_norms:
            errors.append(
                f"Unknown norm_type '{self.norm_type}'. Valid: {valid_norms}"
            )

        valid_pos = [pt.value for pt in PositionEncodingType]
        if self.position_encoding_type.lower() not in valid_pos:
            errors.append(
                f"Unknown position_encoding_type '{self.position_encoding_type}'. "
                f"Valid: {valid_pos}"
            )

        valid_enc = [ea.value for ea in EncoderArchType]
        if self.encoder_type.lower() not in valid_enc:
            errors.append(
                f"Unknown encoder_type '{self.encoder_type}'. Valid: {valid_enc}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dictionary.

        Returns:
            Dictionary representation with all fields serialized.
        """
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VisionConfig":
        """Deserialize a VisionConfig from a dictionary.

        Args:
            config_dict: Dictionary of configuration values.

        Returns:
            VisionConfig instance.

        Raises:
            TypeError: If config_dict is not a dict.
        """
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Args:
            indent: JSON indentation level. Default: 2.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "VisionConfig":
        """Deserialize from a JSON string.

        Args:
            json_str: JSON configuration string.

        Returns:
            VisionConfig instance.
        """
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "VisionConfig":
        """Create a deep copy of this configuration.

        Returns:
            Independent copy of the configuration.
        """
        return copy.deepcopy(self)

    def merge_with(self, other: "VisionConfig", override: bool = True) -> "VisionConfig":
        """Merge with another VisionConfig, creating a new combined config.

        Args:
            other: Another VisionConfig to merge.
            override: If True, other's values take precedence.

        Returns:
            New merged VisionConfig instance.
        """
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return VisionConfig.from_dict(merged)

    @property
    def num_patches(self) -> int:
        """Number of spatial patches (grid_size^2).

        Returns:
            Total number of patches.
        """
        grid = self.grid_size
        return grid * grid

    @property
    def grid_size(self) -> int:
        """Grid size (patches per side).

        Returns:
            Number of patches along each spatial dimension.
        """
        return self.image_size // self.patch_size

    @property
    def head_dim(self) -> int:
        """Dimension per attention head.

        Returns:
            Head dimension.
        """
        return self.embed_dim // self.num_heads

    @property
    def sequence_length(self) -> int:
        """Total sequence length including CLS and register tokens.

        Returns:
            Transformer input sequence length.
        """
        seq_len = self.num_patches
        if self.use_cls_token:
            seq_len += 1
        seq_len += self.num_registers
        return seq_len

    @property
    def num_parameters(self) -> int:
        """Estimate the number of parameters in the vision encoder.

        Returns:
            Approximate parameter count.
        """
        patch_embed_params = (self.patch_size ** 2) * self.num_channels * self.embed_dim + self.embed_dim
        pos_embed_params = self.sequence_length * self.embed_dim if self.use_position_encoding else 0
        cls_params = self.embed_dim if self.use_cls_token else 0
        per_layer = (
            4 * self.embed_dim * self.intermediate_size
            + 4 * self.embed_dim ** 2
            + 2 * self.embed_dim
        )
        transformer_params = self.num_layers * per_layer
        return patch_embed_params + pos_embed_params + cls_params + transformer_params

    def compute_flops(self, resolution: Optional[int] = None) -> int:
        """Estimate FLOPs for the vision encoder at a given resolution.

        Args:
            resolution: Override resolution. Default: self.image_size.

        Returns:
            Estimated FLOPs.
        """
        res = resolution or self.image_size
        grid = res // self.patch_size
        seq_len = grid * grid
        if self.use_cls_token:
            seq_len += 1

        patch_flops = (self.patch_size ** 2) * self.num_channels * self.embed_dim * seq_len
        attn_flops_per_layer = 2 * seq_len * self.embed_dim * self.head_dim * self.num_heads + seq_len ** 2 * self.num_heads
        mlp_flops_per_layer = 2 * seq_len * self.embed_dim * self.intermediate_size
        total = patch_flops + self.num_layers * (attn_flops_per_layer + mlp_flops_per_layer)
        return total

    def get_compatible_resolutions(self) -> List[int]:
        """Get list of resolutions compatible with the current patch size.

        Returns:
            Sorted list of valid resolutions.
        """
        return sorted([r for r in range(self.patch_size, 2048, self.patch_size)])

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VisionConfig(image_size={self.image_size}, patch_size={self.patch_size}, "
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, encoder_type='{self.encoder_type}')"
        )


# =============================================================================
# Audio Configuration
# =============================================================================

@dataclass
class AudioConfig:
    """Configuration for audio encoder components.

    Controls audio processing parameters for spectrogram computation,
    feature extraction, and audio transformer architectures.

    Attributes:
        sample_rate: Audio sample rate in Hz. Default: 16000.
        n_fft: FFT window size. Default: 400.
        hop_length: Hop length for STFT. Default: 160.
        n_mels: Number of mel filter banks. Default: 128.
        fmin: Minimum frequency for mel filterbank. Default: 0.0.
        fmax: Maximum frequency for mel filterbank. Default: None (Nyquist).
        encoder_dim: Audio encoder hidden dimension. Default: 768.
        num_heads: Number of attention heads. Default: 12.
        num_layers: Number of transformer layers. Default: 12.
        encoder_type: Encoder architecture. Default: "whisper".
        hidden_act: Activation function. Default: "gelu".
        mlp_ratio: MLP hidden dim ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.1.
        attention_dropout: Attention dropout. Default: 0.1.
        layer_norm_eps: Layer norm epsilon. Default: 1e-5.
        max_audio_length: Maximum audio length in seconds. Default: 30.0.
        num_audio_tokens: Discrete audio token count. Default: 8192.
        use_audio_tokens: Use discrete tokenization. Default: False.
        use_gradient_checkpointing: Enable gradient checkpointing. Default: False.
        init_std: Weight initialization std. Default: 0.02.
        use_conv_stem: Convolutional stem. Default: True.
        use_spec_augment: Apply SpecAugment. Default: True.
        spec_augment_time_mask_param: SpecAugment time mask param. Default: 100.
        spec_augment_freq_mask_param: SpecAugment freq mask param. Default: 80.
        use_mfcc: Extract MFCC features. Default: False.
        num_mfcc: Number of MFCC coefficients. Default: 40.
        output_hidden_states: Return all hidden states. Default: False.
        output_attentions: Return attention weights. Default: False.
    """
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 128
    fmin: float = 0.0
    fmax: Optional[float] = None
    encoder_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    encoder_type: str = "whisper"
    hidden_act: str = "gelu"
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    max_audio_length: float = 30.0
    num_audio_tokens: int = 8192
    use_audio_tokens: bool = False
    use_gradient_checkpointing: bool = False
    init_std: float = 0.02
    use_conv_stem: bool = True
    use_spec_augment: bool = True
    spec_augment_time_mask_param: int = 100
    spec_augment_freq_mask_param: int = 80
    use_mfcc: bool = False
    num_mfcc: int = 40
    output_hidden_states: bool = False
    output_attentions: bool = False

    def __post_init__(self):
        """Post-initialization defaults."""
        if self.fmax is None:
            self.fmax = self.sample_rate / 2.0

    def validate(self) -> List[str]:
        """Validate all audio configuration parameters.

        Returns:
            List of validation error strings.
        """
        errors = []

        err = _is_positive(self.sample_rate, "sample_rate")
        if err:
            errors.append(err)

        err = _is_positive(self.n_fft, "n_fft")
        if err:
            errors.append(err)

        err = _is_positive(self.hop_length, "hop_length")
        if err:
            errors.append(err)

        if self.hop_length > self.n_fft:
            errors.append(
                f"hop_length ({self.hop_length}) should be <= n_fft ({self.n_fft})"
            )

        err = _is_positive(self.n_mels, "n_mels")
        if err:
            errors.append(err)

        err = _is_positive(self.fmin, "fmin", allow_zero=True)
        if err:
            errors.append(err)

        if self.fmax is not None and self.fmax <= self.fmin:
            errors.append(
                f"fmax ({self.fmax}) must be > fmin ({self.fmin})"
            )

        err = _is_positive(self.encoder_dim, "encoder_dim")
        if err:
            errors.append(err)

        err = _is_positive(self.num_heads, "num_heads")
        if err:
            errors.append(err)

        err = _is_divisible(self.encoder_dim, self.num_heads, "encoder_dim", "num_heads")
        if err:
            errors.append(err)

        err = _is_positive(self.num_layers, "num_layers")
        if err:
            errors.append(err)

        err = _is_probability(self.dropout, "dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.attention_dropout, "attention_dropout")
        if err:
            errors.append(err)

        err = _is_positive(self.max_audio_length, "max_audio_length")
        if err:
            errors.append(err)

        err = _is_positive(self.num_audio_tokens, "num_audio_tokens")
        if err:
            errors.append(err)

        err = _is_positive(self.init_std, "init_std", allow_zero=True)
        if err:
            errors.append(err)

        valid_enc = [ea.value for ea in EncoderArchType]
        if self.encoder_type.lower() not in valid_enc:
            errors.append(
                f"Unknown encoder_type '{self.encoder_type}'. Valid: {valid_enc}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioConfig":
        """Deserialize AudioConfig from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AudioConfig":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "AudioConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "AudioConfig", override: bool = True) -> "AudioConfig":
        """Merge with another AudioConfig."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return AudioConfig.from_dict(merged)

    @property
    def max_samples(self) -> int:
        """Maximum audio samples.

        Returns:
            max_audio_length * sample_rate.
        """
        return int(self.sample_rate * self.max_audio_length)

    @property
    def num_spectral_frames(self) -> int:
        """Number of mel spectrogram time frames for max audio length.

        Returns:
            Approximate frame count.
        """
        return self.max_samples // self.hop_length + 1

    @property
    def frequency_bins(self) -> int:
        """Number of frequency bins.

        Returns:
            n_fft // 2 + 1.
        """
        return self.n_fft // 2 + 1

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.encoder_dim // self.num_heads

    @property
    def intermediate_size(self) -> int:
        """MLP intermediate dimension."""
        return int(self.encoder_dim * self.mlp_ratio)

    @property
    def bandwidth(self) -> float:
        """Mel filterbank bandwidth in Hz.

        Returns:
            fmax - fmin.
        """
        return (self.fmax or self.sample_rate / 2.0) - self.fmin

    @property
    def mel_spacing(self) -> float:
        """Average mel filter spacing.

        Returns:
            Bandwidth divided by n_mels.
        """
        return self.bandwidth / max(self.n_mels, 1)

    def estimate_spectrogram_shape(self) -> Tuple[int, int]:
        """Estimate the mel spectrogram shape for max audio.

        Returns:
            (n_mels, num_spectral_frames).
        """
        return (self.n_mels, self.num_spectral_frames)

    @property
    def num_parameters(self) -> int:
        """Estimate vision encoder parameter count."""
        per_layer = (
            4 * self.encoder_dim * self.intermediate_size
            + 4 * self.encoder_dim ** 2
            + 2 * self.encoder_dim
        )
        return self.num_layers * per_layer

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AudioConfig(sample_rate={self.sample_rate}, n_fft={self.n_fft}, "
            f"n_mels={self.n_mels}, encoder_dim={self.encoder_dim}, "
            f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"encoder_type='{self.encoder_type}')"
        )


# =============================================================================
# Video Configuration
# =============================================================================

@dataclass
class VideoConfig:
    """Configuration for video encoder components.

    Controls video processing parameters for temporal and spatial encoding,
    including frame sampling, 3D patch embedding, and video transformer variants.

    Attributes:
        frame_rate: Target frame rate for processing. Default: 30.
        num_frames: Number of frames to sample per video. Default: 8.
        image_size: Spatial resolution per frame. Default: 224.
        patch_size: Spatial patch size. Default: 16.
        temporal_patch_size: Temporal patch size. Default: 2.
        embed_dim: Video token embedding dimension. Default: 768.
        num_heads: Number of attention heads. Default: 12.
        num_layers: Number of transformer layers. Default: 12.
        mlp_ratio: MLP hidden dim ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.0.
        attention_dropout: Attention dropout. Default: 0.0.
        layer_norm_eps: Layer norm epsilon. Default: 1e-6.
        use_spatial_temporal_attn: Joint spatiotemporal attention. Default: True.
        attention_pattern: Video attention pattern. Default: "full".
        use_gradient_checkpointing: Gradient checkpointing. Default: False.
        init_std: Weight init std. Default: 0.02.
        use_cls_token: CLS token. Default: True.
        use_tubelet_embedding: 3D tubelet embeddings. Default: True.
        tubelet_size: Tubelet dimensions (t, h, w). Default: (2, 16, 16).
        use_sep_spatial_temporal: Separate spatial/temporal attention. Default: False.
        encoder_type: Video encoder architecture. Default: "timesformer".
        hidden_act: Activation function. Default: "gelu".
        max_frames: Maximum frames per video. Default: 100.
        frame_sampling_strategy: Frame sampling method. Default: "uniform".
        use_temporal_position_encoding: Temporal position embeds. Default: True.
        use_spatial_position_encoding: Spatial position embeds. Default: True.
        temporal_embed_dim: Override temporal embed dim. Default: None (equals embed_dim).
        output_hidden_states: Return all hidden states. Default: False.
        output_attentions: Return attention weights. Default: False.
    """
    frame_rate: int = 30
    num_frames: int = 8
    image_size: int = 224
    patch_size: int = 16
    temporal_patch_size: int = 2
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    use_spatial_temporal_attn: bool = True
    attention_pattern: str = "full"
    use_gradient_checkpointing: bool = False
    init_std: float = 0.02
    use_cls_token: bool = True
    use_tubelet_embedding: bool = True
    tubelet_size: Tuple[int, int, int] = (2, 16, 16)
    use_sep_spatial_temporal: bool = False
    encoder_type: str = "timesformer"
    hidden_act: str = "gelu"
    max_frames: int = 100
    frame_sampling_strategy: str = "uniform"
    use_temporal_position_encoding: bool = True
    use_spatial_position_encoding: bool = True
    temporal_embed_dim: Optional[int] = None
    output_hidden_states: bool = False
    output_attentions: bool = False

    def __post_init__(self):
        """Post-initialization setup."""
        if self.temporal_embed_dim is None:
            self.temporal_embed_dim = self.embed_dim
        if isinstance(self.tubelet_size, (list, tuple)):
            self.tubelet_size = tuple(self.tubelet_size)

    def validate(self) -> List[str]:
        """Validate video configuration parameters.

        Returns:
            List of validation error strings.
        """
        errors = []

        err = _is_positive(self.frame_rate, "frame_rate")
        if err:
            errors.append(err)

        err = _is_positive(self.num_frames, "num_frames")
        if err:
            errors.append(err)

        err = _is_positive(self.image_size, "image_size")
        if err:
            errors.append(err)

        err = _is_positive(self.patch_size, "patch_size")
        if err:
            errors.append(err)

        err = _is_positive(self.temporal_patch_size, "temporal_patch_size")
        if err:
            errors.append(err)

        if self.image_size % self.patch_size != 0:
            errors.append(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )

        err = _is_positive(self.embed_dim, "embed_dim")
        if err:
            errors.append(err)

        err = _is_positive(self.num_heads, "num_heads")
        if err:
            errors.append(err)

        err = _is_divisible(self.embed_dim, self.num_heads, "embed_dim", "num_heads")
        if err:
            errors.append(err)

        err = _is_positive(self.num_layers, "num_layers")
        if err:
            errors.append(err)

        err = _is_probability(self.dropout, "dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.attention_dropout, "attention_dropout")
        if err:
            errors.append(err)

        if self.max_frames < self.num_frames:
            errors.append(
                f"max_frames ({self.max_frames}) must be >= "
                f"num_frames ({self.num_frames})"
            )

        if self.use_tubelet_embedding and len(self.tubelet_size) != 3:
            errors.append(
                f"tubelet_size must have 3 elements, got {len(self.tubelet_size)}"
            )
        if self.use_tubelet_embedding:
            for i, ts in enumerate(self.tubelet_size):
                err = _is_positive(ts, f"tubelet_size[{i}]")
                if err:
                    errors.append(err)

        valid_patterns = [ap.value for ap in AttentionPattern]
        if self.attention_pattern.lower() not in valid_patterns:
            errors.append(
                f"Unknown attention_pattern '{self.attention_pattern}'. "
                f"Valid: {valid_patterns}"
            )

        valid_enc = [ea.value for ea in EncoderArchType]
        if self.encoder_type.lower() not in valid_enc:
            errors.append(
                f"Unknown encoder_type '{self.encoder_type}'. Valid: {valid_enc}"
            )

        if self.frame_sampling_strategy.lower() not in ("uniform", "random", "consecutive", "keyframe"):
            errors.append(
                f"Unknown frame_sampling_strategy '{self.frame_sampling_strategy}'. "
                f"Valid: uniform, random, consecutive, keyframe"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VideoConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "VideoConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "VideoConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "VideoConfig", override: bool = True) -> "VideoConfig":
        """Merge with another VideoConfig."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return VideoConfig.from_dict(merged)

    @property
    def num_spatial_patches(self) -> int:
        """Spatial patches per frame.

        Returns:
            grid_size^2.
        """
        return self.spatial_grid_size ** 2

    @property
    def num_temporal_patches(self) -> int:
        """Temporal patch count.

        Returns:
            Number of temporal segments.
        """
        if self.use_tubelet_embedding:
            return self.num_frames // self.tubelet_size[0]
        return self.num_frames // self.temporal_patch_size

    @property
    def total_patches(self) -> int:
        """Total spatiotemporal patches.

        Returns:
            num_spatial_patches * num_temporal_patches.
        """
        return self.num_spatial_patches * self.num_temporal_patches

    @property
    def sequence_length(self) -> int:
        """Total sequence length including CLS token.

        Returns:
            Transformer input sequence length.
        """
        seq_len = self.total_patches
        if self.use_cls_token:
            seq_len += 1
        return seq_len

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.embed_dim // self.num_heads

    @property
    def intermediate_size(self) -> int:
        """MLP intermediate dimension."""
        return int(self.embed_dim * self.mlp_ratio)

    @property
    def temporal_grid_size(self) -> int:
        """Temporal grid size.

        Returns:
            Number of temporal patches.
        """
        if self.use_tubelet_embedding:
            return self.num_frames // self.tubelet_size[0]
        return self.num_frames // self.temporal_patch_size

    @property
    def spatial_grid_size(self) -> int:
        """Spatial grid size.

        Returns:
            Patches per spatial dimension.
        """
        return self.image_size // self.patch_size

    @property
    def video_duration(self) -> float:
        """Video duration at target frame rate.

        Returns:
            num_frames / frame_rate in seconds.
        """
        return self.num_frames / max(self.frame_rate, 1)

    @property
    def max_duration(self) -> float:
        """Maximum video duration.

        Returns:
            max_frames / frame_rate in seconds.
        """
        return self.max_frames / max(self.frame_rate, 1)

    def estimate_tensor_shape(self) -> Tuple[int, int, int, int, int]:
        """Estimate the input video tensor shape.

        Returns:
            (num_frames, num_channels, image_size, image_size, embed_dim).
        """
        return (self.num_frames, 3, self.image_size, self.image_size, self.embed_dim)

    @property
    def num_parameters(self) -> int:
        """Estimate parameter count."""
        per_layer = (
            4 * self.embed_dim * self.intermediate_size
            + 4 * self.embed_dim ** 2
            + 2 * self.embed_dim
        )
        return self.num_layers * per_layer

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VideoConfig(frame_rate={self.frame_rate}, num_frames={self.num_frames}, "
            f"image_size={self.image_size}, embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, num_layers={self.num_layers}, "
            f"encoder_type='{self.encoder_type}')"
        )


# =============================================================================
# Cross-Modal Configuration
# =============================================================================

@dataclass
class CrossModalConfig:
    """Configuration for cross-modal fusion components.

    Controls how information from different modalities is combined,
    projected, and integrated into a unified representation.

    Attributes:
        vision_dim: Vision encoder output dimension. Default: 1024.
        audio_dim: Audio encoder output dimension. Default: 768.
        text_dim: Text/LLM hidden dimension. Default: 4096.
        video_dim: Video encoder output dimension. Default: 768.
        fusion_type: Fusion mechanism type. Default: "cross_attention".
        projection_dim: Common projection dimension. Default: 768.
        num_cross_layers: Number of cross-modal layers. Default: 2.
        gating_type: Gating mechanism type. Default: "soft".
        num_queries: Learned queries for Q-Former. Default: 32.
        query_dim: Query dimension. Default: 768.
        cross_attention_heads: Cross-attention head count. Default: 8.
        cross_attention_dropout: Cross-attention dropout. Default: 0.1.
        fusion_dropout: General fusion dropout. Default: 0.1.
        modality_dropout: Modality dropout for training. Default: 0.1.
        use_modality_embedding: Modality type embeddings. Default: True.
        modality_embed_dim: Modality embedding dimension. Default: 768.
        compact_bilinear_dim: Compact bilinear dimension. Default: 16000.
        compact_bilinear_out_dim: Compact bilinear output dim. Default: 4096.
        adaptive_fusion_temperature: Adaptive fusion softmax temp. Default: 1.0.
        residual_connection: Residual connections in fusion. Default: True.
        layer_norm_fusion: Layer norm after fusion. Default: True.
        pre_norm_fusion: Pre-norm fusion. Default: True.
        use_modality_token_type_ids: Modality token type IDs. Default: True.
        output_hidden_states: Return all hidden states. Default: False.
        output_attentions: Return attention weights. Default: False.
    """
    vision_dim: int = 1024
    audio_dim: int = 768
    text_dim: int = 4096
    video_dim: int = 768
    fusion_type: str = "cross_attention"
    projection_dim: int = 768
    num_cross_layers: int = 2
    gating_type: str = "soft"
    num_queries: int = 32
    query_dim: int = 768
    cross_attention_heads: int = 8
    cross_attention_dropout: float = 0.1
    fusion_dropout: float = 0.1
    modality_dropout: float = 0.1
    use_modality_embedding: bool = True
    modality_embed_dim: int = 768
    compact_bilinear_dim: int = 16000
    compact_bilinear_out_dim: int = 4096
    adaptive_fusion_temperature: float = 1.0
    residual_connection: bool = True
    layer_norm_fusion: bool = True
    pre_norm_fusion: bool = True
    use_modality_token_type_ids: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False

    def validate(self) -> List[str]:
        """Validate cross-modal configuration.

        Returns:
            List of validation error strings.
        """
        errors = []

        for dim_name in ("vision_dim", "audio_dim", "text_dim", "video_dim"):
            err = _is_positive(getattr(self, dim_name), dim_name)
            if err:
                errors.append(err)

        err = _is_positive(self.projection_dim, "projection_dim")
        if err:
            errors.append(err)

        err = _is_positive(self.num_cross_layers, "num_cross_layers")
        if err:
            errors.append(err)

        err = _is_positive(self.num_queries, "num_queries")
        if err:
            errors.append(err)

        err = _is_positive(self.query_dim, "query_dim")
        if err:
            errors.append(err)

        err = _is_positive(self.cross_attention_heads, "cross_attention_heads")
        if err:
            errors.append(err)

        err = _is_divisible(
            self.projection_dim, self.cross_attention_heads,
            "projection_dim", "cross_attention_heads"
        )
        if err:
            errors.append(err)

        err = _is_probability(self.cross_attention_dropout, "cross_attention_dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.fusion_dropout, "fusion_dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.modality_dropout, "modality_dropout")
        if err:
            errors.append(err)

        err = _is_positive(self.adaptive_fusion_temperature, "adaptive_fusion_temperature")
        if err:
            errors.append(err)

        valid_fusion = [ft.value for ft in FusionType]
        if self.fusion_type.lower() not in valid_fusion:
            errors.append(
                f"Unknown fusion_type '{self.fusion_type}'. Valid: {valid_fusion}"
            )

        valid_gating = [gt.value for gt in GatingType]
        if self.gating_type.lower() not in valid_gating:
            errors.append(
                f"Unknown gating_type '{self.gating_type}'. Valid: {valid_gating}"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CrossModalConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "CrossModalConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "CrossModalConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "CrossModalConfig", override: bool = True) -> "CrossModalConfig":
        """Merge with another CrossModalConfig."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return CrossModalConfig.from_dict(merged)

    @property
    def cross_attention_head_dim(self) -> int:
        """Cross-attention head dimension."""
        return self.projection_dim // self.cross_attention_heads

    @property
    def num_modalities(self) -> int:
        """Count of modalities with positive dimensions."""
        count = 0
        if self.vision_dim > 0:
            count += 1
        if self.audio_dim > 0:
            count += 1
        if self.text_dim > 0:
            count += 1
        if self.video_dim > 0:
            count += 1
        return count

    def get_modality_dims(self) -> Dict[str, int]:
        """Get mapping of modality names to dimensions.

        Returns:
            Dict of {modality_name: dimension}.
        """
        return {
            "vision": self.vision_dim,
            "audio": self.audio_dim,
            "text": self.text_dim,
            "video": self.video_dim,
        }

    def get_max_dim(self) -> int:
        """Get the maximum modality dimension.

        Returns:
            Largest encoder output dimension.
        """
        return max(self.vision_dim, self.audio_dim, self.text_dim, self.video_dim)

    def get_min_dim(self) -> int:
        """Get the minimum positive modality dimension.

        Returns:
            Smallest encoder output dimension.
        """
        dims = [d for d in [self.vision_dim, self.audio_dim, self.text_dim, self.video_dim] if d > 0]
        return min(dims) if dims else 0

    def needs_projection(self, modality_dim: int) -> bool:
        """Check if a modality needs projection to the common dimension.

        Args:
            modality_dim: The modality's encoder output dimension.

        Returns:
            True if the dimension differs from projection_dim.
        """
        return modality_dim != self.projection_dim

    def get_all_dims(self) -> List[int]:
        """Get list of all modality dimensions.

        Returns:
            [vision_dim, audio_dim, text_dim, video_dim].
        """
        return [self.vision_dim, self.audio_dim, self.text_dim, self.video_dim]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CrossModalConfig(vision_dim={self.vision_dim}, audio_dim={self.audio_dim}, "
            f"text_dim={self.text_dim}, video_dim={self.video_dim}, "
            f"fusion_type='{self.fusion_type}', projection_dim={self.projection_dim}, "
            f"num_cross_layers={self.num_cross_layers})"
        )


# =============================================================================
# Processor Configurations
# =============================================================================

@dataclass
class ImageProcessorConfig:
    """Configuration for image processing pipeline.

    Controls image resizing, cropping, normalization, and augmentation.

    Attributes:
        image_mean: Normalization mean (RGB). Default: ImageNet.
        image_std: Normalization std (RGB). Default: ImageNet.
        rescale_factor: Pixel value rescale factor. Default: 1/255.
        do_resize: Whether to resize. Default: True.
        size: Target image size. Default: (224, 224).
        do_center_crop: Center crop. Default: True.
        crop_size: Crop size. Default: (224, 224).
        do_normalize: Normalize pixel values. Default: True.
        do_rescale: Rescale to [0, 1]. Default: True.
        do_pad: Pad to square. Default: False.
        pad_value: Padding value. Default: 0.
        do_convert_rgb: Convert to RGB. Default: True.
    """
    image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    rescale_factor: float = 1.0 / 255.0
    do_resize: bool = True
    size: Tuple[int, int] = (224, 224)
    do_center_crop: bool = True
    crop_size: Tuple[int, int] = (224, 224)
    do_normalize: bool = True
    do_rescale: bool = True
    do_pad: bool = False
    pad_value: int = 0
    do_convert_rgb: bool = True

    def validate(self) -> List[str]:
        """Validate image processor config."""
        errors = []
        if len(self.image_mean) != 3:
            errors.append(f"image_mean must have 3 elements, got {len(self.image_mean)}")
        if len(self.image_std) != 3:
            errors.append(f"image_std must have 3 elements, got {len(self.image_std)}")
        for i, s in enumerate(self.image_std):
            if s <= 0:
                errors.append(f"image_std[{i}] must be positive, got {s}")
        err = _is_positive(self.rescale_factor, "rescale_factor")
        if err:
            errors.append(err)
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ImageProcessorConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ImageProcessorConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "ImageProcessorConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "ImageProcessorConfig", override: bool = True) -> "ImageProcessorConfig":
        """Merge with another config."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return ImageProcessorConfig.from_dict(merged)


@dataclass
class AudioProcessorConfig:
    """Configuration for audio processing pipeline.

    Attributes:
        sampling_rate: Target sampling rate. Default: 16000.
        do_resample: Whether to resample. Default: True.
        do_normalize: Normalize waveform. Default: True.
        normalization_strategy: Normalization approach. Default: "peak".
        max_length: Maximum audio length in samples. Default: 480000.
        truncation: Truncate to max length. Default: True.
        padding: Padding strategy. Default: "longest".
        pad_to_multiple_of: Pad to multiple of. Default: None.
        return_attention_mask: Return attention mask. Default: True.
    """
    sampling_rate: int = 16000
    do_resample: bool = True
    do_normalize: bool = True
    normalization_strategy: str = "peak"
    max_length: int = 480000
    truncation: bool = True
    padding: str = "longest"
    pad_to_multiple_of: Optional[int] = None
    return_attention_mask: bool = True

    def validate(self) -> List[str]:
        """Validate audio processor config."""
        errors = []
        err = _is_positive(self.sampling_rate, "sampling_rate")
        if err:
            errors.append(err)
        err = _is_positive(self.max_length, "max_length")
        if err:
            errors.append(err)
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AudioProcessorConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AudioProcessorConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "AudioProcessorConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "AudioProcessorConfig", override: bool = True) -> "AudioProcessorConfig":
        """Merge with another config."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return AudioProcessorConfig.from_dict(merged)


@dataclass
class VideoProcessorConfig:
    """Configuration for video processing pipeline.

    Attributes:
        num_frames: Number of frames to extract. Default: 8.
        frame_rate: Target frame rate. Default: 30.
        do_resize: Resize frames. Default: True.
        size: Target frame size. Default: (224, 224).
        do_center_crop: Center crop frames. Default: True.
        crop_size: Crop size. Default: (224, 224).
        do_normalize: Normalize frames. Default: True.
        image_mean: Normalization mean. Default: ImageNet.
        image_std: Normalization std. Default: ImageNet.
        do_rescale: Rescale pixel values. Default: True.
        rescale_factor: Rescale factor. Default: 1/255.
        sampling_strategy: Frame sampling strategy. Default: "uniform".
    """
    num_frames: int = 8
    frame_rate: int = 30
    do_resize: bool = True
    size: Tuple[int, int] = (224, 224)
    do_center_crop: bool = True
    crop_size: Tuple[int, int] = (224, 224)
    do_normalize: bool = True
    image_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    do_rescale: bool = True
    rescale_factor: float = 1.0 / 255.0
    sampling_strategy: str = "uniform"

    def validate(self) -> List[str]:
        """Validate video processor config."""
        errors = []
        err = _is_positive(self.num_frames, "num_frames")
        if err:
            errors.append(err)
        err = _is_positive(self.frame_rate, "frame_rate")
        if err:
            errors.append(err)
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VideoProcessorConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "VideoProcessorConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "VideoProcessorConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "VideoProcessorConfig", override: bool = True) -> "VideoProcessorConfig":
        """Merge with another config."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return VideoProcessorConfig.from_dict(merged)


# =============================================================================
# Processor Config
# =============================================================================

@dataclass
class ProcessorConfig:
    """Combined configuration for all modality processors.

    Groups image, audio, and video processor configurations with global
    normalization and data loading settings.

    Attributes:
        image_processor: Image processing config. Default: ImageProcessorConfig().
        audio_processor: Audio processing config. Default: AudioProcessorConfig().
        video_processor: Video processing config. Default: VideoProcessorConfig().
        normalization_strategy: Global normalization strategy. Default: "imagenet".
        mixed_precision: Mixed precision processing. Default: False.
        device: Target device. Default: "cpu".
        batch_size: Processing batch size. Default: 1.
        num_workers: Data loading workers. Default: 4.
        pin_memory: Pin memory. Default: False.
        prefetch_factor: Prefetch factor. Default: 2.
        persistent_workers: Persistent workers. Default: False.
    """
    image_processor: ImageProcessorConfig = field(default_factory=ImageProcessorConfig)
    audio_processor: AudioProcessorConfig = field(default_factory=AudioProcessorConfig)
    video_processor: VideoProcessorConfig = field(default_factory=VideoProcessorConfig)
    normalization_strategy: str = "imagenet"
    mixed_precision: bool = False
    device: str = "cpu"
    batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = False

    def validate(self) -> List[str]:
        """Validate all processor sub-configurations.

        Returns:
            List of validation error strings.
        """
        errors = []

        img_errors = self.image_processor.validate()
        errors.extend([f"image_processor.{e}" for e in img_errors])

        aud_errors = self.audio_processor.validate()
        errors.extend([f"audio_processor.{e}" for e in aud_errors])

        vid_errors = self.video_processor.validate()
        errors.extend([f"video_processor.{e}" for e in vid_errors])

        valid_strategies = [ns.value for ns in NormalizationStrategy]
        if self.normalization_strategy.lower() not in valid_strategies:
            errors.append(
                f"Unknown normalization_strategy '{self.normalization_strategy}'. "
                f"Valid: {valid_strategies}"
            )

        err = _is_positive(self.batch_size, "batch_size")
        if err:
            errors.append(err)

        err = _is_positive(self.num_workers, "num_workers", allow_zero=True)
        if err:
            errors.append(err)

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ProcessorConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ProcessorConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "ProcessorConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "ProcessorConfig", override: bool = True) -> "ProcessorConfig":
        """Merge with another ProcessorConfig."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return ProcessorConfig.from_dict(merged)

    def get_dataloader_config(self) -> Dict[str, Any]:
        """Get DataLoader-compatible configuration dict.

        Returns:
            Dictionary of kwargs for DataLoader constructor.
        """
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor if self.num_workers > 0 else None,
            "persistent_workers": self.persistent_workers if self.num_workers > 0 else False,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProcessorConfig(normalization_strategy='{self.normalization_strategy}', "
            f"batch_size={self.batch_size}, device='{self.device}')"
        )


# =============================================================================
# Multimodal Configuration
# =============================================================================

@dataclass
class MultimodalConfig:
    """Master configuration for the entire multimodal system.

    Combines all sub-configurations and provides global settings for
    the multimodal LLM pipeline. This is the primary configuration
    that users interact with.

    Attributes:
        vision: Vision encoder configuration.
        audio: Audio encoder configuration.
        video: Video encoder configuration.
        cross_modal: Cross-modal fusion configuration.
        supported_modalities: Enabled modality names.
        default_modality: Default fallback modality.
        max_tokens_per_modality: Max tokens per modality.
        modality_embeddings_dim: Modality type embedding dim. Default: 768.
        modality_fusion_strategy: Global fusion strategy. Default: "cross_attention".
        projector_type: Projector type. Default: "mlp".
        use_modality_embeddings: Add modality type embeddings. Default: True.
        use_modality_dropout: Modality dropout. Default: True.
        modality_dropout_prob: Dropout probability. Default: 0.1.
        enable_vision: Enable vision. Default: True.
        enable_audio: Enable audio. Default: False.
        enable_video: Enable video. Default: False.
        enable_text: Enable text. Default: True.
        llm_hidden_size: LLM hidden dimension. Default: 4096.
        llm_num_heads: LLM attention heads. Default: 32.
        llm_num_layers: LLM layers. Default: 32.
        llm_vocab_size: LLM vocabulary size. Default: 32000.
        llm_max_position_embeddings: Max position embeddings. Default: 4096.
        gradient_checkpointing: Global gradient checkpointing. Default: False.
        use_flash_attention: Global flash attention. Default: False.
        mixed_precision_dtype: Mixed precision dtype. Default: "bf16".
        torch_dtype: Default torch dtype. Default: "float32".
        init_std: Global init std. Default: 0.02.
        dropout: Global dropout. Default: 0.1.
        attention_dropout: Global attention dropout. Default: 0.0.
        classifier_dropout: Classifier head dropout. Default: None.
    """
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    cross_modal: CrossModalConfig = field(default_factory=CrossModalConfig)
    supported_modalities: List[str] = field(
        default_factory=lambda: ["vision", "text"]
    )
    default_modality: str = "text"
    max_tokens_per_modality: Dict[str, int] = field(
        default_factory=lambda: {
            "vision": 576,
            "audio": 1500,
            "text": 2048,
            "video": 1152,
        }
    )
    modality_embeddings_dim: int = 768
    modality_fusion_strategy: str = "cross_attention"
    projector_type: str = "mlp"
    use_modality_embeddings: bool = True
    use_modality_dropout: bool = True
    modality_dropout_prob: float = 0.1
    enable_vision: bool = True
    enable_audio: bool = False
    enable_video: bool = False
    enable_text: bool = True
    llm_hidden_size: int = 4096
    llm_num_heads: int = 32
    llm_num_layers: int = 32
    llm_vocab_size: int = 32000
    llm_max_position_embeddings: int = 4096
    gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    mixed_precision_dtype: str = "bf16"
    torch_dtype: str = "float32"
    init_std: float = 0.02
    dropout: float = 0.1
    attention_dropout: float = 0.0
    classifier_dropout: Optional[float] = None

    def __post_init__(self):
        """Propagate global settings to sub-configs."""
        if self.gradient_checkpointing:
            self.vision.use_gradient_checkpointing = True
            self.audio.use_gradient_checkpointing = True
            self.video.use_gradient_checkpointing = True
        if self.use_flash_attention:
            self.vision.use_flash_attn = True

        for mod, flag in [("vision", self.enable_vision), ("audio", self.enable_audio),
                          ("video", self.enable_video), ("text", self.enable_text)]:
            if flag and mod not in self.supported_modalities:
                self.supported_modalities.append(mod)

    def validate(self) -> List[str]:
        """Validate all sub-configurations and cross-component compatibility.

        Performs deep validation of:
        - Each sub-configuration (vision, audio, video, cross_modal).
        - Modality enable flags vs supported_modalities.
        - Cross-component dimension compatibility.
        - LLM architecture constraints.
        - Dropout range constraints.

        Returns:
            List of validation error strings. Empty if valid.
        """
        errors = []

        for sub_name in ("vision", "audio", "video", "cross_modal"):
            sub_config = getattr(self, sub_name)
            sub_errors = sub_config.validate()
            errors.extend([f"{sub_name}.{e}" for e in sub_errors])

        valid_modalities = [m.value for m in ModalityType]
        for mod in self.supported_modalities:
            if mod.lower() not in valid_modalities:
                errors.append(
                    f"Unknown modality '{mod}' in supported_modalities. "
                    f"Valid: {valid_modalities}"
                )

        if self.default_modality.lower() not in valid_modalities:
            errors.append(
                f"Unknown default_modality '{self.default_modality}'. "
                f"Valid: {valid_modalities}"
            )

        if self.default_modality.lower() not in [m.lower() for m in self.supported_modalities]:
            errors.append(
                f"default_modality '{self.default_modality}' must be in "
                f"supported_modalities {self.supported_modalities}"
            )

        for mod_name, max_tokens in self.max_tokens_per_modality.items():
            err = _is_positive(max_tokens, f"max_tokens_per_modality['{mod_name}']")
            if err:
                errors.append(err)

        err = _is_positive(self.llm_hidden_size, "llm_hidden_size")
        if err:
            errors.append(err)

        err = _is_positive(self.llm_num_heads, "llm_num_heads")
        if err:
            errors.append(err)

        err = _is_divisible(self.llm_hidden_size, self.llm_num_heads, "llm_hidden_size", "llm_num_heads")
        if err:
            errors.append(err)

        err = _is_positive(self.llm_num_layers, "llm_num_layers")
        if err:
            errors.append(err)

        err = _is_positive(self.llm_vocab_size, "llm_vocab_size")
        if err:
            errors.append(err)

        err = _is_probability(self.dropout, "dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.attention_dropout, "attention_dropout")
        if err:
            errors.append(err)

        err = _is_probability(self.modality_dropout_prob, "modality_dropout_prob")
        if err:
            errors.append(err)

        if self.classifier_dropout is not None:
            err = _is_probability(self.classifier_dropout, "classifier_dropout")
            if err:
                errors.append(err)

        if self.enable_vision and self.cross_modal.vision_dim != self.vision.embed_dim:
            errors.append(
                f"cross_modal.vision_dim ({self.cross_modal.vision_dim}) "
                f"should match vision.embed_dim ({self.vision.embed_dim})"
            )

        if self.enable_audio and self.cross_modal.audio_dim != self.audio.encoder_dim:
            errors.append(
                f"cross_modal.audio_dim ({self.cross_modal.audio_dim}) "
                f"should match audio.encoder_dim ({self.audio.encoder_dim})"
            )

        valid_fusion = [ft.value for ft in FusionType]
        if self.modality_fusion_strategy.lower() not in valid_fusion:
            errors.append(
                f"Unknown modality_fusion_strategy '{self.modality_fusion_strategy}'. "
                f"Valid: {valid_fusion}"
            )

        valid_proj = [pt.value for pt in ProjectionType]
        if self.projector_type.lower() not in valid_proj:
            errors.append(
                f"Unknown projector_type '{self.projector_type}'. "
                f"Valid: {valid_proj}"
            )

        if self.mixed_precision_dtype not in ("bf16", "fp16", "fp32", "none"):
            errors.append(
                f"Unknown mixed_precision_dtype '{self.mixed_precision_dtype}'. "
                f"Valid: bf16, fp16, fp32, none"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {}
        for f in fields(self):
            result[f.name] = _serialize_value(getattr(self, f.name))
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MultimodalConfig":
        """Deserialize from dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(f"Expected dict, got {type(config_dict).__name__}")
        kwargs = {}
        for f in fields(cls):
            if f.name in config_dict:
                kwargs[f.name] = _deserialize_value(config_dict[f.name], f.type)
        return cls(**kwargs)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "MultimodalConfig":
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def deep_copy(self) -> "MultimodalConfig":
        """Deep copy."""
        return copy.deepcopy(self)

    def merge_with(self, other: "MultimodalConfig", override: bool = True) -> "MultimodalConfig":
        """Merge with another MultimodalConfig."""
        merged = _deep_merge_dicts(self.to_dict(), other.to_dict(), override=override)
        return MultimodalConfig.from_dict(merged)

    def get_enabled_modalities(self) -> List[str]:
        """Get list of enabled modality names.

        Returns:
            List of enabled modality strings.
        """
        modalities = []
        if self.enable_vision:
            modalities.append("vision")
        if self.enable_audio:
            modalities.append("audio")
        if self.enable_video:
            modalities.append("video")
        if self.enable_text:
            modalities.append("text")
        return modalities

    def get_modality_config(self, modality: str) -> Any:
        """Get sub-config for a specific modality.

        Args:
            modality: Modality name ("vision", "audio", "video", "text").

        Returns:
            Corresponding configuration object or None.

        Raises:
            ValueError: If modality is not recognized.
        """
        modality_map = {
            "vision": self.vision,
            "audio": self.audio,
            "video": self.video,
            "text": None,
        }
        if modality.lower() not in modality_map:
            raise ValueError(
                f"Unknown modality '{modality}'. Valid: {list(modality_map.keys())}"
            )
        return modality_map[modality.lower()]

    def get_total_sequence_length(self) -> int:
        """Estimate total sequence length across enabled modalities.

        Returns:
            Sum of max_tokens_per_modality for enabled modalities.
        """
        total = 0
        for modality in self.get_enabled_modalities():
            total += self.max_tokens_per_modality.get(modality, 0)
        return total

    @property
    def llm_head_dim(self) -> int:
        """LLM dimension per attention head."""
        return self.llm_hidden_size // self.llm_num_heads

    @property
    def num_enabled_modalities(self) -> int:
        """Count of enabled modalities."""
        return len(self.get_enabled_modalities())

    @property
    def is_multimodal(self) -> bool:
        """Check if more than one modality is enabled.

        Returns:
            True if 2+ modalities are enabled.
        """
        return self.num_enabled_modalities >= 2

    def get_torch_dtype(self) -> torch.dtype:
        """Get the torch dtype corresponding to torch_dtype setting.

        Returns:
            torch.dtype value.
        """
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype.lower(), torch.float32)

    def get_mixed_precision_dtype(self) -> Optional[torch.dtype]:
        """Get the mixed precision dtype.

        Returns:
            torch.dtype or None if mixed precision is disabled.
        """
        if self.mixed_precision_dtype == "none":
            return None
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        return dtype_map.get(self.mixed_precision_dtype.lower())

    def estimate_total_parameters(self) -> int:
        """Estimate total model parameters across all components.

        Returns:
            Approximate total parameter count.
        """
        total = 0
        if self.enable_vision:
            total += self.vision.num_parameters
        if self.enable_audio:
            total += self.audio.num_parameters
        if self.enable_video:
            total += self.video.num_parameters

        llm_per_layer = (
            4 * self.llm_hidden_size * int(self.llm_hidden_size * 4)
            + 4 * self.llm_hidden_size ** 2
        )
        total += self.llm_num_layers * llm_per_layer
        total += self.llm_vocab_size * self.llm_hidden_size

        return total

    @classmethod
    def get_default_configs(cls) -> Dict[str, "MultimodalConfig"]:
        """Get predefined configuration presets.

        Returns:
            Dict of preset name -> MultimodalConfig.
        """
        small = cls(
            vision=VisionConfig(
                image_size=224, patch_size=16, embed_dim=384,
                num_heads=6, num_layers=6,
            ),
            audio=AudioConfig(
                encoder_dim=384, num_heads=6, num_layers=6,
            ),
            video=VideoConfig(
                image_size=224, embed_dim=384, num_heads=6, num_layers=6,
                num_frames=4,
            ),
            cross_modal=CrossModalConfig(
                vision_dim=384, audio_dim=384, text_dim=1024,
                projection_dim=384, num_cross_layers=1,
                cross_attention_heads=6,
            ),
            llm_hidden_size=1024, llm_num_heads=8, llm_num_layers=12,
        )

        base = cls(
            vision=VisionConfig(
                image_size=224, patch_size=14, embed_dim=1024,
                num_heads=16, num_layers=24,
            ),
            audio=AudioConfig(
                encoder_dim=768, num_heads=12, num_layers=12,
            ),
            video=VideoConfig(
                image_size=224, embed_dim=768, num_heads=12, num_layers=12,
            ),
            cross_modal=CrossModalConfig(
                vision_dim=1024, audio_dim=768, text_dim=4096,
                projection_dim=768, num_cross_layers=2,
            ),
            llm_hidden_size=4096, llm_num_heads=32, llm_num_layers=32,
        )

        large = cls(
            vision=VisionConfig(
                image_size=336, patch_size=14, embed_dim=1280,
                num_heads=16, num_layers=32,
            ),
            audio=AudioConfig(
                encoder_dim=1024, num_heads=16, num_layers=24,
            ),
            video=VideoConfig(
                image_size=336, embed_dim=1024, num_heads=16, num_layers=24,
                num_frames=16,
            ),
            cross_modal=CrossModalConfig(
                vision_dim=1280, audio_dim=1024, text_dim=6144,
                projection_dim=1024, num_cross_layers=4,
            ),
            llm_hidden_size=6144, llm_num_heads=48, llm_num_layers=40,
            llm_vocab_size=64000,
            supported_modalities=["vision", "audio", "text"],
        )

        return {"small": small, "base": base, "large": large}

    def save_pretrained(self, save_directory: str) -> str:
        """Save configuration to a directory.

        Args:
            save_directory: Directory to save into.

        Returns:
            Path to the saved config file.
        """
        os.makedirs(save_directory, exist_ok=True)
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=2))
        return config_path

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "MultimodalConfig":
        """Load configuration from a pretrained model directory.

        Args:
            model_name_or_path: Path containing config.json.

        Returns:
            MultimodalConfig instance.

        Raises:
            FileNotFoundError: If config file not found.
        """
        config_path = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_name_or_path, "multimodal_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found in {model_name_or_path}"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __repr__(self) -> str:
        """String representation."""
        enabled = self.get_enabled_modalities()
        return (
            f"MultimodalConfig("
            f"modalities={enabled}, "
            f"llm_hidden_size={self.llm_hidden_size}, "
            f"llm_num_layers={self.llm_num_layers}, "
            f"projector_type='{self.projector_type}')"
        )
