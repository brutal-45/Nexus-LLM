"""
Optimization Configuration Module
==================================

Comprehensive configuration dataclasses for all optimization techniques
supported by the Nexus LLM optimization toolkit. Each config class provides
validated, serializable, and well-documented settings.

Classes:
    QuantizationConfig: Settings for model quantization (GPTQ, AWQ, BitsAndBytes, FP8, etc.)
    PruningConfig: Settings for model pruning (magnitude, structured, SparseGPT, Wanda, etc.)
    DistillationConfig: Settings for knowledge distillation.
    NASConfig: Settings for neural architecture search.
    CompilationConfig: Settings for model compilation.
    CompressionConfig: Settings for model compression.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class QuantizationMethod(str, Enum):
    """Supported quantization methods."""
    GPTQ = "gptq"
    AWQ = "awq"
    BITSANDBYTES = "bitsandbytes"
    LLM_INT8 = "llm.int8"
    LLM_INT4 = "llm.int4"
    NF4 = "nf4"
    FP8 = "fp8"
    SMOOTHQUANT = "smoothquant"
    RTN = "rtn"
    AQQ = "aqq"

    @classmethod
    def from_string(cls, value: str) -> QuantizationMethod:
        """Parse a quantization method from a string."""
        value_clean = value.strip().lower().replace("-", "_").replace(".", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown quantization method '{value}'. Valid methods: {valid}"
        )


class PruningMethod(str, Enum):
    """Supported pruning methods."""
    MAGNITUDE = "magnitude"
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    LOTA = "lota"
    SPARSE_GPT = "sparsegpt"
    WANDA = "wanda"
    LORA = "lora"
    LOTTERY = "lottery"
    GRADUAL = "gradual"

    @classmethod
    def from_string(cls, value: str) -> PruningMethod:
        """Parse a pruning method from a string."""
        value_clean = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown pruning method '{value}'. Valid methods: {valid}"
        )


class DistillationLossType(str, Enum):
    """Types of distillation losses."""
    HARD = "hard"
    SOFT = "soft"
    FEATURE = "feature"
    ATTENTION = "attention"
    COMBINED = "combined"

    @classmethod
    def from_string(cls, value: str) -> DistillationLossType:
        value_clean = value.strip().lower()
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown distillation loss type '{value}'. Valid types: {valid}"
        )


class CompilationBackend(str, Enum):
    """Supported compilation backends."""
    TORCH_COMPILE = "torch.compile"
    TRITON = "triton"
    INDUCTOR = "inductor"
    CUDAGRAPHS = "cudagraphs"
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    TVM = "tvm"
    OPENVINO = "openvino"

    @classmethod
    def from_string(cls, value: str) -> CompilationBackend:
        value_clean = value.strip().lower().replace("-", "_").replace(".", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown compilation backend '{value}'. Valid backends: {valid}"
        )


class CompilationMode(str, Enum):
    """torch.compile optimization modes."""
    DEFAULT = "default"
    REDUCE_OVERHEAD = "reduce-overhead"
    MAX_AUTOTUNE = "max-autotune"

    @classmethod
    def from_string(cls, value: str) -> CompilationMode:
        value_clean = value.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown compilation mode '{value}'. Valid modes: {valid}"
        )


class SearchAlgorithm(str, Enum):
    """Neural architecture search algorithms."""
    EVOLUTIONARY = "evolutionary"
    DIFFERENTIABLE = "differentiable"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    ONE_SHOT = "one_shot"
    HARDWARE_AWARE = "hardware_aware"
    REINFORCEMENT = "reinforcement"
    REGULARIZED = "regularized"

    @classmethod
    def from_string(cls, value: str) -> SearchAlgorithm:
        value_clean = value.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown search algorithm '{value}'. Valid algorithms: {valid}"
        )


class PruningScope(str, Enum):
    """Scope for pruning operations."""
    LOCAL = "local"
    GLOBAL = "global"
    LAYER_WISE = "layer_wise"
    BLOCK_WISE = "block_wise"

    @classmethod
    def from_string(cls, value: str) -> PruningScope:
        value_clean = value.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown pruning scope '{value}'. Valid scopes: {valid}"
        )


class PruningSchedule(str, Enum):
    """Schedules for gradual pruning."""
    COSINE = "cosine"
    LINEAR = "linear"
    STEP = "step"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"

    @classmethod
    def from_string(cls, value: str) -> PruningSchedule:
        value_clean = value.strip().lower()
        for member in cls:
            if member.value == value_clean:
                return member
        valid = [m.value for m in cls]
        raise ValueError(
            f"Unknown pruning schedule '{value}'. Valid schedules: {valid}"
        )


class FP8Format(str, Enum):
    """FP8 floating-point formats."""
    E4M3 = "e4m3"
    E5M2 = "e5m2"

    @classmethod
    def from_string(cls, value: str) -> FP8Format:
        value_clean = value.strip().lower()
        for member in cls:
            if member.value == value_clean:
                return member
        raise ValueError(
            f"Unknown FP8 format '{value}'. Valid formats: e4m3, e5m2"
        )


class NASObjectType(str, Enum):
    """Types of NAS configuration objects."""
    CHOICE = "choice"
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BOOLEAN = "boolean"

    @classmethod
    def from_string(cls, value: str) -> NASObjectType:
        value_clean = value.strip().lower()
        for member in cls:
            if member.value == value_clean:
                return member
        raise ValueError(
            f"Unknown NAS object type '{value}'. Valid types: choice, continuous, integer, boolean"
        )


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_float_range(
    value: float,
    name: str,
    min_val: float,
    max_val: float,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> float:
    """Validate that a float value is within a given range.

    Args:
        value: The float value to validate.
        name: Parameter name for error messages.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        inclusive_min: Whether min_val is inclusive.
        inclusive_max: Whether max_val is inclusive.

    Returns:
        The validated float value.

    Raises:
        ValueError: If the value is outside the valid range.
        TypeError: If the value is not a float or int.
    """
    if not isinstance(value, (float, int)):
        raise TypeError(f"'{name}' must be a float or int, got {type(value).__name__}")
    value = float(value)
    if inclusive_min and inclusive_max:
        if value < min_val or value > max_val:
            raise ValueError(f"'{name}' must be in [{min_val}, {max_val}], got {value}")
    elif inclusive_min and not inclusive_max:
        if value < min_val or value >= max_val:
            raise ValueError(f"'{name}' must be in [{min_val}, {max_val}), got {value}")
    elif not inclusive_min and inclusive_max:
        if value <= min_val or value > max_val:
            raise ValueError(f"'{name}' must be in ({min_val}, {max_val}], got {value}")
    else:
        if value <= min_val or value >= max_val:
            raise ValueError(f"'{name}' must be in ({min_val}, {max_val}), got {value}")
    return value


def validate_int_positive(value: int, name: str) -> int:
    """Validate that an integer value is positive.

    Args:
        value: The integer value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated integer value.

    Raises:
        ValueError: If the value is not positive.
        TypeError: If the value is not an integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"'{name}' must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"'{name}' must be positive, got {value}")
    return value


def validate_int_non_negative(value: int, name: str) -> int:
    """Validate that an integer value is non-negative.

    Args:
        value: The integer value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated integer value.

    Raises:
        ValueError: If the value is negative.
        TypeError: If the value is not an integer.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"'{name}' must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"'{name}' must be non-negative, got {value}")
    return value


def validate_bits(value: int, name: str = "bits") -> int:
    """Validate that the number of bits is a power of 2 in the range [1, 32].

    Args:
        value: The bits value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated bits value.

    Raises:
        ValueError: If the value is not a valid bit width.
    """
    value = validate_int_positive(value, name)
    valid_bits = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32}
    if value not in valid_bits:
        raise ValueError(
            f"'{name}' must be one of {sorted(valid_bits)}, got {value}"
        )
    return value


def validate_sparsity(value: float, name: str = "sparsity") -> float:
    """Validate that sparsity is in [0, 1).

    Args:
        value: The sparsity value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated sparsity value.

    Raises:
        ValueError: If the value is outside [0, 1).
    """
    return validate_float_range(value, name, 0.0, 1.0, inclusive_min=True, inclusive_max=False)


def validate_probability(value: float, name: str = "probability") -> float:
    """Validate that a probability value is in [0, 1].

    Args:
        value: The probability value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated probability value.

    Raises:
        ValueError: If the value is outside [0, 1].
    """
    return validate_float_range(value, name, 0.0, 1.0, inclusive_min=True, inclusive_max=True)


def validate_temperature(value: float, name: str = "temperature") -> float:
    """Validate that a temperature value is positive.

    Args:
        value: The temperature value to validate.
        name: Parameter name for error messages.

    Returns:
        The validated temperature value.

    Raises:
        ValueError: If the value is not positive.
    """
    if not isinstance(value, (float, int)):
        raise TypeError(f"'{name}' must be a float or int, got {type(value).__name__}")
    value = float(value)
    if value <= 0:
        raise ValueError(f"'{name}' must be positive, got {value}")
    return value


def validate_group_size(value: int, name: str = "group_size") -> int:
    """Validate that group_size is a positive power of 2 or -1 (per-channel).

    Args:
        value: The group size value.
        name: Parameter name for error messages.

    Returns:
        The validated group size.

    Raises:
        ValueError: If the value is invalid.
    """
    if value == -1:
        return value
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"'{name}' must be an integer or -1, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"'{name}' must be positive or -1, got {value}")
    if value & (value - 1) != 0:
        logger.warning(
            f"'{name}'={value} is not a power of 2. Some backends may be slower."
        )
    return value


def validate_output_path(path: str, name: str = "output_path") -> str:
    """Validate and normalize an output file path.

    Args:
        path: The path to validate.
        name: Parameter name for error messages.

    Returns:
        The normalized absolute path.

    Raises:
        ValueError: If the path is empty.
        TypeError: If the path is not a string.
    """
    if not isinstance(path, str):
        raise TypeError(f"'{name}' must be a string, got {type(path).__name__}")
    path = path.strip()
    if not path:
        raise ValueError(f"'{name}' must not be empty")
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return path


def validate_dtype_string(value: str, name: str = "dtype") -> str:
    """Validate that a dtype string is a recognized PyTorch dtype.

    Args:
        value: The dtype string to validate.
        name: Parameter name for error messages.

    Returns:
        The validated dtype string.

    Raises:
        ValueError: If the dtype is not recognized.
    """
    import torch
    valid_dtypes = {
        "float32", "float64", "float16", "bfloat16", "int8", "int4", "uint8",
        "fp32", "fp64", "fp16", "bf16", "f32", "f64", "f16",
    }
    value_clean = value.strip().lower()
    if value_clean not in valid_dtypes:
        raise ValueError(
            f"'{name}' must be one of {sorted(valid_dtypes)}, got '{value}'"
        )
    return value_clean


# =============================================================================
# QuantizationConfig
# =============================================================================

@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Supports multiple quantization methods including GPTQ, AWQ, BitsAndBytes,
    LLM.int8, LLM.int4, NF4, and FP8 with comprehensive parameter control.

    Attributes:
        method: Quantization method to use.
        bits: Number of bits for quantization (4 or 8 typical).
        group_size: Group size for grouped quantization (-1 for per-channel).
        desc_act: Whether to use descending activation order (GPTQ).
        damp_percent: Damping percentage for Hessian computation (GPTQ).
        sym: Whether to use symmetric quantization.
        use_cuda_fp16: Whether to use CUDA FP16 for quantization compute.
        fp8_format: FP8 format (e4m3 or e5m2) when method is fp8.
        block_size: Block size for block-wise quantization.
        calibrate_batches: Number of calibration batches.
        batch_size: Batch size for calibration.
        seq_length: Sequence length for calibration inputs.
        pad_token_id: Padding token ID for calibration data.
        fallback_to_cpu: Whether to fall back to CPU if GPU quantization fails.
        tokenizer: Optional tokenizer for calibration data preparation.
        dataset: Optional calibration dataset name or path.
        dataset_num_samples: Number of samples from calibration dataset.
        dataset_split: Dataset split to use for calibration.
        output_dir: Directory to save quantized model.
        save_quantization_config: Whether to save quantization config with model.
        device: Target device for quantization.
        torch_dtype: Target torch dtype for dequantized weights.
        cache_file: Cache file for storing quantization state.
        double_quantization: Whether to apply double quantization (nested).
        double_quant_bits: Bits for outer quantization in double quantization.
        weight_quant_params: Additional quantization-specific parameters.
        activation_quant_params: Additional activation quantization parameters.
        use_flash_attention: Whether to use flash attention during calibration.
        num_calibration_workers: Number of dataloader workers for calibration.
        seed: Random seed for reproducibility.
        verbose: Whether to print detailed quantization logs.
    """
    method: str = "gptq"
    bits: int = 4
    group_size: int = 128
    desc_act: bool = True
    damp_percent: float = 0.01
    sym: bool = True
    use_cuda_fp16: bool = True
    fp8_format: str = "e4m3"
    block_size: int = 128
    calibrate_batches: int = 10
    batch_size: int = 1
    seq_length: int = 2048
    pad_token_id: int = 0
    fallback_to_cpu: bool = False
    tokenizer: Optional[Any] = None
    dataset: Optional[str] = None
    dataset_num_samples: int = 128
    dataset_split: str = "train"
    output_dir: str = "./quantized_model"
    save_quantization_config: bool = True
    device: str = "cuda"
    torch_dtype: str = "float16"
    cache_file: Optional[str] = None
    double_quantization: bool = False
    double_quant_bits: int = 8
    weight_quant_params: Dict[str, Any] = field(default_factory=dict)
    activation_quant_params: Dict[str, Any] = field(default_factory=dict)
    use_flash_attention: bool = False
    num_calibration_workers: int = 4
    seed: int = 42
    verbose: bool = False

    def __post_init__(self):
        """Validate all configuration parameters after initialization."""
        self.method = QuantizationMethod.from_string(self.method).value
        self.bits = validate_bits(self.bits)
        self.group_size = validate_group_size(self.group_size)
        self.damp_percent = validate_float_range(
            self.damp_percent, "damp_percent", 0.0, 1.0
        )
        self.sparsity = validate_sparsity(
            getattr(self, "sparsity", 0.0)
        )
        if self.fp8_format.lower() not in ("e4m3", "e5m2"):
            raise ValueError(
                f"fp8_format must be 'e4m3' or 'e5m2', got '{self.fp8_format}'"
            )
        self.fp8_format = self.fp8_format.lower()
        self.block_size = validate_int_positive(self.block_size, "block_size")
        self.calibrate_batches = validate_int_non_negative(self.calibrate_batches, "calibrate_batches")
        self.batch_size = validate_int_positive(self.batch_size, "batch_size")
        self.seq_length = validate_int_positive(self.seq_length, "seq_length")
        self.pad_token_id = validate_int_non_negative(self.pad_token_id, "pad_token_id")
        self.dataset_num_samples = validate_int_positive(self.dataset_num_samples, "dataset_num_samples")
        if self.dataset_split.strip() not in ("train", "validation", "test", "val"):
            raise ValueError(
                f"dataset_split must be 'train', 'validation', 'test', or 'val', "
                f"got '{self.dataset_split}'"
            )
        self.output_dir = validate_output_path(self.output_dir, "output_dir")
        if self.torch_dtype:
            validate_dtype_string(self.torch_dtype, "torch_dtype")
        if self.double_quantization:
            self.double_quant_bits = validate_bits(self.double_quant_bits, "double_quant_bits")
        if self.cache_file:
            self.cache_file = validate_output_path(self.cache_file, "cache_file")
        self.num_calibration_workers = validate_int_non_negative(
            self.num_calibration_workers, "num_calibration_workers"
        )
        if self.device not in ("cuda", "cpu", "auto", "mps"):
            raise ValueError(
                f"device must be 'cuda', 'cpu', 'auto', or 'mps', got '{self.device}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif value is None or isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)
        return result

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file.

        Args:
            path: Optional file path. If None, returns JSON string.

        Returns:
            JSON string if path is None, otherwise None (writes to file).
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved QuantizationConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> QuantizationConfig:
        """Create configuration from a dictionary.

        Args:
            config_dict: Dictionary of configuration values.

        Returns:
            Instantiated QuantizationConfig.

        Raises:
            TypeError: If config_dict is not a dictionary.
        """
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> QuantizationConfig:
        """Create configuration from a JSON file or JSON string.

        Args:
            path_or_str: Either a file path or a JSON string.

        Returns:
            Instantiated QuantizationConfig.

        Raises:
            FileNotFoundError: If path_or_str is a file path that does not exist.
            json.JSONDecodeError: If the content is not valid JSON.
        """
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> QuantizationConfig:
        """Create a deep copy of this configuration.

        Returns:
            Deep copy of the QuantizationConfig.
        """
        return copy.deepcopy(self)


# =============================================================================
# PruningConfig
# =============================================================================

@dataclass
class PruningConfig:
    """Configuration for model pruning.

    Supports various pruning methods with scheduling, scope control, and
    comprehensive parameter management.

    Attributes:
        method: Pruning method to use.
        sparsity: Target sparsity ratio (0.0 to 1.0).
        schedule: Pruning schedule (cosine, linear, step, etc.).
        pruning_frequency: How often to prune (in epochs/steps).
        scope: Pruning scope (local, global, layer_wise, block_wise).
        start_epoch: Epoch at which to start pruning.
        end_epoch: Epoch at which to finish pruning (reaching target sparsity).
        importance_metric: Metric for computing importance scores (l1, l2, fisher).
        minimal_layer_sparsity: Minimum sparsity per layer to prevent over-pruning.
        maximal_layer_sparsity: Maximum sparsity per layer.
        use_global_pruning: Whether to use global magnitude thresholds.
        exclusion_patterns: Layer name patterns to exclude from pruning.
        inclusion_patterns: Layer name patterns to include (if set, only these).
        keep_first_layer: Whether to skip pruning the first layer.
        keep_last_layer: Whether to skip pruning the last layer.
        keep_attention_layers: Whether to skip pruning attention layers.
        keep_embedding_layers: Whether to skip pruning embedding layers.
        keep_layernorm_layers: Whether to skip pruning layer norm layers.
        initial_sparsity: Initial sparsity before scheduling starts.
        step_size: Step size for step schedule pruning.
        num_steps: Number of steps for step schedule pruning.
        min_params_to_prune: Minimum number of parameters a layer must have.
        calibrate_batches: Number of batches for data-dependent pruning methods.
        batch_size: Batch size for calibration.
        seq_length: Sequence length for calibration inputs.
        prune_bias: Whether to prune bias terms.
        prune_norm_layers: Whether to prune normalization layer parameters.
        finetune_after_pruning: Whether to fine-tune after pruning.
        finetune_epochs: Number of fine-tuning epochs after pruning.
        finetune_lr: Learning rate for post-pruning fine-tuning.
        output_dir: Directory to save pruning results.
        device: Target device for pruning computations.
        seed: Random seed for reproducibility.
        verbose: Whether to print detailed pruning logs.
        record_metrics: Whether to record pruning metrics throughout training.
        checkpoint_every_n_steps: Save checkpoint every N pruning steps.
    """
    method: str = "magnitude"
    sparsity: float = 0.5
    schedule: str = "cosine"
    pruning_frequency: int = 1
    scope: str = "global"
    start_epoch: int = 0
    end_epoch: int = 10
    importance_metric: str = "l1"
    minimal_layer_sparsity: float = 0.0
    maximal_layer_sparsity: float = 0.95
    use_global_pruning: bool = False
    exclusion_patterns: List[str] = field(default_factory=lambda: [
        "embed", "lm_head", "output"
    ])
    inclusion_patterns: List[str] = field(default_factory=list)
    keep_first_layer: bool = True
    keep_last_layer: bool = True
    keep_attention_layers: bool = False
    keep_embedding_layers: bool = True
    keep_layernorm_layers: bool = True
    initial_sparsity: float = 0.0
    step_size: float = 0.1
    num_steps: int = 10
    min_params_to_prune: int = 256
    calibrate_batches: int = 10
    batch_size: int = 4
    seq_length: int = 512
    prune_bias: bool = False
    prune_norm_layers: bool = False
    finetune_after_pruning: bool = False
    finetune_epochs: int = 3
    finetune_lr: float = 1e-5
    output_dir: str = "./pruned_model"
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False
    record_metrics: bool = True
    checkpoint_every_n_steps: int = 0

    def __post_init__(self):
        """Validate all configuration parameters after initialization."""
        self.method = PruningMethod.from_string(self.method).value
        self.sparsity = validate_sparsity(self.sparsity)
        self.schedule = PruningSchedule.from_string(self.schedule).value
        self.pruning_frequency = validate_int_positive(self.pruning_frequency, "pruning_frequency")
        self.scope = PruningScope.from_string(self.scope).value
        self.start_epoch = validate_int_non_negative(self.start_epoch, "start_epoch")
        self.end_epoch = validate_int_non_negative(self.end_epoch, "end_epoch")
        if self.end_epoch <= self.start_epoch:
            raise ValueError(
                f"end_epoch ({self.end_epoch}) must be > start_epoch ({self.start_epoch})"
            )
        if self.importance_metric.lower() not in ("l1", "l2", "fisher", "gradient", "taylor"):
            raise ValueError(
                f"importance_metric must be one of l1, l2, fisher, gradient, taylor, "
                f"got '{self.importance_metric}'"
            )
        self.importance_metric = self.importance_metric.lower()
        self.minimal_layer_sparsity = validate_float_range(
            self.minimal_layer_sparsity, "minimal_layer_sparsity", 0.0, 1.0
        )
        self.maximal_layer_sparsity = validate_float_range(
            self.maximal_layer_sparsity, "maximal_layer_sparsity", 0.0, 1.0
        )
        if self.minimal_layer_sparsity >= self.maximal_layer_sparsity:
            raise ValueError(
                f"minimal_layer_sparsity ({self.minimal_layer_sparsity}) must be < "
                f"maximal_layer_sparsity ({self.maximal_layer_sparsity})"
            )
        self.initial_sparsity = validate_float_range(
            self.initial_sparsity, "initial_sparsity", 0.0, self.sparsity
        )
        self.step_size = validate_float_range(self.step_size, "step_size", 0.0, 1.0)
        self.num_steps = validate_int_positive(self.num_steps, "num_steps")
        self.min_params_to_prune = validate_int_positive(self.min_params_to_prune, "min_params_to_prune")
        self.calibrate_batches = validate_int_non_negative(self.calibrate_batches, "calibrate_batches")
        self.batch_size = validate_int_positive(self.batch_size, "batch_size")
        self.seq_length = validate_int_positive(self.seq_length, "seq_length")
        self.finetune_epochs = validate_int_non_negative(self.finetune_epochs, "finetune_epochs")
        if self.finetune_lr <= 0:
            raise ValueError(f"finetune_lr must be positive, got {self.finetune_lr}")
        self.output_dir = validate_output_path(self.output_dir, "output_dir")
        if self.device not in ("cuda", "cpu", "auto", "mps"):
            raise ValueError(
                f"device must be 'cuda', 'cpu', 'auto', or 'mps', got '{self.device}'"
            )
        self.checkpoint_every_n_steps = validate_int_non_negative(
            self.checkpoint_every_n_steps, "checkpoint_every_n_steps"
        )
        for idx, pattern in enumerate(self.exclusion_patterns):
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"exclusion_patterns[{idx}] must be a non-empty string"
                )
            self.exclusion_patterns[idx] = pattern.strip()
        for idx, pattern in enumerate(self.inclusion_patterns):
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"inclusion_patterns[{idx}] must be a non-empty string"
                )
            self.inclusion_patterns[idx] = pattern.strip()

    def should_prune_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be pruned based on inclusion/exclusion patterns.

        Args:
            layer_name: Fully qualified name of the layer.

        Returns:
            True if the layer should be pruned, False otherwise.
        """
        if self.inclusion_patterns:
            matched = any(
                re.search(re.escape(p), layer_name) for p in self.inclusion_patterns
            )
            if not matched:
                return False
        for pattern in self.exclusion_patterns:
            if re.search(re.escape(pattern), layer_name):
                return False
        return True

    def get_sparsity_at_step(self, step: int, total_steps: Optional[int] = None) -> float:
        """Compute the target sparsity at a given step based on the schedule.

        Args:
            step: Current step (0-indexed).
            total_steps: Total number of steps. If None, uses end_epoch - start_epoch.

        Returns:
            Target sparsity at this step.
        """
        if total_steps is None:
            total_steps = max(1, self.end_epoch - self.start_epoch)
        if total_steps <= 0:
            total_steps = 1
        progress = min(1.0, step / total_steps)
        if self.schedule == "constant":
            return self.sparsity
        elif self.schedule == "linear":
            return self.initial_sparsity + (self.sparsity - self.initial_sparsity) * progress
        elif self.schedule == "cosine":
            import math
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.initial_sparsity + (self.sparsity - self.initial_sparsity) * (1.0 - cosine_decay)
        elif self.schedule == "step":
            num_steps_done = int(progress * self.num_steps)
            current_sparsity = self.initial_sparsity + num_steps_done * self.step_size
            return min(self.sparsity, current_sparsity)
        elif self.schedule == "exponential":
            decay_rate = math.log(1.0 - self.sparsity + 1e-10) / total_steps
            return 1.0 - (1.0 - self.initial_sparsity) * math.exp(decay_rate * step)
        else:
            return self.sparsity * progress

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved PruningConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> PruningConfig:
        """Create configuration from a dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> PruningConfig:
        """Create configuration from a JSON file or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> PruningConfig:
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)


# =============================================================================
# DistillationConfig
# =============================================================================

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation.

    Controls teacher-student model training with various distillation strategies
    including response-based, feature-based, attention-based, and multi-teacher
    distillation.

    Attributes:
        teacher_model: Teacher model name or path.
        student_model: Student model name or path.
        temperature: Softmax temperature for soft target distillation.
        alpha: Weight for distillation loss vs hard label loss (0-1).
        loss_type: Type of distillation loss.
        intermediate_loss_weight: Weight for intermediate feature losses.
        num_layers_to_distill: Number of intermediate layers to distill.
        attention_hidden_dim: Hidden dimension for attention matching projection.
        feature_hidden_dim: Hidden dimension for feature matching projection.
        teacher_layer_mapping: Manual mapping of student layers to teacher layers.
        use_cosine_loss: Whether to use cosine similarity loss for features.
        use_align_dimensions: Whether to project features to matching dimensions.
        num_teachers: Number of teachers for multi-teacher distillation.
        teacher_weights: Predefined weights for each teacher (None = auto-compute).
        progressive_stages: Number of stages for progressive distillation.
        progressive_start_alpha: Starting alpha for progressive distillation.
        progressive_end_alpha: Ending alpha for progressive distillation.
        data_augmentation: Whether to augment training data.
        augmentation_methods: Data augmentation methods to use.
        augmentation_strength: Strength of data augmentation (0-1).
        use_focal_loss: Whether to use focal loss for hard examples.
        focal_gamma: Gamma parameter for focal loss.
        label_smoothing: Label smoothing factor for hard labels.
        gradient_matching: Whether to match gradients between teacher and student.
        gradient_matching_weight: Weight for gradient matching loss.
        num_epochs: Number of distillation training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate for student training.
        lr_scheduler: Learning rate scheduler type.
        warmup_steps: Number of warmup steps.
        weight_decay: Weight decay for optimizer.
        max_grad_norm: Maximum gradient norm for clipping.
        optimizer: Optimizer type.
        eval_steps: Evaluation frequency (in steps).
        save_steps: Model checkpoint save frequency (in steps).
        output_dir: Directory to save distilled model.
        logging_dir: Directory for logging.
        device: Training device.
        seed: Random seed.
        verbose: Whether to print detailed distillation logs.
        mixed_precision: Whether to use mixed precision training.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        early_stopping_patience: Patience for early stopping (0 = disabled).
        early_stopping_metric: Metric for early stopping.
        early_stopping_threshold: Minimum improvement for early stopping.
    """
    teacher_model: str = ""
    student_model: str = ""
    temperature: float = 4.0
    alpha: float = 0.5
    loss_type: str = "soft"
    intermediate_loss_weight: float = 1.0
    num_layers_to_distill: int = 6
    attention_hidden_dim: int = 256
    feature_hidden_dim: int = 512
    teacher_layer_mapping: Optional[Dict[int, int]] = None
    use_cosine_loss: bool = True
    use_align_dimensions: bool = True
    num_teachers: int = 1
    teacher_weights: Optional[List[float]] = None
    progressive_stages: int = 1
    progressive_start_alpha: float = 0.3
    progressive_end_alpha: float = 0.7
    data_augmentation: bool = False
    augmentation_methods: List[str] = field(default_factory=lambda: ["random_mask", "token_substitute"])
    augmentation_strength: float = 0.3
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    gradient_matching: bool = False
    gradient_matching_weight: float = 0.1
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "./distilled_model"
    logging_dir: str = "./logs/distillation"
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 0
    early_stopping_metric: str = "eval_loss"
    early_stopping_threshold: float = 1e-4

    def __post_init__(self):
        """Validate all configuration parameters."""
        if not self.teacher_model or not isinstance(self.teacher_model, str):
            raise ValueError("teacher_model must be a non-empty string")
        if not self.student_model or not isinstance(self.student_model, str):
            raise ValueError("student_model must be a non-empty string")
        self.temperature = validate_temperature(self.temperature)
        self.alpha = validate_probability(self.alpha)
        self.loss_type = DistillationLossType.from_string(self.loss_type).value
        self.intermediate_loss_weight = validate_float_range(
            self.intermediate_loss_weight, "intermediate_loss_weight", 0.0, 10.0
        )
        self.num_layers_to_distill = validate_int_positive(self.num_layers_to_distill, "num_layers_to_distill")
        self.attention_hidden_dim = validate_int_positive(self.attention_hidden_dim, "attention_hidden_dim")
        self.feature_hidden_dim = validate_int_positive(self.feature_hidden_dim, "feature_hidden_dim")
        if self.teacher_layer_mapping is not None:
            if not isinstance(self.teacher_layer_mapping, dict):
                raise TypeError("teacher_layer_mapping must be a dict or None")
            for k, v in self.teacher_layer_mapping.items():
                if not isinstance(k, int) or not isinstance(v, int):
                    raise TypeError("teacher_layer_mapping keys and values must be integers")
        self.num_teachers = validate_int_positive(self.num_teachers, "num_teachers")
        if self.teacher_weights is not None:
            if len(self.teacher_weights) != self.num_teachers:
                raise ValueError(
                    f"teacher_weights length ({len(self.teacher_weights)}) must match "
                    f"num_teachers ({self.num_teachers})"
                )
            total_weight = sum(self.teacher_weights)
            if abs(total_weight - 1.0) > 1e-6:
                raise ValueError(f"teacher_weights must sum to 1.0, got {total_weight}")
        self.progressive_stages = validate_int_positive(self.progressive_stages, "progressive_stages")
        self.progressive_start_alpha = validate_probability(self.progressive_start_alpha, "progressive_start_alpha")
        self.progressive_end_alpha = validate_probability(self.progressive_end_alpha, "progressive_end_alpha")
        if self.progressive_start_alpha >= self.progressive_end_alpha:
            raise ValueError(
                f"progressive_start_alpha ({self.progressive_start_alpha}) must be < "
                f"progressive_end_alpha ({self.progressive_end_alpha})"
            )
        self.augmentation_strength = validate_probability(self.augmentation_strength, "augmentation_strength")
        if self.focal_gamma < 0:
            raise ValueError(f"focal_gamma must be non-negative, got {self.focal_gamma}")
        self.label_smoothing = validate_float_range(self.label_smoothing, "label_smoothing", 0.0, 1.0)
        self.gradient_matching_weight = validate_float_range(
            self.gradient_matching_weight, "gradient_matching_weight", 0.0, 10.0
        )
        self.num_epochs = validate_int_positive(self.num_epochs, "num_epochs")
        self.batch_size = validate_int_positive(self.batch_size, "batch_size")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        self.warmup_steps = validate_int_non_negative(self.warmup_steps, "warmup_steps")
        self.weight_decay = validate_float_range(self.weight_decay, "weight_decay", 0.0, 1.0)
        self.max_grad_norm = validate_float_range(self.max_grad_norm, "max_grad_norm", 0.0, 1000.0)
        if self.optimizer.lower() not in ("adamw", "adam", "sgd", "rmsprop"):
            raise ValueError(
                f"optimizer must be one of adamw, adam, sgd, rmsprop, got '{self.optimizer}'"
            )
        self.optimizer = self.optimizer.lower()
        self.eval_steps = validate_int_positive(self.eval_steps, "eval_steps")
        self.save_steps = validate_int_positive(self.save_steps, "save_steps")
        self.output_dir = validate_output_path(self.output_dir, "output_dir")
        self.logging_dir = validate_output_path(self.logging_dir, "logging_dir")
        self.gradient_accumulation_steps = validate_int_positive(
            self.gradient_accumulation_steps, "gradient_accumulation_steps"
        )
        self.early_stopping_patience = validate_int_non_negative(
            self.early_stopping_patience, "early_stopping_patience"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved DistillationConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> DistillationConfig:
        """Create configuration from a dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> DistillationConfig:
        """Create configuration from a JSON file or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> DistillationConfig:
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)


# =============================================================================
# NASConfig
# =============================================================================

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search.

    Controls the search space, search algorithm, and evaluation criteria
    for finding optimal model architectures.

    Attributes:
        search_space: Dictionary defining the architecture search space.
        search_algorithm: NAS algorithm to use.
        max_trials: Maximum number of architectures to evaluate.
        population_size: Population size for evolutionary algorithms.
        mutation_rate: Mutation probability for evolutionary algorithms.
        crossover_rate: Crossover probability for evolutionary algorithms.
        tournament_size: Tournament size for selection in evolutionary NAS.
        num_generations: Number of generations for evolutionary search.
        arch_learning_rate: Learning rate for architecture parameters (DARTS).
        arch_weight_decay: Weight decay for architecture parameters.
        num_arch_steps: Number of architecture optimization steps per epoch.
        num_weight_steps: Number of weight optimization steps per epoch.
        bayesian_initial_points: Number of initial random samples for Bayesian optimization.
        bayesian_acquisition: Acquisition function for Bayesian optimization.
        supernet_epochs: Number of epochs to train the supernet (one-shot NAS).
        budget_flops: Maximum FLOPs budget for found architectures.
        budget_params: Maximum parameter count for found architectures.
        budget_latency_ms: Maximum latency in milliseconds.
        budget_memory_mb: Maximum memory usage in MB.
        hardware_target: Target hardware platform.
        metric: Evaluation metric to optimize.
        metric_mode: Whether to maximize or minimize the metric.
        warmup_epochs: Warmup epochs before starting architecture search.
        eval_frequency: How often to evaluate during search.
        num_workers: Number of parallel evaluation workers.
        seed: Random seed.
        output_dir: Directory to save NAS results.
        verbose: Whether to print detailed NAS logs.
        checkpoint_every_n_trials: Save checkpoint every N trials.
        resume_from: Checkpoint file to resume from.
    """
    search_space: Dict[str, Any] = field(default_factory=lambda: {
        "num_layers": {"type": "integer", "low": 2, "high": 12},
        "hidden_dim": {"type": "choice", "options": [256, 512, 768, 1024]},
        "num_heads": {"type": "choice", "options": [4, 8, 12, 16]},
        "ffn_dim": {"type": "choice", "options": [1024, 2048, 3072, 4096]},
        "dropout": {"type": "continuous", "low": 0.0, "high": 0.3},
    })
    search_algorithm: str = "evolutionary"
    max_trials: int = 100
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.9
    tournament_size: int = 5
    num_generations: int = 20
    arch_learning_rate: float = 3e-4
    arch_weight_decay: float = 1e-3
    num_arch_steps: int = 5
    num_weight_steps: int = 20
    bayesian_initial_points: int = 10
    bayesian_acquisition: str = "expected_improvement"
    supernet_epochs: int = 10
    budget_flops: float = 0.0
    budget_params: float = 0.0
    budget_latency_ms: float = 0.0
    budget_memory_mb: float = 0.0
    hardware_target: str = "cuda"
    metric: str = "accuracy"
    metric_mode: str = "maximize"
    warmup_epochs: int = 1
    eval_frequency: int = 1
    num_workers: int = 4
    seed: int = 42
    output_dir: str = "./nas_results"
    verbose: bool = False
    checkpoint_every_n_trials: int = 10
    resume_from: Optional[str] = None

    def __post_init__(self):
        """Validate all configuration parameters."""
        if not isinstance(self.search_space, dict):
            raise TypeError(
                f"search_space must be a dict, got {type(self.search_space).__name__}"
            )
        for name, spec in self.search_space.items():
            if not isinstance(spec, dict):
                raise TypeError(f"search_space['{name}'] must be a dict")
            if "type" not in spec:
                raise ValueError(f"search_space['{name}'] must have a 'type' key")
            spec_type = spec["type"].lower()
            if spec_type not in ("choice", "continuous", "integer", "boolean"):
                raise ValueError(
                    f"search_space['{name}']['type'] must be one of "
                    f"choice, continuous, integer, boolean, got '{spec_type}'"
                )
            if spec_type == "choice":
                if "options" not in spec or not isinstance(spec["options"], (list, tuple)):
                    raise ValueError(
                        f"search_space['{name}'] with type 'choice' must have 'options' list"
                    )
                if len(spec["options"]) < 2:
                    raise ValueError(
                        f"search_space['{name}']['options'] must have at least 2 elements"
                    )
            elif spec_type == "continuous":
                if "low" not in spec or "high" not in spec:
                    raise ValueError(
                        f"search_space['{name}'] with type 'continuous' must have 'low' and 'high'"
                    )
                if spec["low"] >= spec["high"]:
                    raise ValueError(
                        f"search_space['{name}']['low'] must be < 'high'"
                    )
            elif spec_type == "integer":
                if "low" not in spec or "high" not in spec:
                    raise ValueError(
                        f"search_space['{name}'] with type 'integer' must have 'low' and 'high'"
                    )
                if spec["low"] >= spec["high"]:
                    raise ValueError(
                        f"search_space['{name}']['low'] must be < 'high'"
                    )
        self.search_algorithm = SearchAlgorithm.from_string(self.search_algorithm).value
        self.max_trials = validate_int_positive(self.max_trials, "max_trials")
        self.population_size = validate_int_positive(self.population_size, "population_size")
        self.mutation_rate = validate_probability(self.mutation_rate, "mutation_rate")
        self.crossover_rate = validate_probability(self.crossover_rate, "crossover_rate")
        self.tournament_size = validate_int_positive(self.tournament_size, "tournament_size")
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) must be <= "
                f"population_size ({self.population_size})"
            )
        self.num_generations = validate_int_positive(self.num_generations, "num_generations")
        self.arch_learning_rate = validate_float_range(
            self.arch_learning_rate, "arch_learning_rate", 0.0, 1.0
        )
        self.arch_weight_decay = validate_float_range(
            self.arch_weight_decay, "arch_weight_decay", 0.0, 1.0
        )
        self.num_arch_steps = validate_int_positive(self.num_arch_steps, "num_arch_steps")
        self.num_weight_steps = validate_int_positive(self.num_weight_steps, "num_weight_steps")
        self.bayesian_initial_points = validate_int_positive(
            self.bayesian_initial_points, "bayesian_initial_points"
        )
        if self.bayesian_acquisition.lower() not in (
            "expected_improvement", "upper_confidence_bound",
            "probability_of_improvement", "ei", "ucb", "pi",
        ):
            raise ValueError(
                f"bayesian_acquisition must be one of expected_improvement, "
                f"upper_confidence_bound, probability_of_improvement, got '{self.bayesian_acquisition}'"
            )
        self.bayesian_acquisition = self.bayesian_acquisition.lower()
        self.supernet_epochs = validate_int_positive(self.supernet_epochs, "supernet_epochs")
        self.num_workers = validate_int_positive(self.num_workers, "num_workers")
        self.eval_frequency = validate_int_positive(self.eval_frequency, "eval_frequency")
        self.output_dir = validate_output_path(self.output_dir, "output_dir")
        if self.metric_mode.lower() not in ("maximize", "minimize"):
            raise ValueError(
                f"metric_mode must be 'maximize' or 'minimize', got '{self.metric_mode}'"
            )
        self.metric_mode = self.metric_mode.lower()
        self.warmup_epochs = validate_int_non_negative(self.warmup_epochs, "warmup_epochs")
        self.checkpoint_every_n_trials = validate_int_non_negative(
            self.checkpoint_every_n_trials, "checkpoint_every_n_trials"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved NASConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> NASConfig:
        """Create configuration from a dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> NASConfig:
        """Create configuration from a JSON file or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> NASConfig:
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)


# =============================================================================
# CompilationConfig
# =============================================================================

@dataclass
class CompilationConfig:
    """Configuration for model compilation.

    Controls compilation with torch.compile, Triton, inductor, CUDA graphs,
    and other backends.

    Attributes:
        backend: Compilation backend to use.
        mode: Optimization mode for torch.compile.
        fullgraph: Whether to require full graph capture (no graph breaks).
        dynamic: Whether to use dynamic shapes.
        enable_cpp_wrapper: Whether to use C++ wrapper.
        memory_format: Memory format for compiled model.
        guard_type: Guard type for torch.compile.
        disable_mode: Whether to disable the optimization pass.
        use_dynamo: Whether to use TorchDynamo for tracing.
        num_warups: Number of warmup iterations before benchmarking.
        num_iterations: Number of benchmark iterations.
        triton_autotune: Whether to auto-tune Triton kernels.
        triton_autotune_keys: Keys for Triton autotuning.
        triton_block_sizes: Block sizes for Triton kernels.
        cudagraphs_warmup: Number of warmup steps for CUDA graphs.
        cudagraphs_pool_size: CUDA memory pool size for CUDA graphs.
        inductor_fusion: Whether to enable operator fusion in inductor.
        inductor_memory_planning: Whether to enable memory planning in inductor.
        inductor_shape_padding: Whether to pad shapes for vectorization.
        cache_dir: Directory for caching compiled kernels.
        cache_key: Optional cache key for kernel caching.
        profile_compilation: Whether to profile compilation time.
        benchmark_compiled: Whether to benchmark after compilation.
        save_compiled: Whether to save compiled model.
        save_path: Path to save compiled model.
        device: Target device.
        verbose: Whether to print compilation logs.
        seed: Random seed.
    """
    backend: str = "torch.compile"
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    enable_cpp_wrapper: bool = True
    memory_format: str = "contiguous_format"
    guard_type: str = "default"
    disable_mode: bool = False
    use_dynamo: bool = True
    num_warups: int = 3
    num_iterations: int = 100
    triton_autotune: bool = True
    triton_autotune_keys: List[str] = field(default_factory=lambda: ["M", "N", "K"])
    triton_block_sizes: Dict[str, int] = field(default_factory=lambda: {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32})
    cudagraphs_warmup: int = 3
    cudagraphs_pool_size: int = 0
    inductor_fusion: bool = True
    inductor_memory_planning: bool = True
    inductor_shape_padding: bool = True
    cache_dir: str = "./compilation_cache"
    cache_key: Optional[str] = None
    profile_compilation: bool = False
    benchmark_compiled: bool = True
    save_compiled: bool = False
    save_path: str = "./compiled_model"
    device: str = "cuda"
    verbose: bool = False
    seed: int = 42

    def __post_init__(self):
        """Validate all configuration parameters."""
        self.backend = CompilationBackend.from_string(self.backend).value
        self.mode = CompilationMode.from_string(self.mode).value
        if self.memory_format.lower() not in (
            "contiguous_format", "channels_last", "channels_last_3d",
            "preserve_format",
        ):
            raise ValueError(
                f"memory_format must be one of contiguous_format, channels_last, "
                f"channels_last_3d, preserve_format, got '{self.memory_format}'"
            )
        self.memory_format = self.memory_format.lower()
        if self.guard_type.lower() not in ("default", "debug_assert", "static", "strict"):
            raise ValueError(
                f"guard_type must be one of default, debug_assert, static, strict, "
                f"got '{self.guard_type}'"
            )
        self.guard_type = self.guard_type.lower()
        self.num_warups = validate_int_non_negative(self.num_warups, "num_warups")
        self.num_iterations = validate_int_positive(self.num_iterations, "num_iterations")
        if self.cudagraphs_warmup < 1:
            raise ValueError(f"cudagraphs_warmup must be >= 1, got {self.cudagraphs_warmup}")
        self.cudagraphs_pool_size = validate_int_non_negative(
            self.cudagraphs_pool_size, "cudagraphs_pool_size"
        )
        self.cache_dir = validate_output_path(self.cache_dir, "cache_dir")
        self.save_path = validate_output_path(self.save_path, "save_path")
        if self.device not in ("cuda", "cpu", "auto", "mps"):
            raise ValueError(
                f"device must be 'cuda', 'cpu', 'auto', or 'mps', got '{self.device}'"
            )

    def get_torch_compile_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for torch.compile().

        Returns:
            Dictionary of kwargs suitable for torch.compile().
        """
        kwargs = {
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
        }
        if hasattr(__import__("torch"), "compile"):
            import torch
            if self.backend in ("torch.compile", "inductor"):
                kwargs["backend"] = "inductor" if self.backend == "inductor" else "inductor"
        return kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved CompilationConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CompilationConfig:
        """Create configuration from a dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> CompilationConfig:
        """Create configuration from a JSON file or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> CompilationConfig:
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)


# =============================================================================
# CompressionConfig
# =============================================================================

@dataclass
class CompressionConfig:
    """Configuration for model compression.

    High-level configuration for applying multiple compression techniques
    to a model to achieve a target compression ratio while preserving
    accuracy.

    Attributes:
        target_ratio: Target compression ratio (0.0 to 1.0).
        preserve_accuracy_threshold: Minimum accuracy to preserve.
        max_flops_reduction: Maximum FLOPs reduction ratio.
        methods: Ordered list of compression methods to apply.
        quantization_config: Quantization configuration (if applicable).
        pruning_config: Pruning configuration (if applicable).
        low_rank_rank: Rank for low-rank factorization.
        vocabulary_size_target: Target vocabulary size after pruning.
        merge_threshold: Threshold for layer merging.
        hash_bits: Number of bits for hash-based compression.
        product_quantization_subvectors: Number of subvectors for product quantization.
        iterative: Whether to apply compression iteratively.
        max_iterations: Maximum number of compression iterations.
        tolerance: Convergence tolerance for iterative compression.
        evaluation_dataloader: Reference to evaluation dataloader.
        evaluation_metric: Metric to track during compression.
        baseline_accuracy: Baseline accuracy before compression.
        benchmark_original: Whether to benchmark original model first.
        export_format: Format for compressed model export.
        save_intermediate: Whether to save intermediate models.
        intermediate_dir: Directory for intermediate model checkpoints.
        output_dir: Directory to save final compressed model.
        device: Target device.
        seed: Random seed.
        verbose: Whether to print detailed compression logs.
    """
    target_ratio: float = 0.5
    preserve_accuracy_threshold: float = 0.95
    max_flops_reduction: float = 0.5
    methods: List[str] = field(default_factory=lambda: ["pruning", "quantization", "low_rank"])
    quantization_config: Optional[Dict[str, Any]] = None
    pruning_config: Optional[Dict[str, Any]] = None
    low_rank_rank: int = 64
    vocabulary_size_target: int = 32000
    merge_threshold: float = 0.95
    hash_bits: int = 16
    product_quantization_subvectors: int = 8
    iterative: bool = True
    max_iterations: int = 10
    tolerance: float = 1e-4
    evaluation_dataloader: Optional[Any] = None
    evaluation_metric: str = "accuracy"
    baseline_accuracy: float = 0.0
    benchmark_original: bool = True
    export_format: str = "pytorch"
    save_intermediate: bool = False
    intermediate_dir: str = "./compression_intermediate"
    output_dir: str = "./compressed_model"
    device: str = "cuda"
    seed: int = 42
    verbose: bool = False

    def __post_init__(self):
        """Validate all configuration parameters."""
        self.target_ratio = validate_float_range(
            self.target_ratio, "target_ratio", 0.0, 1.0
        )
        self.preserve_accuracy_threshold = validate_float_range(
            self.preserve_accuracy_threshold, "preserve_accuracy_threshold", 0.0, 1.0
        )
        self.max_flops_reduction = validate_float_range(
            self.max_flops_reduction, "max_flops_reduction", 0.0, 1.0
        )
        valid_methods = {
            "pruning", "quantization", "low_rank", "weight_sharing",
            "hash_embedding", "product_quantization", "vocabulary_pruning",
            "layer_fusion",
        }
        if not isinstance(self.methods, list) or len(self.methods) == 0:
            raise ValueError("methods must be a non-empty list")
        for idx, method in enumerate(self.methods):
            if method not in valid_methods:
                raise ValueError(
                    f"methods[{idx}] '{method}' is not valid. Valid methods: {valid_methods}"
                )
        self.low_rank_rank = validate_int_positive(self.low_rank_rank, "low_rank_rank")
        self.vocabulary_size_target = validate_int_positive(
            self.vocabulary_size_target, "vocabulary_size_target"
        )
        self.merge_threshold = validate_probability(self.merge_threshold, "merge_threshold")
        self.hash_bits = validate_int_positive(self.hash_bits, "hash_bits")
        if self.hash_bits > 64:
            raise ValueError(f"hash_bits must be <= 64, got {self.hash_bits}")
        self.product_quantization_subvectors = validate_int_positive(
            self.product_quantization_subvectors, "product_quantization_subvectors"
        )
        self.max_iterations = validate_int_positive(self.max_iterations, "max_iterations")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}")
        if self.export_format.lower() not in ("pytorch", "onnx", "tensorrt", "torchscript"):
            raise ValueError(
                f"export_format must be one of pytorch, onnx, tensorrt, torchscript, "
                f"got '{self.export_format}'"
            )
        self.export_format = self.export_format.lower()
        self.intermediate_dir = validate_output_path(self.intermediate_dir, "intermediate_dir")
        self.output_dir = validate_output_path(self.output_dir, "output_dir")
        if self.device not in ("cuda", "cpu", "auto", "mps"):
            raise ValueError(
                f"device must be 'cuda', 'cpu', 'auto', or 'mps', got '{self.device}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        result = asdict(self)
        result["quantization_config"] = self.quantization_config
        result["pruning_config"] = self.pruning_config
        return result

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """Serialize configuration to JSON string or file."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, sort_keys=True)
        if path is not None:
            path = validate_output_path(path, "path")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            logger.info("Saved CompressionConfig to %s", path)
            return None
        return json_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> CompressionConfig:
        """Create configuration from a dictionary."""
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"config_dict must be a dict, got {type(config_dict).__name__}"
            )
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path_or_str: str) -> CompressionConfig:
        """Create configuration from a JSON file or JSON string."""
        if os.path.isfile(path_or_str):
            with open(path_or_str, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
        else:
            config_dict = json.loads(path_or_str)
        return cls.from_dict(config_dict)

    def copy(self) -> CompressionConfig:
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)
