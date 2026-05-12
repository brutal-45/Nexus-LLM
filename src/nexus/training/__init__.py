"""
Training Package Init
=====================
Complete training infrastructure for Nexus:
    - trainer: Distributed training loop with mixed precision, FSDP, gradient checkpointing
    - optimizers: All optimizers built from scratch (SGD, Adam, AdamW, LAMB, Adafactor, LION, Sophia, Shampoo, 8-bit)
    - scheduler: All LR schedulers (cosine, linear, WSD, inverse sqrt, OneCycle, polynomial, multi-phase, etc.)
    - parallel: 3D parallelism (TP/PP/DP), FSDP/ZeRO-3
    - checkpoint: Safetensors-based checkpoint management
    - convergence: Loss landscape analysis, saddle point detection, gradient noise tracking, critical batch size
    - kernels: Triton fused kernels (RMSNorm, SwiGLU, RoPE, CrossEntropy, Adam, INT8/INT4, Flash Attention)
    - lora: LoRA, QLoRA, DoRA fine-tuning infrastructure
"""

from .trainer import Trainer, TrainingArguments
from .scheduler import (
    CosineAnnealingWithWarmup,
    LinearWarmupWithDecay,
    WarmupStableDecay,
    CosineAnnealingWithWarmRestarts,
    InverseSquareRootDecay,
    OneCycleLR,
    PolynomialDecay,
    MultiPhaseSchedule,
    ConstantWithWarmup,
    ExponentialDecayWithWarmup,
    WarmupHoldDecay,
    get_scheduler,
)
from .optimizers import (
    SGD,
    Adam,
    AdamW,
    LAMB,
    Adafactor,
    LION,
    Sophia,
    Shampoo,
    AdamW8bit,
)
from .parallel import setup_distributed, cleanup_distributed, get_world_info
from .checkpoint import CheckpointManager
from .convergence import (
    LossLandscapeAnalyzer,
    SaddlePointDetector,
    GradientNoiseTracker,
    CriticalBatchSizeEstimator,
    TrainingDynamicsMonitor,
)
from .kernels import (
    fused_rms_norm,
    fused_swiglu,
    fused_rope,
    fused_cross_entropy,
    quantize_int8,
    dequantize_int8,
    int8_matmul,
    quantize_int4,
    dequantize_int4,
    fused_adam,
    flash_attention_fwd,
)
from .lora import (
    LoRAConfig,
    LoRALinear,
    DoRALinear,
    QLoRALinear,
    NF4Quantizer,
    LoRAWrapper,
    LoRAFineTuner,
    apply_lora_to_model,
    merge_lora_weights,
    lora_state_dict,
    load_lora_state_dict,
    find_target_modules,
    create_lora_config_from_dict,
    print_lora_info,
)

__all__ = [
    # Trainer
    "Trainer",
    "TrainingArguments",
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    "LAMB",
    "Adafactor",
    "LION",
    "Sophia",
    "Shampoo",
    "AdamW8bit",
    # Schedulers
    "CosineAnnealingWithWarmup",
    "LinearWarmupWithDecay",
    "WarmupStableDecay",
    "CosineAnnealingWithWarmRestarts",
    "InverseSquareRootDecay",
    "OneCycleLR",
    "PolynomialDecay",
    "MultiPhaseSchedule",
    "ConstantWithWarmup",
    "ExponentialDecayWithWarmup",
    "WarmupHoldDecay",
    "get_scheduler",
    # Parallel
    "setup_distributed",
    "cleanup_distributed",
    "get_world_info",
    # Checkpoint
    "CheckpointManager",
    # Convergence
    "LossLandscapeAnalyzer",
    "SaddlePointDetector",
    "GradientNoiseTracker",
    "CriticalBatchSizeEstimator",
    "TrainingDynamicsMonitor",
    # Triton Kernels
    "fused_rms_norm",
    "fused_swiglu",
    "fused_rope",
    "fused_cross_entropy",
    "quantize_int8",
    "dequantize_int8",
    "int8_matmul",
    "quantize_int4",
    "dequantize_int4",
    "fused_adam",
    "flash_attention_fwd",
    # LoRA / QLoRA / DoRA
    "LoRAConfig",
    "LoRALinear",
    "DoRALinear",
    "QLoRALinear",
    "NF4Quantizer",
    "LoRAWrapper",
    "LoRAFineTuner",
    "apply_lora_to_model",
    "merge_lora_weights",
    "lora_state_dict",
    "load_lora_state_dict",
    "find_target_modules",
    "create_lora_config_from_dict",
    "print_lora_info",
]
