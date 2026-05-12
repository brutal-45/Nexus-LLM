"""
Nexus LLM Optimization Module
==============================

Production-grade optimization toolkit for large language models including:
- Quantization: GPTQ, AWQ, NF4, FP8, SmoothQuant, mixed-precision, calibration
- Pruning: Magnitude, Structured, SparseGPT, Wanda, Lottery Ticket, Gradual
- Knowledge Distillation: Multi-level, Multi-Teacher, Progressive, TinyBERT
- Mixed Precision Training: BF16, FP16, dynamic loss scaling
- Gradient Checkpointing: Block, Selective, Sequential, Recompute policies
- Memory Optimization: Tracking, CPU offloading, Activation offloading, Memory pools
- Inference Optimization: Paged KV cache, Continuous batching, CUDA graphs, Speculative decoding
- Model Compression: Weight sharing, Low-rank factorization, Layer fusion
- Compilation: torch.compile, Triton kernels, Inductor optimization
"""

from nexus.optimization.quantization_pruning import (
    # ── Abstract base ──────────────────────────────────────────────────
    BaseQuantizer,
    # ── GPTQ Quantizer ─────────────────────────────────────────────────
    GPTQQuantizer,
    # ── AWQ Quantizer ──────────────────────────────────────────────────
    AWQQuantizer,
    # ── NF4 Quantizer (bitsandbytes-style) ─────────────────────────────
    NF4Quantizer,
    NF4Linear,
    DoubleQuantization,
    # ── FP8 Quantizer ──────────────────────────────────────────────────
    FP8Quantizer,
    FP8Linear,
    # ── SmoothQuant ────────────────────────────────────────────────────
    SmoothQuantizer,
    # ── Mixed-Precision Quantizer ──────────────────────────────────────
    MixedPrecisionQuantizer,
    # ── Quantization helpers ───────────────────────────────────────────
    QuantizationSimulator,
    QuantizationCalibrator,
    # ── Magnitude Pruner ───────────────────────────────────────────────
    MagnitudePruner,
    # ── Structured Pruner ──────────────────────────────────────────────
    StructuredPruner,
    # ── SparseGPT Pruner ──────────────────────────────────────────────
    SparseGPTPruner,
    # ── Wanda Pruner ───────────────────────────────────────────────────
    WandaPruner,
    # ── Lottery Ticket Pruner ──────────────────────────────────────────
    LotteryTicketPruner,
    # ── Gradual Pruner ─────────────────────────────────────────────────
    GradualPruner,
    # ── Pruning helpers ────────────────────────────────────────────────
    PruningScheduler,
    PruningMetrics,
    SparseLinear,
    # ── Distillation ───────────────────────────────────────────────────
    DistillationTrainer,
    FeatureDistiller,
    MultiTeacherDistiller,
    ProgressiveDistiller,
    TinyBERTDistiller,
)

from nexus.optimization.memory_inference import (
    # ── Mixed Precision Training ───────────────────────────────────────
    MixedPrecisionTrainer,
    BF16Trainer,
    FP16Trainer,
    LossScaler,
    PrecisionOptimizer,
    # ── Gradient Checkpointing ─────────────────────────────────────────
    CheckpointFunction,
    BlockCheckpointing,
    SelectiveCheckpointing,
    SequentialCheckpointing,
    RecomputePolicy,
    MemoryEstimator,
    # ── Memory Optimization ────────────────────────────────────────────
    MemoryTracker,
    CPUOffloader,
    ActivationsOffloader,
    OptimizerStateOffloader,
    MemoryPool,
    GarbageCollector,
    # ── Inference Optimization ─────────────────────────────────────────
    PagedKVCache,
    ContinuousBatching,
    CUDAGraphRunner,
    KVCachePrefixSharing,
    SpeculativeDraftModel,
    DraftVerifier,
    # ── Model Compression ──────────────────────────────────────────────
    ModelCompressor,
    WeightSharing,
    LowRankFactorization,
    LayerFusion,
    CompressionAnalyzer,
    # ── Compilation ────────────────────────────────────────────────────
    ModelCompiler,
    TritonKernelCompiler,
    InductorOptimizer,
    CompilationCache,
    DynamicShapeHandler,
)

# ── Convenience aliases ────────────────────────────────────────────────

# Quantization shortcuts
GPTQ = GPTQQuantizer
AWQ = AWQQuantizer
NF4 = NF4Quantizer
FP8Q = FP8Quantizer
SmoothQ = SmoothQuantizer
MPQ = MixedPrecisionQuantizer
QSim = QuantizationSimulator
QCalib = QuantizationCalibrator

# Pruning shortcuts
MagPrune = MagnitudePruner
StructPrune = StructuredPruner
SparseGPTP = SparseGPTPruner
WandaP = WandaPruner
LTP = LotteryTicketPruner
GradPrune = GradualPruner
PSched = PruningScheduler
PMetrics = PruningMetrics

# Distillation shortcuts
Distill = DistillationTrainer
FeatDistill = FeatureDistiller
MultiTDistill = MultiTeacherDistiller
ProgDistill = ProgressiveDistiller
TinyBERTD = TinyBERTDistiller

# Mixed precision shortcuts
MPT = MixedPrecisionTrainer
BF16T = BF16Trainer
FP16T = FP16Trainer
LS = LossScaler
PrecOpt = PrecisionOptimizer

# Memory shortcuts
MemTrack = MemoryTracker
CPUOff = CPUOffloader
ActOff = ActivationsOffloader
OptOff = OptimizerStateOffloader
MemPool = MemoryPool
SmartGC = GarbageCollector

# Inference shortcuts
PagedKV = PagedKVCache
ContBatch = ContinuousBatching
CUDAGraph = CUDAGraphRunner
KVShare = KVCachePrefixSharing
SpecDraft = SpeculativeDraftModel
DraftV = DraftVerifier

# Compression shortcuts
MCompress = ModelCompressor
WShare = WeightSharing
LRF = LowRankFactorization
LFuse = LayerFusion
CAnalyzer = CompressionAnalyzer

# Compilation shortcuts
MCompiler = ModelCompiler
TritonKC = TritonKernelCompiler
IndOpt = InductorOptimizer
CompCache = CompilationCache
DynShape = DynamicShapeHandler

# ── Version ────────────────────────────────────────────────────────────

__version__ = "1.0.0"
__all__ = [
    # Quantization
    "BaseQuantizer",
    "GPTQQuantizer",
    "GPTQ",
    "AWQQuantizer",
    "AWQ",
    "NF4Quantizer",
    "NF4",
    "NF4Linear",
    "DoubleQuantization",
    "FP8Quantizer",
    "FP8Q",
    "FP8Linear",
    "SmoothQuantizer",
    "SmoothQ",
    "MixedPrecisionQuantizer",
    "MPQ",
    "QuantizationSimulator",
    "QSim",
    "QuantizationCalibrator",
    "QCalib",
    # Pruning
    "MagnitudePruner",
    "MagPrune",
    "StructuredPruner",
    "StructPrune",
    "SparseGPTPruner",
    "SparseGPTP",
    "WandaPruner",
    "WandaP",
    "LotteryTicketPruner",
    "LTP",
    "GradualPruner",
    "GradPrune",
    "PruningScheduler",
    "PSched",
    "PruningMetrics",
    "PMetrics",
    "SparseLinear",
    # Distillation
    "DistillationTrainer",
    "Distill",
    "FeatureDistiller",
    "FeatDistill",
    "MultiTeacherDistiller",
    "MultiTDistill",
    "ProgressiveDistiller",
    "ProgDistill",
    "TinyBERTDistiller",
    "TinyBERTD",
    # Mixed Precision
    "MixedPrecisionTrainer",
    "MPT",
    "BF16Trainer",
    "BF16T",
    "FP16Trainer",
    "FP16T",
    "LossScaler",
    "LS",
    "PrecisionOptimizer",
    "PrecOpt",
    # Gradient Checkpointing
    "CheckpointFunction",
    "BlockCheckpointing",
    "SelectiveCheckpointing",
    "SequentialCheckpointing",
    "RecomputePolicy",
    "MemoryEstimator",
    # Memory Optimization
    "MemoryTracker",
    "MemTrack",
    "CPUOffloader",
    "CPUOff",
    "ActivationsOffloader",
    "ActOff",
    "OptimizerStateOffloader",
    "OptOff",
    "MemoryPool",
    "MemPool",
    "GarbageCollector",
    "SmartGC",
    # Inference
    "PagedKVCache",
    "PagedKV",
    "ContinuousBatching",
    "ContBatch",
    "CUDAGraphRunner",
    "CUDAGraph",
    "KVCachePrefixSharing",
    "KVShare",
    "SpeculativeDraftModel",
    "SpecDraft",
    "DraftVerifier",
    "DraftV",
    # Compression
    "ModelCompressor",
    "MCompress",
    "WeightSharing",
    "WShare",
    "LowRankFactorization",
    "LRF",
    "LayerFusion",
    "LFuse",
    "CompressionAnalyzer",
    "CAnalyzer",
    # Compilation
    "ModelCompiler",
    "MCompiler",
    "TritonKernelCompiler",
    "TritonKC",
    "InductorOptimizer",
    "IndOpt",
    "CompilationCache",
    "CompCache",
    "DynamicShapeHandler",
    "DynShape",
]
