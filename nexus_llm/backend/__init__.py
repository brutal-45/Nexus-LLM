"""Nexus-LLM Backend Module.

Provides model management, inference, scheduling, caching, quantization,
streaming, adapters, health checks, metrics, and more for high-performance
LLM inference serving.
"""

# Model management
from .model_manager import ModelManager, ModelInfo, ModelStatus, ModelRegistry

# Inference engine
from .inference import InferenceEngine, GenerationResult

# Tokenizer utilities
from .tokenizer_utils import TokenizerWrapper, load_tokenizer, count_message_tokens, truncate_messages

# Server
from .server import create_app, app

# KV Cache
from .cache import PagedKVCache, CacheBlock, CacheEntry, CacheStatus, estimate_cache_size, compute_optimal_num_blocks

# Scheduler
from .scheduler import RequestScheduler, InferenceRequest, RequestPriority, RequestStatus, Batch

# Quantization
from .quantization import QuantizationConfig, QuantizationType, QuantizationManager

# Pipeline
from .pipeline import (
    Pipeline, TextGenerationPipeline, Text2TextGenerationPipeline,
    ConversationalPipeline, CustomPipeline, PipelineFactory, PipelineConfig, PipelineType,
)

# Generation config
from .generation import GenerationConfig, GenerationPresets

# Sampling
from .sampling import (
    SamplingConfig, SamplingStrategy, GreedySampling, MultinomialSampling,
    TopKSampling, TopPSampling, NucleusSampling, TypicalSampling,
    EtaSampling, EpsilonSampling, CombinedSampling, create_sampler,
)

# Stopping criteria
from .stopping import (
    StoppingCriterion, MaxLengthCriteria, MaxNewTokensCriteria,
    EosTokenCriteria, StringMatchCriteria, StopTokenIdsCriteria,
    MinLengthCriteria, MinNewTokensCriteria, TimeLimitCriteria,
    CustomFunctionCriteria, CompositeStoppingCriteria, StoppingCriteriaBuilder,
)

# Logits processors
from .logits_process import (
    LogitsProcessor, RepetitionPenaltyLogitsProcessor, FrequencyPenaltyLogitsProcessor,
    PresencePenaltyLogitsProcessor, TemperatureLogitsProcessor, TopKLogitsProcessor,
    TopPLogitsProcessor, MinLengthLogitsProcessor, MinNewTokensLogitsProcessor,
    NoRepeatNGramLogitsProcessor, NoBadWordsLogitsProcessor, EpsilonLogitsProcessor,
    EtaLogitsProcessor, EncoderRepetitionPenaltyLogitsProcessor, CustomLogitsProcessor,
    LogitsProcessorList,
)

# Beam search
from .beam_search import BeamSearchScorer, DiverseBeamSearchScorer, ConstrainedBeamSearchScorer, BeamHypothesis

# Streamer
from .streamer import TextIteratorStreamer, CallbackStreamer, AsyncStreamer, create_streamer

# Adapter
from .adapter import AdapterManager, AdapterInfo, AdapterType, AdapterStatus

# Loader
from .loader import ModelLoader, ModelLoadConfig, LoadFormat, LoadProgress, ProgressCallback

# Offload
from .offload import LayerOffloader, OffloadConfig, OffloadStrategy, OffloadTarget, LayerPlacement

# Memory
from .memory import MemoryTracker, MemoryEstimator, GarbageCollector, MemoryMappedLoader, MemorySnapshot

# Benchmark
from .benchmark import (
    BenchmarkResult, LatencyMeasurer, ThroughputBenchmarker,
    MemoryBenchmarker, run_full_benchmark,
)

# Health
from .health import ServiceHealthCheck, ModelHealthCheck, GPUHealthCheck, MemoryHealthCheck, HealthStatus, HealthCheckResult

# Metrics
from .metrics import Counter, Gauge, Histogram, MetricsRegistry, BackendMetrics

__all__ = [
    # Model management
    "ModelManager", "ModelInfo", "ModelStatus", "ModelRegistry",
    # Inference
    "InferenceEngine", "GenerationResult",
    # Tokenizer
    "TokenizerWrapper", "load_tokenizer", "count_message_tokens", "truncate_messages",
    # Server
    "create_app", "app",
    # Cache
    "PagedKVCache", "CacheBlock", "CacheEntry", "CacheStatus",
    "estimate_cache_size", "compute_optimal_num_blocks",
    # Scheduler
    "RequestScheduler", "InferenceRequest", "RequestPriority", "RequestStatus", "Batch",
    # Quantization
    "QuantizationConfig", "QuantizationType", "QuantizationManager",
    # Pipeline
    "Pipeline", "TextGenerationPipeline", "Text2TextGenerationPipeline",
    "ConversationalPipeline", "CustomPipeline", "PipelineFactory",
    "PipelineConfig", "PipelineType",
    # Generation
    "GenerationConfig", "GenerationPresets",
    # Sampling
    "SamplingConfig", "SamplingStrategy", "GreedySampling", "MultinomialSampling",
    "TopKSampling", "TopPSampling", "NucleusSampling", "TypicalSampling",
    "EtaSampling", "EpsilonSampling", "CombinedSampling", "create_sampler",
    # Stopping
    "StoppingCriterion", "MaxLengthCriteria", "MaxNewTokensCriteria",
    "EosTokenCriteria", "StringMatchCriteria", "StopTokenIdsCriteria",
    "MinLengthCriteria", "MinNewTokensCriteria", "TimeLimitCriteria",
    "CustomFunctionCriteria", "CompositeStoppingCriteria", "StoppingCriteriaBuilder",
    # Logits
    "LogitsProcessor", "RepetitionPenaltyLogitsProcessor", "FrequencyPenaltyLogitsProcessor",
    "PresencePenaltyLogitsProcessor", "TemperatureLogitsProcessor", "TopKLogitsProcessor",
    "TopPLogitsProcessor", "MinLengthLogitsProcessor", "MinNewTokensLogitsProcessor",
    "NoRepeatNGramLogitsProcessor", "NoBadWordsLogitsProcessor", "EpsilonLogitsProcessor",
    "EtaLogitsProcessor", "EncoderRepetitionPenaltyLogitsProcessor",
    "CustomLogitsProcessor", "LogitsProcessorList",
    # Beam search
    "BeamSearchScorer", "DiverseBeamSearchScorer", "ConstrainedBeamSearchScorer", "BeamHypothesis",
    # Streamer
    "TextIteratorStreamer", "CallbackStreamer", "AsyncStreamer", "create_streamer",
    # Adapter
    "AdapterManager", "AdapterInfo", "AdapterType", "AdapterStatus",
    # Loader
    "ModelLoader", "ModelLoadConfig", "LoadFormat", "LoadProgress", "ProgressCallback",
    # Offload
    "LayerOffloader", "OffloadConfig", "OffloadStrategy", "OffloadTarget", "LayerPlacement",
    # Memory
    "MemoryTracker", "MemoryEstimator", "GarbageCollector", "MemoryMappedLoader", "MemorySnapshot",
    # Benchmark
    "BenchmarkResult", "LatencyMeasurer", "ThroughputBenchmarker",
    "MemoryBenchmarker", "run_full_benchmark",
    # Health
    "ServiceHealthCheck", "ModelHealthCheck", "GPUHealthCheck", "MemoryHealthCheck",
    "HealthStatus", "HealthCheckResult",
    # Metrics
    "Counter", "Gauge", "Histogram", "MetricsRegistry", "BackendMetrics",
]
