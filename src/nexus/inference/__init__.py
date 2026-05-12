"""
Inference Package Init
======================
Complete inference infrastructure for Nexus.

Modules:
    - generator: Autoregressive text generation (temperature, top-k, top-p, beam search)
    - quantize: Model quantization (GPTQ, AWQ)
    - server: OpenAI-compatible FastAPI inference server
    - kv_cache: KV cache management (standard, quantized, sliding window, cross-layer)
    - speculative: Speculative decoding (draft model, self-spec, Medusa, EAGLE)
    - multi_token_pred: Multi-token prediction heads and inference engine
"""

from .generator import TextGenerator, GenerationConfig, GenerationResult
from .quantize import Quantizer, GPTQQuantizer, AWQQuantizer
from .server import InferenceServer
from .kv_cache import (
    StandardKVCache,
    KVCacheQuantizer,
    QuantizedKVCache,
    SlidingWindowKVCache,
    CrossLayerKVCache,
    MultiTokenPredictionCache,
    CacheManager,
)
from .speculative import (
    SpeculativeDecodingStats,
    DraftModelSpeculativeDecoder,
    SelfSpeculativeDecoder,
    MedusaHead,
    MedusaTree,
    MedusaTrainer,
    EAGLE,
    EAGLEDecoder,
    create_speculative_decoder,
)
from .multi_token_pred import (
    MultiTokenPredictionHead,
    SharedTrunkMultiTokenHead,
    MultiTokenTrainingWrapper,
    MultiTokenInferenceEngine,
    NgramFallbackPredictor,
)

__all__ = [
    # Core generation
    "TextGenerator", "GenerationConfig", "GenerationResult",
    "Quantizer", "GPTQQuantizer", "AWQQuantizer",
    "InferenceServer",
    # KV Cache
    "StandardKVCache",
    "KVCacheQuantizer",
    "QuantizedKVCache",
    "SlidingWindowKVCache",
    "CrossLayerKVCache",
    "MultiTokenPredictionCache",
    "CacheManager",
    # Speculative Decoding
    "SpeculativeDecodingStats",
    "DraftModelSpeculativeDecoder",
    "SelfSpeculativeDecoder",
    "MedusaHead",
    "MedusaTree",
    "MedusaTrainer",
    "EAGLE",
    "EAGLEDecoder",
    "create_speculative_decoder",
    # Multi-Token Prediction
    "MultiTokenPredictionHead",
    "SharedTrunkMultiTokenHead",
    "MultiTokenTrainingWrapper",
    "MultiTokenInferenceEngine",
    "NgramFallbackPredictor",
]
