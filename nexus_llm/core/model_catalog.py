"""Model catalog for Nexus-LLM - 39 supported models."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about a supported model."""
    id: str
    name: str
    hf_id: str
    category: str
    size: str
    params: str
    description: str
    model_type: str = "causal"  # causal or seq2seq
    recommended: bool = False
    min_ram_gb: int = 4


MODEL_CATALOG: Dict[str, ModelInfo] = {
    # === GPT-2 Family ===
    "gpt2": ModelInfo(
        id="gpt2", name="GPT-2 Small", hf_id="openai-community/gpt2",
        category="gpt2", size="124M", params="124M",
        description="Original GPT-2 small model, fast but limited",
        recommended=False, min_ram_gb=2,
    ),
    "gpt2-medium": ModelInfo(
        id="gpt2-medium", name="GPT-2 Medium", hf_id="openai-community/gpt2-medium",
        category="gpt2", size="355M", params="355M",
        description="GPT-2 Medium - good balance of speed and quality (DEFAULT)",
        model_type="causal", recommended=True, min_ram_gb=4,
    ),
    "gpt2-large": ModelInfo(
        id="gpt2-large", name="GPT-2 Large", hf_id="openai-community/gpt2-large",
        category="gpt2", size="774M", params="774M",
        description="GPT-2 Large - better quality, slower",
        min_ram_gb=6,
    ),
    "gpt2-xl": ModelInfo(
        id="gpt2-xl", name="GPT-2 XL", hf_id="openai-community/gpt2-xl",
        category="gpt2", size="1.5B", params="1.5B",
        description="GPT-2 XL - best in GPT-2 family",
        min_ram_gb=8,
    ),
    # === DialoGPT Family ===
    "dialogpt-small": ModelInfo(
        id="dialogpt-small", name="DialoGPT Small", hf_id="microsoft/DialoGPT-small",
        category="dialogpt", size="117M", params="117M",
        description="Microsoft conversational model, small",
        min_ram_gb=2,
    ),
    "dialogpt-medium": ModelInfo(
        id="dialogpt-medium", name="DialoGPT Medium", hf_id="microsoft/DialoGPT-medium",
        category="dialogpt", size="345M", params="345M",
        description="Microsoft conversational model, medium",
        min_ram_gb=4,
    ),
    "dialogpt-large": ModelInfo(
        id="dialogpt-large", name="DialoGPT Large", hf_id="microsoft/DialoGPT-large",
        category="dialogpt", size="762M", params="762M",
        description="Microsoft conversational model, large",
        min_ram_gb=6,
    ),
    # === Phi Family ===
    "phi-1": ModelInfo(
        id="phi-1", name="Phi-1", hf_id="microsoft/phi-1",
        category="phi", size="1.3B", params="1.3B",
        description="Microsoft Phi-1, code-focused",
        min_ram_gb=6,
    ),
    "phi-1.5": ModelInfo(
        id="phi-1.5", name="Phi-1.5", hf_id="microsoft/phi-1_5",
        category="phi", size="1.3B", params="1.3B",
        description="Microsoft Phi-1.5, improved reasoning",
        min_ram_gb=6,
    ),
    "phi-2": ModelInfo(
        id="phi-2", name="Phi-2", hf_id="microsoft/phi-2",
        category="phi", size="2.7B", params="2.7B",
        description="Microsoft Phi-2, excellent for its size",
        recommended=True, min_ram_gb=10,
    ),
    # === Pythia Family ===
    "pythia-70m": ModelInfo(
        id="pythia-70m", name="Pythia 70M", hf_id="EleutherAI/pythia-70m",
        category="pythia", size="70M", params="70M",
        description="EleutherAI small model, very fast",
        min_ram_gb=2,
    ),
    "pythia-160m": ModelInfo(
        id="pythia-160m", name="Pythia 160M", hf_id="EleutherAI/pythia-160m",
        category="pythia", size="160M", params="160M",
        description="EleutherAI small model",
        min_ram_gb=2,
    ),
    "pythia-410m": ModelInfo(
        id="pythia-410m", name="Pythia 410M", hf_id="EleutherAI/pythia-410m",
        category="pythia", size="410M", params="410M",
        description="EleutherAI medium model",
        min_ram_gb=4,
    ),
    "pythia-1b": ModelInfo(
        id="pythia-1b", name="Pythia 1B", hf_id="EleutherAI/pythia-1b",
        category="pythia", size="1B", params="1B",
        description="EleutherAI 1B parameter model",
        min_ram_gb=6,
    ),
    "pythia-1.4b": ModelInfo(
        id="pythia-1.4b", name="Pythia 1.4B", hf_id="EleutherAI/pythia-1.4b",
        category="pythia", size="1.4B", params="1.4B",
        description="EleutherAI 1.4B parameter model",
        min_ram_gb=8,
    ),
    "pythia-2.8b": ModelInfo(
        id="pythia-2.8b", name="Pythia 2.8B", hf_id="EleutherAI/pythia-2.8b",
        category="pythia", size="2.8B", params="2.8B",
        description="EleutherAI 2.8B parameter model",
        min_ram_gb=12,
    ),
    # === OPT Family ===
    "opt-125m": ModelInfo(
        id="opt-125m", name="OPT 125M", hf_id="facebook/opt-125m",
        category="opt", size="125M", params="125M",
        description="Meta OPT small model, very fast",
        min_ram_gb=2,
    ),
    "opt-350m": ModelInfo(
        id="opt-350m", name="OPT 350M", hf_id="facebook/opt-350m",
        category="opt", size="350M", params="350M",
        description="Meta OPT medium model",
        min_ram_gb=4,
    ),
    "opt-1.3b": ModelInfo(
        id="opt-1.3b", name="OPT 1.3B", hf_id="facebook/opt-1.3b",
        category="opt", size="1.3B", params="1.3B",
        description="Meta OPT 1.3B model",
        min_ram_gb=6,
    ),
    "opt-2.7b": ModelInfo(
        id="opt-2.7b", name="OPT 2.7B", hf_id="facebook/opt-2.7b",
        category="opt", size="2.7B", params="2.7B",
        description="Meta OPT 2.7B model",
        min_ram_gb=10,
    ),
    # === TinyLlama ===
    "tinyllama": ModelInfo(
        id="tinyllama", name="TinyLlama 1.1B", hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        category="llama", size="1.1B", params="1.1B",
        description="TinyLlama chat model, great for limited hardware",
        recommended=True, min_ram_gb=4,
    ),
    # === Qwen ===
    "qwen2.5-0.5b": ModelInfo(
        id="qwen2.5-0.5b", name="Qwen2.5 0.5B", hf_id="Qwen/Qwen2.5-0.5B",
        category="qwen", size="0.5B", params="0.5B",
        description="Alibaba Qwen2.5 small model",
        min_ram_gb=2,
    ),
    "qwen2.5-1.5b": ModelInfo(
        id="qwen2.5-1.5b", name="Qwen2.5 1.5B", hf_id="Qwen/Qwen2.5-1.5B",
        category="qwen", size="1.5B", params="1.5B",
        description="Alibaba Qwen2.5 medium model",
        min_ram_gb=6,
    ),
    "qwen2.5-3b": ModelInfo(
        id="qwen2.5-3b", name="Qwen2.5 3B", hf_id="Qwen/Qwen2.5-3B",
        category="qwen", size="3B", params="3B",
        description="Alibaba Qwen2.5 3B model",
        min_ram_gb=12,
    ),
    # === SmolLM ===
    "smollm-135m": ModelInfo(
        id="smollm-135m", name="SmolLM 135M", hf_id="HuggingFaceTB/SmolLM-135M",
        category="smollm", size="135M", params="135M",
        description="HuggingFace tiny model, very fast",
        min_ram_gb=2,
    ),
    "smollm-360m": ModelInfo(
        id="smollm-360m", name="SmolLM 360M", hf_id="HuggingFaceTB/SmolLM-360M",
        category="smollm", size="360M", params="360M",
        description="HuggingFace small model",
        min_ram_gb=2,
    ),
    "smollm-1.7b": ModelInfo(
        id="smollm-1.7b", name="SmolLM 1.7B", hf_id="HuggingFaceTB/SmolLM-1.7B",
        category="smollm", size="1.7B", params="1.7B",
        description="HuggingFace medium model",
        min_ram_gb=6,
    ),
    # === Gemma ===
    "gemma-2b": ModelInfo(
        id="gemma-2b", name="Gemma 2B", hf_id="google/gemma-2b",
        category="gemma", size="2B", params="2B",
        description="Google Gemma 2B model",
        min_ram_gb=8,
    ),
    "gemma-2b-it": ModelInfo(
        id="gemma-2b-it", name="Gemma 2B IT", hf_id="google/gemma-2b-it",
        category="gemma", size="2B", params="2B",
        description="Google Gemma 2B instruction-tuned",
        recommended=True, min_ram_gb=8,
    ),
    # === Mamba ===
    "mamba-130m": ModelInfo(
        id="mamba-130m", name="Mamba 130M", hf_id="state-spaces/mamba-130m",
        category="mamba", size="130M", params="130M",
        description="State-space model, very fast inference",
        min_ram_gb=2,
    ),
    "mamba-370m": ModelInfo(
        id="mamba-370m", name="Mamba 370M", hf_id="state-spaces/mamba-370m",
        category="mamba", size="370M", params="370M",
        description="State-space model, medium size",
        min_ram_gb=4,
    ),
    "mamba-790m": ModelInfo(
        id="mamba-790m", name="Mamba 790M", hf_id="state-spaces/mamba-790m",
        category="mamba", size="790M", params="790M",
        description="State-space model, large",
        min_ram_gb=6,
    ),
    # === StableLM ===
    "stablelm-2-1.6b": ModelInfo(
        id="stablelm-2-1.6b", name="StableLM 2 1.6B", hf_id="stabilityai/stablelm-2-1_6b",
        category="stablelm", size="1.6B", params="1.6B",
        description="StabilityAI StableLM 2",
        min_ram_gb=6,
    ),
    "stablelm-2-zephyr": ModelInfo(
        id="stablelm-2-zephyr", name="StableLM 2 Zephyr", hf_id="stabilityai/stablelm-2-zephyr-1_6b",
        category="stablelm", size="1.6B", params="1.6B",
        description="StableLM 2 Zephyr chat model",
        recommended=True, min_ram_gb=6,
    ),
    # === Bloom ===
    "bloom-560m": ModelInfo(
        id="bloom-560m", name="BLOOM 560M", hf_id="bigscience/bloom-560m",
        category="bloom", size="560M", params="560M",
        description="BigScience BLOOM multilingual model",
        min_ram_gb=4,
    ),
    "bloom-1b1": ModelInfo(
        id="bloom-1b1", name="BLOOM 1.1B", hf_id="bigscience/bloom-1b1",
        category="bloom", size="1.1B", params="1.1B",
        description="BigScience BLOOM 1.1B multilingual",
        min_ram_gb=6,
    ),
    "bloom-1b7": ModelInfo(
        id="bloom-1b7", name="BLOOM 1.7B", hf_id="bigscience/bloom-1b7",
        category="bloom", size="1.7B", params="1.7B",
        description="BigScience BLOOM 1.7B multilingual",
        min_ram_gb=8,
    ),
    # === FLAN-T5 ===
    "flan-t5-small": ModelInfo(
        id="flan-t5-small", name="FLAN-T5 Small", hf_id="google/flan-t5-small",
        category="flan-t5", size="80M", params="80M",
        description="Google FLAN-T5 instruction-following model",
        model_type="seq2seq", min_ram_gb=2,
    ),
    "flan-t5-base": ModelInfo(
        id="flan-t5-base", name="FLAN-T5 Base", hf_id="google/flan-t5-base",
        category="flan-t5", size="250M", params="250M",
        description="Google FLAN-T5 base instruction model",
        model_type="seq2seq", recommended=True, min_ram_gb=4,
    ),
    "flan-t5-large": ModelInfo(
        id="flan-t5-large", name="FLAN-T5 Large", hf_id="google/flan-t5-large",
        category="flan-t5", size="780M", params="780M",
        description="Google FLAN-T5 large instruction model",
        model_type="seq2seq", min_ram_gb=6,
    ),
}


def get_model_info(model_id: str) -> ModelInfo:
    """Get model info by ID. Raises ModelNotFoundError if not found."""
    from nexus_llm.core.exceptions import ModelNotFoundError
    if model_id not in MODEL_CATALOG:
        available = ", ".join(sorted(MODEL_CATALOG.keys()))
        raise ModelNotFoundError(
            f"Model '{model_id}' not found. Available models: {available}"
        )
    return MODEL_CATALOG[model_id]


def list_models(category: Optional[str] = None) -> List[ModelInfo]:
    """List all available models, optionally filtered by category."""
    models = list(MODEL_CATALOG.values())
    if category:
        models = [m for m in models if m.category == category]
    return sorted(models, key=lambda m: m.name)


def list_categories() -> List[str]:
    """List all model categories."""
    return sorted(set(m.category for m in MODEL_CATALOG.values()))


def get_recommended_models() -> List[ModelInfo]:
    """Get recommended models."""
    return [m for m in MODEL_CATALOG.values() if m.recommended]
