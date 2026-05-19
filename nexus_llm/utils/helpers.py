"""Helper functions: MODEL_CATALOG with 39 models across 10 categories, utility functions."""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # GPT-2 Family
    "gpt2": {
        "name": "GPT-2 Small",
        "category": "gpt2",
        "provider": "OpenAI",
        "parameters": "124M",
        "param_count": 124_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "gpt2",
        "description": "Original GPT-2 small model with 124M parameters.",
    },
    "gpt2-medium": {
        "name": "GPT-2 Medium",
        "category": "gpt2",
        "provider": "OpenAI",
        "parameters": "355M",
        "param_count": 355_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "gpt2-medium",
        "description": "GPT-2 medium model with 355M parameters.",
    },
    "gpt2-large": {
        "name": "GPT-2 Large",
        "category": "gpt2",
        "provider": "OpenAI",
        "parameters": "774M",
        "param_count": 774_000_000,
        "hidden_size": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "gpt2-large",
        "description": "GPT-2 large model with 774M parameters.",
    },
    "gpt2-xl": {
        "name": "GPT-2 XL",
        "category": "gpt2",
        "provider": "OpenAI",
        "parameters": "1.5B",
        "param_count": 1_500_000_000,
        "hidden_size": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "gpt2-xl",
        "description": "GPT-2 extra-large model with 1.5B parameters.",
    },

    # GPT-Neo / GPT-J Family
    "EleutherAI/gpt-neo-125M": {
        "name": "GPT-Neo 125M",
        "category": "gpt_neo",
        "provider": "EleutherAI",
        "parameters": "125M",
        "param_count": 125_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 2048,
        "vocab_size": 50257,
        "hf_id": "EleutherAI/gpt-neo-125M",
        "description": "GPT-Neo 125M, an open-source GPT-2 equivalent.",
    },
    "EleutherAI/gpt-neo-1.3B": {
        "name": "GPT-Neo 1.3B",
        "category": "gpt_neo",
        "provider": "EleutherAI",
        "parameters": "1.3B",
        "param_count": 1_300_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 50257,
        "hf_id": "EleutherAI/gpt-neo-1.3B",
        "description": "GPT-Neo 1.3B, larger open-source language model.",
    },
    "EleutherAI/gpt-neo-2.7B": {
        "name": "GPT-Neo 2.7B",
        "category": "gpt_neo",
        "provider": "EleutherAI",
        "parameters": "2.7B",
        "param_count": 2_700_000_000,
        "hidden_size": 2560,
        "num_layers": 32,
        "num_heads": 20,
        "max_length": 2048,
        "vocab_size": 50257,
        "hf_id": "EleutherAI/gpt-neo-2.7B",
        "description": "GPT-Neo 2.7B, one of the largest Neo models.",
    },
    "EleutherAI/gpt-j-6B": {
        "name": "GPT-J 6B",
        "category": "gpt_neo",
        "provider": "EleutherAI",
        "parameters": "6B",
        "param_count": 6_000_000_000,
        "hidden_size": 4096,
        "num_layers": 28,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 50400,
        "hf_id": "EleutherAI/gpt-j-6B",
        "description": "GPT-J 6B, a powerful open-source model with parallel attention.",
    },

    # LLaMA Family
    "meta-llama/Llama-2-7b-hf": {
        "name": "LLaMA 2 7B",
        "category": "llama",
        "provider": "Meta",
        "parameters": "7B",
        "param_count": 7_000_000_000,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "max_length": 4096,
        "vocab_size": 32000,
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "description": "LLaMA 2 7B base model from Meta.",
    },
    "meta-llama/Llama-2-13b-hf": {
        "name": "LLaMA 2 13B",
        "category": "llama",
        "provider": "Meta",
        "parameters": "13B",
        "param_count": 13_000_000_000,
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "max_length": 4096,
        "vocab_size": 32000,
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "description": "LLaMA 2 13B base model from Meta.",
    },

    # Phi Family
    "microsoft/phi-1": {
        "name": "Phi-1",
        "category": "phi",
        "provider": "Microsoft",
        "parameters": "1.3B",
        "param_count": 1_300_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 50257,
        "hf_id": "microsoft/phi-1",
        "description": "Phi-1, a small but capable code model from Microsoft.",
    },
    "microsoft/phi-1_5": {
        "name": "Phi-1.5",
        "category": "phi",
        "provider": "Microsoft",
        "parameters": "1.3B",
        "param_count": 1_300_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 50257,
        "hf_id": "microsoft/phi-1_5",
        "description": "Phi-1.5, improved general language model from Microsoft.",
    },
    "microsoft/phi-2": {
        "name": "Phi-2",
        "category": "phi",
        "provider": "Microsoft",
        "parameters": "2.7B",
        "param_count": 2_700_000_000,
        "hidden_size": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 51200,
        "hf_id": "microsoft/phi-2",
        "description": "Phi-2, a powerful small model from Microsoft.",
    },

    # Pythia Family
    "EleutherAI/pythia-70m": {
        "name": "Pythia 70M",
        "category": "pythia",
        "provider": "EleutherAI",
        "parameters": "70M",
        "param_count": 70_000_000,
        "hidden_size": 512,
        "num_layers": 6,
        "num_heads": 8,
        "max_length": 2048,
        "vocab_size": 50304,
        "hf_id": "EleutherAI/pythia-70m",
        "description": "Pythia 70M, smallest in the Pythia suite.",
    },
    "EleutherAI/pythia-160m": {
        "name": "Pythia 160M",
        "category": "pythia",
        "provider": "EleutherAI",
        "parameters": "160M",
        "param_count": 160_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 2048,
        "vocab_size": 50304,
        "hf_id": "EleutherAI/pythia-160m",
        "description": "Pythia 160M, small-scale research model.",
    },
    "EleutherAI/pythia-410m": {
        "name": "Pythia 410M",
        "category": "pythia",
        "provider": "EleutherAI",
        "parameters": "410M",
        "param_count": 410_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 50304,
        "hf_id": "EleutherAI/pythia-410m",
        "description": "Pythia 410M, medium-scale research model.",
    },
    "EleutherAI/pythia-1b": {
        "name": "Pythia 1B",
        "category": "pythia",
        "provider": "EleutherAI",
        "parameters": "1B",
        "param_count": 1_000_000_000,
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 8,
        "max_length": 2048,
        "vocab_size": 50304,
        "hf_id": "EleutherAI/pythia-1b",
        "description": "Pythia 1B, billion-parameter research model.",
    },

    # BLOOM Family
    "bigscience/bloom-560m": {
        "name": "BLOOM 560M",
        "category": "bloom",
        "provider": "BigScience",
        "parameters": "560M",
        "param_count": 560_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 250680,
        "hf_id": "bigscience/bloom-560m",
        "description": "BLOOM 560M, multilingual model from BigScience.",
    },
    "bigscience/bloom-1b1": {
        "name": "BLOOM 1.1B",
        "category": "bloom",
        "provider": "BigScience",
        "parameters": "1.1B",
        "param_count": 1_100_000_000,
        "hidden_size": 1536,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 250680,
        "hf_id": "bigscience/bloom-1b1",
        "description": "BLOOM 1.1B, multilingual language model.",
    },
    "bigscience/bloom-1b7": {
        "name": "BLOOM 1.7B",
        "category": "bloom",
        "provider": "BigScience",
        "parameters": "1.7B",
        "param_count": 1_700_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 250680,
        "hf_id": "bigscience/bloom-1b7",
        "description": "BLOOM 1.7B, mid-scale multilingual model.",
    },
    "bigscience/bloom-3b": {
        "name": "BLOOM 3B",
        "category": "bloom",
        "provider": "BigScience",
        "parameters": "3B",
        "param_count": 3_000_000_000,
        "hidden_size": 2560,
        "num_layers": 30,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 250680,
        "hf_id": "bigscience/bloom-3b",
        "description": "BLOOM 3B, larger multilingual model.",
    },

    # OPT Family
    "facebook/opt-125m": {
        "name": "OPT 125M",
        "category": "opt",
        "provider": "Meta",
        "parameters": "125M",
        "param_count": 125_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 2048,
        "vocab_size": 50272,
        "hf_id": "facebook/opt-125m",
        "description": "OPT 125M, smallest in the OPT family.",
    },
    "facebook/opt-350m": {
        "name": "OPT 350M",
        "category": "opt",
        "provider": "Meta",
        "parameters": "350M",
        "param_count": 350_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 50272,
        "hf_id": "facebook/opt-350m",
        "description": "OPT 350M, small open pre-trained transformer.",
    },
    "facebook/opt-1.3b": {
        "name": "OPT 1.3B",
        "category": "opt",
        "provider": "Meta",
        "parameters": "1.3B",
        "param_count": 1_300_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 50272,
        "hf_id": "facebook/opt-1.3b",
        "description": "OPT 1.3B, mid-range open pre-trained transformer.",
    },
    "facebook/opt-2.7b": {
        "name": "OPT 2.7B",
        "category": "opt",
        "provider": "Meta",
        "parameters": "2.7B",
        "param_count": 2_700_000_000,
        "hidden_size": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 50272,
        "hf_id": "facebook/opt-2.7b",
        "description": "OPT 2.7B, larger open pre-trained transformer.",
    },

    # DialoGPT Family
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT Medium",
        "category": "dialogpt",
        "provider": "Microsoft",
        "parameters": "355M",
        "param_count": 355_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "microsoft/DialoGPT-medium",
        "description": "DialoGPT Medium, conversational AI model.",
    },
    "microsoft/DialoGPT-large": {
        "name": "DialoGPT Large",
        "category": "dialogpt",
        "provider": "Microsoft",
        "parameters": "774M",
        "param_count": 774_000_000,
        "hidden_size": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "max_length": 1024,
        "vocab_size": 50257,
        "hf_id": "microsoft/DialoGPT-large",
        "description": "DialoGPT Large, larger conversational AI model.",
    },

    # Qwen Family
    "Qwen/Qwen1.5-0.5B": {
        "name": "Qwen 1.5 0.5B",
        "category": "qwen",
        "provider": "Alibaba",
        "parameters": "0.5B",
        "param_count": 500_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 32768,
        "vocab_size": 151936,
        "hf_id": "Qwen/Qwen1.5-0.5B",
        "description": "Qwen 1.5 0.5B, compact multilingual model.",
    },
    "Qwen/Qwen1.5-1.8B": {
        "name": "Qwen 1.5 1.8B",
        "category": "qwen",
        "provider": "Alibaba",
        "parameters": "1.8B",
        "param_count": 1_800_000_000,
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "max_length": 32768,
        "vocab_size": 151936,
        "hf_id": "Qwen/Qwen1.5-1.8B",
        "description": "Qwen 1.5 1.8B, capable multilingual model.",
    },

    # TinyLlama
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "name": "TinyLlama 1.1B Chat",
        "category": "tinyllama",
        "provider": "TinyLlama",
        "parameters": "1.1B",
        "param_count": 1_100_000_000,
        "hidden_size": 2048,
        "num_layers": 22,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 32000,
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B Chat, compact chat model.",
    },

    # Fine-tuned / Specialized
    "abhishek/llama-2-7b-finetuned": {
        "name": "LLaMA 2 7B Fine-tuned",
        "category": "finetuned",
        "provider": "Community",
        "parameters": "7B",
        "param_count": 7_000_000_000,
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "max_length": 4096,
        "vocab_size": 32000,
        "hf_id": "abhishek/llama-2-7b-finetuned",
        "description": "Fine-tuned LLaMA 2 7B for specific tasks.",
    },

    # Code Models
    "bigcode/tiny_starcoder_py": {
        "name": "Tiny StarCoder",
        "category": "code",
        "provider": "BigCode",
        "parameters": "164M",
        "param_count": 164_000_000,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "max_length": 8192,
        "vocab_size": 49152,
        "hf_id": "bigcode/tiny_starcoder_py",
        "description": "Tiny StarCoder, compact code generation model.",
    },
    "Salesforce/codegen-350M-mono": {
        "name": "CodeGen 350M Mono",
        "category": "code",
        "provider": "Salesforce",
        "parameters": "350M",
        "param_count": 350_000_000,
        "hidden_size": 1024,
        "num_layers": 20,
        "num_heads": 16,
        "max_length": 2048,
        "vocab_size": 51200,
        "hf_id": "Salesforce/codegen-350M-mono",
        "description": "CodeGen 350M, Python-focused code generation model.",
    },
    "Salesforce/codegen-2B-mono": {
        "name": "CodeGen 2B Mono",
        "category": "code",
        "provider": "Salesforce",
        "parameters": "2B",
        "param_count": 2_000_000_000,
        "hidden_size": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "max_length": 2048,
        "vocab_size": 51200,
        "hf_id": "Salesforce/codegen-2B-mono",
        "description": "CodeGen 2B, larger Python-focused code generation model.",
    },
}

# Category mapping
CATEGORIES = {
    "gpt2": "GPT-2 Family",
    "gpt_neo": "GPT-Neo / GPT-J Family",
    "llama": "LLaMA Family",
    "phi": "Phi Family",
    "pythia": "Pythia Family",
    "bloom": "BLOOM Family",
    "opt": "OPT Family",
    "dialogpt": "DialoGPT Family",
    "qwen": "Qwen Family",
    "tinyllama": "TinyLlama",
    "finetuned": "Fine-tuned Models",
    "code": "Code Models",
}


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific model.

    Args:
        model_id: Model identifier (e.g., "gpt2", "EleutherAI/gpt-neo-125M").

    Returns:
        Model info dictionary, or None if not found.
    """
    return MODEL_CATALOG.get(model_id)


def list_models_by_category(category: str) -> List[Dict[str, Any]]:
    """List all models in a given category.

    Args:
        category: Category name.

    Returns:
        List of model info dictionaries.
    """
    return [
        {"id": model_id, **info}
        for model_id, info in MODEL_CATALOG.items()
        if info.get("category") == category
    ]


def list_all_categories() -> Dict[str, str]:
    """List all model categories with their display names.

    Returns:
        Dictionary mapping category keys to display names.
    """
    return dict(CATEGORIES)


def search_models(query: str) -> List[Dict[str, Any]]:
    """Search models by name, provider, or description.

    Args:
        query: Search query string.

    Returns:
        List of matching model info dictionaries.
    """
    query_lower = query.lower()
    results = []
    for model_id, info in MODEL_CATALOG.items():
        searchable = f"{model_id} {info.get('name', '')} {info.get('provider', '')} {info.get('description', '')}".lower()
        if query_lower in searchable:
            results.append({"id": model_id, **info})
    return results


def get_models_by_size(min_params: int = 0, max_params: int = float("inf")) -> List[Dict[str, Any]]:
    """Filter models by parameter count.

    Args:
        min_params: Minimum parameter count.
        max_params: Maximum parameter count.

    Returns:
        List of models within the specified parameter range.
    """
    return [
        {"id": model_id, **info}
        for model_id, info in MODEL_CATALOG.items()
        if min_params <= info.get("param_count", 0) <= max_params
    ]


def format_model_info(model_id: str) -> str:
    """Format model information as a readable string.

    Args:
        model_id: Model identifier.

    Returns:
        Formatted string with model details.
    """
    info = get_model_info(model_id)
    if info is None:
        return f"Model '{model_id}' not found in catalog."

    lines = [
        f"Model: {info.get('name', model_id)}",
        f"  ID: {model_id}",
        f"  Category: {CATEGORIES.get(info.get('category', ''), info.get('category', ''))}",
        f"  Provider: {info.get('provider', 'Unknown')}",
        f"  Parameters: {info.get('parameters', 'Unknown')}",
        f"  Hidden Size: {info.get('hidden_size', 'Unknown')}",
        f"  Layers: {info.get('num_layers', 'Unknown')}",
        f"  Heads: {info.get('num_heads', 'Unknown')}",
        f"  Max Length: {info.get('max_length', 'Unknown')}",
        f"  Vocab Size: {info.get('vocab_size', 'Unknown')}",
        f"  Description: {info.get('description', 'No description available.')}",
    ]
    return "\n".join(lines)
