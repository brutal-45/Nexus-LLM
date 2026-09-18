"""Helper Utilities - Common utility functions for the Nexus-LLM application."""

import re
import time
from typing import List, Dict, Optional


def format_bytes(bytes_value: int) -> str:
    """
    Format a byte value into a human-readable string.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB", "256 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format a time value in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2m 30s", "1h 15m")
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with a suffix.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append when truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """
    Count the number of words in a text string.

    Args:
        text: Input text

    Returns:
        Word count
    """
    return len(text.split())


def validate_model_name(model_name: str) -> bool:
    """
    Validate a HuggingFace model name.

    Args:
        model_name: Model name to validate

    Returns:
        True if valid format
    """
    # HuggingFace model names: "org/model-name" or just "model-name"
    pattern = r"^[a-zA-Z0-9_\-]+(/[a-zA-Z0-9_\-.]+)?$"
    return bool(re.match(pattern, model_name))


def get_available_models() -> List[Dict[str, str]]:
    """
    Get a list of recommended models that can be used.

    Returns:
        List of model info dicts with name, description, and size
    """
    return [
        # GPT-2 Family
        {"name": "gpt2", "description": "GPT-2 Small (124M) - Fast, testing", "size": "~500MB"},
        {"name": "gpt2-medium", "description": "GPT-2 Medium (355M) - 3x bigger [DEFAULT]", "size": "~1.4GB"},
        {"name": "gpt2-large", "description": "GPT-2 Large (774M) - 6x bigger", "size": "~3GB"},
        {"name": "gpt2-xl", "description": "GPT-2 XL (1.5B) - 12x bigger", "size": "~6GB"},
        # GPT-Neo / GPT-J
        {"name": "EleutherAI/gpt-neo-125M", "description": "GPT-Neo 125M - Open source alternative", "size": "~500MB"},
        {"name": "EleutherAI/gpt-neo-1.3B", "description": "GPT-Neo 1.3B - 10x bigger, great quality", "size": "~5GB"},
        {"name": "EleutherAI/gpt-neo-2.7B", "description": "GPT-Neo 2.7B - 22x bigger, strong", "size": "~10GB"},
        {"name": "EleutherAI/gpt-j-6b", "description": "GPT-J 6B - Near GPT-3 quality", "size": "~22GB"},
        # LLaMA / OpenLLaMA
        {"name": "openlm-research/open_llama_3b", "description": "OpenLLaMA 3B - Open LLaMA reproduction", "size": "~13GB"},
        {"name": "openlm-research/open_llama_7b", "description": "OpenLLaMA 7B - Full LLaMA reproduction", "size": "~26GB"},
        # Phi
        {"name": "microsoft/phi-1", "description": "Phi-1 (1.3B) - Powerful for coding", "size": "~5GB"},
        {"name": "microsoft/phi-1_5", "description": "Phi-1.5 (1.3B) - Improved reasoning", "size": "~5GB"},
        {"name": "microsoft/phi-2", "description": "Phi-2 (2.7B) - Excellent reasoning", "size": "~10GB"},
        # Pythia
        {"name": "EleutherAI/pythia-70m", "description": "Pythia 70M - Ultra-fast tiny model", "size": "~280MB"},
        {"name": "EleutherAI/pythia-160m", "description": "Pythia 160M - Small & fast", "size": "~640MB"},
        {"name": "EleutherAI/pythia-410m", "description": "Pythia 410M - Good balance", "size": "~1.6GB"},
        {"name": "EleutherAI/pythia-1b", "description": "Pythia 1B - Strong performance", "size": "~4GB"},
        {"name": "EleutherAI/pythia-1.4b", "description": "Pythia 1.4B - Great for most tasks", "size": "~5.5GB"},
        {"name": "EleutherAI/pythia-2.8b", "description": "Pythia 2.8B - Powerful open model", "size": "~11GB"},
        # BLOOM Multilingual
        {"name": "bigscience/bloom-560m", "description": "BLOOM 560M - 46 languages", "size": "~2GB"},
        {"name": "bigscience/bloom-1b1", "description": "BLOOM 1.1B - Bigger multilingual", "size": "~4GB"},
        {"name": "bigscience/bloom-1b7", "description": "BLOOM 1.7B - Strong multilingual", "size": "~7GB"},
        {"name": "bigscience/bloom-3b", "description": "BLOOM 3B - Large multilingual", "size": "~12GB"},
        # OPT
        {"name": "facebook/opt-125m", "description": "OPT 125M - Meta's open, small", "size": "~500MB"},
        {"name": "facebook/opt-350m", "description": "OPT 350M - Speed + quality balance", "size": "~1.3GB"},
        {"name": "facebook/opt-1.3b", "description": "OPT 1.3B - GPT-3 class, open", "size": "~5GB"},
        {"name": "facebook/opt-2.7b", "description": "OPT 2.7B - Near GPT-3 quality", "size": "~10GB"},
        {"name": "facebook/opt-6.7b", "description": "OPT 6.7B - Excellent quality", "size": "~25GB"},
        # Chat Optimized
        {"name": "microsoft/DialoGPT-small", "description": "DialoGPT Small - Chat, fast", "size": "~470MB"},
        {"name": "microsoft/DialoGPT-medium", "description": "DialoGPT Medium - Chat, better", "size": "~1.3GB"},
        {"name": "microsoft/DialoGPT-large", "description": "DialoGPT Large - Best for chat", "size": "~3GB"},
        {"name": "facebook/blenderbot-400M-distill", "description": "BlenderBot 400M - Empathetic chat", "size": "~1.5GB"},
        {"name": "facebook/blenderbot-1B-distill", "description": "BlenderBot 1B - Knowledgeable chat", "size": "~4GB"},
        # Tiny
        {"name": "sshleifer/tiny-gpt2", "description": "Tiny GPT-2 (5M) - Instant testing", "size": "~20MB"},
        {"name": "distilbert/distilgpt2", "description": "DistilGPT-2 (82M) - 2x faster", "size": "~330MB"},
        # Code
        {"name": "Salesforce/codegen-350M-mono", "description": "CodeGen 350M - Python code gen", "size": "~1.4GB"},
        {"name": "Salesforce/codegen-2B-mono", "description": "CodeGen 2B - Powerful code gen", "size": "~8GB"},
        {"name": "bigcode/tiny_starcoder_py", "description": "Tiny StarCoder - Fast code model", "size": "~650MB"},
        {"name": "bigcode/starcoderbase-1b", "description": "StarCoder 1B - 80+ languages", "size": "~4GB"},
    ]


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.

    Args:
        text: Text containing markdown code blocks

    Returns:
        List of dicts with 'language' and 'code' keys
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"language": lang or "text", "code": code.strip()} for lang, code in matches]


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially harmful characters.

    Args:
        text: Raw user input

    Returns:
        Sanitized text
    """
    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Limit length
    if len(text) > 10000:
        text = text[:10000]
    return text.strip()
