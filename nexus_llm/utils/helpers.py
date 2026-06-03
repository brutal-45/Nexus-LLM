"""Utility helper functions for Nexus-LLM.

Provides common formatting, validation, and model-management helpers used
across the project.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.core.exceptions import ModelNotFoundError, ModelLoadError
from nexus_llm.core.model_catalog import MODEL_CATALOG, ModelInfo, get_model_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Byte / time formatting
# ---------------------------------------------------------------------------

def format_bytes(n: int) -> str:
    """Convert a byte count to a human-readable string.

    Args:
        n: Number of bytes.

    Returns:
        Formatted string, e.g. "1.23 GiB".
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.2f} EiB"


def format_time(seconds: float) -> str:
    """Convert seconds to a human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string, e.g. "2h 15m 30s" or "45.2s".
    """
    if seconds < 0:
        return "—"

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts: List[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs and not days:  # skip seconds when showing days
        parts.append(f"{secs}s")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_len: int = 100, suffix: str = "…") -> str:
    """Truncate text to *max_len* characters, appending *suffix* if needed.

    Args:
        text: The input string.
        max_len: Maximum length of the returned string (including suffix).
        suffix: Suffix to append when truncating.

    Returns:
        Truncated string.
    """
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def count_words(text: str) -> int:
    """Count the number of words in *text*.

    Handles common edge cases such as excessive whitespace and punctuation.
    """
    # Strip markdown / code fences for a more accurate count
    cleaned = re.sub(r"[`#*_~\[\]()]", " ", text)
    return len(cleaned.split())


# ---------------------------------------------------------------------------
# Model validation and discovery
# ---------------------------------------------------------------------------

def validate_model_name(name: str) -> ModelInfo:
    """Validate a model name against the catalog.

    Args:
        name: Model ID (e.g. "gpt2-medium") or HuggingFace ID
              (e.g. "openai-community/gpt2-medium").

    Returns:
        The matching ``ModelInfo``.

    Raises:
        ModelNotFoundError: If the model is not in the catalog.
    """
    # Try direct lookup by short ID
    if name in MODEL_CATALOG:
        return MODEL_CATALOG[name]

    # Try matching by HuggingFace ID
    for info in MODEL_CATALOG.values():
        if info.hf_id == name:
            return info

    # Fuzzy match: try to find a partial match
    name_lower = name.lower()
    candidates = [
        info for info in MODEL_CATALOG.values()
        if name_lower in info.id.lower() or name_lower in info.hf_id.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]

    available = ", ".join(sorted(MODEL_CATALOG.keys()))
    raise ModelNotFoundError(
        f"Model '{name}' not found in catalog. Available models: {available}"
    )


def get_available_models(
    category: Optional[str] = None,
    recommended_only: bool = False,
    max_ram_gb: Optional[int] = None,
) -> List[ModelInfo]:
    """List available models, optionally filtered.

    Args:
        category:        Filter by model category (e.g. "gpt2", "phi").
        recommended_only: Return only recommended models.
        max_ram_gb:       Maximum RAM requirement in GB.

    Returns:
        Sorted list of ``ModelInfo`` objects.
    """
    models = list(MODEL_CATALOG.values())

    if category:
        models = [m for m in models if m.category == category]

    if recommended_only:
        models = [m for m in models if m.recommended]

    if max_ram_gb is not None:
        models = [m for m in models if m.min_ram_gb <= max_ram_gb]

    return sorted(models, key=lambda m: m.name)


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    """Download a model from HuggingFace Hub.

    Only downloads the model files; does not load the model into memory.

    Args:
        model_id: Short model ID from the catalog or a HuggingFace repo ID.
        cache_dir: Directory to cache downloaded files.
        revision: Optional Git revision (branch, tag, or commit hash).

    Returns:
        The local path where the model was downloaded.

    Raises:
        ModelNotFoundError: If *model_id* is not in the catalog.
        ModelLoadError: If the download fails.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelLoadError(
            "The 'huggingface_hub' package is required for downloading models. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    # Resolve model ID → HuggingFace repo ID
    try:
        info = validate_model_name(model_id)
        hf_id = info.hf_id
    except ModelNotFoundError:
        # Allow direct HuggingFace IDs not in the catalog
        hf_id = model_id

    logger.info("Downloading model %s (hf_id=%s) …", model_id, hf_id)

    kwargs: Dict[str, Any] = {"repo_id": hf_id}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision

    try:
        local_path = snapshot_download(**kwargs)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to download model '{hf_id}': {exc}"
        ) from exc

    logger.info("Model %s downloaded to %s", model_id, local_path)
    return str(local_path)
