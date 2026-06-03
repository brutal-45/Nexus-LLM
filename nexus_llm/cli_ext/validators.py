"""
CLI Argument Validators for Nexus-LLM

Provides validation functions for CLI arguments including model names,
file paths, URLs, ports, numeric ranges, and file extensions.
Each validator returns a normalized value or raises ValueError.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Collection, List, Optional, Sequence, Union
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Model name validator
# ---------------------------------------------------------------------------

def validate_model_name(value: str) -> str:
    """Validate a model name or HuggingFace model ID.

    Accepts:
    - HuggingFace IDs: "org/model-name", "model-name"
    - Local paths: "/path/to/model", "./model"
    - Predefined aliases: "llama-7b", "mistral-7b", etc.

    Args:
        value: Model name string.

    Returns:
        Normalized model name.

    Raises:
        ValueError: If the model name is invalid.
    """
    if not value or not value.strip():
        raise ValueError("Model name cannot be empty")

    value = value.strip()

    # Allow local paths
    if value.startswith("/") or value.startswith("./") or value.startswith("~"):
        path = Path(value).expanduser()
        if not path.exists():
            raise ValueError(f"Model path does not exist: {value}")
        return str(path)

    # Validate HuggingFace model ID format: org/model-name or model-name
    pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?(/[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?)?$"
    if not re.match(pattern, value):
        raise ValueError(
            f"Invalid model name '{value}'. Must be a HuggingFace model ID "
            f"(e.g., 'meta-llama/Llama-2-7b') or a local path."
        )

    return value


# ---------------------------------------------------------------------------
# Path validator
# ---------------------------------------------------------------------------

def validate_path(
    value: str,
    *,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    must_be_writable: bool = False,
    create_parent: bool = False,
) -> Path:
    """Validate a file system path.

    Args:
        value: Path string.
        must_exist: Whether the path must already exist.
        must_be_file: Whether the path must be a file.
        must_be_dir: Whether the path must be a directory.
        must_be_writable: Whether the path must be writable.
        create_parent: Whether to create parent directories.

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If validation fails.
    """
    if not value or not value.strip():
        raise ValueError("Path cannot be empty")

    path = Path(value).expanduser().resolve()

    if must_exist and not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    if must_be_file and path.exists() and not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if must_be_dir and path.exists() and not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    if must_be_writable:
        parent = path.parent
        if not parent.exists():
            if create_parent:
                parent.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(f"Parent directory does not exist: {parent}")
        if not os.access(parent, os.W_OK):
            raise ValueError(f"Directory is not writable: {parent}")

    return path


# ---------------------------------------------------------------------------
# URL validator
# ---------------------------------------------------------------------------

def validate_url(
    value: str,
    *,
    allowed_schemes: Optional[List[str]] = None,
) -> str:
    """Validate a URL.

    Args:
        value: URL string.
        allowed_schemes: List of allowed schemes (e.g., ['http', 'https']).
                         Defaults to ['http', 'https'].

    Returns:
        Validated URL string.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not value or not value.strip():
        raise ValueError("URL cannot be empty")

    schemes = allowed_schemes or ["http", "https"]

    try:
        parsed = urlparse(value)
    except Exception as exc:
        raise ValueError(f"Invalid URL '{value}': {exc}")

    if not parsed.scheme:
        raise ValueError(f"URL must include a scheme (e.g., https://): {value}")

    if parsed.scheme not in schemes:
        raise ValueError(
            f"URL scheme '{parsed.scheme}' not allowed. "
            f"Allowed schemes: {', '.join(schemes)}"
        )

    if not parsed.hostname:
        raise ValueError(f"URL must include a hostname: {value}")

    return value


# ---------------------------------------------------------------------------
# Port validator
# ---------------------------------------------------------------------------

def validate_port(value: Union[str, int]) -> int:
    """Validate a network port number.

    Args:
        value: Port number as string or int.

    Returns:
        Validated port number as int.

    Raises:
        ValueError: If the port is invalid.
    """
    try:
        port = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Port must be a number: {value}")

    if not (1 <= port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {port}")

    # Warn about well-known ports
    if port < 1024:
        import warnings
        warnings.warn(
            f"Port {port} is a well-known port and may require elevated privileges.",
            UserWarning,
            stacklevel=2,
        )

    return port


# ---------------------------------------------------------------------------
# Numeric validators
# ---------------------------------------------------------------------------

def validate_positive_int(
    value: Union[str, int],
    *,
    min_value: int = 1,
    max_value: Optional[int] = None,
) -> int:
    """Validate a positive integer.

    Args:
        value: Integer value as string or int.
        min_value: Minimum allowed value (default 1).
        max_value: Maximum allowed value (optional).

    Returns:
        Validated integer.

    Raises:
        ValueError: If the value is invalid.
    """
    try:
        num = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"Expected an integer, got: {value}")

    if num < min_value:
        raise ValueError(f"Value must be at least {min_value}, got {num}")

    if max_value is not None and num > max_value:
        raise ValueError(f"Value must be at most {max_value}, got {num}")

    return num


def validate_float_range(
    value: Union[str, float],
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "value",
) -> float:
    """Validate a float within an optional range.

    Args:
        value: Float value as string or number.
        min_value: Minimum allowed value (optional).
        max_value: Maximum allowed value (optional).
        name: Parameter name for error messages.

    Returns:
        Validated float.

    Raises:
        ValueError: If the value is invalid.
    """
    try:
        num = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a number, got: {value}")

    if min_value is not None and num < min_value:
        raise ValueError(f"{name} must be at least {min_value}, got {num}")

    if max_value is not None and num > max_value:
        raise ValueError(f"{name} must be at most {max_value}, got {num}")

    return num


# ---------------------------------------------------------------------------
# Choice validator
# ---------------------------------------------------------------------------

def validate_choice(
    value: str,
    choices: Collection[str],
    *,
    case_sensitive: bool = False,
    name: str = "value",
) -> str:
    """Validate that a value is one of the allowed choices.

    Args:
        value: The value to check.
        choices: Collection of allowed values.
        case_sensitive: Whether comparison is case-sensitive.
        name: Parameter name for error messages.

    Returns:
        The matched choice value (original casing from choices).

    Raises:
        ValueError: If the value is not in the allowed choices.
    """
    if not case_sensitive:
        lower_choices = {c.lower(): c for c in choices}
        matched = lower_choices.get(value.lower())
        if matched is not None:
            return matched
    else:
        if value in choices:
            return value

    choices_str = ", ".join(f"'{c}'" for c in sorted(choices))
    raise ValueError(f"{name} must be one of: {choices_str}, got '{value}'")


# ---------------------------------------------------------------------------
# File extension validator
# ---------------------------------------------------------------------------

def validate_file_extension(
    value: str,
    extensions: Collection[str],
    *,
    case_sensitive: bool = False,
) -> str:
    """Validate that a filename has an allowed extension.

    Args:
        value: Filename or path string.
        extensions: Collection of allowed extensions (with or without dots).
        case_sensitive: Whether extension comparison is case-sensitive.

    Returns:
        The original value if valid.

    Raises:
        ValueError: If the file extension is not allowed.
    """
    # Normalize extensions to include leading dot
    normalized_exts = set()
    for ext in extensions:
        ext = ext if ext.startswith(".") else f".{ext}"
        if case_sensitive:
            normalized_exts.add(ext)
        else:
            normalized_exts.add(ext.lower())

    # Extract file extension
    _, file_ext = os.path.splitext(value)
    compare_ext = file_ext if case_sensitive else file_ext.lower()

    if compare_ext not in normalized_exts:
        ext_str = ", ".join(f"'{e}'" for e in sorted(normalized_exts))
        raise ValueError(
            f"File extension '{file_ext}' not allowed. "
            f"Allowed extensions: {ext_str}"
        )

    return value
