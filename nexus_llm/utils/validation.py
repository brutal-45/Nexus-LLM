"""Input validation: type checking, range validation, path validation, URL validation."""

import os
import re
import logging
from typing import Any, Optional, Type, Union, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_type(
    value: Any,
    expected_type: Union[Type, tuple],
    name: str = "value",
    raise_error: bool = True,
) -> bool:
    """Validate that a value is of the expected type.

    Args:
        value: Value to validate.
        expected_type: Expected type or tuple of types.
        name: Name of the value for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails and raise_error is True.
    """
    if not isinstance(value, expected_type):
        msg = f"{name} must be of type {expected_type}, got {type(value).__name__}"
        if raise_error:
            raise ValidationError(msg)
        return False
    return True


def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    name: str = "value",
    raise_error: bool = True,
) -> bool:
    """Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Name of the value for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(value, (int, float)):
        msg = f"{name} must be numeric, got {type(value).__name__}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if min_val is not None and value < min_val:
        msg = f"{name} must be >= {min_val}, got {value}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if max_val is not None and value > max_val:
        msg = f"{name} must be <= {max_val}, got {value}"
        if raise_error:
            raise ValidationError(msg)
        return False

    return True


def validate_path(
    path: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    must_be_readable: bool = False,
    must_be_writable: bool = False,
    name: str = "path",
    raise_error: bool = True,
) -> bool:
    """Validate a file system path.

    Args:
        path: Path to validate.
        must_exist: Whether the path must exist.
        must_be_file: Whether the path must be a file.
        must_be_dir: Whether the path must be a directory.
        must_be_readable: Whether the path must be readable.
        must_be_writable: Whether the path must be writable.
        name: Name for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(path, str) or not path.strip():
        msg = f"{name} must be a non-empty string"
        if raise_error:
            raise ValidationError(msg)
        return False

    # Check for path traversal
    if ".." in os.path.normpath(path).split(os.sep):
        msg = f"{name} contains potentially unsafe path traversal"
        if raise_error:
            raise ValidationError(msg)
        return False

    if must_exist and not os.path.exists(path):
        msg = f"{name} does not exist: {path}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if must_be_file and not os.path.isfile(path):
        msg = f"{name} is not a file: {path}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if must_be_dir and not os.path.isdir(path):
        msg = f"{name} is not a directory: {path}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if must_be_readable and os.path.exists(path) and not os.access(path, os.R_OK):
        msg = f"{name} is not readable: {path}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if must_be_writable:
        parent = os.path.dirname(path) or "."
        if os.path.exists(parent) and not os.access(parent, os.W_OK):
            msg = f"{name} is not writable: {path}"
            if raise_error:
                raise ValidationError(msg)
            return False

    return True


def validate_url(
    url: str,
    allowed_schemes: Optional[List[str]] = None,
    require_netloc: bool = True,
    name: str = "url",
    raise_error: bool = True,
) -> bool:
    """Validate a URL.

    Args:
        url: URL to validate.
        allowed_schemes: List of allowed schemes (e.g., ["http", "https"]).
        require_netloc: Whether the URL must have a network location.
        name: Name for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(url, str) or not url.strip():
        msg = f"{name} must be a non-empty string"
        if raise_error:
            raise ValidationError(msg)
        return False

    try:
        parsed = urlparse(url)
    except Exception as e:
        msg = f"{name} is not a valid URL: {e}"
        if raise_error:
            raise ValidationError(msg)
        return False

    if not parsed.scheme:
        msg = f"{name} must have a scheme (e.g., http://, https://)"
        if raise_error:
            raise ValidationError(msg)
        return False

    if allowed_schemes and parsed.scheme not in allowed_schemes:
        msg = f"{name} must use one of these schemes: {allowed_schemes}, got '{parsed.scheme}'"
        if raise_error:
            raise ValidationError(msg)
        return False

    if require_netloc and not parsed.netloc:
        msg = f"{name} must have a network location (domain)"
        if raise_error:
            raise ValidationError(msg)
        return False

    return True


def validate_choice(
    value: Any,
    choices: List[Any],
    name: str = "value",
    raise_error: bool = True,
) -> bool:
    """Validate that a value is one of the allowed choices.

    Args:
        value: Value to validate.
        choices: List of allowed values.
        name: Name for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.
    """
    if value not in choices:
        msg = f"{name} must be one of {choices}, got '{value}'"
        if raise_error:
            raise ValidationError(msg)
        return False
    return True


def validate_non_empty(
    value: Any,
    name: str = "value",
    raise_error: bool = True,
) -> bool:
    """Validate that a value is not empty.

    Args:
        value: Value to validate.
        name: Name for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.
    """
    if value is None or (hasattr(value, "__len__") and len(value) == 0):
        msg = f"{name} must not be empty"
        if raise_error:
            raise ValidationError(msg)
        return False
    return True


def validate_email(
    email: str,
    name: str = "email",
    raise_error: bool = True,
) -> bool:
    """Validate an email address format.

    Args:
        email: Email address to validate.
        name: Name for error messages.
        raise_error: Whether to raise on failure.

    Returns:
        True if valid.
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        msg = f"{name} is not a valid email address: {email}"
        if raise_error:
            raise ValidationError(msg)
        return False
    return True
