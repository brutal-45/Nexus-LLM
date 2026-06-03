"""Input sanitization: HTML escape, path sanitization, filename cleanup."""

import os
import re
import html
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Characters not allowed in filenames on various platforms
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Input text.

    Returns:
        HTML-escaped text.
    """
    return html.escape(text, quote=True)


def unescape_html(text: str) -> str:
    """Unescape HTML entities.

    Args:
        text: HTML-escaped text.

    Returns:
        Unescaped text.
    """
    return html.unescape(text)


def sanitize_path(
    path: str,
    allow_absolute: bool = False,
    base_dir: Optional[str] = None,
) -> str:
    """Sanitize a file system path to prevent path traversal attacks.

    Args:
        path: Input path string.
        allow_absolute: Whether to allow absolute paths.
        base_dir: If provided, resolve the path relative to this directory.

    Returns:
        Sanitized path string.

    Raises:
        ValueError: If the path is potentially unsafe.
    """
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")

    # Normalize path separators
    path = path.replace("\\", "/")

    # Remove null bytes
    path = path.replace("\x00", "")

    # Check for path traversal patterns
    if ".." in path.split("/"):
        raise ValueError(f"Path traversal detected: {path}")

    if not allow_absolute and os.path.isabs(path):
        raise ValueError(f"Absolute paths not allowed: {path}")

    # Normalize the path
    path = os.path.normpath(path)

    # If base_dir is provided, ensure the resolved path is within it
    if base_dir:
        full_path = os.path.normpath(os.path.join(base_dir, path))
        if not full_path.startswith(os.path.normpath(base_dir)):
            raise ValueError(f"Path escapes base directory: {path}")
        return os.path.relpath(full_path, base_dir)

    return path


def sanitize_filename(
    filename: str,
    replacement: str = "_",
    max_length: int = 255,
    preserve_extension: bool = True,
) -> str:
    """Sanitize a filename to be safe for file system use.

    Args:
        filename: Input filename.
        replacement: Character to replace unsafe characters with.
        max_length: Maximum filename length.
        preserve_extension: Whether to preserve the file extension when truncating.

    Returns:
        Sanitized filename string.
    """
    if not filename:
        return "unnamed"

    # Normalize unicode
    filename = unicodedata.normalize("NFKD", filename)

    # Replace unsafe characters
    filename = _UNSAFE_FILENAME_CHARS.sub(replacement, filename)

    # Remove leading/trailing whitespace and dots
    filename = filename.strip(" .")

    # Handle Windows reserved names
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in _WINDOWS_RESERVED_NAMES:
        filename = f"_{filename}"

    # Remove consecutive replacements
    if replacement:
        pattern = re.escape(replacement) + r"+"
        filename = re.sub(pattern, replacement, filename)

    # Truncate if too long
    if len(filename) > max_length:
        if preserve_extension:
            ext = os.path.splitext(filename)[1]
            name_max = max_length - len(ext)
            filename = filename[:name_max] + ext
        else:
            filename = filename[:max_length]

    if not filename:
        filename = "unnamed"

    return filename


def sanitize_shell_arg(arg: str) -> str:
    """Sanitize a shell argument to prevent injection.

    Args:
        arg: Shell argument string.

    Returns:
        Sanitized shell argument (quoted).
    """
    # Remove any null bytes
    arg = arg.replace("\x00", "")

    # Use single quotes and escape any existing single quotes
    sanitized = arg.replace("'", "'\\''")
    return f"'{sanitized}'"


def sanitize_url(url: str) -> str:
    """Sanitize a URL by removing or encoding dangerous components.

    Args:
        url: Input URL string.

    Returns:
        Sanitized URL.
    """
    if not url:
        return ""

    # Remove whitespace and null bytes
    url = url.strip().replace("\x00", "")

    # Remove javascript: protocol
    if url.lower().startswith("javascript:"):
        return ""

    # Remove data: protocol (except for common safe types)
    if url.lower().startswith("data:") and not url.lower().startswith("data:image/"):
        return ""

    return url


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Args:
        text: Input text potentially containing ANSI codes.

    Returns:
        Text with ANSI codes removed.
    """
    ansi_pattern = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    return ansi_pattern.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_control_chars(text: str, keep_newlines: bool = True) -> str:
    """Remove control characters from text.

    Args:
        text: Input text.
        keep_newlines: Whether to keep newline characters.

    Returns:
        Text with control characters removed.
    """
    if keep_newlines:
        return "".join(c for c in text if c == "\n" or unicodedata.category(c)[0] != "C")
    return "".join(c for c in text if unicodedata.category(c)[0] != "C")
