"""Output sanitizer for Nexus-LLM.

Cleans model outputs by removing harmful content markers, truncating
dangerous code patterns, and enforcing length limits.
"""

from __future__ import annotations

import re
from typing import List, Optional

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sanitisation rules
# ---------------------------------------------------------------------------

_HARMFUL_MARKERS: List[re.Pattern] = [
    re.compile(r"\[HARMFUL\]", re.IGNORECASE),
    re.compile(r"\[UNSAFE\]", re.IGNORECASE),
    re.compile(r"\[DANGEROUS\]", re.IGNORECASE),
    re.compile(r"\[WARNING:?\s*[^]]*\]", re.IGNORECASE),
]

_DANGEROUS_CODE_PATTERNS: List[re.Pattern] = [
    # Shell / system commands
    re.compile(r"(rm\s+-rf\s+/)", re.IGNORECASE),
    re.compile(r"(del\s+/[sS]\s+/[qQ]\s+)", re.IGNORECASE),
    re.compile(r"(format\s+[cC]:)", re.IGNORECASE),
    # Network exfiltration
    re.compile(r"(curl\s+.+\|\s*sh)", re.IGNORECASE),
    re.compile(r"(wget\s+.+\|\s*bash)", re.IGNORECASE),
    # Python dangerous calls
    re.compile(r"(os\.system\s*\(.+\))", re.IGNORECASE),
    re.compile(r"(subprocess\.call\s*\(.+\))", re.IGNORECASE),
    re.compile(r"(eval\s*\(.+\))", re.IGNORECASE),
    re.compile(r"(exec\s*\(.+\))", re.IGNORECASE),
]

_TRUNCATION_MARKER = "[CODE_TRUNCATED]"

_DEFAULT_MAX_LENGTH = 4096


class OutputSanitizer:
    """Sanitise model outputs for safe display.

    Handles:
    - Removal of harmful content markers.
    - Truncation of dangerous code patterns.
    - Enforcing maximum output length.

    Args:
        max_length: Maximum allowed character length for output.
        remove_markers: Whether to remove harmful content markers.
        truncate_code: Whether to truncate dangerous code patterns.
    """

    def __init__(
        self,
        max_length: int = _DEFAULT_MAX_LENGTH,
        remove_markers: bool = True,
        truncate_code: bool = True,
    ) -> None:
        self.max_length = max_length
        self.remove_markers = remove_markers
        self.truncate_code = truncate_code
        logger.info(
            "OutputSanitizer initialised (max_length=%d, remove_markers=%s, truncate_code=%s)",
            max_length, remove_markers, truncate_code,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(self, text: str) -> str:
        """Return a cleaned version of *text*.

        Applies all configured sanitisation steps in sequence.
        """
        result = text

        if self.remove_markers:
            result = self._strip_harmful_markers(result)

        if self.truncate_code:
            result = self._truncate_dangerous_code(result)

        result = self._enforce_length(result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strip_harmful_markers(self, text: str) -> str:
        """Remove harmful content marker tags from the text."""
        cleaned = text
        for pattern in _HARMFUL_MARKERS:
            cleaned = pattern.sub("", cleaned)
        return cleaned

    def _truncate_dangerous_code(self, text: str) -> str:
        """Replace dangerous code patterns with a truncation marker.

        Only the matched portion is replaced; surrounding context is
        preserved so the output remains readable.
        """
        cleaned = text
        for pattern in _DANGEROUS_CODE_PATTERNS:
            cleaned = pattern.sub(_TRUNCATION_MARKER, cleaned)
        return cleaned

    def _enforce_length(self, text: str) -> str:
        """Truncate text to ``self.max_length`` characters if needed.

        A truncation notice is appended when text is cut.
        """
        if len(text) <= self.max_length:
            return text

        truncated = text[: self.max_length]
        truncation_notice = "\n[OUTPUT_TRUNCATED: exceeded maximum length]"
        return truncated + truncation_notice
