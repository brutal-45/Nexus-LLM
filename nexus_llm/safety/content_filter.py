"""Content filter for Nexus-LLM.

Blocks harmful content, violence, and illegal activities based on
configurable strictness levels.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Tuple

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


class StrictnessLevel(str, Enum):
    """Strictness levels for content filtering."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Pattern databases – organised by category
# ---------------------------------------------------------------------------

_HARMFUL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how\s+to\s+harm|ways\s+to\s+hurt)\b", re.IGNORECASE),
    re.compile(r"\b(harmful|dangerous)\s+(substances|chemicals|materials)\b", re.IGNORECASE),
]

_VIOLENCE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how\s+to\s+kill|ways\s+to\s+kill)\b", re.IGNORECASE),
    re.compile(r"\b(make\s+a\s+bomb|build\s+a\s+bomb|create\s+explosive)\b", re.IGNORECASE),
    re.compile(r"\b(violent\s+attack|mass\s+shooting|terrorist\s+attack)\b", re.IGNORECASE),
    re.compile(r"\b(how\s+to\s+make\s+a\s+weapon|weaponize)\b", re.IGNORECASE),
]

_ILLEGAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(how\s+to\s+steal|shoplifting\s+tips|credit\s+card\s+fraud)\b", re.IGNORECASE),
    re.compile(r"\b(hack\s+into|break\s+into|bypass\s+security)\b", re.IGNORECASE),
    re.compile(r"\b(illegal\s+drug|drug\s+manufacturing|synthesis\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(counterfeit\s+money|money\s+laundering)\b", re.IGNORECASE),
]

# Additional patterns used only at HIGH strictness
_HIGH_STRICTNESS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(kill|murder|assassinate|execute)\b", re.IGNORECASE),
    re.compile(r"\b(rob|burglar|theft|larceny)\b", re.IGNORECASE),
    re.compile(r"\b(exploit|vulnerability|backdoor)\b", re.IGNORECASE),
]

# MEDIUM-strictness adds these on top of the base patterns
_MEDIUM_STRICTNESS_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(instructions?\s+for\s+violence)\b", re.IGNORECASE),
    re.compile(r"\b(step[- ]by[- ]step\s+illegal)\b", re.IGNORECASE),
]

_REASON_MAP = {
    "harmful": "Content contains potentially harmful instructions",
    "violence": "Content contains violent or threatening language",
    "illegal": "Content references illegal activities",
    "high_strictness": "Content blocked under high strictness policy",
    "medium_strictness": "Content blocked under medium strictness policy",
}


class ContentFilter:
    """Filter and check text content for safety violations.

    Args:
        strictness: The filtering strictness level (low, medium, high).
    """

    def __init__(self, strictness: str = "medium") -> None:
        self.strictness = StrictnessLevel(strictness)
        self._patterns = self._build_patterns()
        logger.info("ContentFilter initialised with strictness=%s", self.strictness.value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter_text(self, text: str) -> str:
        """Return *text* with blocked content replaced by ``[FILTERED]``.

        Each matched pattern is replaced; overlapping matches are resolved
        by replacing from left to right.
        """
        filtered = text
        for pattern in self._patterns:
            filtered = pattern.sub("[FILTERED]", filtered)
        return filtered

    def check_prompt(self, prompt: str) -> Tuple[bool, str]:
        """Check whether *prompt* is safe to send to the model.

        Returns:
            A tuple of ``(safe, reason)``.  When *safe* is ``True``,
            *reason* is an empty string.
        """
        return self._check(prompt)

    def check_response(self, response: str) -> Tuple[bool, str]:
        """Check whether *response* from the model is safe to show.

        Returns:
            A tuple of ``(safe, reason)``.  When *safe* is ``True``,
            *reason* is an empty string.
        """
        return self._check(response)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_patterns(self) -> List[re.Pattern]:
        """Assemble the pattern list based on the strictness level."""
        patterns: List[re.Pattern] = []
        patterns.extend(_HARMFUL_PATTERNS)
        patterns.extend(_VIOLENCE_PATTERNS)
        patterns.extend(_ILLEGAL_PATTERNS)

        if self.strictness in (StrictnessLevel.MEDIUM, StrictnessLevel.HIGH):
            patterns.extend(_MEDIUM_STRICTNESS_PATTERNS)

        if self.strictness == StrictnessLevel.HIGH:
            patterns.extend(_HIGH_STRICTNESS_PATTERNS)

        return patterns

    def _check(self, text: str) -> Tuple[bool, str]:
        """Run all patterns against *text* and return the first match."""
        # Check category-specific patterns first
        for pattern in _HARMFUL_PATTERNS:
            if pattern.search(text):
                return False, _REASON_MAP["harmful"]

        for pattern in _VIOLENCE_PATTERNS:
            if pattern.search(text):
                return False, _REASON_MAP["violence"]

        for pattern in _ILLEGAL_PATTERNS:
            if pattern.search(text):
                return False, _REASON_MAP["illegal"]

        # Medium-level extras
        if self.strictness in (StrictnessLevel.MEDIUM, StrictnessLevel.HIGH):
            for pattern in _MEDIUM_STRICTNESS_PATTERNS:
                if pattern.search(text):
                    return False, _REASON_MAP["medium_strictness"]

        # High-level extras
        if self.strictness == StrictnessLevel.HIGH:
            for pattern in _HIGH_STRICTNESS_PATTERNS:
                if pattern.search(text):
                    return False, _REASON_MAP["high_strictness"]

        return True, ""
