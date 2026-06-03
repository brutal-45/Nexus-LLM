"""PII (Personally Identifiable Information) filter for Nexus-LLM.

Detects and redacts PII such as emails, phone numbers, SSNs, credit
card numbers, and common name patterns from text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# PII pattern definitions
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[Dict[str, str | re.Pattern]] = [
    {
        "name": "email",
        "pattern": re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
        ),
        "replacement": "[REDACTED_EMAIL]",
    },
    {
        "name": "phone_us",
        "pattern": re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "replacement": "[REDACTED_PHONE]",
    },
    {
        "name": "ssn",
        "pattern": re.compile(
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"
        ),
        "replacement": "[REDACTED_SSN]",
    },
    {
        "name": "credit_card",
        "pattern": re.compile(
            r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"
        ),
        "replacement": "[REDACTED_CC]",
    },
    {
        "name": "ip_address",
        "pattern": re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
        "replacement": "[REDACTED_IP]",
    },
    {
        "name": "date_of_birth",
        "pattern": re.compile(
            r"\b(?:DOB|Date\s+of\s+Birth|Born)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            re.IGNORECASE,
        ),
        "replacement": "[REDACTED_DOB]",
    },
]

# Name detection heuristic – looks for common title + name patterns
_NAME_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"
)

_GENERIC_REDACTED = "[REDACTED]"


class PIIFilter:
    """Detect and redact personally identifiable information from text.

    Supports email addresses, phone numbers, SSNs, credit-card numbers,
    IP addresses, dates of birth, and titled names (Mr/Mrs/Dr …).

    Args:
        redaction_style: If ``"specific"`` (default), each PII type
            gets its own redaction label (e.g. ``[REDACTED_EMAIL]``).
            If ``"generic"``, all PII is replaced with ``[REDACTED]``.
    """

    def __init__(self, redaction_style: str = "specific") -> None:
        self.redaction_style = redaction_style
        self._patterns = _PII_PATTERNS
        logger.info("PIIFilter initialised with style=%s", redaction_style)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, text: str) -> str:
        """Return *text* with all detected PII replaced by redaction markers.

        The marker style depends on ``self.redaction_style``.
        """
        filtered = text
        for entry in self._patterns:
            pattern: re.Pattern = entry["pattern"]  # type: ignore[assignment]
            replacement = (
                _GENERIC_REDACTED
                if self.redaction_style == "generic"
                else str(entry["replacement"])
            )
            filtered = pattern.sub(replacement, filtered)

        # Name detection
        name_replacement = (
            _GENERIC_REDACTED
            if self.redaction_style == "generic"
            else "[REDACTED_NAME]"
        )
        filtered = _NAME_PATTERN.sub(name_replacement, filtered)

        return filtered

    def redact_pii(self, text: str) -> str:
        """Convenience alias that redacts using generic ``[REDACTED]`` markers.

        Equivalent to calling :meth:`filter` with ``redaction_style="generic"``.
        """
        original_style = self.redaction_style
        self.redaction_style = "generic"
        try:
            return self.filter(text)
        finally:
            self.redaction_style = original_style

    def detect_pii(self, text: str) -> List[Dict[str, str]]:
        """Return a list of detected PII entries without modifying the text.

        Each entry contains ``"type"``, ``"value"`` (the matched text),
        and ``"start"`` / ``"end"`` positions.
        """
        findings: List[Dict[str, str]] = []

        for entry in self._patterns:
            pattern: re.Pattern = entry["pattern"]  # type: ignore[assignment]
            for match in pattern.finditer(text):
                findings.append({
                    "type": str(entry["name"]),
                    "value": match.group(),
                    "start": str(match.start()),
                    "end": str(match.end()),
                })

        # Names
        for match in _NAME_PATTERN.finditer(text):
            findings.append({
                "type": "name",
                "value": match.group(),
                "start": str(match.start()),
                "end": str(match.end()),
            })

        return findings

    def has_pii(self, text: str) -> bool:
        """Return ``True`` if *text* contains any detectable PII."""
        return len(self.detect_pii(text)) > 0
