"""Safety checker for Nexus-LLM.

Evaluates text safety using rule-based and pattern-matching heuristics
and returns structured ``SafetyReport`` objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safety report
# ---------------------------------------------------------------------------

@dataclass
class SafetyReport:
    """Structured safety evaluation result.

    Attributes:
        text: The evaluated text (truncated to 200 chars for storage).
        safe: Whether the text passed all safety checks.
        score: Safety score between 0.0 (unsafe) and 1.0 (safe).
        flags: Human-readable list of safety flag descriptions.
        categories: Categories of violations detected (e.g. violence, pii).
    """

    text: str
    safe: bool
    score: float
    flags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Truncate stored text for memory efficiency
        if len(self.text) > 200:
            self.text = self.text[:200] + "…"

    def summary(self) -> str:
        """Return a one-line summary of the report."""
        status = "SAFE" if self.safe else "UNSAFE"
        return f"[{status}] score={self.score:.2f} flags={self.flags}"


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

_SAFETY_RULES: List[dict] = [
    {
        "name": "violence",
        "patterns": [
            re.compile(r"\b(kill|murder|assault|stab|shoot|attack)\b", re.IGNORECASE),
            re.compile(r"\b(bomb|explosive|weapon|firearm)\b", re.IGNORECASE),
        ],
        "weight": 0.3,
    },
    {
        "name": "hate_speech",
        "patterns": [
            re.compile(r"\b(hate\s+speech|racial\s+slur|ethnic\s+cleansing)\b", re.IGNORECASE),
            re.compile(r"\b(discriminate|bigot|supremac)\b", re.IGNORECASE),
        ],
        "weight": 0.3,
    },
    {
        "name": "self_harm",
        "patterns": [
            re.compile(r"\b(suicide|self[- ]harm|kill\s+myself)\b", re.IGNORECASE),
            re.compile(r"\b(cut\s+myself|hurt\s+myself|end\s+my\s+life)\b", re.IGNORECASE),
        ],
        "weight": 0.35,
    },
    {
        "name": "illegal",
        "patterns": [
            re.compile(r"\b(illegal|fraud|counterfeit|laundering)\b", re.IGNORECASE),
            re.compile(r"\b(drug\s+trafficking|smuggling)\b", re.IGNORECASE),
        ],
        "weight": 0.25,
    },
    {
        "name": "sexual",
        "patterns": [
            re.compile(r"\b(sexual\s+assault|nonconsensual|exploitation)\b", re.IGNORECASE),
            re.compile(r"\b(child\s+abuse|minor\s+exploitation)\b", re.IGNORECASE),
        ],
        "weight": 0.35,
    },
    {
        "name": "pii",
        "patterns": [
            re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),  # SSN-like
            re.compile(r"\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b"),  # CC-like
        ],
        "weight": 0.15,
    },
]

_THRESHOLD = 0.5  # Below this score the text is considered unsafe

# Categories that immediately mark text as unsafe when matched,
# regardless of the numeric score.
_CRITICAL_CATEGORIES = {"violence", "self_harm", "sexual", "hate_speech"}


class SafetyChecker:
    """Rule-based safety evaluator.

    Uses pattern matching across multiple safety categories to produce
    ``SafetyReport`` instances with numeric safety scores.
    """

    def __init__(self, threshold: float = _THRESHOLD) -> None:
        self.threshold = threshold
        self._rules = _SAFETY_RULES
        logger.info("SafetyChecker initialised with threshold=%.2f", threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> SafetyReport:
        """Evaluate a single piece of text and return a ``SafetyReport``.

        The safety score starts at 1.0 and is reduced by the *weight* of
        every matched rule.  A score below ``self.threshold`` means the
        text is considered unsafe.
        """
        score = 1.0
        flags: List[str] = []
        categories: List[str] = []

        for rule in self._rules:
            for pattern in rule["patterns"]:
                if pattern.search(text):
                    score -= rule["weight"]
                    flags.append(f"Matched rule: {rule['name']}")
                    if rule["name"] not in categories:
                        categories.append(rule["name"])
                    break  # one match per rule is enough

        score = max(score, 0.0)

        # Any critical category match forces unsafe regardless of score
        has_critical = bool(set(categories) & _CRITICAL_CATEGORIES)
        safe = score >= self.threshold and not has_critical

        report = SafetyReport(
            text=text,
            safe=safe,
            score=round(score, 4),
            flags=flags,
            categories=categories,
        )
        logger.debug("Safety check: %s", report.summary())
        return report

    def batch_check(self, texts: List[str]) -> List[SafetyReport]:
        """Evaluate a list of texts and return a list of reports."""
        return [self.check(text) for text in texts]
