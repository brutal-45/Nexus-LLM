"""Toxicity detector for Nexus-LLM.

Classifies text across toxicity categories and returns a toxicity score
with category-level breakdowns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

@dataclass
class _ToxicityCategory:
    """Internal representation of a toxicity category with its patterns."""
    name: str
    patterns: List[re.Pattern]
    threshold: float  # per-category score threshold for flagging


_CATEGORIES: List[_ToxicityCategory] = [
    _ToxicityCategory(
        name="hate",
        patterns=[
            re.compile(r"\b(hate\s+speech|racial\s+slur|xenophob)\b", re.IGNORECASE),
            re.compile(r"\b(bigot|supremac|ethnic\s+cleansing)\b", re.IGNORECASE),
            re.compile(r"\b(slur|derogatory)\b", re.IGNORECASE),
        ],
        threshold=0.3,
    ),
    _ToxicityCategory(
        name="violence",
        patterns=[
            re.compile(r"\b(kill|murder|assault|stab|shoot|attack)\b", re.IGNORECASE),
            re.compile(r"\b(bomb|massacre|torture|mutilat)\b", re.IGNORECASE),
            re.compile(r"\b(weapon|firearm|bloodbath)\b", re.IGNORECASE),
        ],
        threshold=0.3,
    ),
    _ToxicityCategory(
        name="sexual",
        patterns=[
            re.compile(r"\b(pornograph|sexual\s+explicit|nsfw)\b", re.IGNORECASE),
            re.compile(r"\b(exploit|nonconsensual|coerce)\b", re.IGNORECASE),
            re.compile(r"\b(obscene|indecent)\b", re.IGNORECASE),
        ],
        threshold=0.3,
    ),
    _ToxicityCategory(
        name="harassment",
        patterns=[
            re.compile(r"\b(harass|bully|intimidat|stalk)\b", re.IGNORECASE),
            re.compile(r"\b(threat|dox|swat|shame)\b", re.IGNORECASE),
            re.compile(r"\b(insult|belittle|humiliate)\b", re.IGNORECASE),
        ],
        threshold=0.3,
    ),
    _ToxicityCategory(
        name="self-harm",
        patterns=[
            re.compile(r"\b(suicide|self[- ]harm|kill\s+myself)\b", re.IGNORECASE),
            re.compile(r"\b(cut\s+myself|hurt\s+myself|end\s+my\s+life)\b", re.IGNORECASE),
            re.compile(r"\b(overdose|suicidal)\b", re.IGNORECASE),
        ],
        threshold=0.25,
    ),
]

_DEFAULT_THRESHOLD = 0.5


class ToxicityDetector:
    """Detect toxic content across multiple categories.

    Args:
        threshold: Overall toxicity score above which text is flagged.
                   Defaults to 0.5.
    """

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._categories = _CATEGORIES
        logger.info("ToxicityDetector initialised with threshold=%.2f", threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> Tuple[bool, float, List[str]]:
        """Analyse *text* for toxicity.

        Returns:
            A 3-tuple of ``(is_toxic, score, categories)`` where:

            - *is_toxic* is ``True`` when the overall score exceeds the
              threshold.
            - *score* is a float between 0.0 and 1.0.
            - *categories* is the list of flagged category names.
        """
        category_scores = self._score_categories(text)
        overall_score = self._compute_overall_score(category_scores)
        flagged = [
            cat for cat, score in category_scores.items()
            if score >= self._category_threshold(cat)
        ]

        is_toxic = overall_score >= self.threshold or len(flagged) > 0

        logger.debug(
            "Toxicity detect: score=%.3f is_toxic=%s categories=%s",
            overall_score, is_toxic, flagged,
        )
        return is_toxic, round(overall_score, 4), flagged

    def detect_detailed(self, text: str) -> Dict[str, float]:
        """Return per-category toxicity scores for *text*.

        Useful for dashboard displays and detailed reporting.
        """
        return self._score_categories(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_categories(self, text: str) -> Dict[str, float]:
        """Score each category based on pattern matches."""
        scores: Dict[str, float] = {}
        for cat in self._categories:
            match_count = sum(1 for p in cat.patterns if p.search(text))
            # Each match contributes 0.35, capped at 1.0
            score = min(match_count * 0.35, 1.0)
            scores[cat.name] = round(score, 4)
        return scores

    def _compute_overall_score(self, category_scores: Dict[str, float]) -> float:
        """Aggregate per-category scores into a single overall score.

        Uses a weighted average where higher individual scores pull the
        overall score up more aggressively (quadratic weighting).
        """
        if not category_scores:
            return 0.0
        total = sum(score ** 1.5 for score in category_scores.values())
        max_possible = len(category_scores)  # if all were 1.0^1.5 = 1.0
        return min(total / max_possible, 1.0)

    def _category_threshold(self, name: str) -> float:
        """Return the threshold for a given category name."""
        for cat in self._categories:
            if cat.name == name:
                return cat.threshold
        return 0.3  # fallback
