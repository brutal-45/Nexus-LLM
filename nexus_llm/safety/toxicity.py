"""Toxicity detection: toxic content scoring, category classification."""

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ToxicityCategory(Enum):
    """Categories of toxic content."""
    HATE = "hate"
    HARASSMENT = "harassment"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    PROFANITY = "profanity"
    INSULT = "insult"
    THREAT = "threat"
    IDENTITY_ATTACK = "identity_attack"


@dataclass
class CategoryScore:
    """Toxicity score for a specific category."""
    category: ToxicityCategory
    score: float
    confidence: float = 1.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "matched_terms": self.matched_terms[:5],
        }


@dataclass
class ToxicityResult:
    """Result of toxicity detection on text."""
    text: str
    overall_score: float = 0.0
    is_toxic: bool = False
    primary_category: Optional[ToxicityCategory] = None
    category_scores: List[CategoryScore] = field(default_factory=list)
    threshold: float = 0.5
    detection_method: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_toxic": self.is_toxic,
            "overall_score": round(self.overall_score, 4),
            "primary_category": self.primary_category.value if self.primary_category else None,
            "threshold": self.threshold,
            "detection_method": self.detection_method,
            "category_scores": [cs.to_dict() for cs in self.category_scores],
        }


# Lexicon-based toxicity patterns per category
_TOXICITY_LEXICONS: Dict[ToxicityCategory, List[str]] = {
    ToxicityCategory.HATE: [
        r"\bhate\s+(you|them|people|group|race)\b",
        r"\b(racial|ethnic)\s+(slur|epithet|insult)\b",
        r"\bsupremac(y|ist)\b",
        r"\bgenocide\b",
        r"\bethnic\s+cleansing\b",
        r"\bwhite\s+(power|pride|supremacy)\b",
    ],
    ToxicityCategory.HARASSMENT: [
        r"\bstalk\b",
        r"\bdox\b",
        r"\bswat\b",
        r"\bharass\b",
        r"\bbully\b",
        r"\bthreaten\b",
        r"\bintimidat\b",
        r"\bterroriz\b",
    ],
    ToxicityCategory.SELF_HARM: [
        r"\bkill\s+myself\b",
        r"\bend\s+(my\s+)?life\b",
        r"\bcommit\s+suicide\b",
        r"\bself[- ]?harm\b",
        r"\bcut\s+myself\b",
        r"\bsuicidal\b",
        r"\boverdose\b",
        r"\bdon'?t\s+want\s+to\s+(live|be\s+alive)\b",
        r"\bno\s+reason\s+to\s+live\b",
        r"\bhow\s+to\s+(commit|die|kill)\b",
    ],
    ToxicityCategory.VIOLENCE: [
        r"\bkill\s+(you|him|her|them)\b",
        r"\bmurder\b",
        r"\bassault\b",
        r"\bstab\b",
        r"\bshoot\b",
        r"\bbomb\b",
        r"\btortur\b",
        r"\bmutilat\b",
        r"\bmassacre\b",
        r"\bslaughter\b",
    ],
    ToxicityCategory.SEXUAL: [
        r"\bsexual\s+(assault|harassment|violence)\b",
        r"\brape\b",
        r"\bpedophil\b",
        r"\bchild\s+(abuse|molest)\b",
        r"\bnon[- ]?consensual\b",
    ],
    ToxicityCategory.PROFANITY: [
        r"\bfuck\b",
        r"\bshit\b",
        r"\bdamn\b",
        r"\basshole\b",
        r"\bbitch\b",
        r"\bbastard\b",
        r"\bcrap\b",
        r"\bhell\b",
    ],
    ToxicityCategory.INSULT: [
        r"\byou'?re?\s+(stupid|dumb|idiot|moron|fool|ignorant)\b",
        r"\bworthless\b",
        r"\bpathetic\b",
        r"\bincompetent\b",
        r"\buseless\b",
        r"\btrash\b",
    ],
    ToxicityCategory.THREAT: [
        r"\bI\s+will\s+(kill|hurt|destroy|ruin)\b",
        r"\byou'?ll\s+(pay|regret|suffer)\b",
        r"\bcoming\s+for\s+you\b",
        r"\bwatch\s+your\s+back\b",
        r"\byou'?re?\s+(dead|done|finished)\b",
        r"\bI'?m\s+going\s+to\s+(kill|hurt|destroy)\b",
    ],
    ToxicityCategory.IDENTITY_ATTACK: [
        r"\b(go\s+back\s+to\s+your|your\s+kind)\b",
        r"\b(you\s+people|those\s+people)\b",
        r"\bdiscriminat\b",
        r"\bsexist\b",
        r"\bracist\b",
        r"\bhomophob\b",
        r"\btransphob\b",
        r"\bableist\b",
        r"\bxenophob\b",
    ],
}

# Severity weights for each category (higher = more severe)
_CATEGORY_WEIGHTS: Dict[ToxicityCategory, float] = {
    ToxicityCategory.SELF_HARM: 1.5,
    ToxicityCategory.VIOLENCE: 1.4,
    ToxicityCategory.THREAT: 1.3,
    ToxicityCategory.HATE: 1.2,
    ToxicityCategory.IDENTITY_ATTACK: 1.1,
    ToxicityCategory.HARASSMENT: 1.0,
    ToxicityCategory.SEXUAL: 1.0,
    ToxicityCategory.INSULT: 0.7,
    ToxicityCategory.PROFANITY: 0.4,
}


class ToxicityDetector:
    """Detects toxic content in text using lexicon-based heuristics.

    Provides per-category scoring and an overall toxicity score
    with configurable thresholds. Designed to be extended with
    ML-based classifiers.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        use_compound_scoring: bool = True,
    ):
        self.threshold = threshold
        self.use_compound_scoring = use_compound_scoring
        self._compiled_patterns: Dict[ToxicityCategory, List[Tuple[str, Any]]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for each toxicity category."""
        for category, patterns in _TOXICITY_LEXICONS.items():
            compiled = []
            for pattern in patterns:
                try:
                    compiled.append((pattern, re.compile(pattern, re.IGNORECASE)))
                except re.error:
                    continue
            self._compiled_patterns[category] = compiled

    def detect(self, text: str) -> ToxicityResult:
        """Detect toxic content in the given text.

        Args:
            text: Input text to analyze for toxicity.

        Returns:
            ToxicityResult with overall and per-category scores.
        """
        if not text or not text.strip():
            return ToxicityResult(
                text=text,
                overall_score=0.0,
                is_toxic=False,
                threshold=self.threshold,
            )

        category_scores: List[CategoryScore] = []
        primary_category: Optional[ToxicityCategory] = None
        max_weighted_score = 0.0

        for category, patterns in self._compiled_patterns.items():
            score, matched = self._score_category(text, category, patterns)
            weight = _CATEGORY_WEIGHTS.get(category, 1.0)
            weighted_score = score * weight

            category_scores.append(CategoryScore(
                category=category,
                score=score,
                confidence=min(score * 2, 1.0),
                matched_terms=matched,
            ))

            if weighted_score > max_weighted_score:
                max_weighted_score = weighted_score
                primary_category = category

        overall_score = self._compute_overall_score(category_scores)
        is_toxic = overall_score >= self.threshold

        return ToxicityResult(
            text=text,
            overall_score=overall_score,
            is_toxic=is_toxic,
            primary_category=primary_category,
            category_scores=category_scores,
            threshold=self.threshold,
        )

    def _score_category(
        self,
        text: str,
        category: ToxicityCategory,
        patterns: List[Tuple[str, Any]],
    ) -> Tuple[float, List[str]]:
        """Score a single toxicity category.

        Args:
            text: Input text.
            category: Category to score.
            patterns: Compiled regex patterns for this category.

        Returns:
            Tuple of (score, matched_terms).
        """
        matched_terms: List[str] = []
        match_count = 0
        total_patterns = len(patterns)

        if total_patterns == 0:
            return 0.0, []

        for pattern_str, compiled in patterns:
            matches = compiled.findall(text)
            if matches:
                match_count += len(matches)
                matched_terms.extend(matches[:3])

        raw_score = match_count / max(total_patterns, 1)
        score = min(1.0, raw_score * 2.0)

        if match_count > 0 and score < 0.2:
            score = 0.2

        return score, matched_terms

    def _compute_overall_score(self, category_scores: List[CategoryScore]) -> float:
        """Compute the overall toxicity score from category scores.

        Args:
            category_scores: Per-category scores.

        Returns:
            Overall toxicity score between 0 and 1.
        """
        if not category_scores:
            return 0.0

        if self.use_compound_scoring:
            weighted_sum = 0.0
            weight_total = 0.0
            for cs in category_scores:
                weight = _CATEGORY_WEIGHTS.get(cs.category, 1.0)
                weighted_sum += cs.score * weight
                weight_total += weight
            return min(1.0, weighted_sum / weight_total) if weight_total > 0 else 0.0
        else:
            return max(cs.score for cs in category_scores)

    def batch_detect(self, texts: List[str]) -> List[ToxicityResult]:
        """Detect toxicity in a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of ToxicityResult objects.
        """
        return [self.detect(text) for text in texts]

    def get_toxicity_summary(self, results: List[ToxicityResult]) -> Dict[str, Any]:
        """Generate a summary from a batch of detection results.

        Args:
            results: List of ToxicityResult objects.

        Returns:
            Summary statistics.
        """
        total = len(results)
        toxic = sum(1 for r in results if r.is_toxic)
        category_counts: Dict[str, int] = {}

        for result in results:
            if result.primary_category:
                cat = result.primary_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

        scores = [r.overall_score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "total_texts": total,
            "toxic_texts": toxic,
            "toxic_rate": round(toxic / total, 4) if total > 0 else 0.0,
            "average_score": round(avg_score, 4),
            "max_score": round(max(scores), 4) if scores else 0.0,
            "category_distribution": category_counts,
        }

    def set_threshold(self, threshold: float) -> None:
        """Update the toxicity detection threshold.

        Args:
            threshold: New threshold (0.0 to 1.0).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def add_custom_pattern(
        self,
        category: ToxicityCategory,
        pattern: str,
        weight: float = 1.0,
    ) -> None:
        """Add a custom regex pattern for a toxicity category.

        Args:
            category: Target toxicity category.
            pattern: Regex pattern string.
            weight: Optional weight for this pattern.
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            if category not in self._compiled_patterns:
                self._compiled_patterns[category] = []
            self._compiled_patterns[category].append((pattern, compiled))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Return current detector configuration."""
        return {
            "threshold": self.threshold,
            "use_compound_scoring": self.use_compound_scoring,
            "categories": [cat.value for cat in _TOXICITY_LEXICONS.keys()],
            "total_patterns": sum(
                len(patterns) for patterns in _TOXICITY_LEXICONS.values()
            ),
        }
