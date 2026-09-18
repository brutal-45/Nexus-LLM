"""Test toxicity detection for Nexus-LLM."""
import re
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class ToxicityResult:
    is_toxic: bool
    overall_score: float
    categories: Dict[str, float]
    threshold: float

    @property
    def severity(self) -> str:
        if self.overall_score < 0.3:
            return "low"
        elif self.overall_score < 0.7:
            return "medium"
        return "high"


class ToxicityDetector:
    def __init__(self, threshold: float = 0.5):
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold = threshold
        self._patterns: Dict[str, List[str]] = {
            "insult": ["idiot", "stupid", "moron", "fool"],
            "threat": ["kill", "destroy", "hurt", "attack"],
            "obscene": ["damn", "hell"],
        }

    @property
    def threshold(self):
        return self._threshold

    def set_threshold(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        self._threshold = value

    def add_pattern(self, category: str, words: List[str]):
        self._patterns[category] = self._patterns.get(category, []) + words

    def _score_text(self, text: str, category: str) -> float:
        words = self._patterns.get(category, [])
        if not words:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for w in words if w in text_lower)
        return min(1.0, matches / max(len(words), 1) * 2)

    def detect(self, text: str) -> ToxicityResult:
        if not text:
            return ToxicityResult(is_toxic=False, overall_score=0.0, categories={}, threshold=self._threshold)

        categories = {}
        for cat in self._patterns:
            score = self._score_text(text, cat)
            categories[cat] = round(score, 3)

        overall = max(categories.values()) if categories else 0.0
        return ToxicityResult(
            is_toxic=overall >= self._threshold,
            overall_score=overall,
            categories=categories,
            threshold=self._threshold,
        )

    def is_toxic(self, text: str) -> bool:
        return self.detect(text).is_toxic

    def get_categories(self) -> List[str]:
        return list(self._patterns.keys())


class TestToxicityResult:
    def test_low_severity(self):
        result = ToxicityResult(is_toxic=False, overall_score=0.1, categories={}, threshold=0.5)
        assert result.severity == "low"

    def test_medium_severity(self):
        result = ToxicityResult(is_toxic=True, overall_score=0.5, categories={}, threshold=0.5)
        assert result.severity == "medium"

    def test_high_severity(self):
        result = ToxicityResult(is_toxic=True, overall_score=0.8, categories={}, threshold=0.5)
        assert result.severity == "high"


class TestToxicityDetector:
    def test_clean_text(self):
        detector = ToxicityDetector()
        result = detector.detect("Hello, how are you today?")
        assert result.is_toxic is False
        assert result.overall_score < 0.5

    def test_insult_detected(self):
        detector = ToxicityDetector(threshold=0.1)
        result = detector.detect("You are an idiot and a moron")
        assert result.is_toxic is True
        assert result.categories.get("insult", 0) > 0

    def test_threat_detected(self):
        detector = ToxicityDetector(threshold=0.1)
        result = detector.detect("I will destroy everything")
        assert result.categories.get("threat", 0) > 0

    def test_empty_text(self):
        detector = ToxicityDetector()
        result = detector.detect("")
        assert result.is_toxic is False

    def test_custom_threshold(self):
        detector = ToxicityDetector(threshold=0.9)
        result = detector.detect("You are an idiot")
        assert result.threshold == 0.9

    def test_set_threshold(self):
        detector = ToxicityDetector()
        detector.set_threshold(0.1)
        assert detector.threshold == 0.1

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ToxicityDetector(threshold=1.5)

    def test_add_pattern(self):
        detector = ToxicityDetector(threshold=0.1)
        detector.add_pattern("custom", ["badword"])
        result = detector.detect("This is badword")
        assert result.is_toxic is True
        assert "custom" in result.categories

    def test_get_categories(self):
        detector = ToxicityDetector()
        categories = detector.get_categories()
        assert "insult" in categories
        assert "threat" in categories

    def test_is_toxic_method(self):
        detector = ToxicityDetector(threshold=0.1)
        assert detector.is_toxic("clean text") is False
        assert detector.is_toxic("idiot moron") is True
