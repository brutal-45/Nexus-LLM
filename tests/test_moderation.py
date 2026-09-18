"""Test content moderation for Nexus-LLM."""
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class ModerationCategory(Enum):
    HATE = "hate"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    MISINFORMATION = "misinformation"
    SPAM = "spam"
    NONE = "none"


@dataclass
class ModerationResult:
    flagged: bool
    categories: Dict[str, float]
    action: str = "allow"  # allow, warn, block

    @property
    def max_score(self) -> float:
        return max(self.categories.values()) if self.categories else 0.0

    @property
    def top_category(self) -> str:
        if not self.categories:
            return "none"
        return max(self.categories, key=self.categories.get)


class ContentModerator:
    def __init__(self, thresholds: Dict[str, float] = None):
        self._thresholds = thresholds or {
            "hate": 0.7,
            "harassment": 0.6,
            "violence": 0.5,
            "sexual": 0.7,
            "self_harm": 0.5,
            "misinformation": 0.6,
            "spam": 0.8,
        }

    @property
    def thresholds(self):
        return dict(self._thresholds)

    def update_threshold(self, category: str, threshold: float):
        if category not in self._thresholds:
            raise ValueError(f"Unknown category: {category}")
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")
        self._thresholds[category] = threshold

    def moderate(self, scores: Dict[str, float]) -> ModerationResult:
        flagged_categories = {}
        max_over_threshold = 0.0
        for category, score in scores.items():
            threshold = self._thresholds.get(category, 1.0)
            if score >= threshold:
                flagged_categories[category] = score
                max_over_threshold = max(max_over_threshold, score - threshold)

        flagged = len(flagged_categories) > 0
        if not flagged:
            action = "allow"
        elif max_over_threshold > 0.3:
            action = "block"
        else:
            action = "warn"

        return ModerationResult(flagged=flagged, categories=flagged_categories, action=action)

    def get_categories(self) -> List[str]:
        return list(self._thresholds.keys())


class TestModerationCategory:
    def test_all_categories(self):
        expected = {"hate", "harassment", "violence", "sexual", "self_harm", "misinformation", "spam", "none"}
        actual = {c.value for c in ModerationCategory}
        assert actual == expected


class TestModerationResult:
    def test_max_score(self):
        result = ModerationResult(flagged=True, categories={"hate": 0.8, "violence": 0.5})
        assert result.max_score == 0.8

    def test_top_category(self):
        result = ModerationResult(flagged=True, categories={"hate": 0.8, "violence": 0.5})
        assert result.top_category == "hate"

    def test_empty_categories(self):
        result = ModerationResult(flagged=False, categories={})
        assert result.max_score == 0.0
        assert result.top_category == "none"


class TestContentModerator:
    def test_clean_content_allowed(self):
        moderator = ContentModerator()
        result = moderator.moderate({"hate": 0.1, "violence": 0.1, "sexual": 0.1})
        assert result.flagged is False
        assert result.action == "allow"

    def test_moderate_violation_warned(self):
        moderator = ContentModerator()
        result = moderator.moderate({"hate": 0.75, "violence": 0.1})
        assert result.flagged is True
        assert result.action in ("warn", "block")

    def test_severe_violation_blocked(self):
        moderator = ContentModerator()
        result = moderator.moderate({"hate": 0.99, "violence": 0.95})
        assert result.flagged is True
        assert result.action == "block"

    def test_custom_thresholds(self):
        moderator = ContentModerator(thresholds={"hate": 0.9, "violence": 0.9})
        result = moderator.moderate({"hate": 0.8, "violence": 0.8})
        assert result.flagged is False

    def test_update_threshold(self):
        moderator = ContentModerator()
        moderator.update_threshold("hate", 0.3)
        result = moderator.moderate({"hate": 0.5})
        assert result.flagged is True

    def test_update_invalid_category(self):
        moderator = ContentModerator()
        with pytest.raises(ValueError, match="Unknown"):
            moderator.update_threshold("nonexistent", 0.5)

    def test_update_invalid_threshold(self):
        moderator = ContentModerator()
        with pytest.raises(ValueError, match="between 0 and 1"):
            moderator.update_threshold("hate", 1.5)

    def test_get_categories(self):
        moderator = ContentModerator()
        categories = moderator.get_categories()
        assert "hate" in categories
        assert "violence" in categories

    def test_thresholds_property(self):
        moderator = ContentModerator()
        t = moderator.thresholds
        assert isinstance(t, dict)
        t["hate"] = 0.0  # modifying copy
        assert moderator.thresholds["hate"] != 0.0
