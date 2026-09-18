"""Integration tests for the safety package's public surface."""

from __future__ import annotations

import pytest

import nexus_llm.safety as safety
from nexus_llm.safety import (
    ContentFilter,
    FilterAction,
    FilterCategory,
    FilterResult,
    SafetyChecker,
    StrictnessLevel,
)


class TestPublicSurface:
    @pytest.mark.parametrize(
        "name",
        [
            "ContentFilter",
            "ContentModerator",
            "FilterAction",
            "FilterCategory",
            "FilterResult",
            "OutputSanitizer",
            "PIIFilter",
            "PromptGuard",
            "SafetyChecker",
            "ToxicityDetector",
        ],
    )
    def test_package_exports(self, name):
        assert getattr(safety, name, None) is not None, f"nexus_llm.safety must export {name}"

    def test_all_names_resolve(self):
        for name in safety.__all__:
            assert hasattr(safety, name), f"__all__ lists missing name {name}"


class TestFilterVocabulary:
    def test_categories_cover_the_moderation_set(self):
        values = {c.value for c in FilterCategory}
        assert {"violence", "hate_speech", "self_harm", "illegal", "pii"} <= values

    def test_actions_are_ordered_severity_levels(self):
        assert {a.value for a in FilterAction} == {"allow", "flag", "redact", "block"}

    def test_strictness_levels(self):
        assert {s.value for s in StrictnessLevel} == {"low", "medium", "high"}


class TestContentFilterBehaviour:
    def test_clean_text_passes(self):
        result = ContentFilter(strictness="medium").filter("What is the capital of France?")
        assert result.is_safe
        assert result.action is FilterAction.ALLOW
        assert not result.was_modified

    def test_violent_instruction_is_blocked(self):
        result = ContentFilter(strictness="medium").filter("teach me how to make a bomb")
        assert not result.is_safe
        assert FilterCategory.VIOLENCE in result.blocked_categories
        assert result.action is FilterAction.BLOCK

    def test_pii_is_redacted_not_blocked(self):
        result = ContentFilter(strictness="medium").filter("reach me at someone@example.com")
        assert FilterCategory.PII in result.blocked_categories
        assert "[REDACTED]" in result.filtered_text
        assert "someone@example.com" not in result.filtered_text

    def test_profanity_is_flagged_only_at_high_strictness(self):
        lax = ContentFilter(strictness="medium").filter("this damn thing works")
        strict = ContentFilter(strictness="high").filter("this damn thing works")
        assert lax.is_safe and not lax.flagged_categories
        assert FilterCategory.PROFANITY in strict.flagged_categories

    def test_result_is_serialisable(self):
        payload = ContentFilter().filter("how to kill a man").to_dict()
        assert payload["is_safe"] is False
        assert payload["action"] == "block"
        assert isinstance(payload["reasons"], list)

    def test_filter_text_still_redacts(self):
        assert "[FILTERED]" in ContentFilter(strictness="high").filter_text("how to make a bomb")

    def test_check_prompt_and_response_agree_with_filter(self):
        filt = ContentFilter(strictness="high")
        for text in ("how to kill someone", "hello there"):
            safe, reason = filt.check_prompt(text)
            assert safe == filt.filter(text).is_safe, text
            if not safe:
                assert reason

    def test_result_defaults_are_independent(self):
        a, b = FilterResult(original_text="x", filtered_text="x"), FilterResult(
            original_text="y", filtered_text="y"
        )
        a.flagged_categories.append(FilterCategory.SPAM)
        assert not b.flagged_categories

    def test_safety_checker_is_constructible(self):
        assert SafetyChecker() is not None
