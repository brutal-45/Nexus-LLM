"""Test content filtering for Nexus-LLM."""
import re
import pytest
from dataclasses import dataclass
from typing import List, Dict, Optional, Pattern


class ContentFilterError(Exception):
    pass


@dataclass
class FilterRule:
    name: str
    pattern: str
    action: str = "block"  # block, flag, replace
    replacement: str = "[FILTERED]"
    severity: str = "medium"  # low, medium, high

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, text: str) -> bool:
        return bool(self._compiled.search(text))

    def apply(self, text: str) -> str:
        if self.action == "replace":
            return self._compiled.sub(self.replacement, text)
        return text


BUILTIN_RULES = [
    FilterRule(name="pii_ssn", pattern=r'\b\d{3}-\d{2}-\d{4}\b', action="replace", severity="high"),
    FilterRule(name="pii_credit_card", pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', action="replace", severity="high"),
    FilterRule(name="pii_email", pattern=r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', action="flag", severity="medium"),
    FilterRule(name="pii_phone", pattern=r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', action="replace", severity="medium"),
    FilterRule(name="profanity_slur", pattern=r'\b(badword1|badword2|badword3)\b', action="block", severity="high"),
]


@dataclass
class FilterResult:
    passed: bool
    filtered_text: str
    matched_rules: List[str]
    severity: str = "none"

    @property
    def was_filtered(self):
        return len(self.matched_rules) > 0


class ContentFilter:
    def __init__(self, rules: List[FilterRule] = None, enabled: bool = True):
        self._rules = rules or list(BUILTIN_RULES)
        self._enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    def add_rule(self, rule: FilterRule):
        self._rules.append(rule)

    def remove_rule(self, name: str):
        self._rules = [r for r in self._rules if r.name != name]

    def filter(self, text: str) -> FilterResult:
        if not self._enabled:
            return FilterResult(passed=True, filtered_text=text, matched_rules=[])

        matched_rules = []
        result_text = text
        max_severity = "none"
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3}

        for rule in self._rules:
            if rule.matches(text):
                matched_rules.append(rule.name)
                if severity_order.get(rule.severity, 0) > severity_order.get(max_severity, 0):
                    max_severity = rule.severity
                if rule.action == "block":
                    return FilterResult(passed=False, filtered_text="", matched_rules=matched_rules, severity=max_severity)
                elif rule.action == "replace":
                    result_text = rule.apply(result_text)

        return FilterResult(
            passed=len(matched_rules) == 0 or all(r.action != "block" for r in self._rules if r.name in matched_rules),
            filtered_text=result_text,
            matched_rules=matched_rules,
            severity=max_severity,
        )

    def check_only(self, text: str) -> bool:
        result = self.filter(text)
        return result.passed


class TestFilterRule:
    def test_matches(self):
        rule = FilterRule(name="test", pattern=r"bad")
        assert rule.matches("this is bad") is True
        assert rule.matches("this is good") is False

    def test_apply_replace(self):
        rule = FilterRule(name="test", pattern=r"secret", action="replace", replacement="[REDACTED]")
        assert rule.apply("this is secret info") == "this is [REDACTED] info"

    def test_apply_block(self):
        rule = FilterRule(name="test", pattern=r"forbidden", action="block")
        assert rule.apply("forbidden content") == "forbidden content"

    def test_case_insensitive(self):
        rule = FilterRule(name="test", pattern=r"bad")
        assert rule.matches("BAD") is True

    def test_severity(self):
        rule = FilterRule(name="test", pattern=r"test", severity="high")
        assert rule.severity == "high"


class TestContentFilter:
    def test_clean_text_passes(self):
        cf = ContentFilter()
        result = cf.filter("Hello, how are you today?")
        assert result.passed is True
        assert result.was_filtered is False

    def test_ssn_filtered(self):
        cf = ContentFilter()
        result = cf.filter("My SSN is 123-45-6789")
        assert result.was_filtered is True
        assert "123-45-6789" not in result.filtered_text

    def test_credit_card_filtered(self):
        cf = ContentFilter()
        result = cf.filter("Card: 4111-1111-1111-1111")
        assert result.was_filtered is True

    def test_email_flagged(self):
        cf = ContentFilter()
        result = cf.filter("Contact user@example.com")
        assert result.was_filtered is True

    def test_blocked_content(self):
        cf = ContentFilter()
        result = cf.filter("This contains badword1 here")
        assert result.passed is False

    def test_disabled_filter(self):
        cf = ContentFilter(enabled=False)
        result = cf.filter("My SSN is 123-45-6789")
        assert result.passed is True
        assert "123-45-6789" in result.filtered_text

    def test_add_custom_rule(self):
        cf = ContentFilter(rules=[])
        cf.add_rule(FilterRule(name="custom", pattern=r"forbidden", action="block"))
        result = cf.filter("this is forbidden")
        assert result.passed is False

    def test_remove_rule(self):
        cf = ContentFilter()
        cf.remove_rule("profanity_slur")
        # After removing profanity rule, the word should pass
        result = cf.filter("badword1 is here")
        assert "profanity_slur" not in result.matched_rules

    def test_check_only(self):
        cf = ContentFilter()
        assert cf.check_only("clean text") is True
        assert cf.check_only("badword1 here") is False

    def test_multiple_matches(self):
        cf = ContentFilter(rules=[
            FilterRule(name="r1", pattern=r"secret", action="replace", replacement="XXX"),
            FilterRule(name="r2", pattern=r"confidential", action="flag"),
        ])
        result = cf.filter("secret and confidential info")
        assert len(result.matched_rules) == 2
