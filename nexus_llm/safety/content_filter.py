"""Content filtering: keyword filter, regex filter, category filter, custom rules."""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple


class FilterAction(Enum):
    """Action to take when a filter matches."""
    BLOCK = "block"
    FLAG = "flag"
    REPLACE = "replace"
    ALLOW = "allow"


class FilterCategory(Enum):
    """Categories for content filters."""
    PROFANITY = "profanity"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    HARASSMENT = "harassment"
    ILLEGAL = "illegal"
    PII = "pii"
    SPAM = "spam"
    CUSTOM = "custom"


@dataclass
class FilterMatch:
    """Represents a single filter match in text."""
    rule_name: str
    category: FilterCategory
    matched_text: str
    start_index: int
    end_index: int
    action: FilterAction
    confidence: float = 1.0
    replacement: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "category": self.category.value,
            "matched_text": self.matched_text[:50] + "..." if len(self.matched_text) > 50 else self.matched_text,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "action": self.action.value,
            "confidence": self.confidence,
            "replacement": self.replacement,
        }


@dataclass
class FilterRule:
    """A single content filter rule."""
    name: str
    category: FilterCategory
    action: FilterAction = FilterAction.BLOCK
    replacement: Optional[str] = None
    case_sensitive: bool = False
    whole_word: bool = False
    enabled: bool = True
    priority: int = 0
    confidence: float = 1.0

    def matches(self, text: str) -> List[FilterMatch]:
        """Check if this rule matches the given text. Override in subclasses."""
        raise NotImplementedError


@dataclass
class KeywordRule(FilterRule):
    """Filter rule that matches against a list of keywords."""
    keywords: List[str] = field(default_factory=list)

    def matches(self, text: str) -> List[FilterMatch]:
        if not self.enabled:
            return []

        results = []
        search_text = text if self.case_sensitive else text.lower()

        for keyword in self.keywords:
            kw = keyword if self.case_sensitive else keyword.lower()

            start = 0
            while True:
                idx = search_text.find(kw, start)
                if idx == -1:
                    break

                if self.whole_word:
                    before_ok = idx == 0 or not search_text[idx - 1].isalnum()
                    after_idx = idx + len(kw)
                    after_ok = after_idx >= len(search_text) or not search_text[after_idx].isalnum()
                    if not (before_ok and after_ok):
                        start = idx + 1
                        continue

                results.append(FilterMatch(
                    rule_name=self.name,
                    category=self.category,
                    matched_text=text[idx:idx + len(kw)],
                    start_index=idx,
                    end_index=idx + len(kw),
                    action=self.action,
                    confidence=self.confidence,
                    replacement=self.replacement,
                ))
                start = idx + len(kw)

        return results


@dataclass
class RegexRule(FilterRule):
    """Filter rule that matches using a regular expression pattern."""
    pattern: str = ""
    _compiled: Optional[Pattern] = field(default=None, repr=False)

    def _get_compiled(self) -> Pattern:
        if self._compiled is None:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._compiled = re.compile(self.pattern, flags)
        return self._compiled

    def matches(self, text: str) -> List[FilterMatch]:
        if not self.enabled or not self.pattern:
            return []

        compiled = self._get_compiled()
        results = []

        for match in compiled.finditer(text):
            results.append(FilterMatch(
                rule_name=self.name,
                category=self.category,
                matched_text=match.group(),
                start_index=match.start(),
                end_index=match.end(),
                action=self.action,
                confidence=self.confidence,
                replacement=self.replacement,
            ))

        return results


@dataclass
class CategoryRule(FilterRule):
    """Filter rule that applies a custom matching function."""
    match_fn: Optional[Callable[[str], List[Tuple[int, int, float]]]] = None

    def matches(self, text: str) -> List[FilterMatch]:
        if not self.enabled or self.match_fn is None:
            return []

        results = []
        matches = self.match_fn(text)
        for start, end, confidence in matches:
            results.append(FilterMatch(
                rule_name=self.name,
                category=self.category,
                matched_text=text[start:end],
                start_index=start,
                end_index=end,
                action=self.action,
                confidence=confidence,
                replacement=self.replacement,
            ))

        return results


@dataclass
class FilterResult:
    """Result of applying content filters to text."""
    original_text: str
    filtered_text: str
    matches: List[FilterMatch] = field(default_factory=list)
    is_safe: bool = True
    blocked_categories: List[FilterCategory] = field(default_factory=list)
    flagged_categories: List[FilterCategory] = field(default_factory=list)
    replaced_count: int = 0

    @property
    def was_modified(self) -> bool:
        return self.original_text != self.filtered_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "was_modified": self.was_modified,
            "filtered_text": self.filtered_text,
            "blocked_categories": [c.value for c in self.blocked_categories],
            "flagged_categories": [c.value for c in self.flagged_categories],
            "match_count": len(self.matches),
            "replaced_count": self.replaced_count,
            "matches": [m.to_dict() for m in self.matches],
        }


class ContentFilter:
    """Multi-rule content filter with keyword, regex, and category support.

    Applies a configurable set of filter rules to text, supporting
    block, flag, and replace actions with priority ordering.
    """

    def __init__(self) -> None:
        self._rules: List[FilterRule] = []
        self._custom_filters: Dict[str, Callable] = {}
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load built-in default filter rules."""
        pii_patterns = [
            r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
            r"\b(?:\d[ -]?){13,16}\b",
        ]

        self.add_rule(RegexRule(
            name="pii_phone",
            category=FilterCategory.PII,
            pattern=pii_patterns[1],
            action=FilterAction.REPLACE,
            replacement="[PHONE]",
            priority=10,
        ))

        self.add_rule(RegexRule(
            name="pii_email",
            category=FilterCategory.PII,
            pattern=pii_patterns[2],
            action=FilterAction.REPLACE,
            replacement="[EMAIL]",
            priority=10,
        ))

        self.add_rule(RegexRule(
            name="pii_ssn",
            category=FilterCategory.PII,
            pattern=pii_patterns[3],
            action=FilterAction.REPLACE,
            replacement="[SSN]",
            priority=10,
        ))

        self.add_rule(RegexRule(
            name="pii_credit_card",
            category=FilterCategory.PII,
            pattern=pii_patterns[4],
            action=FilterAction.REPLACE,
            replacement="[CARD]",
            priority=10,
        ))

    def add_rule(self, rule: FilterRule) -> None:
        """Add a filter rule.

        Args:
            rule: FilterRule to add.
        """
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        """Remove a filter rule by name.

        Args:
            name: Name of the rule to remove.

        Returns:
            True if the rule was found and removed.
        """
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def enable_rule(self, name: str) -> None:
        """Enable a filter rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = True
                return

    def disable_rule(self, name: str) -> None:
        """Disable a filter rule by name."""
        for rule in self._rules:
            if rule.name == name:
                rule.enabled = False
                return

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all registered filter rules."""
        return [
            {
                "name": r.name,
                "category": r.category.value,
                "action": r.action.value,
                "enabled": r.enabled,
                "priority": r.priority,
            }
            for r in self._rules
        ]

    def add_custom_filter(
        self, name: str, filter_fn: Callable[[str], FilterResult]
    ) -> None:
        """Add a custom filter function.

        Args:
            name: Unique name for the filter.
            filter_fn: Callable that takes text and returns FilterResult.
        """
        self._custom_filters[name] = filter_fn

    def filter(self, text: str) -> FilterResult:
        """Apply all filter rules to the input text.

        Args:
            text: Input text to filter.

        Returns:
            FilterResult with filtered text, matches, and safety status.
        """
        all_matches: List[FilterMatch] = []
        blocked_categories: List[FilterCategory] = []
        flagged_categories: List[FilterCategory] = []
        replaced_count = 0

        for rule in self._rules:
            if not rule.enabled:
                continue
            matches = rule.matches(text)
            all_matches.extend(matches)

        for name, filter_fn in self._custom_filters.items():
            try:
                result = filter_fn(text)
                all_matches.extend(result.matches)
                if not result.is_safe:
                    blocked_categories.extend(result.blocked_categories)
            except Exception:
                pass

        filtered_text = text
        replace_matches = sorted(
            [m for m in all_matches if m.action == FilterAction.REPLACE],
            key=lambda m: m.start_index,
            reverse=True,
        )
        for match in replace_matches:
            if match.replacement:
                filtered_text = (
                    filtered_text[:match.start_index]
                    + match.replacement
                    + filtered_text[match.end_index:]
                )
                replaced_count += 1

        for match in all_matches:
            if match.action == FilterAction.BLOCK:
                if match.category not in blocked_categories:
                    blocked_categories.append(match.category)
            elif match.action == FilterAction.FLAG:
                if match.category not in flagged_categories:
                    flagged_categories.append(match.category)

        is_safe = len(blocked_categories) == 0

        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            matches=all_matches,
            is_safe=is_safe,
            blocked_categories=blocked_categories,
            flagged_categories=flagged_categories,
            replaced_count=replaced_count,
        )

    def is_safe(self, text: str) -> bool:
        """Quick check if text passes all filters.

        Args:
            text: Text to check.

        Returns:
            True if text is safe (no blocking matches).
        """
        return self.filter(text).is_safe

    def add_keyword_blocklist(
        self,
        keywords: List[str],
        category: FilterCategory = FilterCategory.CUSTOM,
        name: str = "custom_blocklist",
        case_sensitive: bool = False,
        whole_word: bool = True,
    ) -> None:
        """Add a keyword blocklist as a filter rule.

        Args:
            keywords: List of keywords to block.
            category: Filter category for the blocklist.
            name: Rule name.
            case_sensitive: Whether matching is case-sensitive.
            whole_word: Whether to match whole words only.
        """
        self.add_rule(KeywordRule(
            name=name,
            category=category,
            keywords=keywords,
            action=FilterAction.BLOCK,
            case_sensitive=case_sensitive,
            whole_word=whole_word,
        ))

    def add_regex_blocklist(
        self,
        patterns: List[str],
        category: FilterCategory = FilterCategory.CUSTOM,
        name: str = "custom_regex",
        action: FilterAction = FilterAction.BLOCK,
        replacement: Optional[str] = None,
    ) -> None:
        """Add multiple regex patterns as filter rules.

        Args:
            patterns: List of regex pattern strings.
            category: Filter category.
            name: Base rule name (pattern index appended).
            action: Filter action.
            replacement: Replacement text for replace action.
        """
        for i, pattern in enumerate(patterns):
            self.add_rule(RegexRule(
                name=f"{name}_{i}",
                category=category,
                pattern=pattern,
                action=action,
                replacement=replacement,
            ))

    def clear_rules(self) -> None:
        """Remove all filter rules."""
        self._rules.clear()
        self._custom_filters.clear()
