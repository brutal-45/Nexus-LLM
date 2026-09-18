"""Content filter for Nexus-LLM.

Blocks harmful content, violence, and illegal activities based on
configurable strictness levels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


class StrictnessLevel(str, Enum):
    """Strictness levels for content filtering."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FilterCategory(str, Enum):
    """Categories of unsafe content recognised by the filter.

    Used as the shared vocabulary between :class:`ContentFilter`,
    :class:`~nexus_llm.safety.moderation.ContentModerator` and the policy
    engine, so a rule can be expressed once and applied everywhere.
    """

    HARMFUL = "harmful"
    SELF_HARM = "self_harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    ILLEGAL = "illegal"
    SEXUAL = "sexual"
    PROFANITY = "profanity"
    SPAM = "spam"
    PII = "pii"


class FilterAction(str, Enum):
    """What to do with content that matched a filter category."""

    ALLOW = "allow"
    FLAG = "flag"
    REDACT = "redact"
    BLOCK = "block"


#: Categories that are always hard-blocked, whatever the strictness.
_HARD_BLOCKED = frozenset(
    {FilterCategory.SELF_HARM, FilterCategory.VIOLENCE, FilterCategory.HATE_SPEECH}
)

#: Categories that are only reported (flagged) unless strictness is HIGH.
_SOFT_CATEGORIES = frozenset({FilterCategory.PROFANITY, FilterCategory.SPAM})


@dataclass
class FilterResult:
    """Outcome of running :meth:`ContentFilter.filter` on a piece of text.

    Attributes:
        original_text: The text as supplied.
        filtered_text: The text with matched spans replaced by ``[FILTERED]``.
        blocked_categories: Categories that must not be shown to the user.
        flagged_categories: Categories worth recording but not blocking.
        action: The most severe action implied by the matches.
        reasons: Human-readable explanation per matched category.
        matches: Matched substrings per category (useful for auditing).
    """

    original_text: str
    filtered_text: str
    blocked_categories: List[FilterCategory] = field(default_factory=list)
    flagged_categories: List[FilterCategory] = field(default_factory=list)
    action: FilterAction = FilterAction.ALLOW
    reasons: List[str] = field(default_factory=list)
    matches: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        """Whether :attr:`filtered_text` differs from the input."""
        return self.filtered_text != self.original_text

    @property
    def is_safe(self) -> bool:
        """Whether nothing was blocked."""
        return not self.blocked_categories

    @property
    def categories(self) -> List[FilterCategory]:
        """All categories touched by the text, blocked first."""
        return [*self.blocked_categories, *self.flagged_categories]

    def to_dict(self) -> Dict[str, object]:
        """Serialise the result for logging or an API response."""
        return {
            "is_safe": self.is_safe,
            "action": self.action.value,
            "blocked_categories": [c.value for c in self.blocked_categories],
            "flagged_categories": [c.value for c in self.flagged_categories],
            "reasons": list(self.reasons),
            "was_modified": self.was_modified,
        }


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

_SELF_HARM_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(how\s+to\s+(kill|harm)\s+myself|want\s+to\s+die|end\s+my\s+life)\b", re.IGNORECASE
    ),
    re.compile(r"\b(self[- ]harm|hurt\s+myself|suicidal\s+thoughts)\b", re.IGNORECASE),
]

_HATE_SPEECH_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(racially\s+pure|ethnic\s+cleansing|genocide\s+of)\b", re.IGNORECASE),
    re.compile(r"\b(hate\s+speech|dehumanis(?:e|ing)\s+group)\b", re.IGNORECASE),
]

_HARASSMENT_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(repeatedly\s+(stalk|harass|bother)|bully\s+online|cyberbully)\b", re.IGNORECASE
    ),
    re.compile(r"\b(doxx(?:ing|ed)?|revenge\s+porn)\b", re.IGNORECASE),
]

_SEXUAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(child\s+(porn|exploitation)|csam|non[- ]consensual\s+sexual)\b", re.IGNORECASE),
]

_PROFANITY_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(fuck(?:ing|ed|s)?|shit(?:e|y|s)?|cunt|nigger|bitch)\b", re.IGNORECASE),
]

_SPAM_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:\bfree\s+(?:money|cash|prize|lottery)\b[\s\S]{0,40}){2,}", re.IGNORECASE),
    re.compile(r"(?:https?://\S+\s*){6,}", re.IGNORECASE),
    re.compile(r"\b(?:click\s+here\b[\s\S]{0,30}){3,}", re.IGNORECASE),
]

_PII_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b\+?\d{1,3}[ .-]?\(?\d{2,4}\)?[ .-]?\d{3,4}[ .-]?\d{3,4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
]

#: Which patterns belong to which category, ordered from most to least severe.
_CATEGORY_PATTERNS: Dict[FilterCategory, List[re.Pattern]] = {
    FilterCategory.SELF_HARM: _SELF_HARM_PATTERNS,
    FilterCategory.HATE_SPEECH: _HATE_SPEECH_PATTERNS,
    FilterCategory.VIOLENCE: _VIOLENCE_PATTERNS,
    FilterCategory.SEXUAL: _SEXUAL_PATTERNS,
    FilterCategory.HARASSMENT: _HARASSMENT_PATTERNS,
    FilterCategory.ILLEGAL: _ILLEGAL_PATTERNS,
    FilterCategory.HARMFUL: _HARMFUL_PATTERNS,
    FilterCategory.PII: _PII_PATTERNS,
    FilterCategory.SPAM: _SPAM_PATTERNS,
    FilterCategory.PROFANITY: _PROFANITY_PATTERNS,
}

_REASON_MAP = {
    "self_harm": "Content references self-harm",
    "hate_speech": "Content contains hate speech",
    "harassment": "Content contains harassment or targeting",
    "sexual": "Content contains prohibited sexual material",
    "profanity": "Content contains profanity",
    "spam": "Content looks like spam",
    "pii": "Content contains personally identifiable information",
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

    def filter(self, text: str) -> FilterResult:
        """Analyse *text* and return a structured :class:`FilterResult`.

        Unlike :meth:`filter_text`, which only redacts, this classifies every
        match by category and decides an action, so the moderation and policy
        layers can act on it.

        Args:
            text: The text to inspect.

        Returns:
            A :class:`FilterResult` describing what matched and what to do.
        """
        blocked: List[FilterCategory] = []
        flagged: List[FilterCategory] = []
        reasons: List[str] = []
        matches: Dict[str, List[str]] = {}

        for category, patterns in _CATEGORY_PATTERNS.items():
            hits: List[str] = []
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = match.group(0)
                    if span and span not in hits:
                        hits.append(span)
            if not hits:
                continue
            if self._should_skip(category):
                continue
            matches[category.value] = hits
            reasons.append(_REASON_MAP.get(category.value, f"Matched {category.value} content"))
            if category in _SOFT_CATEGORIES:
                flagged.append(category)
            else:
                blocked.append(category)

        action = self._decide_action(blocked, flagged)
        filtered = self.filter_text(text) if matches else text
        if FilterCategory.PII.value in matches:
            for pattern in _PII_PATTERNS:
                filtered = pattern.sub("[REDACTED]", filtered)

        return FilterResult(
            original_text=text,
            filtered_text=filtered,
            blocked_categories=blocked,
            flagged_categories=flagged,
            action=action,
            reasons=reasons,
            matches=matches,
        )

    def _should_skip(self, category: FilterCategory) -> bool:
        """Return whether *category* is below the configured strictness."""
        if category in _SOFT_CATEGORIES:
            return self.strictness is not StrictnessLevel.HIGH
        if category is FilterCategory.PII:
            return self.strictness is StrictnessLevel.LOW
        return False

    @staticmethod
    def _decide_action(
        blocked: List[FilterCategory],
        flagged: List[FilterCategory],
    ) -> FilterAction:
        """Map the matched categories onto a single action."""
        if any(category in _HARD_BLOCKED for category in blocked):
            return FilterAction.BLOCK
        if FilterCategory.PII in blocked:
            return FilterAction.REDACT
        if blocked:
            return FilterAction.BLOCK
        if flagged:
            return FilterAction.FLAG
        return FilterAction.ALLOW

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
