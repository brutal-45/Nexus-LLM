"""Nexus-LLM Response Validator.

Provides validation of LLM responses against expected formats,
schemas, and quality criteria.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Severity level of validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found in a response.

    Attributes:
        level: Severity level.
        message: Description of the issue.
        field: Optional field name where the issue was found.
        value: Optional value that caused the issue.
    """

    level: ValidationLevel = ValidationLevel.ERROR
    message: str = ""
    field: str = ""
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "message": self.message,
            "field": self.field,
            "value": self.value,
        }


@dataclass
class ValidationResult:
    """Result of a response validation.

    Attributes:
        is_valid: Whether the response passed validation.
        issues: List of validation issues found.
        score: Quality score (0.0 to 1.0).
    """

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 1.0

    def add_issue(self, level: ValidationLevel, message: str, field: str = "", value: Any = None) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(level=level, message=message, field=field, value=value)
        self.issues.append(issue)
        if level == ValidationLevel.ERROR:
            self.is_valid = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "score": self.score,
        }


class ResponseValidator:
    """Validates LLM responses against configurable criteria.

    Example::

        validator = ResponseValidator()
        validator.add_rule("min_length", min_length=10)
        result = validator.validate("Hello, this is a response.")
    """

    def __init__(self) -> None:
        self._rules: List[Callable] = []
        logger.debug("ResponseValidator initialized")

    def add_rule(self, rule_name: str, **kwargs: Any) -> None:
        """Add a validation rule.

        Args:
            rule_name: Name of the built-in rule.
            **kwargs: Rule-specific parameters.
        """
        rule_map = {
            "min_length": self._min_length_rule,
            "max_length": self._max_length_rule,
            "not_empty": self._not_empty_rule,
            "matches_regex": self._regex_rule,
            "is_json": self._json_rule,
            "no_profanity": self._profanity_rule,
        }

        if rule_name not in rule_map:
            raise ValueError(f"Unknown rule: {rule_name}")

        rule_fn = rule_map[rule_name]
        self._rules.append(lambda response, r=rule_fn, kw=kwargs: r(response, **kw))
        logger.debug("Added validation rule: %s", rule_name)

    def add_custom_rule(self, rule_fn: Callable) -> None:
        """Add a custom validation rule function.

        Args:
            rule_fn: Function that takes a response string and returns a list of ValidationIssues.
        """
        self._rules.append(rule_fn)

    def validate(self, response: str) -> ValidationResult:
        """Validate a response against all registered rules.

        Args:
            response: The response text to validate.

        Returns:
            A ValidationResult with issues and score.
        """
        result = ValidationResult()
        total_issues = 0

        for rule in self._rules:
            try:
                issues = rule(response)
                if isinstance(issues, list):
                    for issue in issues:
                        result.add_issue(issue.level, issue.message, issue.field, issue.value)
                        total_issues += 1
            except Exception as exc:
                result.add_issue(ValidationLevel.WARNING, f"Rule execution error: {exc}")

        # Calculate score
        if total_issues > 0:
            error_count = sum(1 for i in result.issues if i.level == ValidationLevel.ERROR)
            result.score = max(0.0, 1.0 - (error_count * 0.2))

        return result

    def _min_length_rule(self, response: str, min_length: int = 1) -> List[ValidationIssue]:
        if len(response) < min_length:
            return [ValidationIssue(ValidationLevel.ERROR, f"Response too short ({len(response)} < {min_length})")]
        return []

    def _max_length_rule(self, response: str, max_length: int = 10000) -> List[ValidationIssue]:
        if len(response) > max_length:
            return [ValidationIssue(ValidationLevel.WARNING, f"Response too long ({len(response)} > {max_length})")]
        return []

    def _not_empty_rule(self, response: str) -> List[ValidationIssue]:
        if not response.strip():
            return [ValidationIssue(ValidationLevel.ERROR, "Response is empty")]
        return []

    def _regex_rule(self, response: str, pattern: str = "") -> List[ValidationIssue]:
        if pattern and not re.search(pattern, response):
            return [ValidationIssue(ValidationLevel.ERROR, f"Response does not match pattern: {pattern}")]
        return []

    def _json_rule(self, response: str) -> List[ValidationIssue]:
        try:
            json.loads(response)
        except json.JSONDecodeError:
            return [ValidationIssue(ValidationLevel.ERROR, "Response is not valid JSON")]
        return []

    def _profanity_rule(self, response: str, words: Optional[List[str]] = None) -> List[ValidationIssue]:
        bad_words = words or []
        found = [w for w in bad_words if w.lower() in response.lower()]
        if found:
            return [ValidationIssue(ValidationLevel.WARNING, f"Potentially inappropriate content found")]
        return []
