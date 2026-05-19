"""Nexus-LLM Input Sanitization.

Provides the InputSanitizer for cleaning and validating user input
to prevent injection attacks, XSS, and other security issues.
"""

import html
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern

logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result from input sanitization.

    Attributes:
        original: Original input string.
        sanitized: Sanitized output string.
        issues: List of detected issues.
        is_safe: Whether the input was determined to be safe.
        changes_made: Number of changes made during sanitization.
    """

    original: str = ""
    sanitized: str = ""
    issues: List[str] = field(default_factory=list)
    is_safe: bool = True
    changes_made: int = 0


class InputSanitizer:
    """Sanitizes and validates user input for security.

    The InputSanitizer applies a configurable pipeline of sanitization
    rules, including HTML escaping, SQL injection detection, path
    traversal prevention, and custom pattern matching.

    Example::

        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("<script>alert('xss')</script>")
        assert result.is_safe is False
    """

    def __init__(self, max_length: int = 10000, strict_mode: bool = False) -> None:
        self._max_length = max_length
        self._strict_mode = strict_mode
        self._custom_patterns: List[Pattern[str]] = []
        logger.debug("InputSanitizer initialized (max_length=%d, strict=%s)", max_length, strict_mode)

    def add_pattern(self, pattern: str, flags: int = 0) -> None:
        """Add a custom regex pattern to flag as dangerous.

        Args:
            pattern: Regular expression pattern string.
            flags: Regex flags.
        """
        compiled = re.compile(pattern, flags)
        self._custom_patterns.append(compiled)

    def sanitize(self, text: str) -> SanitizationResult:
        """Sanitize input text through the full pipeline.

        Args:
            text: Input text to sanitize.

        Returns:
            A SanitizationResult with the cleaned text and issues.
        """
        if not text:
            return SanitizationResult(original=text, sanitized=text)

        result = SanitizationResult(original=text)
        current = text

        # Length check
        if len(current) > self._max_length:
            result.issues.append(f"Input exceeds max length ({len(current)} > {self._max_length})")
            current = current[:self._max_length]

        # HTML/XSS detection and escaping
        current, html_issues = self._sanitize_html(current)
        result.issues.extend(html_issues)

        # SQL injection detection
        sql_issues = self._detect_sql_injection(current)
        result.issues.extend(sql_issues)

        # Path traversal detection
        path_issues = self._detect_path_traversal(current)
        result.issues.extend(path_issues)

        # Prompt injection detection (basic)
        prompt_issues = self._detect_prompt_injection(current)
        result.issues.extend(prompt_issues)

        # Custom pattern checks
        for pattern in self._custom_patterns:
            if pattern.search(current):
                result.issues.append(f"Matched dangerous pattern: {pattern.pattern}")

        # Null byte removal
        if "\x00" in current:
            result.issues.append("Null bytes removed")
            current = current.replace("\x00", "")

        # Control character removal (except common whitespace)
        cleaned = "".join(
            ch for ch in current
            if ch.isprintable() or ch in "\n\r\t"
        )
        if len(cleaned) != len(current):
            result.issues.append("Control characters removed")

        result.sanitized = cleaned
        result.is_safe = len(result.issues) == 0
        result.changes_made = sum(1 for a, b in zip(text, cleaned) if a != b) + abs(len(text) - len(cleaned))

        if not result.is_safe:
            logger.warning("Input sanitization found %d issues: %s", len(result.issues), result.issues[:3])

        return result

    def validate(self, text: str) -> List[str]:
        """Validate input without modifying it.

        Args:
            text: Input text to validate.

        Returns:
            List of validation issues (empty if valid).
        """
        result = self.sanitize(text)
        return result.issues

    def _sanitize_html(self, text: str) -> tuple:
        """Detect and escape HTML/XSS content."""
        issues: List[str] = []
        dangerous_tags = re.findall(r'<\s*(script|iframe|object|embed|form|input)[^>]*>', text, re.IGNORECASE)
        if dangerous_tags:
            issues.append(f"Dangerous HTML tags detected: {dangerous_tags[:3]}")

        # Check for event handlers
        event_handlers = re.findall(r'on\w+\s*=', text, re.IGNORECASE)
        if event_handlers:
            issues.append(f"Event handlers detected: {event_handlers[:3]}")

        # Escape HTML entities
        if "<" in text or ">" in text or "&" in text:
            escaped = html.escape(text)
            return escaped, issues

        return text, issues

    def _detect_sql_injection(self, text: str) -> List[str]:
        """Detect common SQL injection patterns."""
        issues: List[str] = []
        sql_patterns = [
            r"(?i)(\b(union|select|insert|update|delete|drop|alter)\b.*\b(from|into|table|where)\b)",
            r"(?i)(\bor\b\s+\d+\s*=\s*\d+)",
            r"(?i)(\band\b\s+\d+\s*=\s*\d+)",
            r"'--",
            r";\s*(drop|delete|update|insert)",
        ]
        for pattern in sql_patterns:
            if re.search(pattern, text):
                issues.append("Potential SQL injection pattern detected")
                break
        return issues

    def _detect_path_traversal(self, text: str) -> List[str]:
        """Detect path traversal attempts."""
        issues: List[str] = []
        traversal_patterns = [r"\.\.\/", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e/"]
        for pattern in traversal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append("Path traversal pattern detected")
                break
        return issues

    def _detect_prompt_injection(self, text: str) -> List[str]:
        """Detect basic prompt injection patterns."""
        issues: List[str] = []
        injection_patterns = [
            r"(?i)ignore\s+(previous|above|all)\s+instructions",
            r"(?i)forget\s+(everything|all|previous)",
            r"(?i)you\s+are\s+now\s+",
            r"(?i)system\s*:\s*",
            r"(?i)new\s+instructions?\s*:",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text):
                issues.append("Potential prompt injection pattern detected")
                break
        return issues
