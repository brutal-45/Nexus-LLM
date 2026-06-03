"""Prompt guard for Nexus-LLM.

Validates prompts against injection attacks, jailbreak attempts, and
system-prompt extraction queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Injection / jailbreak pattern databases
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: List[re.Pattern] = [
    # Classic prompt-injection phrases
    re.compile(r"\bignore\s+(all\s+)?previous\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\bignore\s+(the\s+)?above\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?previous\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(all\s+)?previous\s+instructions?\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"\bassistant\s*:\s*", re.IGNORECASE),
    re.compile(r"\b<\|im_start\|>", re.IGNORECASE),
    re.compile(r"\b<\|im_end\|>", re.IGNORECASE),
]

_JAILBREAK_PATTERNS: List[re.Pattern] = [
    # DAN-style and roleplay jailbreaks
    re.compile(r"\bDAN\s+mode\b", re.IGNORECASE),
    re.compile(r"\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(all\s+)?restrictions?\b", re.IGNORECASE),
    re.compile(r"\bremove\s+(all\s+)?filters?\b", re.IGNORECASE),
    re.compile(r"\bno\s+(more\s+)?rules\b", re.IGNORECASE),
    re.compile(r"\byou\s+have\s+no\s+(moral|ethical)\s+(boundaries|restrictions)\b", re.IGNORECASE),
    re.compile(r"\bpretend\s+you\s+(are|have)\s+no\s+restrictions\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+if\s+you\s+have\s+no\b", re.IGNORECASE),
]

_EXTRACTION_PATTERNS: List[re.Pattern] = [
    # System-prompt extraction attempts
    re.compile(r"\b(reveal|show|display|print|output)\s+(your|the)\s+(system|initial)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(are|were|is)\s+your\s+original\s+instructions\b", re.IGNORECASE),
    re.compile(r"\brepeat\s+(your|the)\s+(system|initial)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+your\s+(system|hidden)\s+prompt\b", re.IGNORECASE),
    re.compile(r"\boutput\s+(everything\s+)?above\b", re.IGNORECASE),
    re.compile(r"\bdebug\s+mode\b", re.IGNORECASE),
]

_SYSTEM_PROMPT_WARN_PATTERNS: List[re.Pattern] = [
    # Patterns that should not appear in *system prompts themselves*
    re.compile(r"\bignore\s+all\s+instructions\b", re.IGNORECASE),
    re.compile(r"\buser\s+input\s+is\s+always\s+safe\b", re.IGNORECASE),
    re.compile(r"\bnever\s+refuse\s+a\s+request\b", re.IGNORECASE),
    re.compile(r"\balways\s+comply\b", re.IGNORECASE),
]


@dataclass
class ValidationIssue:
    """A single validation issue found in a prompt.

    Attributes:
        category: The type of issue (injection, jailbreak, extraction).
        description: Human-readable description.
        severity: How serious the issue is (low, medium, high, critical).
        matched_text: The text snippet that triggered the issue.
    """

    category: str
    description: str
    severity: str
    matched_text: str


class PromptGuard:
    """Guard against prompt injection, jailbreaks, and extraction attempts.

    All checks are pattern-based and run locally without any external
    model calls.
    """

    def __init__(self) -> None:
        self._injection = _INJECTION_PATTERNS
        self._jailbreak = _JAILBREAK_PATTERNS
        self._extraction = _EXTRACTION_PATTERNS
        self._system_warn = _SYSTEM_PROMPT_WARN_PATTERNS
        logger.info("PromptGuard initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_prompt(self, prompt: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate *prompt* and return issues.

        Returns:
            A tuple of ``(valid, issues)`` where *valid* is ``True``
            only when *issues* is empty.
        """
        issues: List[ValidationIssue] = []

        for pattern in self._injection:
            match = pattern.search(prompt)
            if match:
                issues.append(ValidationIssue(
                    category="injection",
                    description="Potential prompt injection detected",
                    severity="high",
                    matched_text=match.group(),
                ))

        for pattern in self._jailbreak:
            match = pattern.search(prompt)
            if match:
                issues.append(ValidationIssue(
                    category="jailbreak",
                    description="Potential jailbreak attempt detected",
                    severity="critical",
                    matched_text=match.group(),
                ))

        for pattern in self._extraction:
            match = pattern.search(prompt)
            if match:
                issues.append(ValidationIssue(
                    category="extraction",
                    description="Potential system-prompt extraction attempt",
                    severity="medium",
                    matched_text=match.group(),
                ))

        valid = len(issues) == 0
        if not valid:
            logger.warning(
                "Prompt validation failed: %d issue(s) found", len(issues)
            )
        return valid, issues

    def check_system_prompt(self, system_prompt: str) -> Tuple[bool, List[str]]:
        """Check a system prompt for unsafe directives.

        Returns:
            A tuple of ``(safe, warnings)`` where *safe* is ``True``
            only when *warnings* is empty.
        """
        warnings: List[str] = []

        for pattern in self._system_warn:
            match = pattern.search(system_prompt)
            if match:
                warnings.append(
                    f"Unsafe directive in system prompt: '{match.group()}'"
                )

        safe = len(warnings) == 0
        if not safe:
            logger.warning(
                "System prompt check: %d warning(s)", len(warnings)
            )
        return safe, warnings
