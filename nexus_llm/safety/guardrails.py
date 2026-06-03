"""Safety guardrails: input validation, output validation, topic restrictions, length limits."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple


@dataclass
class GuardrailViolation:
    """Represents a single guardrail violation."""
    rule_name: str
    violation_type: str
    message: str
    severity: str = "warning"
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "violation_type": self.violation_type,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
        }


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    is_valid: bool = True
    violations: List[GuardrailViolation] = field(default_factory=list)
    sanitized_text: str = ""
    warnings: List[str] = field(default_factory=list)

    def add_violation(
        self,
        rule_name: str,
        violation_type: str,
        message: str,
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a violation and update validity status."""
        self.violations.append(GuardrailViolation(
            rule_name=rule_name,
            violation_type=violation_type,
            message=message,
            severity=severity,
            details=details,
        ))
        if severity in ("error", "critical"):
            self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a non-blocking warning."""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "warnings": self.warnings,
        }


@dataclass
class TopicRestriction:
    """Defines a restricted topic with patterns and enforcement."""
    name: str
    description: str
    patterns: List[str]
    action: str = "block"
    _compiled: List[Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def matches(self, text: str) -> bool:
        """Check if the text matches any restricted topic pattern."""
        return any(p.search(text) for p in self._compiled)


class SafetyGuardrails:
    """Comprehensive safety guardrails for LLM input/output validation.

    Provides:
    - Input validation (length, format, content)
    - Output validation (length, format, content)
    - Topic restrictions (configurable blocked topics)
    - Length limits (input/output tokens)
    - Pattern-based validation
    - Custom validation hooks
    """

    def __init__(
        self,
        max_input_length: int = 10000,
        max_output_length: int = 50000,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 4096,
        allow_code_execution: bool = False,
        allow_url_generation: bool = True,
        allow_system_prompt_injection: bool = False,
    ):
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.allow_code_execution = allow_code_execution
        self.allow_url_generation = allow_url_generation
        self.allow_system_prompt_injection = allow_system_prompt_injection

        self._topic_restrictions: List[TopicRestriction] = []
        self._input_hooks: List[Callable[[str], GuardrailResult]] = []
        self._output_hooks: List[Callable[[str], GuardrailResult]] = []
        self._blocked_patterns: List[Tuple[str, Pattern, str]] = []

        self._load_default_guardrails()

    def _load_default_guardrails(self) -> None:
        """Load default safety guardrails."""
        if not self.allow_system_prompt_injection:
            self._blocked_patterns.extend([
                (
                    "system_prompt_injection_ignore",
                    re.compile(
                        r"\b(ignore|disregard|forget)\s+(previous|above|all|your|the)\s+(instructions|prompts|rules|directives)",
                        re.IGNORECASE,
                    ),
                    "error",
                ),
                (
                    "system_prompt_injection_pretend",
                    re.compile(
                        r"\b(pretend|act\s+as|roleplay\s+as|simulate)\s+(you\s+are|you're|being)\s+(not|a\s+different|an\s+unfiltered)",
                        re.IGNORECASE,
                    ),
                    "error",
                ),
                (
                    "system_prompt_injection_new_instructions",
                    re.compile(
                        r"\bnew\s+instructions?\s*:\s*",
                        re.IGNORECASE,
                    ),
                    "warning",
                ),
            ])

        self._topic_restrictions.extend([
            TopicRestriction(
                name="weapons_creation",
                description="Instructions for creating weapons or explosives",
                patterns=[
                    r"\bhow\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive|gun|firearm)\b",
                    r"\b(improvised|homemade)\s+(explosive|weapon|firearm)\b",
                ],
                action="block",
            ),
            TopicRestriction(
                name="illegal_drugs",
                description="Instructions for manufacturing illegal drugs",
                patterns=[
                    r"\bhow\s+to\s+(make|synthesize|manufacture|cook)\s+(meth|cocaine|heroin|fentanyl|lsd|ecstasy)\b",
                    r"\b(drug|narcotic)\s+(recipe|synthesis|manufacture)\b",
                ],
                action="block",
            ),
            TopicRestriction(
                name="hacking_instructions",
                description="Instructions for unauthorized system access",
                patterns=[
                    r"\bhow\s+to\s+hack\s+(into|a\s+system)\b",
                    r"\b(exploit|vulnerability)\s+(tutorial|guide|instructions)\b",
                ],
                action="block",
            ),
        ])

    def validate_input(self, text: str) -> GuardrailResult:
        """Validate user input before model processing.

        Args:
            text: User input text.

        Returns:
            GuardrailResult with validation outcome.
        """
        result = GuardrailResult(sanitized_text=text)

        # Length checks
        if len(text) > self.max_input_length:
            result.add_violation(
                rule_name="max_input_length",
                violation_type="length",
                message=f"Input exceeds maximum length of {self.max_input_length} characters",
                severity="error",
                details={"actual_length": len(text), "max_length": self.max_input_length},
            )

        if len(text.split()) > self.max_input_tokens * 1.3:
            result.add_violation(
                rule_name="max_input_tokens",
                violation_type="length",
                message=f"Input likely exceeds maximum of {self.max_input_tokens} tokens",
                severity="warning",
            )

        # Empty input check
        if not text.strip():
            result.add_violation(
                rule_name="empty_input",
                violation_type="format",
                message="Input cannot be empty or whitespace-only",
                severity="error",
            )
            return result

        # Blocked pattern checks
        for pattern_name, compiled, severity in self._blocked_patterns:
            if compiled.search(text):
                result.add_violation(
                    rule_name=pattern_name,
                    violation_type="blocked_pattern",
                    message=f"Input matches blocked pattern: {pattern_name}",
                    severity=severity,
                )

        # Topic restriction checks
        for restriction in self._topic_restrictions:
            if restriction.matches(text):
                if restriction.action == "block":
                    result.add_violation(
                        rule_name=f"topic_{restriction.name}",
                        violation_type="restricted_topic",
                        message=f"Input discusses restricted topic: {restriction.description}",
                        severity="error",
                    )
                else:
                    result.add_warning(
                        f"Input may discuss restricted topic: {restriction.description}"
                    )

        # Code execution check
        if not self.allow_code_execution:
            code_exec_patterns = [
                re.compile(r"\bexec\s*\(", re.IGNORECASE),
                re.compile(r"\beval\s*\(", re.IGNORECASE),
                re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
                re.compile(r"\bsubprocess\.(run|call|Popen)", re.IGNORECASE),
            ]
            for pattern in code_exec_patterns:
                if pattern.search(text):
                    result.add_violation(
                        rule_name="code_execution",
                        violation_type="restricted_operation",
                        message="Code execution patterns detected in input",
                        severity="warning",
                    )

        # Custom hooks
        for hook in self._input_hooks:
            try:
                hook_result = hook(text)
                result.violations.extend(hook_result.violations)
                result.warnings.extend(hook_result.warnings)
                if not hook_result.is_valid:
                    result.is_valid = False
            except Exception:
                pass

        return result

    def validate_output(self, text: str) -> GuardrailResult:
        """Validate model output before returning to user.

        Args:
            text: Model output text.

        Returns:
            GuardrailResult with validation outcome.
        """
        result = GuardrailResult(sanitized_text=text)

        # Length checks
        if len(text) > self.max_output_length:
            result.add_violation(
                rule_name="max_output_length",
                violation_type="length",
                message=f"Output exceeds maximum length of {self.max_output_length} characters",
                severity="warning",
                details={"actual_length": len(text), "max_length": self.max_output_length},
            )

        # URL generation check
        if not self.allow_url_generation:
            url_pattern = re.compile(r"https?://[^\s<>\"]+")
            urls = url_pattern.findall(text)
            if urls:
                result.add_warning(f"Output contains {len(urls)} URL(s)")

        # PII leak check in output
        pii_patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
        }
        for pii_type, pattern in pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                result.add_warning(
                    f"Output may contain {pii_type} ({len(matches)} match(es))"
                )

        # Hallucinated confidence check
        overconfident_patterns = [
            re.compile(r"\bI\s+am\s+100%\s+(sure|certain|confident)\b", re.IGNORECASE),
            re.compile(r"\bthis\s+is\s+definitely\s+(true|correct|accurate)\b", re.IGNORECASE),
        ]
        for pattern in overconfident_patterns:
            if pattern.search(text):
                result.add_warning("Output contains overconfident assertions")

        # Custom hooks
        for hook in self._output_hooks:
            try:
                hook_result = hook(text)
                result.violations.extend(hook_result.violations)
                result.warnings.extend(hook_result.warnings)
                if not hook_result.is_valid:
                    result.is_valid = False
            except Exception:
                pass

        return result

    def validate(self, text: str) -> GuardrailResult:
        """Run both input and output validation.

        Args:
            text: Text to validate.

        Returns:
            Combined GuardrailResult.
        """
        input_result = self.validate_input(text)
        output_result = self.validate_output(text)

        combined = GuardrailResult(
            is_valid=input_result.is_valid and output_result.is_valid,
            sanitized_text=text,
        )
        combined.violations = input_result.violations + output_result.violations
        combined.warnings = input_result.warnings + output_result.warnings

        return combined

    def add_topic_restriction(self, restriction: TopicRestriction) -> None:
        """Add a topic restriction.

        Args:
            restriction: TopicRestriction to add.
        """
        self._topic_restrictions.append(restriction)

    def remove_topic_restriction(self, name: str) -> bool:
        """Remove a topic restriction by name."""
        before = len(self._topic_restrictions)
        self._topic_restrictions = [
            r for r in self._topic_restrictions if r.name != name
        ]
        return len(self._topic_restrictions) < before

    def add_input_hook(self, hook: Callable[[str], GuardrailResult]) -> None:
        """Add a custom input validation hook."""
        self._input_hooks.append(hook)

    def add_output_hook(self, hook: Callable[[str], GuardrailResult]) -> None:
        """Add a custom output validation hook."""
        self._output_hooks.append(hook)

    def add_blocked_pattern(
        self,
        name: str,
        pattern: str,
        severity: str = "error",
    ) -> None:
        """Add a blocked regex pattern.

        Args:
            name: Rule name.
            pattern: Regex pattern string.
            severity: Violation severity if matched.
        """
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
            self._blocked_patterns.append((name, compiled, severity))
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

    def list_topic_restrictions(self) -> List[Dict[str, Any]]:
        """List all topic restrictions."""
        return [
            {
                "name": r.name,
                "description": r.description,
                "action": r.action,
                "pattern_count": len(r.patterns),
            }
            for r in self._topic_restrictions
        ]

    def list_blocked_patterns(self) -> List[Dict[str, Any]]:
        """List all blocked patterns."""
        return [
            {"name": name, "severity": severity}
            for name, _, severity in self._blocked_patterns
        ]

    def get_config(self) -> Dict[str, Any]:
        """Return current guardrail configuration."""
        return {
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "allow_code_execution": self.allow_code_execution,
            "allow_url_generation": self.allow_url_generation,
            "allow_system_prompt_injection": self.allow_system_prompt_injection,
            "topic_restrictions": len(self._topic_restrictions),
            "blocked_patterns": len(self._blocked_patterns),
            "input_hooks": len(self._input_hooks),
            "output_hooks": len(self._output_hooks),
        }
