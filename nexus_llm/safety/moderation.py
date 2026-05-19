"""Content moderation: input/output moderation, severity levels, action policies."""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from nexus_llm.safety.content_filter import (
    ContentFilter,
    FilterAction,
    FilterCategory,
    FilterResult,
)
from nexus_llm.safety.toxicity import ToxicityDetector, ToxicityResult
from nexus_llm.safety.guardrails import SafetyGuardrails

logger = logging.getLogger("nexus_llm.safety.moderation")


class SeverityLevel(Enum):
    """Severity levels for moderated content."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ModerationAction(Enum):
    """Actions the moderator can take."""
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"
    REDACT = "redact"
    REPLACE = "replace"


@dataclass
class ModerationPolicy:
    """Defines how to handle content at different severity levels."""
    category: FilterCategory
    severity_thresholds: Dict[ModerationAction, SeverityLevel] = field(default_factory=dict)
    custom_handler: Optional[Callable] = None
    enabled: bool = True
    log_violations: bool = True
    notify_admin: bool = False

    def __post_init__(self):
        if not self.severity_thresholds:
            self.severity_thresholds = {
                ModerationAction.ALLOW: SeverityLevel.NONE,
                ModerationAction.FLAG: SeverityLevel.LOW,
                ModerationAction.BLOCK: SeverityLevel.HIGH,
            }

    def determine_action(self, severity: SeverityLevel) -> ModerationAction:
        """Determine the moderation action for a given severity level.

        Args:
            severity: The detected severity level.

        Returns:
            The appropriate ModerationAction.
        """
        if self.custom_handler:
            return self.custom_handler(severity)

        action = ModerationAction.BLOCK
        for mod_action, threshold in sorted(
            self.severity_thresholds.items(),
            key=lambda x: x[1].value,
        ):
            if severity.value >= threshold.value:
                action = mod_action
        return action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity_thresholds": {
                a.value: s.value for a, s in self.severity_thresholds.items()
            },
            "enabled": self.enabled,
            "log_violations": self.log_violations,
            "notify_admin": self.notify_admin,
        }


@dataclass
class ModerationResult:
    """Result of content moderation."""
    original_text: str
    moderated_text: str
    is_allowed: bool = True
    action: ModerationAction = ModerationAction.ALLOW
    severity: SeverityLevel = SeverityLevel.NONE
    categories_flagged: List[FilterCategory] = field(default_factory=list)
    categories_blocked: List[FilterCategory] = field(default_factory=list)
    filter_result: Optional[FilterResult] = None
    toxicity_result: Optional[ToxicityResult] = None
    guardrail_result: Optional[Any] = None
    reason: str = ""
    moderation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "action": self.action.value,
            "severity": self.severity.value,
            "severity_label": self.severity.name,
            "categories_flagged": [c.value for c in self.categories_flagged],
            "categories_blocked": [c.value for c in self.categories_blocked],
            "reason": self.reason,
            "moderation_time_ms": round(self.moderation_time_ms, 2),
            "was_modified": self.original_text != self.moderated_text,
        }


@dataclass
class ModerationAuditEntry:
    """An entry in the moderation audit log."""
    timestamp: float
    direction: str
    action_taken: str
    severity: str
    categories: List[str]
    text_snippet: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "action_taken": self.action_taken,
            "severity": self.severity,
            "categories": self.categories,
            "text_snippet": self.text_snippet[:100],
            "user_id": self.user_id,
            "request_id": self.request_id,
        }


class ContentModerator:
    """Comprehensive content moderation system.

    Combines content filtering, toxicity detection, and safety guardrails
    with configurable severity-based action policies and audit logging.
    """

    def __init__(
        self,
        content_filter: Optional[ContentFilter] = None,
        toxicity_detector: Optional[ToxicityDetector] = None,
        guardrails: Optional[SafetyGuardrails] = None,
        audit_log_size: int = 10000,
    ):
        self.content_filter = content_filter or ContentFilter()
        self.toxicity_detector = toxicity_detector or ToxicityDetector()
        self.guardrails = guardrails or SafetyGuardrails()
        self._policies: Dict[FilterCategory, ModerationPolicy] = {}
        self._audit_log: List[ModerationAuditEntry] = []
        self._audit_log_size = audit_log_size
        self._stats = {
            "total_checked": 0,
            "total_blocked": 0,
            "total_flagged": 0,
            "total_allowed": 0,
        }
        self._load_default_policies()

    def _load_default_policies(self) -> None:
        """Load default moderation policies for each category."""
        critical_categories = [
            FilterCategory.SELF_HARM,
            FilterCategory.VIOLENCE,
        ]
        for cat in critical_categories:
            self._policies[cat] = ModerationPolicy(
                category=cat,
                severity_thresholds={
                    ModerationAction.ALLOW: SeverityLevel.NONE,
                    ModerationAction.FLAG: SeverityLevel.LOW,
                    ModerationAction.BLOCK: SeverityLevel.MEDIUM,
                },
                notify_admin=True,
            )

        high_categories = [
            FilterCategory.HATE_SPEECH,
            FilterCategory.HARASSMENT,
            FilterCategory.ILLEGAL,
            FilterCategory.SEXUAL,
        ]
        for cat in high_categories:
            self._policies[cat] = ModerationPolicy(
                category=cat,
                severity_thresholds={
                    ModerationAction.ALLOW: SeverityLevel.NONE,
                    ModerationAction.FLAG: SeverityLevel.LOW,
                    ModerationAction.BLOCK: SeverityLevel.HIGH,
                },
            )

        medium_categories = [
            FilterCategory.PROFANITY,
            FilterCategory.SPAM,
        ]
        for cat in medium_categories:
            self._policies[cat] = ModerationPolicy(
                category=cat,
                severity_thresholds={
                    ModerationAction.ALLOW: SeverityLevel.NONE,
                    ModerationAction.FLAG: SeverityLevel.MEDIUM,
                    ModerationAction.BLOCK: SeverityLevel.CRITICAL,
                },
            )

        self._policies[FilterCategory.PII] = ModerationPolicy(
            category=FilterCategory.PII,
            severity_thresholds={
                ModerationAction.ALLOW: SeverityLevel.NONE,
                ModerationAction.REDACT: SeverityLevel.LOW,
                ModerationAction.BLOCK: SeverityLevel.HIGH,
            },
        )

    def moderate_input(
        self,
        text: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ModerationResult:
        """Moderate user input before processing by the model.

        Args:
            text: Input text to moderate.
            user_id: Optional user identifier.
            request_id: Optional request identifier.
            model_name: Optional model name.

        Returns:
            ModerationResult with action decision and modified text.
        """
        return self._moderate(
            text, direction="input", user_id=user_id,
            request_id=request_id, model_name=model_name,
        )

    def moderate_output(
        self,
        text: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ModerationResult:
        """Moderate model output before returning to the user.

        Args:
            text: Output text to moderate.
            user_id: Optional user identifier.
            request_id: Optional request identifier.
            model_name: Optional model name.

        Returns:
            ModerationResult with action decision and modified text.
        """
        return self._moderate(
            text, direction="output", user_id=user_id,
            request_id=request_id, model_name=model_name,
        )

    def _moderate(
        self,
        text: str,
        direction: str = "input",
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> ModerationResult:
        """Core moderation logic combining all checks.

        Args:
            text: Text to moderate.
            direction: "input" or "output".
            user_id: Optional user identifier.
            request_id: Optional request identifier.
            model_name: Optional model name.

        Returns:
            ModerationResult with comprehensive moderation decision.
        """
        start_time = time.time()

        filter_result = self.content_filter.filter(text)
        toxicity_result = self.toxicity_detector.detect(text)
        guardrail_result = self.guardrails.validate(text)

        max_severity = SeverityLevel.NONE
        categories_flagged: List[FilterCategory] = []
        categories_blocked: List[FilterCategory] = []
        final_action = ModerationAction.ALLOW
        reasons: List[str] = []
        moderated_text = filter_result.filtered_text

        if filter_result.blocked_categories:
            for cat in filter_result.blocked_categories:
                policy = self._policies.get(cat)
                severity = SeverityLevel.HIGH if policy is None else SeverityLevel.HIGH
                action = policy.determine_action(severity) if policy else ModerationAction.BLOCK
                if action.value > final_action.value:
                    final_action = action
                if severity.value > max_severity.value:
                    max_severity = severity
                if action in (ModerationAction.BLOCK, ModerationAction.REDACT):
                    categories_blocked.append(cat)
                else:
                    categories_flagged.append(cat)
                reasons.append(f"Content filter: {cat.value}")

        if filter_result.flagged_categories:
            for cat in filter_result.flagged_categories:
                categories_flagged.append(cat)
                if max_severity.value < SeverityLevel.LOW.value:
                    max_severity = SeverityLevel.LOW
                if final_action == ModerationAction.ALLOW:
                    final_action = ModerationAction.FLAG

        if toxicity_result and toxicity_result.is_toxic:
            toxicity_severity = self._map_toxicity_to_severity(toxicity_result)
            if toxicity_severity.value > max_severity.value:
                max_severity = toxicity_severity
            policy = self._policies.get(FilterCategory.HATE_SPEECH)
            action = policy.determine_action(toxicity_severity) if policy else ModerationAction.BLOCK
            if action.value > final_action.value:
                final_action = action
            reasons.append(
                f"Toxicity detected: {toxicity_result.primary_category} "
                f"(score: {toxicity_result.overall_score:.2f})"
            )
            if action in (ModerationAction.BLOCK,):
                categories_blocked.append(FilterCategory.HATE_SPEECH)

        if guardrail_result and not guardrail_result.is_valid:
            for violation in guardrail_result.violations:
                reasons.append(f"Guardrail: {violation}")
            if final_action == ModerationAction.ALLOW:
                final_action = ModerationAction.FLAG
            if max_severity.value < SeverityLevel.MEDIUM.value:
                max_severity = SeverityLevel.MEDIUM

        if final_action == ModerationAction.REDACT or final_action == ModerationAction.REPLACE:
            if filter_result.was_modified:
                moderated_text = filter_result.filtered_text
            else:
                moderated_text = "[REDACTED]"

        is_allowed = final_action not in (ModerationAction.BLOCK, ModerationAction.REDACT)

        elapsed_ms = (time.time() - start_time) * 1000.0

        result = ModerationResult(
            original_text=text,
            moderated_text=moderated_text if is_allowed else "",
            is_allowed=is_allowed,
            action=final_action,
            severity=max_severity,
            categories_flagged=categories_flagged,
            categories_blocked=categories_blocked,
            filter_result=filter_result,
            toxicity_result=toxicity_result,
            guardrail_result=guardrail_result,
            reason="; ".join(reasons) if reasons else "Content passed moderation",
            moderation_time_ms=elapsed_ms,
        )

        self._update_stats(result)
        self._log_audit(direction, result, user_id, request_id, model_name)

        return result

    @staticmethod
    def _map_toxicity_to_severity(toxicity_result: ToxicityResult) -> SeverityLevel:
        """Map a toxicity score to a severity level."""
        score = toxicity_result.overall_score
        if score >= 0.9:
            return SeverityLevel.CRITICAL
        elif score >= 0.7:
            return SeverityLevel.HIGH
        elif score >= 0.5:
            return SeverityLevel.MEDIUM
        elif score >= 0.3:
            return SeverityLevel.LOW
        return SeverityLevel.NONE

    def _update_stats(self, result: ModerationResult) -> None:
        """Update moderation statistics."""
        self._stats["total_checked"] += 1
        if result.action == ModerationAction.BLOCK:
            self._stats["total_blocked"] += 1
        elif result.action == ModerationAction.FLAG:
            self._stats["total_flagged"] += 1
        else:
            self._stats["total_allowed"] += 1

    def _log_audit(
        self,
        direction: str,
        result: ModerationResult,
        user_id: Optional[str],
        request_id: Optional[str],
        model_name: Optional[str],
    ) -> None:
        """Add an entry to the moderation audit log."""
        if result.action == ModerationAction.ALLOW:
            return

        entry = ModerationAuditEntry(
            timestamp=time.time(),
            direction=direction,
            action_taken=result.action.value,
            severity=result.severity.name,
            categories=[c.value for c in result.categories_blocked + result.categories_flagged],
            text_snippet=result.original_text[:100],
            user_id=user_id,
            request_id=request_id,
            model_name=model_name,
        )
        self._audit_log.append(entry)

        if len(self._audit_log) > self._audit_log_size:
            self._audit_log = self._audit_log[-self._audit_log_size:]

        if result.severity.value >= SeverityLevel.HIGH.value:
            logger.warning(
                "Moderation action: %s (severity: %s, direction: %s, categories: %s)",
                result.action.value,
                result.severity.name,
                direction,
                [c.value for c in result.categories_blocked],
            )

    def set_policy(self, policy: ModerationPolicy) -> None:
        """Set or update a moderation policy for a category."""
        self._policies[policy.category] = policy

    def get_policy(self, category: FilterCategory) -> Optional[ModerationPolicy]:
        """Get the moderation policy for a category."""
        return self._policies.get(category)

    def get_stats(self) -> Dict[str, Any]:
        """Return moderation statistics."""
        return dict(self._stats)

    def get_audit_log(
        self,
        limit: int = 100,
        direction: Optional[str] = None,
        min_severity: Optional[SeverityLevel] = None,
    ) -> List[ModerationAuditEntry]:
        """Retrieve entries from the audit log.

        Args:
            limit: Maximum number of entries to return.
            direction: Filter by "input" or "output".
            min_severity: Filter by minimum severity level.

        Returns:
            List of matching audit entries.
        """
        entries = list(reversed(self._audit_log))

        if direction:
            entries = [e for e in entries if e.direction == direction]

        if min_severity:
            entries = [e for e in entries if SeverityLevel[e.severity].value >= min_severity.value]

        return entries[:limit]

    def get_config(self) -> Dict[str, Any]:
        """Return current moderation configuration."""
        return {
            "policies": {
                cat.value: policy.to_dict()
                for cat, policy in self._policies.items()
            },
            "stats": self.get_stats(),
            "audit_log_size": len(self._audit_log),
        }
