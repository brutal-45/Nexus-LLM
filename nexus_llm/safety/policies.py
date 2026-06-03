"""Safety policies: configurable policies, policy enforcement, audit logging."""

import json
import logging
import os
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
from nexus_llm.safety.moderation import (
    ContentModerator,
    ModerationAction,
    ModerationResult,
    ModerationPolicy,
    SeverityLevel,
)
from nexus_llm.safety.toxicity import ToxicityCategory, ToxicityDetector, ToxicityResult
from nexus_llm.safety.guardrails import SafetyGuardrails, GuardrailResult

logger = logging.getLogger("nexus_llm.safety.policies")


class PolicyEnforcementMode(Enum):
    """How strictly policies are enforced."""
    PERMISSIVE = "permissive"
    MODERATE = "moderate"
    STRICT = "strict"
    LOCKDOWN = "lockdown"


@dataclass
class SafetyPolicy:
    """A configurable safety policy with rules and enforcement settings."""
    name: str
    description: str
    enforcement_mode: PolicyEnforcementMode = PolicyEnforcementMode.MODERATE
    enabled: bool = True
    priority: int = 0
    blocked_categories: List[FilterCategory] = field(default_factory=list)
    toxicity_threshold: float = 0.5
    max_input_length: int = 10000
    max_output_length: int = 50000
    enable_content_filter: bool = True
    enable_toxicity_detection: bool = True
    enable_guardrails: bool = True
    enable_audit_logging: bool = True
    custom_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "enforcement_mode": self.enforcement_mode.value,
            "enabled": self.enabled,
            "priority": self.priority,
            "blocked_categories": [c.value for c in self.blocked_categories],
            "toxicity_threshold": self.toxicity_threshold,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "enable_content_filter": self.enable_content_filter,
            "enable_toxicity_detection": self.enable_toxicity_detection,
            "enable_guardrails": self.enable_guardrails,
            "enable_audit_logging": self.enable_audit_logging,
            "custom_rules": self.custom_rules,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyPolicy":
        """Create a SafetyPolicy from a dictionary."""
        data["enforcement_mode"] = PolicyEnforcementMode(
            data.get("enforcement_mode", "moderate")
        )
        data["blocked_categories"] = [
            FilterCategory(c) for c in data.get("blocked_categories", [])
        ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AuditLogEntry:
    """An entry in the safety audit log."""
    timestamp: float
    policy_name: str
    action: str
    direction: str
    severity: str
    categories: List[str]
    text_hash: str
    details: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "policy_name": self.policy_name,
            "action": self.action,
            "direction": self.direction,
            "severity": self.severity,
            "categories": self.categories,
            "text_hash": self.text_hash,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "details": self.details,
        }


class AuditLogger:
    """Persistent audit logger for safety policy enforcement.

    Supports in-memory logging with optional file persistence,
    query filtering, and log rotation.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        max_entries: int = 100000,
        persist_interval: int = 100,
    ):
        self._entries: List[AuditLogEntry] = []
        self._log_dir = log_dir
        self._max_entries = max_entries
        self._persist_interval = persist_interval
        self._entries_since_persist = 0

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, entry: AuditLogEntry) -> None:
        """Add an entry to the audit log.

        Args:
            entry: AuditLogEntry to add.
        """
        self._entries.append(entry)
        self._entries_since_persist += 1

        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        if self._log_dir and self._entries_since_persist >= self._persist_interval:
            self.persist()
            self._entries_since_persist = 0

    def query(
        self,
        policy_name: Optional[str] = None,
        action: Optional[str] = None,
        direction: Optional[str] = None,
        min_severity: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """Query the audit log with filters.

        Args:
            policy_name: Filter by policy name.
            action: Filter by action taken.
            direction: Filter by "input" or "output".
            min_severity: Filter by minimum severity.
            user_id: Filter by user ID.
            start_time: Filter by start timestamp.
            end_time: Filter by end timestamp.
            limit: Maximum entries to return.

        Returns:
            List of matching AuditLogEntry objects.
        """
        results = list(reversed(self._entries))

        if policy_name:
            results = [e for e in results if e.policy_name == policy_name]
        if action:
            results = [e for e in results if e.action == action]
        if direction:
            results = [e for e in results if e.direction == direction]
        if user_id:
            results = [e for e in results if e.user_id == user_id]
        if start_time:
            results = [e for e in results if e.timestamp >= start_time]
        if end_time:
            results = [e for e in results if e.timestamp <= end_time]

        return results[:limit]

    def persist(self) -> None:
        """Persist audit log entries to disk."""
        if not self._log_dir:
            return

        filename = time.strftime("audit_%Y%m%d_%H%M%S.json", time.localtime())
        filepath = os.path.join(self._log_dir, filename)

        try:
            entries_data = [e.to_dict() for e in self._entries]
            with open(filepath, "w") as f:
                json.dump(entries_data, f, indent=2)
            logger.info("Audit log persisted to %s (%d entries)", filepath, len(entries_data))
        except OSError as e:
            logger.error("Failed to persist audit log: %s", e)

    def load(self, filepath: str) -> int:
        """Load audit log entries from a file.

        Args:
            filepath: Path to the JSON audit log file.

        Returns:
            Number of entries loaded.
        """
        try:
            with open(filepath, "r") as f:
                entries_data = json.load(f)

            loaded = 0
            for entry_dict in entries_data:
                entry = AuditLogEntry(
                    timestamp=entry_dict.get("timestamp", time.time()),
                    policy_name=entry_dict.get("policy_name", "unknown"),
                    action=entry_dict.get("action", "unknown"),
                    direction=entry_dict.get("direction", "unknown"),
                    severity=entry_dict.get("severity", "none"),
                    categories=entry_dict.get("categories", []),
                    text_hash=entry_dict.get("text_hash", ""),
                    details=entry_dict.get("details", {}),
                    user_id=entry_dict.get("user_id"),
                    request_id=entry_dict.get("request_id"),
                )
                self._entries.append(entry)
                loaded += 1

            return loaded
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load audit log: %s", e)
            return 0

    def get_entry_count(self) -> int:
        """Return the total number of log entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all audit log entries."""
        self._entries.clear()
        self._entries_since_persist = 0


class PolicyEnforcer:
    """Enforces safety policies on input and output text.

    Combines content filtering, toxicity detection, and guardrails
    with configurable policy rules and audit logging.
    """

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self._policies: Dict[str, SafetyPolicy] = {}
        self._active_policy: Optional[str] = None
        self._content_filter = ContentFilter()
        self._toxicity_detector = ToxicityDetector()
        self._guardrails = SafetyGuardrails()
        self._moderator = ContentModerator(
            content_filter=self._content_filter,
            toxicity_detector=self._toxicity_detector,
            guardrails=self._guardrails,
        )
        self._audit_logger = audit_logger or AuditLogger()
        self._enforcement_stats = {
            "total_enforcements": 0,
            "blocked": 0,
            "flagged": 0,
            "allowed": 0,
        }

        self._load_default_policies()

    def _load_default_policies(self) -> None:
        """Load built-in default safety policies."""
        default_policy = SafetyPolicy(
            name="default",
            description="Default safety policy with moderate enforcement",
            enforcement_mode=PolicyEnforcementMode.MODERATE,
            priority=0,
            blocked_categories=[
                FilterCategory.SELF_HARM,
                FilterCategory.VIOLENCE,
                FilterCategory.HATE_SPEECH,
            ],
            toxicity_threshold=0.5,
        )
        self._policies["default"] = default_policy
        self._active_policy = "default"

        strict_policy = SafetyPolicy(
            name="strict",
            description="Strict safety policy blocking most sensitive content",
            enforcement_mode=PolicyEnforcementMode.STRICT,
            priority=10,
            blocked_categories=[
                FilterCategory.SELF_HARM,
                FilterCategory.VIOLENCE,
                FilterCategory.HATE_SPEECH,
                FilterCategory.HARASSMENT,
                FilterCategory.SEXUAL,
                FilterCategory.ILLEGAL,
                FilterCategory.PROFANITY,
            ],
            toxicity_threshold=0.3,
            max_input_length=5000,
            max_output_length=10000,
        )
        self._policies["strict"] = strict_policy

        permissive_policy = SafetyPolicy(
            name="permissive",
            description="Permissive policy allowing most content with flagging",
            enforcement_mode=PolicyEnforcementMode.PERMISSIVE,
            priority=-10,
            blocked_categories=[
                FilterCategory.SELF_HARM,
            ],
            toxicity_threshold=0.8,
        )
        self._policies["permissive"] = permissive_policy

    def check_input(
        self,
        text: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ModerationResult:
        """Check input text against the active safety policy.

        Args:
            text: User input text.
            user_id: Optional user identifier.
            request_id: Optional request identifier.

        Returns:
            ModerationResult with enforcement decision.
        """
        policy = self._get_active_policy()
        if policy is None or not policy.enabled:
            return ModerationResult(
                original_text=text,
                moderated_text=text,
                is_allowed=True,
                reason="No active policy or policy disabled",
            )

        result = self._moderator.moderate_input(
            text, user_id=user_id, request_id=request_id,
        )

        result = self._apply_policy_enforcement(result, policy, "input")

        self._log_enforcement(result, policy, "input", user_id, request_id)
        self._update_stats(result)

        return result

    def check_output(
        self,
        text: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> ModerationResult:
        """Check output text against the active safety policy.

        Args:
            text: Model output text.
            user_id: Optional user identifier.
            request_id: Optional request identifier.

        Returns:
            ModerationResult with enforcement decision.
        """
        policy = self._get_active_policy()
        if policy is None or not policy.enabled:
            return ModerationResult(
                original_text=text,
                moderated_text=text,
                is_allowed=True,
                reason="No active policy or policy disabled",
            )

        result = self._moderator.moderate_output(
            text, user_id=user_id, request_id=request_id,
        )

        result = self._apply_policy_enforcement(result, policy, "output")

        self._log_enforcement(result, policy, "output", user_id, request_id)
        self._update_stats(result)

        return result

    def _apply_policy_enforcement(
        self,
        result: ModerationResult,
        policy: SafetyPolicy,
        direction: str,
    ) -> ModerationResult:
        """Apply policy-specific enforcement adjustments.

        Args:
            result: Current moderation result.
            policy: Active safety policy.
            direction: "input" or "output".

        Returns:
            Adjusted ModerationResult.
        """
        mode = policy.enforcement_mode

        for blocked_cat in policy.blocked_categories:
            if blocked_cat in result.categories_flagged and blocked_cat not in result.categories_blocked:
                if mode in (PolicyEnforcementMode.STRICT, PolicyEnforcementMode.LOCKDOWN):
                    result.categories_blocked.append(blocked_cat)
                    result.categories_flagged.remove(blocked_cat)
                    result.is_allowed = False
                    result.action = ModerationAction.BLOCK

        if mode == PolicyEnforcementMode.LOCKDOWN:
            if result.severity.value >= SeverityLevel.LOW.value:
                result.is_allowed = False
                result.action = ModerationAction.BLOCK

        if mode == PolicyEnforcementMode.PERMISSIVE:
            if result.action == ModerationAction.BLOCK:
                non_critical = all(
                    cat not in [FilterCategory.SELF_HARM, FilterCategory.VIOLENCE]
                    for cat in result.categories_blocked
                )
                if non_critical:
                    result.action = ModerationAction.FLAG
                    result.is_allowed = True
                    result.moderated_text = result.original_text

        if not policy.enable_content_filter:
            result.filter_result = None

        if not policy.enable_toxicity_detection:
            result.toxicity_result = None

        if not policy.enable_guardrails:
            result.guardrail_result = None

        return result

    def _get_active_policy(self) -> Optional[SafetyPolicy]:
        """Get the currently active safety policy."""
        if self._active_policy and self._active_policy in self._policies:
            return self._policies[self._active_policy]
        return None

    def _log_enforcement(
        self,
        result: ModerationResult,
        policy: SafetyPolicy,
        direction: str,
        user_id: Optional[str],
        request_id: Optional[str],
    ) -> None:
        """Log a policy enforcement action."""
        if not policy.enable_audit_logging:
            return

        import hashlib
        text_hash = hashlib.sha256(result.original_text.encode()).hexdigest()[:16]

        entry = AuditLogEntry(
            timestamp=time.time(),
            policy_name=policy.name,
            action=result.action.value,
            direction=direction,
            severity=result.severity.name,
            categories=[c.value for c in result.categories_blocked + result.categories_flagged],
            text_hash=text_hash,
            details={"reason": result.reason},
            user_id=user_id,
            request_id=request_id,
        )
        self._audit_logger.log(entry)

    def _update_stats(self, result: ModerationResult) -> None:
        """Update enforcement statistics."""
        self._enforcement_stats["total_enforcements"] += 1
        if result.action == ModerationAction.BLOCK:
            self._enforcement_stats["blocked"] += 1
        elif result.action == ModerationAction.FLAG:
            self._enforcement_stats["flagged"] += 1
        else:
            self._enforcement_stats["allowed"] += 1

    def set_active_policy(self, name: str) -> None:
        """Set the active policy by name.

        Args:
            name: Name of the policy to activate.

        Raises:
            KeyError: If the policy name doesn't exist.
        """
        if name not in self._policies:
            raise KeyError(f"Policy '{name}' not found. Available: {list(self._policies.keys())}")
        self._active_policy = name
        logger.info("Active safety policy set to: %s", name)

    def add_policy(self, policy: SafetyPolicy) -> None:
        """Add or update a safety policy.

        Args:
            policy: SafetyPolicy to add.
        """
        self._policies[policy.name] = policy

    def remove_policy(self, name: str) -> bool:
        """Remove a safety policy.

        Args:
            name: Policy name to remove.

        Returns:
            True if the policy was found and removed.
        """
        if name in self._policies:
            del self._policies[name]
            if self._active_policy == name:
                self._active_policy = "default" if "default" in self._policies else None
            return True
        return False

    def list_policies(self) -> List[Dict[str, Any]]:
        """List all registered policies."""
        return [
            {**policy.to_dict(), "is_active": policy.name == self._active_policy}
            for policy in self._policies.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Return enforcement statistics."""
        return dict(self._enforcement_stats)

    def get_audit_entries(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Query audit log entries."""
        entries = self._audit_logger.query(**kwargs)
        return [e.to_dict() for e in entries]

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        active = self._get_active_policy()
        return {
            "active_policy": active.name if active else None,
            "available_policies": list(self._policies.keys()),
            "enforcement_stats": self.get_stats(),
            "audit_log_size": self._audit_logger.get_entry_count(),
            "moderator_config": self._moderator.get_config(),
        }

    def export_policies(self, filepath: str) -> None:
        """Export all policies to a JSON file.

        Args:
            filepath: Output file path.
        """
        data = {name: policy.to_dict() for name, policy in self._policies.items()}
        data["__active_policy"] = self._active_policy
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_policies(self, filepath: str) -> int:
        """Import policies from a JSON file.

        Args:
            filepath: Input file path.

        Returns:
            Number of policies imported.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        active = data.pop("__active_policy", None)
        imported = 0

        for name, policy_data in data.items():
            try:
                policy = SafetyPolicy.from_dict(policy_data)
                self._policies[name] = policy
                imported += 1
            except Exception as e:
                logger.warning("Failed to import policy '%s': %s", name, e)

        if active and active in self._policies:
            self._active_policy = active

        return imported


# Global policy enforcer
_enforcer: Optional[PolicyEnforcer] = None


def get_policy_enforcer() -> PolicyEnforcer:
    """Get the global policy enforcer singleton."""
    global _enforcer
    if _enforcer is None:
        _enforcer = PolicyEnforcer()
    return _enforcer


def init_safety(
    policy_name: str = "default",
    audit_log_dir: Optional[str] = None,
) -> PolicyEnforcer:
    """Initialize the global safety system.

    Args:
        policy_name: Name of the initial active policy.
        audit_log_dir: Optional directory for audit log persistence.

    Returns:
        The initialized PolicyEnforcer.
    """
    global _enforcer
    audit_logger = AuditLogger(log_dir=audit_log_dir)
    _enforcer = PolicyEnforcer(audit_logger=audit_logger)
    if policy_name in _enforcer._policies:
        _enforcer.set_active_policy(policy_name)
    return _enforcer
