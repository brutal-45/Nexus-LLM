"""Nexus-LLM Safety Module.

Provides content filtering, moderation, toxicity detection,
guardrails, and configurable safety policies for LLM serving.
"""

from nexus_llm.safety.content_filter import (
    CategoryRule,
    ContentFilter,
    FilterAction,
    FilterCategory,
    FilterMatch,
    FilterResult,
    FilterRule,
    KeywordRule,
    RegexRule,
)
from nexus_llm.safety.moderation import (
    ContentModerator,
    ModerationAction,
    ModerationAuditEntry,
    ModerationPolicy,
    ModerationResult,
    SeverityLevel,
)
from nexus_llm.safety.toxicity import (
    CategoryScore,
    ToxicityCategory,
    ToxicityDetector,
    ToxicityResult,
)
from nexus_llm.safety.guardrails import (
    GuardrailResult,
    GuardrailViolation,
    SafetyGuardrails,
    TopicRestriction,
)
from nexus_llm.safety.policies import (
    AuditLogger,
    AuditLogEntry,
    PolicyEnforcer,
    PolicyEnforcementMode,
    SafetyPolicy,
    get_policy_enforcer,
    init_safety,
)


__all__ = [
    # Content Filter
    "CategoryRule",
    "ContentFilter",
    "FilterAction",
    "FilterCategory",
    "FilterMatch",
    "FilterResult",
    "FilterRule",
    "KeywordRule",
    "RegexRule",
    # Moderation
    "ContentModerator",
    "ModerationAction",
    "ModerationAuditEntry",
    "ModerationPolicy",
    "ModerationResult",
    "SeverityLevel",
    # Toxicity
    "CategoryScore",
    "ToxicityCategory",
    "ToxicityDetector",
    "ToxicityResult",
    # Guardrails
    "GuardrailResult",
    "GuardrailViolation",
    "SafetyGuardrails",
    "TopicRestriction",
    # Policies
    "AuditLogger",
    "AuditLogEntry",
    "PolicyEnforcer",
    "PolicyEnforcementMode",
    "SafetyPolicy",
    "get_policy_enforcer",
    "init_safety",
]
