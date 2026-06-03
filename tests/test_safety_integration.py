"""Tests for safety module integration."""
import pytest

from nexus_llm.safety import (
    ContentFilter,
    FilterCategory,
    FilterAction,
    ContentModerator,
    SeverityLevel,
    ToxicityDetector,
    SafetyGuardrails,
    PolicyEnforcer,
    SafetyPolicy,
    init_safety,
    get_policy_enforcer,
)


class TestSafetyModuleImports:
    """Test that all safety module components can be imported."""

    def test_content_filter_importable(self):
        assert ContentFilter is not None

    def test_filter_category_importable(self):
        assert FilterCategory is not None

    def test_filter_action_importable(self):
        assert FilterAction is not None

    def test_content_moderator_importable(self):
        assert ContentModerator is not None

    def test_severity_level_importable(self):
        assert SeverityLevel is not None

    def test_toxicity_detector_importable(self):
        assert ToxicityDetector is not None

    def test_safety_guardrails_importable(self):
        assert SafetyGuardrails is not None

    def test_policy_enforcer_importable(self):
        assert PolicyEnforcer is not None

    def test_safety_policy_importable(self):
        assert SafetyPolicy is not None


class TestContentFilterIntegration:
    """Test ContentFilter basic usage."""

    def test_create_filter(self):
        cf = ContentFilter()
        assert cf is not None

    def test_filter_categories_exist(self):
        assert hasattr(FilterCategory, 'HATE_SPEECH') or len(list(FilterCategory)) > 0

    def test_filter_actions_exist(self):
        assert hasattr(FilterAction, 'BLOCK') or len(list(FilterAction)) > 0


class TestContentModeratorIntegration:
    """Test ContentModerator basic usage."""

    def test_create_moderator(self):
        mod = ContentModerator()
        assert mod is not None


class TestToxicityDetectorIntegration:
    """Test ToxicityDetector basic usage."""

    def test_create_detector(self):
        detector = ToxicityDetector()
        assert detector is not None


class TestSafetyGuardrailsIntegration:
    """Test SafetyGuardrails basic usage."""

    def test_create_guardrails(self):
        guardrails = SafetyGuardrails()
        assert guardrails is not None


class TestSafetyPoliciesIntegration:
    """Test safety policies and enforcer."""

    def test_init_safety(self):
        enforcer = init_safety()
        assert enforcer is not None

    def test_get_policy_enforcer(self):
        enforcer = get_policy_enforcer()
        assert enforcer is not None

    def test_safety_policy_creation(self):
        policy = SafetyPolicy(name="test", description="Test policy")
        assert policy.name == "test"
