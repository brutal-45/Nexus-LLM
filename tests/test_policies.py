"""Test safety policies for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class PolicyAction(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    REDACT = "redact"


@dataclass
class Policy:
    name: str
    description: str
    action: PolicyAction = PolicyAction.BLOCK
    enabled: bool = True
    priority: int = 0
    conditions: Dict = field(default_factory=dict)

    def evaluate(self, context: dict) -> PolicyAction:
        if not self.enabled:
            return PolicyAction.ALLOW
        for key, expected in self.conditions.items():
            actual = context.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return PolicyAction.ALLOW
            elif actual != expected:
                return PolicyAction.ALLOW
        return self.action


class PolicyManager:
    def __init__(self):
        self._policies: Dict[str, Policy] = {}

    def add_policy(self, policy: Policy):
        if policy.name in self._policies:
            raise ValueError(f"Policy '{policy.name}' already exists")
        self._policies[policy.name] = policy

    def remove_policy(self, name: str):
        if name not in self._policies:
            raise KeyError(f"Policy '{name}' not found")
        del self._policies[name]

    def get_policy(self, name: str) -> Optional[Policy]:
        return self._policies.get(name)

    def list_policies(self) -> List[Policy]:
        return sorted(self._policies.values(), key=lambda p: p.priority, reverse=True)

    def evaluate(self, context: dict) -> PolicyAction:
        actions = []
        for policy in self.list_policies():
            if policy.enabled:
                action = policy.evaluate(context)
                actions.append(action)
        if PolicyAction.BLOCK in actions:
            return PolicyAction.BLOCK
        if PolicyAction.REDACT in actions:
            return PolicyAction.REDACT
        if PolicyAction.WARN in actions:
            return PolicyAction.WARN
        return PolicyAction.ALLOW

    def enable_policy(self, name: str):
        if name not in self._policies:
            raise KeyError(f"Policy '{name}' not found")
        self._policies[name].enabled = True

    def disable_policy(self, name: str):
        if name not in self._policies:
            raise KeyError(f"Policy '{name}' not found")
        self._policies[name].enabled = False


# Built-in policies
BUILTIN_POLICIES = [
    Policy(name="block_harmful", description="Block harmful content", action=PolicyAction.BLOCK, priority=10,
           conditions={"category": "harmful"}),
    Policy(name="warn_sensitive", description="Warn on sensitive topics", action=PolicyAction.WARN, priority=5,
           conditions={"category": "sensitive"}),
    Policy(name="redact_pii", description="Redact PII", action=PolicyAction.REDACT, priority=8,
           conditions={"contains_pii": True}),
]


class TestPolicy:
    def test_creation(self):
        policy = Policy(name="test", description="Test policy")
        assert policy.name == "test"
        assert policy.action == PolicyAction.BLOCK
        assert policy.enabled is True

    def test_evaluate_matching(self):
        policy = Policy(name="test", description="", action=PolicyAction.BLOCK,
                        conditions={"category": "harmful"})
        assert policy.evaluate({"category": "harmful"}) == PolicyAction.BLOCK

    def test_evaluate_not_matching(self):
        policy = Policy(name="test", description="", action=PolicyAction.BLOCK,
                        conditions={"category": "harmful"})
        assert policy.evaluate({"category": "safe"}) == PolicyAction.ALLOW

    def test_evaluate_disabled(self):
        policy = Policy(name="test", description="", action=PolicyAction.BLOCK, enabled=False,
                        conditions={"category": "harmful"})
        assert policy.evaluate({"category": "harmful"}) == PolicyAction.ALLOW

    def test_evaluate_list_condition(self):
        policy = Policy(name="test", description="", action=PolicyAction.WARN,
                        conditions={"category": ["harmful", "sensitive"]})
        assert policy.evaluate({"category": "harmful"}) == PolicyAction.WARN
        assert policy.evaluate({"category": "safe"}) == PolicyAction.ALLOW

    def test_priority(self):
        policy = Policy(name="test", description="", priority=5)
        assert policy.priority == 5


class TestPolicyManager:
    def test_add_policy(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="test", description="Test"))
        assert pm.get_policy("test") is not None

    def test_add_duplicate(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="test", description="Test"))
        with pytest.raises(ValueError, match="already exists"):
            pm.add_policy(Policy(name="test", description="Dup"))

    def test_remove_policy(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="test", description="Test"))
        pm.remove_policy("test")
        assert pm.get_policy("test") is None

    def test_remove_nonexistent(self):
        pm = PolicyManager()
        with pytest.raises(KeyError, match="not found"):
            pm.remove_policy("nonexistent")

    def test_list_policies_sorted_by_priority(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="low", description="", priority=1))
        pm.add_policy(Policy(name="high", description="", priority=10))
        pm.add_policy(Policy(name="mid", description="", priority=5))
        policies = pm.list_policies()
        assert policies[0].name == "high"
        assert policies[-1].name == "low"

    def test_evaluate_block_takes_precedence(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="warn", description="", action=PolicyAction.WARN, priority=5,
                             conditions={"type": "A"}))
        pm.add_policy(Policy(name="block", description="", action=PolicyAction.BLOCK, priority=10,
                             conditions={"type": "A"}))
        result = pm.evaluate({"type": "A"})
        assert result == PolicyAction.BLOCK

    def test_evaluate_allow_when_no_match(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="block", description="", action=PolicyAction.BLOCK,
                             conditions={"category": "harmful"}))
        result = pm.evaluate({"category": "safe"})
        assert result == PolicyAction.ALLOW

    def test_enable_disable(self):
        pm = PolicyManager()
        pm.add_policy(Policy(name="test", description="", action=PolicyAction.BLOCK,
                             conditions={"x": 1}))
        pm.disable_policy("test")
        result = pm.evaluate({"x": 1})
        assert result == PolicyAction.ALLOW
        pm.enable_policy("test")
        result = pm.evaluate({"x": 1})
        assert result == PolicyAction.BLOCK


class TestBuiltinPolicies:
    def test_block_harmful_exists(self):
        names = [p.name for p in BUILTIN_POLICIES]
        assert "block_harmful" in names

    def test_warn_sensitive_exists(self):
        names = [p.name for p in BUILTIN_POLICIES]
        assert "warn_sensitive" in names

    def test_redact_pii_exists(self):
        names = [p.name for p in BUILTIN_POLICIES]
        assert "redact_pii" in names

    def test_all_enabled(self):
        for policy in BUILTIN_POLICIES:
            assert policy.enabled is True
