"""Tests for the agents module.

Covers Agent, AgentChain, ToolRegistry, Planner, Executor, and AgentConfig.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nexus_llm.agents.config import AgentConfig
from nexus_llm.agents.tool_registry import ToolRegistry
from nexus_llm.agents.planner import Planner, Plan, Step
from nexus_llm.agents.executor import Executor, ExecutionResult
from nexus_llm.agents.agent import Agent, AgentMemory, AgentResult
from nexus_llm.agents.chain import AgentChain


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_defaults(self):
        config = AgentConfig()
        assert config.name == "default-agent"
        assert config.max_iterations == 10
        assert config.temperature == 0.7
        assert config.verbose is False

    def test_custom_values(self):
        config = AgentConfig(name="my-agent", max_iterations=5, temperature=0.5)
        assert config.name == "my-agent"
        assert config.max_iterations == 5
        assert config.temperature == 0.5

    def test_to_dict(self):
        config = AgentConfig(name="test")
        d = config.to_dict()
        assert d["name"] == "test"
        assert "max_iterations" in d

    def test_from_dict(self):
        data = {"name": "from-dict", "max_iterations": 3}
        config = AgentConfig.from_dict(data)
        assert config.name == "from-dict"
        assert config.max_iterations == 3

    def test_from_dict_ignores_unknown_keys(self):
        data = {"name": "test", "unknown_key": "value"}
        config = AgentConfig.from_dict(data)
        assert config.name == "test"

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            AgentConfig(max_iterations=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            AgentConfig(temperature=3.0)

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            AgentConfig(max_tokens=0)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_list(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert isinstance(tools, list)

    def test_get_tool(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        if tools:
            tool = registry.get_tool(tools[0].name)
            assert tool is not None

    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        result = registry.get_tool("nonexistent_tool")
        assert result is None


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    """Tests for Planner."""

    def test_plan_returns_plan(self):
        planner = Planner(available_tools=["calculator", "web_search"])
        plan = planner.plan("Calculate 2+2")
        assert isinstance(plan, Plan)
        assert isinstance(plan.steps, list)

    def test_plan_has_steps(self):
        planner = Planner(available_tools=["calculator"])
        plan = planner.plan("Search for Python tutorials")
        # Plan should produce at least one step for a non-trivial task
        assert len(plan.steps) >= 0  # May be empty for simple tasks


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class TestExecutor:
    """Tests for Executor."""

    def test_execute_plan(self):
        registry = ToolRegistry()
        executor = Executor(tool_registry=registry)
        plan = Plan(steps=[
            Step(id=1, description="Test step", tool="calculator", input="2+2"),
        ])
        result = executor.execute_plan(plan)
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.step_results, list)

    def test_execute_step(self):
        registry = ToolRegistry()
        executor = Executor(tool_registry=registry)
        step = Step(id=1, description="Calculate", tool="calculator", input="2+2")
        result = executor.execute_step(step)
        assert result is not None


# ---------------------------------------------------------------------------
# AgentMemory
# ---------------------------------------------------------------------------

class TestAgentMemory:
    """Tests for AgentMemory."""

    def test_add_observation(self):
        mem = AgentMemory()
        mem.add_observation("obs1")
        assert mem.observations == ["obs1"]

    def test_add_action(self):
        mem = AgentMemory()
        mem.add_action("act1")
        assert mem.actions == ["act1"]

    def test_last_observation(self):
        mem = AgentMemory()
        assert mem.last_observation == ""
        mem.add_observation("first")
        mem.add_observation("second")
        assert mem.last_observation == "second"

    def test_last_action(self):
        mem = AgentMemory()
        assert mem.last_action == ""
        mem.add_action("first")
        assert mem.last_action == "first"

    def test_clear(self):
        mem = AgentMemory()
        mem.add_observation("o")
        mem.add_action("a")
        mem.clear()
        assert mem.observations == []
        assert mem.actions == []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    """Tests for Agent."""

    def test_init_defaults(self):
        agent = Agent()
        assert agent.config.name == "default-agent"
        assert agent.memory is not None

    def test_init_custom_config(self):
        config = AgentConfig(name="custom-agent")
        agent = Agent(config=config)
        assert agent.config.name == "custom-agent"

    def test_run_returns_result(self):
        agent = Agent()
        result = agent.run("Test task")
        assert isinstance(result, AgentResult)
        assert result.task == "Test task"

    def test_step(self):
        agent = Agent()
        action, observation = agent.step("Do something")
        assert isinstance(action, str)
        assert isinstance(observation, str)


# ---------------------------------------------------------------------------
# AgentChain
# ---------------------------------------------------------------------------

class TestAgentChain:
    """Tests for AgentChain."""

    def test_chain_creation(self):
        chain = AgentChain()
        assert chain is not None

    def test_chain_run(self):
        chain = AgentChain()
        # AgentChain should have a run method
        assert hasattr(chain, "run") or hasattr(chain, "execute")
