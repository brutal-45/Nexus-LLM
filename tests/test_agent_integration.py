"""Tests for agent module integration."""
import pytest

from nexus_llm.agents import (
    Agent,
    AgentState,
    AgentConfig,
    ChatAgent,
    CodeAgent,
    ResearchAgent,
    ToolAgent,
    ActionExecutor,
    AgentMemory,
    ShortTermMemory,
    LongTermMemory,
    EpisodicMemory,
    TaskPlanner,
    Plan,
    Step,
    Tool,
    ToolResult,
)


class TestAgentModuleImports:
    """Test that all agent module components can be imported."""

    def test_base_imports(self):
        assert Agent is not None
        assert AgentState is not None
        assert AgentConfig is not None

    def test_agent_imports(self):
        assert ChatAgent is not None
        assert CodeAgent is not None
        assert ResearchAgent is not None
        assert ToolAgent is not None

    def test_executor_import(self):
        assert ActionExecutor is not None

    def test_memory_imports(self):
        assert AgentMemory is not None
        assert ShortTermMemory is not None
        assert LongTermMemory is not None
        assert EpisodicMemory is not None

    def test_planner_imports(self):
        assert TaskPlanner is not None
        assert Plan is not None
        assert Step is not None

    def test_tool_imports(self):
        assert Tool is not None
        assert ToolResult is not None


class TestAgentConfigIntegration:
    """Test AgentConfig creation."""

    def test_create_config(self):
        config = AgentConfig()
        assert config is not None


class TestAgentStateEnum:
    """Test AgentState enum."""

    def test_states_exist(self):
        assert AgentState is not None


class TestMemoryIntegration:
    """Test memory components."""

    def test_short_term_memory(self):
        memory = ShortTermMemory()
        assert memory is not None

    def test_long_term_memory(self):
        memory = LongTermMemory()
        assert memory is not None

    def test_agent_memory_is_abstract(self):
        """AgentMemory is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentMemory()
