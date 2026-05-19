"""Test base agent for Nexus-LLM."""
import pytest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


class AgentError(Exception):
    pass


@dataclass
class AgentConfig:
    name: str = "base-agent"
    description: str = ""
    max_iterations: int = 10
    verbose: bool = False


@dataclass
class AgentState:
    status: str = "idle"  # idle, running, completed, failed
    current_step: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_result(self, result):
        self.results.append(result)

    def add_error(self, error: str):
        self.errors.append(error)

    @property
    def is_complete(self):
        return self.status in ("completed", "failed")


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig = None):
        self._config = config or AgentConfig()
        self._state = AgentState()

    @property
    def config(self):
        return self._config

    @property
    def name(self):
        return self._config.name

    @property
    def state(self):
        return self._state

    def reset(self):
        self._state = AgentState()

    @abstractmethod
    def run(self, task: str, **kwargs) -> Any:
        pass

    def _start(self):
        self._state.status = "running"
        self._state.current_step = 0

    def _complete(self, result=None):
        self._state.status = "completed"
        if result is not None:
            self._state.add_result(result)

    def _fail(self, error: str):
        self._state.status = "failed"
        self._state.add_error(error)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self._config.name,
            "description": self._config.description,
            "status": self._state.status,
            "steps_completed": self._state.current_step,
        }


class SimpleAgent(BaseAgent):
    def run(self, task: str, **kwargs) -> str:
        self._start()
        try:
            if not task:
                self._fail("Task cannot be empty")
                return "Error: Task cannot be empty"
            self._state.current_step = 1
            result = f"Completed task: {task}"
            self._complete(result)
            return result
        except Exception as e:
            self._fail(str(e))
            return f"Error: {e}"


class TestAgentConfig:
    def test_defaults(self):
        config = AgentConfig()
        assert config.name == "base-agent"
        assert config.max_iterations == 10

    def test_custom(self):
        config = AgentConfig(name="custom", max_iterations=5)
        assert config.name == "custom"


class TestAgentState:
    def test_initial_state(self):
        state = AgentState()
        assert state.status == "idle"
        assert state.is_complete is False

    def test_add_result(self):
        state = AgentState()
        state.add_result("result1")
        assert len(state.results) == 1

    def test_add_error(self):
        state = AgentState()
        state.add_error("error1")
        assert len(state.errors) == 1

    def test_is_complete(self):
        state = AgentState(status="completed")
        assert state.is_complete is True
        state.status = "failed"
        assert state.is_complete is True


class TestBaseAgent:
    def test_simple_agent_run(self):
        agent = SimpleAgent()
        result = agent.run("test task")
        assert "test task" in result
        assert agent.state.status == "completed"

    def test_empty_task(self):
        agent = SimpleAgent()
        result = agent.run("")
        assert agent.state.status == "failed"

    def test_reset(self):
        agent = SimpleAgent()
        agent.run("test")
        agent.reset()
        assert agent.state.status == "idle"
        assert len(agent.state.results) == 0

    def test_get_info(self):
        agent = SimpleAgent(AgentConfig(name="test-agent"))
        info = agent.get_info()
        assert info["name"] == "test-agent"
        assert info["status"] == "idle"

    def test_name_property(self):
        agent = SimpleAgent(AgentConfig(name="my-agent"))
        assert agent.name == "my-agent"

    def test_config_property(self):
        config = AgentConfig(name="x")
        agent = SimpleAgent(config)
        assert agent.config is config
