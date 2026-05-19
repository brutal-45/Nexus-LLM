"""Base agent with think/act/observe loop, tool usage, and state management.

Provides the foundational Agent class implementing a reactive agent
architecture with reasoning, action selection, observation processing,
and configurable behavior.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.memory import AgentMemory, ShortTermMemory
from nexus_llm.agents.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Possible states of an agent."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    name: str = "Agent"
    description: str = "A general-purpose agent"
    max_iterations: int = 10
    thinking_model: Optional[str] = None
    verbose: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 300.0  # seconds
    retry_attempts: int = 2
    tools: List[str] = field(default_factory=list)


@dataclass
class AgentAction:
    """An action taken by the agent."""

    action_type: str  # "tool_call", "respond", "think"
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    thought: Optional[str] = None
    response: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "thought": self.thought,
            "response": self.response,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentObservation:
    """An observation from the environment after an action."""

    action: AgentAction
    result: Optional[ToolResult] = None
    observation_text: str = ""
    success: bool = True
    timestamp: float = field(default_factory=time.time)


class Agent(ABC):
    """Base agent implementing a think/act/observe loop.

    Agents operate in an iterative cycle:
    1. THINK: Analyze the current state and decide what to do
    2. ACT: Execute an action (tool call or response)
    3. OBSERVE: Process the result of the action

    This loop continues until the agent reaches a conclusion
    or exhausts its maximum iterations.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
        memory: Optional[AgentMemory] = None,
        llm_fn: Optional[Callable] = None,
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration.
            tools: Dictionary of available tools.
            memory: Agent memory system.
            llm_fn: LLM function for generation. Takes prompt, returns response.
        """
        self.config = config or AgentConfig()
        self.tools: Dict[str, Tool] = tools or {}
        self.memory = memory or ShortTermMemory()
        self.llm_fn = llm_fn

        self.agent_id = str(uuid.uuid4())[:8]
        self.state = AgentState.IDLE
        self.iteration = 0
        self.action_history: List[AgentAction] = []
        self.observation_history: List[AgentObservation] = []
        self._current_task: Optional[str] = None

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with the agent."""
        self.tools[tool.name] = tool
        logger.debug("Agent %s: Added tool '%s'.", self.config.name, tool.name)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent."""
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True
        return False

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Execute the think/act/observe loop for a task.

        Args:
            task: The task or query to process.
            context: Optional context information.

        Returns:
            The agent's final response.
        """
        self._current_task = task
        self.state = AgentState.THINKING
        self.iteration = 0
        self.action_history.clear()
        self.observation_history.clear()

        # Store the task in memory
        self.memory.store("current_task", task)
        if context:
            for key, value in context.items():
                self.memory.store(f"context_{key}", value)

        logger.info("Agent %s starting task: %s", self.config.name, task[:100])

        final_response = ""
        start_time = time.time()

        while self.iteration < self.config.max_iterations:
            if time.time() - start_time > self.config.timeout:
                logger.warning("Agent %s timed out after %.1fs.", self.config.name, self.config.timeout)
                final_response = "I ran out of time while processing your request."
                self.state = AgentState.ERROR
                break

            self.iteration += 1

            # THINK phase
            self.state = AgentState.THINKING
            action = self.think(task, context)

            if action is None:
                logger.debug("Agent %s: No action produced, finishing.", self.config.name)
                break

            self.action_history.append(action)

            # Check if agent wants to respond directly
            if action.action_type == "respond":
                final_response = action.response or ""
                self.state = AgentState.FINISHED
                break

            # ACT phase
            self.state = AgentState.ACTING
            observation = self.act(action)
            self.observation_history.append(observation)

            # OBSERVE phase
            self.state = AgentState.OBSERVING
            self.observe(observation)

            # Store observation in memory
            self.memory.store(
                f"observation_{self.iteration}",
                observation.observation_text,
            )

            # Check for errors
            if not observation.success:
                logger.warning(
                    "Agent %s: Action failed at iteration %d: %s",
                    self.config.name,
                    self.iteration,
                    observation.observation_text,
                )

        if self.state != AgentState.FINISHED:
            if not final_response:
                final_response = self._summarize_observations()
            self.state = AgentState.FINISHED

        logger.info(
            "Agent %s completed task in %d iterations.",
            self.config.name,
            self.iteration,
        )
        return final_response

    @abstractmethod
    def think(self, task: str, context: Optional[Dict[str, Any]] = None) -> Optional[AgentAction]:
        """Decide what action to take based on current state.

        Args:
            task: The current task.
            context: Optional context.

        Returns:
            An AgentAction to execute, or None to stop.
        """
        ...

    def act(self, action: AgentAction) -> AgentObservation:
        """Execute an action.

        Args:
            action: The action to execute.

        Returns:
            An AgentObservation with the result.
        """
        if action.action_type == "tool_call":
            return self._execute_tool(action)
        elif action.action_type == "think":
            return AgentObservation(
                action=action,
                observation_text=action.thought or "Internal thinking step.",
                success=True,
            )
        else:
            return AgentObservation(
                action=action,
                observation_text=f"Unknown action type: {action.action_type}",
                success=False,
            )

    def observe(self, observation: AgentObservation) -> None:
        """Process an observation and update agent state.

        Override in subclasses for custom observation handling.
        """
        pass

    def _execute_tool(self, action: AgentAction) -> AgentObservation:
        """Execute a tool call action."""
        tool_name = action.tool_name
        tool_args = action.tool_args or {}

        if tool_name not in self.tools:
            return AgentObservation(
                action=action,
                observation_text=f"Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}",
                success=False,
            )

        tool = self.tools[tool_name]

        try:
            result = tool.execute(**tool_args)
            return AgentObservation(
                action=action,
                result=result,
                observation_text=result.output if result.success else f"Tool error: {result.error}",
                success=result.success,
            )
        except Exception as e:
            return AgentObservation(
                action=action,
                observation_text=f"Exception executing tool '{tool_name}': {e}",
                success=False,
            )

    def _summarize_observations(self) -> str:
        """Generate a summary from observation history."""
        if not self.observation_history:
            return "I was unable to complete the task."

        parts = []
        for obs in self.observation_history:
            if obs.success and obs.observation_text:
                parts.append(obs.observation_text)

        if parts:
            return "Based on my analysis:\n" + "\n".join(parts[:5])
        return "I was unable to gather sufficient information."

    def _build_prompt(self, task: str, include_history: bool = True) -> str:
        """Build a prompt for the LLM with task and context."""
        parts = [
            f"You are {self.config.name}. {self.config.description}",
            f"\nTask: {task}",
        ]

        if include_history and self.action_history:
            parts.append("\nPrevious actions:")
            for i, action in enumerate(self.action_history[-5:]):
                if action.thought:
                    parts.append(f"  Step {i + 1} - Thought: {action.thought}")
                if action.tool_name:
                    parts.append(f"  Step {i + 1} - Tool: {action.tool_name}({action.tool_args})")

        if self.observation_history:
            parts.append("\nObservations:")
            for i, obs in enumerate(self.observation_history[-5:]):
                parts.append(f"  Step {i + 1}: {obs.observation_text[:200]}")

        available_tools = list(self.tools.keys())
        if available_tools:
            parts.append(f"\nAvailable tools: {', '.join(available_tools)}")

        return "\n".join(parts)

    def reset(self) -> None:
        """Reset the agent to idle state."""
        self.state = AgentState.IDLE
        self.iteration = 0
        self.action_history.clear()
        self.observation_history.clear()
        self._current_task = None

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "state": self.state.value,
            "iteration": self.iteration,
            "max_iterations": self.config.max_iterations,
            "tools": list(self.tools.keys()),
            "current_task": self._current_task,
            "action_count": len(self.action_history),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, state={self.state.value})"
