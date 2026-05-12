"""
Agent Framework Module
=======================

Core agent framework providing the abstract base class and concrete agent
implementations for the Nexus LLM agent system.  Agents follow a
think-plan-act-reflect loop and are composable with tools, memory, and
multi-agent orchestration.

Classes
-------
- ``Message``: Immutable dataclass representing a single conversation turn.
- ``ToolCall``: Represents a pending tool invocation from the LLM.
- ``ToolCallResult``: Result of a completed tool invocation.
- ``AgentState``: Captures the agent's current internal state.
- ``BaseAgent``: Abstract base class with the full reasoning pipeline.
- ``ConversationalAgent``: Chat-focused agent with dialogue state management.
- ``TaskAgent``: Goal-oriented agent with planning and tool use.
- ``ReflectiveAgent``: Self-reflective agent that critiques its own reasoning.

Architecture
------------
Each agent implementation follows the **observe → think → plan → act →
reflect → respond** cycle.  Subclasses override specific stages while
inheriting the orchestration logic from ``BaseAgent.run()``.
"""

from __future__ import annotations

import abc
import copy
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from nexus.agents.agent_config import (
    AgentConfig,
    AgentMemoryConfig,
    ReasoningStrategy,
    ToolConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MessageRole(Enum):
    """Roles that participants can take in a conversation.

    Attributes
    ----------
    SYSTEM : str
        System-level instruction from the framework.
    USER : str
        Human user input.
    ASSISTANT : str
        Agent's own response.
    TOOL : str
        Output returned by an external tool.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the LLM.

    Parameters
    ----------
    id : str
        Unique identifier for this tool call instance.
    name : str
        Name of the tool to invoke.
    arguments : Dict[str, Any]
        Arguments to pass to the tool handler.
    """
    id: str = ""
    name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"call_{uuid.uuid4().hex[:12]}"
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolCall.name must be a non-empty string")
        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall.arguments must be a dict")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the tool call.
        """
        return {
            "id": self.id,
            "name": self.name,
            "arguments": copy.deepcopy(self.arguments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCall:
        """Deserialize from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with tool call fields.

        Returns
        -------
        ToolCall
            Deserialized tool call.
        """
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
        )


@dataclass
class ToolCallResult:
    """Result of a completed tool invocation.

    Parameters
    ----------
    tool_call_id : str
        ID matching the originating ``ToolCall.id``.
    output : str
        The textual output from the tool.
    is_error : bool
        Whether the tool execution encountered an error.
    metadata : Dict[str, Any]
        Additional metadata about the execution.
    """
    tool_call_id: str = ""
    output: str = ""
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation.
        """
        return {
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "is_error": self.is_error,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class Message:
    """Immutable dataclass representing a single conversation turn.

    Messages flow between the user, agent, and tools during the
    reasoning loop.  Each message tracks its role, content, timing,
    optional tool calls, and lineage via ``parent_id``.

    Parameters
    ----------
    role : str
        The participant role (``"user"``, ``"assistant"``, ``"system"``,
        or ``"tool"``).
    content : str
        The text content of the message.
    timestamp : Optional[datetime]
        When the message was created.  Defaults to now.
    metadata : Dict[str, Any]
        Arbitrary key-value metadata attached to this message.
    tool_calls : List[ToolCall]
        Tool invocations requested alongside this message.
    tool_results : List[ToolCallResult]
        Results from previously requested tool calls.
    parent_id : Optional[str]
        ID of the preceding message for conversation threading.
    id : str
        Unique identifier for this message.
    """

    role: str = "user"
    content: str = ""
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolCallResult] = field(default_factory=list)
    parent_id: Optional[str] = None
    id: str = ""

    def __post_init__(self) -> None:
        """Set defaults for optional fields."""
        if not self.id:
            self.id = uuid.uuid4().hex
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if not isinstance(self.role, str):
            raise TypeError(f"Message.role must be a string, got {type(self.role)}")
        valid_roles = {r.value for r in MessageRole}
        if self.role not in valid_roles:
            logger.warning(
                "Message role '%s' is not a standard role (%s)",
                self.role,
                sorted(valid_roles),
            )

    @property
    def is_user(self) -> bool:
        """Whether this message originated from the user."""
        return self.role == MessageRole.USER.value

    @property
    def is_assistant(self) -> bool:
        """Whether this message originated from the agent."""
        return self.role == MessageRole.ASSISTANT.value

    @property
    def is_system(self) -> bool:
        """Whether this message is a system instruction."""
        return self.role == MessageRole.SYSTEM.value

    @property
    def is_tool(self) -> bool:
        """Whether this message is a tool result."""
        return self.role == MessageRole.TOOL.value

    @property
    def has_tool_calls(self) -> bool:
        """Whether this message contains tool call requests."""
        return len(self.tool_calls) > 0

    @property
    def has_tool_results(self) -> bool:
        """Whether this message contains tool call results."""
        return len(self.tool_results) > 0

    @property
    def token_estimate(self) -> int:
        """Rough estimate of token count for this message.

        Uses a simple heuristic of ~4 characters per token.

        Returns
        -------
        int
            Estimated token count.
        """
        text_length = len(self.content)
        tool_call_length = sum(
            len(tc.name) + len(json.dumps(tc.arguments))
            for tc in self.tool_calls
        )
        tool_result_length = sum(len(tr.output) for tr in self.tool_results)
        total = text_length + tool_call_length + tool_result_length
        return max(1, total // 4)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message to a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": copy.deepcopy(self.metadata),
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with message fields.

        Returns
        -------
        Message
            Deserialized message instance.
        """
        timestamp = None
        if data.get("timestamp"):
            ts_str = str(data["timestamp"])
            if "Z" in ts_str or "+" in ts_str:
                timestamp = datetime.fromisoformat(ts_str)
            else:
                timestamp = datetime.fromisoformat(ts_str).replace(
                    tzinfo=timezone.utc
                )
        tool_calls = [
            ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
            for tc in data.get("tool_calls", [])
        ]
        tool_results = [
            ToolCallResult(
                tool_call_id=tr.get("tool_call_id", ""),
                output=tr.get("output", ""),
                is_error=tr.get("is_error", False),
                metadata=tr.get("metadata", {}),
            )
            if isinstance(tr, dict)
            else tr
            for tr in data.get("tool_results", [])
        ]
        return cls(
            id=data.get("id", ""),
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            tool_calls=tool_calls,
            tool_results=tool_results,
            parent_id=data.get("parent_id"),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
            JSON-encoded message.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> Message:
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded message.

        Returns
        -------
        Message
            Deserialized message instance.
        """
        return cls.from_dict(json.loads(json_str))

    def copy(self) -> Message:
        """Create a deep copy of this message.

        Returns
        -------
        Message
            Independent copy of the message.
        """
        return Message.from_dict(self.to_dict())

    def content_hash(self) -> str:
        """Compute a hash of the message content for deduplication.

        Returns
        -------
        str
            SHA-256 hex digest of the content.
        """
        payload = f"{self.role}:{self.content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class AgentState:
    """Captures the agent's current internal state during execution.

    The agent state is updated at each stage of the reasoning pipeline and
    can be inspected for debugging, logging, or evaluation purposes.

    Parameters
    ----------
    current_goal : str
        The goal the agent is currently pursuing.
    plan : List[str]
        Current execution plan as a list of step descriptions.
    memory_snapshot : Dict[str, Any]
        Snapshot of relevant memory contents.
    tool_results : List[Dict[str, Any]]
        Accumulated tool execution results.
    reasoning_trace : List[str]
        Step-by-step reasoning trace for explainability.
    """

    current_goal: str = ""
    plan: List[str] = field(default_factory=list)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    step_count: int = 0
    current_step: str = ""
    error_count: int = 0
    last_error: Optional[str] = None
    start_time: Optional[float] = None
    last_update_time: Optional[float] = None
    is_complete: bool = False
    is_failed: bool = False
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    completed_tool_calls: List[ToolCall] = field(default_factory=list)
    context_tokens_used: int = 0
    output_tokens_used: int = 0
    total_cost_estimate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the agent state.
        """
        return {
            "current_goal": self.current_goal,
            "plan": list(self.plan),
            "memory_snapshot": copy.deepcopy(self.memory_snapshot),
            "tool_results": copy.deepcopy(self.tool_results),
            "reasoning_trace": list(self.reasoning_trace),
            "step_count": self.step_count,
            "current_step": self.current_step,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "start_time": self.start_time,
            "last_update_time": self.last_update_time,
            "is_complete": self.is_complete,
            "is_failed": self.is_failed,
            "pending_tool_calls": [tc.to_dict() for tc in self.pending_tool_calls],
            "completed_tool_calls": [tc.to_dict() for tc in self.completed_tool_calls],
            "context_tokens_used": self.context_tokens_used,
            "output_tokens_used": self.output_tokens_used,
            "total_cost_estimate": self.total_cost_estimate,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentState:
        """Deserialize state from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with state fields.

        Returns
        -------
        AgentState
            Deserialized agent state.
        """
        pending = [
            ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
            for tc in data.get("pending_tool_calls", [])
        ]
        completed = [
            ToolCall.from_dict(tc) if isinstance(tc, dict) else tc
            for tc in data.get("completed_tool_calls", [])
        ]
        return cls(
            current_goal=data.get("current_goal", ""),
            plan=data.get("plan", []),
            memory_snapshot=data.get("memory_snapshot", {}),
            tool_results=data.get("tool_results", []),
            reasoning_trace=data.get("reasoning_trace", []),
            step_count=data.get("step_count", 0),
            current_step=data.get("current_step", ""),
            error_count=data.get("error_count", 0),
            last_error=data.get("last_error"),
            start_time=data.get("start_time"),
            last_update_time=data.get("last_update_time"),
            is_complete=data.get("is_complete", False),
            is_failed=data.get("is_failed", False),
            pending_tool_calls=pending,
            completed_tool_calls=completed,
            context_tokens_used=data.get("context_tokens_used", 0),
            output_tokens_used=data.get("output_tokens_used", 0),
            total_cost_estimate=data.get("total_cost_estimate", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentRunResult:
    """Result of a complete agent execution run.

    Parameters
    ----------
    content : str
        The final text response from the agent.
    messages : List[Message]
        Complete conversation history from the run.
    state : AgentState
        Final agent state after the run.
    success : bool
        Whether the run completed without errors.
    error : Optional[str]
        Error message if the run failed.
    total_steps : int
        Number of reasoning-action cycles executed.
    total_time : float
        Wall-clock time of the entire run in seconds.
    tokens_used : int
        Total tokens consumed across all LLM calls.
    metadata : Dict[str, Any]
        Additional metadata about the run.
    """

    content: str = ""
    messages: List[Message] = field(default_factory=list)
    state: Optional[AgentState] = None
    success: bool = True
    error: Optional[str] = None
    total_steps: int = 0
    total_time: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the run result.
        """
        return {
            "content": self.content,
            "messages": [m.to_dict() for m in self.messages],
            "state": self.state.to_dict() if self.state else None,
            "success": self.success,
            "error": self.error,
            "total_steps": self.total_steps,
            "total_time": self.total_time,
            "tokens_used": self.tokens_used,
            "metadata": copy.deepcopy(self.metadata),
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAgent(abc.ABC):
    """Abstract base class for all agents in the Nexus framework.

    Provides the complete reasoning pipeline (observe → think → plan → act →
    reflect → respond) and manages conversation history, tool execution,
    and memory integration.  Subclasses override specific stages to customize
    behavior while inheriting the orchestration logic.

    Parameters
    ----------
    config : AgentConfig
        Configuration controlling agent behavior.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent with the given configuration.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration specifying behavior, tools, and memory.
        """
        if not isinstance(config, AgentConfig):
            raise TypeError(
                f"config must be AgentConfig, got {type(config).__name__}"
            )
        self._config = config
        self._id: str = uuid.uuid4().hex
        self._name: str = config.name
        self._role: str = config.role
        self._state = AgentState()
        self._messages: List[Message] = []
        self._tools: Dict[str, Callable] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}
        self._memory: Dict[str, Any] = {}
        self._llm_backend: Optional[Any] = None
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}.{self._name}"
        )
        self._run_count: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._conversation_id: str = uuid.uuid4().hex
        self._registered_hooks: Dict[str, List[Callable]] = {
            "pre_observe": [],
            "post_observe": [],
            "pre_think": [],
            "post_think": [],
            "pre_plan": [],
            "post_plan": [],
            "pre_act": [],
            "post_act": [],
            "pre_reflect": [],
            "post_reflect": [],
            "pre_respond": [],
            "post_respond": [],
            "pre_run": [],
            "post_run": [],
            "on_error": [],
            "on_tool_call": [],
            "on_tool_result": [],
        }
        self._register_builtin_tools()

    # --- Properties ---

    @property
    def config(self) -> AgentConfig:
        """Return the agent's configuration.

        Returns
        -------
        AgentConfig
            Read-only reference to the configuration.
        """
        return self._config

    @property
    def name(self) -> str:
        """Return the agent's name.

        Returns
        -------
        str
            Agent name string.
        """
        return self._name

    @property
    def role(self) -> str:
        """Return the agent's role.

        Returns
        -------
        str
            Agent role string.
        """
        return self._role

    @property
    def id(self) -> str:
        """Return the agent's unique identifier.

        Returns
        -------
        str
            Hex-encoded UUID.
        """
        return self._id

    @property
    def state(self) -> AgentState:
        """Return the current agent state.

        Returns
        -------
        AgentState
            Current agent state snapshot.
        """
        return self._state

    @property
    def messages(self) -> List[Message]:
        """Return the conversation history.

        Returns
        -------
        List[Message]
            List of messages in chronological order.
        """
        return list(self._messages)

    @property
    def is_ready(self) -> bool:
        """Check if the agent is ready to process input.

        Returns
        -------
        bool
            True if the agent has been properly initialized.
        """
        return self._llm_backend is not None or not self._config.tools_enabled

    @property
    def conversation_id(self) -> str:
        """Return the unique conversation identifier.

        Returns
        -------
        str
            Conversation ID hex string.
        """
        return self._conversation_id

    # --- Hook management ---

    def register_hook(
        self, event: str, callback: Callable
    ) -> None:
        """Register a callback hook for a specific agent event.

        Parameters
        ----------
        event : str
            Name of the event to hook into.
        callback : Callable
            Function to call when the event fires.

        Raises
        ------
        ValueError
            If the event name is not recognized.
        """
        valid_events = set(self._registered_hooks.keys())
        if event not in valid_events:
            raise ValueError(
                f"Unknown hook event '{event}'. "
                f"Valid events: {sorted(valid_events)}"
            )
        if not callable(callback):
            raise TypeError("Hook callback must be callable")
        self._registered_hooks[event].append(callback)
        self._logger.debug("Registered hook for event '%s': %s", event, callback)

    def remove_hook(self, event: str, callback: Callable) -> bool:
        """Remove a previously registered hook callback.

        Parameters
        ----------
        event : str
            Name of the event.
        callback : Callable
            The callback to remove.

        Returns
        -------
        bool
            True if the callback was found and removed.
        """
        hooks = self._registered_hooks.get(event, [])
        try:
            hooks.remove(callback)
            return True
        except ValueError:
            return False

    def _fire_hook(self, event: str, **kwargs: Any) -> None:
        """Fire all registered callbacks for a given event.

        Parameters
        ----------
        event : str
            Name of the event to fire.
        **kwargs : Any
            Keyword arguments passed to each callback.
        """
        for callback in self._registered_hooks.get(event, []):
            try:
                callback(self, **kwargs)
            except Exception as exc:
                self._logger.error(
                    "Hook callback error for event '%s': %s",
                    event,
                    exc,
                    exc_info=True,
                )

    # --- LLM backend ---

    def set_llm_backend(self, backend: Any) -> None:
        """Set the LLM backend used for inference.

        The backend must implement a ``generate(messages, **kwargs)`` method
        that returns a dict with at least ``"content"`` and ``"tool_calls"``.

        Parameters
        ----------
        backend : Any
            An LLM backend instance.
        """
        if not hasattr(backend, "generate"):
            raise AttributeError(
                "LLM backend must implement a 'generate' method"
            )
        self._llm_backend = backend
        self._logger.info("LLM backend set: %s", type(backend).__name__)

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call the LLM backend with the given message history.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Message history formatted for the LLM API.
        **kwargs : Any
            Additional generation parameters.

        Returns
        -------
        Dict[str, Any]
            LLM response with at least ``content`` and ``tool_calls``.

        Raises
        ------
        RuntimeError
            If no LLM backend has been set.
        """
        if self._llm_backend is None:
            raise RuntimeError(
                f"Agent '{self._name}' has no LLM backend. "
                f"Call set_llm_backend() first."
            )
        generate_kwargs = {
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "top_p": self._config.top_p,
            "frequency_penalty": self._config.frequency_penalty,
            "presence_penalty": self._config.presence_penalty,
            "stop": self._config.stop_sequences if self._config.stop_sequences else None,
        }
        generate_kwargs.update(kwargs)
        try:
            start_time = time.monotonic()
            response = self._llm_backend.generate(messages, **generate_kwargs)
            elapsed = time.monotonic() - start_time
            self._logger.debug(
                "LLM call completed in %.3fs", elapsed
            )
            if isinstance(response, str):
                response = {"content": response, "tool_calls": []}
            if not isinstance(response, dict):
                response = {"content": str(response), "tool_calls": []}
            response.setdefault("content", "")
            response.setdefault("tool_calls", [])
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            self._total_tokens += total_tokens
            self._state.context_tokens_used += prompt_tokens
            self._state.output_tokens_used += completion_tokens
            cost_per_1k = self._estimate_cost_per_1k()
            self._total_cost += (total_tokens / 1000.0) * cost_per_1k
            self._state.total_cost_estimate = self._total_cost
            return response
        except Exception as exc:
            self._logger.error(
                "LLM call failed: %s", exc, exc_info=True
            )
            self._state.error_count += 1
            self._state.last_error = str(exc)
            self._fire_hook("on_error", error=exc, step="llm_call")
            raise

    def _estimate_cost_per_1k(self) -> float:
        """Estimate the cost per 1,000 tokens based on model name.

        Returns
        -------
        float
            Estimated cost in USD per 1,000 tokens.
        """
        model_lower = self._config.model.lower()
        cost_map = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "gpt-3.5-turbo": 0.0005,
            "claude-3-opus": 0.015,
            "claude-3-sonnet": 0.003,
            "claude-3-haiku": 0.00025,
            "claude-3.5-sonnet": 0.003,
            "llama-3-70b": 0.0006,
            "llama-3-8b": 0.00005,
        }
        for key, cost in cost_map.items():
            if key in model_lower:
                return cost
        return 0.002

    # --- Tool management ---

    def _register_builtin_tools(self) -> None:
        """Register built-in tools available to all agents."""
        self.register_tool(
            name="echo",
            description="Echoes the input text back. Useful for testing.",
            parameters={
                "text": {
                    "type": "string",
                    "description": "The text to echo back",
                }
            },
            handler=self._builtin_echo,
            required=["text"],
        )
        self.register_tool(
            name="get_current_time",
            description="Returns the current date and time in ISO format.",
            parameters={},
            handler=self._builtin_get_current_time,
            required=[],
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        required: Optional[List[str]] = None,
        timeout: float = 30.0,
        retry_count: int = 3,
        dangerous: bool = False,
    ) -> None:
        """Register a tool that the agent can invoke.

        Parameters
        ----------
        name : str
            Unique tool name.
        description : str
            Human-readable description of the tool.
        parameters : Dict[str, Any]
            JSON Schema describing expected parameters.
        handler : Callable
            Function to call when the tool is invoked.
        required : List[str], optional
            Required parameter names.
        timeout : float
            Maximum execution time in seconds.
        retry_count : int
            Number of retry attempts on failure.
        dangerous : bool
            Whether the tool requires confirmation.
        """
        if not callable(handler):
            raise TypeError("Tool handler must be callable")
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")
        if name in self._tools:
            self._logger.warning(
                "Overwriting existing tool: %s", name
            )
        self._tools[name] = handler
        self._tool_configs[name] = ToolConfig(
            name=name,
            description=description,
            parameters=parameters,
            required=required or [],
            timeout=timeout,
            retry_count=retry_count,
            dangerous=dangerous,
        )
        self._logger.info("Registered tool: %s", name)

    def unregister_tool(self, name: str) -> bool:
        """Remove a registered tool by name.

        Parameters
        ----------
        name : str
            Name of the tool to remove.

        Returns
        -------
        bool
            True if the tool was found and removed.
        """
        if name in self._tools:
            del self._tools[name]
            self._tool_configs.pop(name, None)
            self._logger.info("Unregistered tool: %s", name)
            return True
        return False

    def get_tool_names(self) -> List[str]:
        """Return the names of all registered tools.

        Returns
        -------
        List[str]
            Sorted list of tool name strings.
        """
        return sorted(self._tools.keys())

    def get_tool_config(self, name: str) -> Optional[ToolConfig]:
        """Look up a tool configuration by name.

        Parameters
        ----------
        name : str
            Name of the tool.

        Returns
        -------
        Optional[ToolConfig]
            Tool configuration if found, else None.
        """
        return self._tool_configs.get(name)

    def _builtin_echo(self, text: str, **_kwargs: Any) -> str:
        """Built-in echo tool for testing.

        Parameters
        ----------
        text : str
            Text to echo.

        Returns
        -------
        str
            The echoed text.
        """
        return str(text)

    def _builtin_get_current_time(self, **_kwargs: Any) -> str:
        """Built-in time tool.

        Returns
        -------
        str
            Current time in ISO format.
        """
        return datetime.now(timezone.utc).isoformat()

    def execute_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolCallResult:
        """Execute a tool call with safety checks, timeout, and retries.

        Parameters
        ----------
        tool_name : str
            Name of the tool to invoke.
        args : Dict[str, Any]
            Arguments to pass to the tool handler.

        Returns
        -------
        ToolCallResult
            Result of the tool execution.
        """
        tool_call = ToolCall(name=tool_name, arguments=args)
        self._fire_hook("on_tool_call", tool_call=tool_call)

        effective_tools = self._config.effective_tools
        config_tool_names = {t.name for t in effective_tools}
        if tool_name not in self._tools:
            error_msg = f"Tool '{tool_name}' is not registered"
            self._logger.error(error_msg)
            self._state.error_count += 1
            self._state.last_error = error_msg
            result = ToolCallResult(
                tool_call_id=tool_call.id,
                output=f"Error: {error_msg}",
                is_error=True,
            )
            self._fire_hook("on_tool_result", result=result)
            return result

        if (
            config_tool_names
            and tool_name not in config_tool_names
        ):
            error_msg = (
                f"Tool '{tool_name}' is not in the allowed tools list"
            )
            self._logger.warning(error_msg)
            result = ToolCallResult(
                tool_call_id=tool_call.id,
                output=f"Error: {error_msg}",
                is_error=True,
            )
            self._fire_hook("on_tool_result", result=result)
            return result

        tool_config = self._tool_configs.get(tool_name)
        if tool_config and tool_config.dangerous:
            if self._config.require_tool_confirmation:
                error_msg = (
                    f"Tool '{tool_name}' requires explicit confirmation "
                    f"but no confirmation mechanism is available"
                )
                result = ToolCallResult(
                    tool_call_id=tool_call.id,
                    output=f"Error: {error_msg}",
                    is_error=True,
                )
                self._fire_hook("on_tool_result", result=result)
                return result

        handler = self._tools[tool_name]
        timeout = tool_config.timeout if tool_config else 30.0
        retry_count = tool_config.retry_count if tool_config else 3

        last_error_msg = ""
        for attempt in range(retry_count + 1):
            try:
                start_time = time.monotonic()
                output = handler(**args)
                elapsed = time.monotonic() - start_time
                if not isinstance(output, str):
                    output = json.dumps(output, default=str)
                self._logger.info(
                    "Tool '%s' executed successfully in %.3fs "
                    "(attempt %d/%d)",
                    tool_name,
                    elapsed,
                    attempt + 1,
                    retry_count + 1,
                )
                self._state.tool_results.append(
                    {
                        "tool": tool_name,
                        "args": copy.deepcopy(args),
                        "output": output[:5000],
                        "success": True,
                        "execution_time": elapsed,
                        "attempt": attempt + 1,
                    }
                )
                self._state.completed_tool_calls.append(tool_call)
                result = ToolCallResult(
                    tool_call_id=tool_call.id,
                    output=output,
                    is_error=False,
                    metadata={
                        "execution_time": elapsed,
                        "tool_name": tool_name,
                        "attempt": attempt + 1,
                    },
                )
                self._fire_hook("on_tool_result", result=result)
                return result
            except Exception as exc:
                last_error_msg = str(exc)
                elapsed = time.monotonic() - start_time if 'start_time' in dir() else 0.0
                self._logger.warning(
                    "Tool '%s' failed on attempt %d/%d: %s",
                    tool_name,
                    attempt + 1,
                    retry_count + 1,
                    last_error_msg,
                )
                if attempt < retry_count:
                    wait_time = min(2.0 ** attempt, 10.0)
                    time.sleep(wait_time)

        self._state.error_count += 1
        self._state.last_error = last_error_msg
        result = ToolCallResult(
            tool_call_id=tool_call.id,
            output=f"Error executing tool '{tool_name}': {last_error_msg}",
            is_error=True,
            metadata={"tool_name": tool_name, "attempts": retry_count + 1},
        )
        self._fire_hook("on_tool_result", result=result)
        return result

    def handle_tool_result(self, result: ToolCallResult) -> str:
        """Process a tool result and extract useful information.

        Parameters
        ----------
        result : ToolCallResult
            The tool execution result to process.

        Returns
        -------
        str
            Processed result text for inclusion in the conversation.
        """
        if result.is_error:
            self._logger.warning(
                "Processing tool error: %s", result.output
            )
            return f"Tool returned an error: {result.output}"
        max_length = self._config.max_input_tokens * 4
        output = result.output
        if len(output) > max_length:
            output = output[:max_length] + f"\n\n[Output truncated, {len(output)} total chars]"
        return output

    # --- Memory management ---

    def update_memory(self, message: Message) -> None:
        """Store a message in the agent's memory.

        Parameters
        ----------
        message : Message
            The message to store in memory.
        """
        if not self._config.memory_enabled:
            return
        msg_key = message.id
        self._memory[msg_key] = {
            "message": message.to_dict(),
            "stored_at": time.time(),
            "access_count": 0,
            "importance_score": self._compute_importance(message),
        }
        self._prune_memory()

    def _compute_importance(self, message: Message) -> float:
        """Compute an importance score for a message.

        Factors in length, tool usage, and role to estimate how important
        a message is for future reference.

        Parameters
        ----------
        message : Message
            The message to score.

        Returns
        -------
        float
            Importance score between 0.0 and 1.0.
        """
        score = 0.5
        content_length = len(message.content)
        if content_length > 500:
            score += 0.1
        if content_length > 2000:
            score += 0.1
        if message.has_tool_calls:
            score += 0.2
        if message.has_tool_results:
            score += 0.15
        if message.is_user:
            score += 0.1
        if message.role == MessageRole.TOOL.value:
            tool_errors = [
                tr for tr in message.tool_results if tr.is_error
            ]
            if tool_errors:
                score += 0.1
        score = min(max(score, 0.0), 1.0)
        return score

    def _prune_memory(self) -> None:
        """Remove old or low-importance memories to stay within capacity."""
        if not self._config.memory_enabled:
            return
        capacity = self._config.memory_config.short_term_capacity
        if len(self._memory) <= capacity:
            return
        sorted_entries = sorted(
            self._memory.items(),
            key=lambda x: (
                x[1]["importance_score"],
                x[1]["access_count"],
                x[1]["stored_at"],
            ),
        )
        while len(self._memory) > capacity:
            key, _ = sorted_entries.pop(0)
            del self._memory[key]

    def clear_memory(self) -> None:
        """Clear all stored memories."""
        self._memory.clear()
        self._logger.info("Memory cleared for agent '%s'", self._name)

    def get_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory entry by key.

        Parameters
        ----------
        key : str
            The memory key (message ID).

        Returns
        -------
        Optional[Dict[str, Any]]
            Memory entry if found, else None.
        """
        entry = self._memory.get(key)
        if entry is not None:
            entry["access_count"] += 1
        return entry

    @property
    def memory_size(self) -> int:
        """Return the number of items in memory.

        Returns
        -------
        int
            Memory entry count.
        """
        return len(self._memory)

    # --- Context building ---

    def get_context(self) -> List[Dict[str, Any]]:
        """Build the context window for the next LLM call.

        Includes the system prompt, conversation history, and relevant
        memory entries, formatted for the LLM API.

        Returns
        -------
        List[Dict[str, Any]]
            List of message dicts ready for the LLM.
        """
        context: List[Dict[str, Any]] = []
        system_prompt = self._config.system_prompt
        if system_prompt:
            role_block = f"You are {self._name}, {self._role}."
            if self._config.description:
                role_block += f"\nDescription: {self._config.description}"
            full_system = f"{system_prompt}\n\n{role_block}"
            if self._config.tools_enabled and self._tools:
                tool_descriptions = self._format_tools_for_context()
                full_system += f"\n\n## Available Tools\n{tool_descriptions}"
            context.append({
                "role": "system",
                "content": full_system,
            })
        for message in self._messages:
            msg_dict = {"role": message.role, "content": message.content}
            if message.has_tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
            context.append(msg_dict)
            for tr in message.tool_results:
                context.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.output,
                })
        max_input_tokens = self._config.max_input_tokens
        if max_input_tokens > 0:
            context = self._trim_context(context, max_input_tokens)
        return context

    def _format_tools_for_context(self) -> str:
        """Format registered tool descriptions for inclusion in the system prompt.

        Returns
        -------
        str
            Formatted tool descriptions block.
        """
        if not self._tool_configs:
            return "No tools available."
        lines: List[str] = []
        for name, config in sorted(self._tool_configs.items()):
            effective_tools = self._config.effective_tools
            config_names = {t.name for t in effective_tools}
            if config_names and name not in config_names:
                continue
            param_str = ""
            if config.parameters:
                param_parts = []
                for param_name, param_info in config.parameters.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    required_marker = ""
                    if param_name in config.required:
                        required_marker = " (required)"
                    param_parts.append(
                        f"  - {param_name} ({param_type}){required_marker}: {param_desc}"
                    )
                param_str = "\n".join(param_parts)
            lines.append(f"### {name}")
            lines.append(config.description)
            if param_str:
                lines.append(f"Parameters:\n{param_str}")
            lines.append("")
        return "\n".join(lines)

    def _trim_context(
        self,
        context: List[Dict[str, Any]],
        max_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Trim the context to fit within the token budget.

        Keeps the system prompt and most recent messages, dropping oldest
        non-system messages first.

        Parameters
        ----------
        context : List[Dict[str, Any]]
            Full context list.
        max_tokens : int
            Maximum allowed tokens.

        Returns
        -------
        List[Dict[str, Any]]
            Trimmed context list.
        """
        total_estimate = sum(
            len(msg.get("content", "")) // 4 for msg in context
        )
        if total_estimate <= max_tokens:
            return context
        system_msgs = [msg for msg in context if msg["role"] == "system"]
        non_system = [msg for msg in context if msg["role"] != "system"]
        budget = max_tokens - sum(
            len(msg.get("content", "")) // 4 for msg in system_msgs
        )
        if budget <= 0:
            return system_msgs
        trimmed: List[Dict[str, Any]] = list(system_msgs)
        for msg in reversed(non_system):
            msg_tokens = len(msg.get("content", "")) // 4
            if budget - msg_tokens >= 0:
                trimmed.insert(len(system_msgs), msg)
                budget -= msg_tokens
            else:
                break
        return trimmed

    # --- Core reasoning pipeline ---

    def observe(self, user_input: str) -> Message:
        """Process user input and update agent state.

        This is the first stage of the reasoning pipeline. It creates a
        Message object, updates the agent's goal, stores the input in
        memory, and prepares the state for thinking.

        Parameters
        ----------
        user_input : str
            The user's input text.

        Returns
        -------
        Message
            The created user message.
        """
        self._fire_hook("pre_observe", user_input=user_input)
        message = Message(
            role=MessageRole.USER.value,
            content=user_input,
        )
        self._messages.append(message)
        self._state.current_goal = user_input
        self._state.reasoning_trace.append(
            f"[observe] Received user input: {user_input[:200]}"
        )
        self.update_memory(message)
        self._fire_hook("post_observe", message=message)
        return message

    @abc.abstractmethod
    def think(self, user_input: str) -> Dict[str, Any]:
        """Perform reasoning on the current input and state.

        This is the core reasoning stage where the agent processes the
        input, retrieves relevant context, and generates a reasoning
        output that may include tool calls.

        Parameters
        ----------
        user_input : str
            The user's input text.

        Returns
        -------
        Dict[str, Any]
            Reasoning output containing at least ``thought`` (str) and
            optionally ``tool_calls`` (List[ToolCall]).
        """

    @abc.abstractmethod
    def plan(self) -> List[str]:
        """Create or revise the execution plan.

        Generates a sequence of steps the agent should follow to achieve
        the current goal.  The plan may be empty if no planning is needed.

        Returns
        -------
        List[str]
            Ordered list of plan step descriptions.
        """

    @abc.abstractmethod
    def act(self) -> Optional[Message]:
        """Execute the next planned action or tool call.

        Returns
        -------
        Optional[Message]
            A message describing the action taken, or None if no action
            is needed.
        """

    def reflect(self) -> Optional[str]:
        """Reflect on the actions taken so far and identify improvements.

        Returns
        -------
        Optional[str]
            Reflection text with observations and potential corrections,
            or None if no reflection is warranted.
        """
        self._fire_hook("pre_reflect")
        if self._state.step_count == 0:
            self._fire_hook("post_reflect", reflection=None)
            return None
        trace = self._state.reasoning_trace
        errors = [
            entry
            for entry in trace
            if "error" in entry.lower() or "failed" in entry.lower()
        ]
        tool_results = self._state.tool_results
        failed_tools = [
            r
            for r in tool_results
            if not r.get("success", True)
        ]
        if not errors and not failed_tools:
            self._fire_hook("post_reflect", reflection=None)
            return None
        reflection_parts: List[str] = []
        reflection_parts.append("I need to reflect on my progress:")
        if errors:
            reflection_parts.append(
                f"- I encountered {len(errors)} errors in my reasoning."
            )
            if self._state.last_error:
                reflection_parts.append(
                    f"- Last error: {self._state.last_error}"
                )
        if failed_tools:
            reflection_parts.append(
                f"- {len(failed_tools)} tool calls failed. "
                f"I should try alternative approaches."
            )
        if self._state.step_count > self._config.max_steps // 2:
            reflection_parts.append(
                f"- I've used {self._state.step_count} of "
                f"{self._config.max_steps} steps. "
                f"I should focus on delivering results."
            )
        reflection = "\n".join(reflection_parts)
        self._state.reasoning_trace.append(f"[reflect] {reflection}")
        self._fire_hook("post_reflect", reflection=reflection)
        return reflection

    def respond(self) -> Message:
        """Generate the final response to the user.

        Synthesizes the agent's reasoning, tool results, and reflections
        into a coherent response message.

        Returns
        -------
        Message
            The agent's final response message.
        """
        self._fire_hook("pre_respond")
        context = self.get_context()
        if self._state.is_failed:
            error_content = (
                f"I encountered an error while processing your request: "
                f"{self._state.last_error or 'Unknown error'}. "
                f"Please try again or rephrase your request."
            )
            message = Message(
                role=MessageRole.ASSISTANT.value,
                content=error_content,
                parent_id=self._messages[-1].id if self._messages else None,
            )
            self._messages.append(message)
            self.update_memory(message)
            self._fire_hook("post_respond", message=message)
            return message
        try:
            response = self._call_llm(context)
            content = response.get("content", "")
            tool_calls_data = response.get("tool_calls", [])
            tool_calls: List[ToolCall] = []
            if isinstance(tool_calls_data, list):
                for tc_data in tool_calls_data:
                    if isinstance(tc_data, dict):
                        func_info = tc_data.get("function", {})
                        tool_calls.append(
                            ToolCall(
                                id=tc_data.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                                name=func_info.get("name", ""),
                                arguments=json.loads(
                                    func_info.get("arguments", "{}")
                                ),
                            )
                        )
                    elif isinstance(tc_data, ToolCall):
                        tool_calls.append(tc_data)
            parent_id = self._messages[-1].id if self._messages else None
            message = Message(
                role=MessageRole.ASSISTANT.value,
                content=content,
                tool_calls=tool_calls,
                parent_id=parent_id,
            )
            self._messages.append(message)
            self.update_memory(message)
            if tool_calls:
                self._state.pending_tool_calls.extend(tool_calls)
            self._state.reasoning_trace.append(
                f"[respond] Generated response: {content[:200]}"
            )
            self._fire_hook("post_respond", message=message)
            return message
        except Exception as exc:
            error_content = (
                f"Failed to generate response: {str(exc)}"
            )
            self._logger.error(
                "Response generation failed: %s", exc, exc_info=True
            )
            message = Message(
                role=MessageRole.ASSISTANT.value,
                content=error_content,
                parent_id=self._messages[-1].id if self._messages else None,
            )
            self._messages.append(message)
            self._fire_hook("post_respond", message=message)
            return message

    # --- Main execution loop ---

    def run(
        self,
        user_input: str,
        max_steps: Optional[int] = None,
    ) -> AgentRunResult:
        """Execute the full agent reasoning loop.

        Runs the observe → think → plan → act → reflect → respond cycle
        until the agent produces a final response or reaches the step limit.

        Parameters
        ----------
        user_input : str
            The user's input text.
        max_steps : int, optional
            Override the configured maximum steps.

        Returns
        -------
        AgentRunResult
            Complete result of the agent run.
        """
        self._fire_hook("pre_run", user_input=user_input)
        start_time = time.monotonic()
        effective_max = max_steps or self._config.max_steps
        self._state = AgentState(start_time=start_time)
        self._logger.info(
            "Agent '%s' starting run (max_steps=%d)", self._name, effective_max
        )
        try:
            self.observe(user_input)
            for step_idx in range(effective_max):
                self._state.step_count = step_idx + 1
                self._state.last_update_time = time.monotonic()
                self._logger.debug(
                    "Agent '%s' step %d/%d",
                    self._name,
                    step_idx + 1,
                    effective_max,
                )
                thinking_result = self.think(user_input)
                self._state.reasoning_trace.append(
                    f"[think] {thinking_result.get('thought', '')[:200]}"
                )
                if self._config.planning_enabled:
                    plan = self.plan()
                    self._state.plan = plan
                    self._state.reasoning_trace.append(
                        f"[plan] Generated plan with {len(plan)} steps"
                    )
                action_msg = self.act()
                if action_msg:
                    self._messages.append(action_msg)
                    self.update_memory(action_msg)
                if self._state.pending_tool_calls:
                    self._process_pending_tool_calls()
                    continue
                if step_idx >= 1:
                    reflection = self.reflect()
                    if reflection:
                        self._state.reasoning_trace.append(reflection)
                if self._should_respond():
                    break
            final_response = self.respond()
            self._state.is_complete = True
            elapsed = time.monotonic() - start_time
            self._run_count += 1
            result = AgentRunResult(
                content=final_response.content,
                messages=list(self._messages),
                state=self._state,
                success=not self._state.is_failed,
                error=self._state.last_error,
                total_steps=self._state.step_count,
                total_time=elapsed,
                tokens_used=self._total_tokens,
            )
            self._logger.info(
                "Agent '%s' run completed in %.3fs (%d steps, %d tokens)",
                self._name,
                elapsed,
                self._state.step_count,
                self._total_tokens,
            )
            self._fire_hook("post_run", result=result)
            return result
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            self._state.is_failed = True
            self._state.last_error = str(exc)
            self._state.is_complete = True
            self._logger.error(
                "Agent '%s' run failed after %.3fs: %s",
                self._name,
                elapsed,
                exc,
                exc_info=True,
            )
            result = AgentRunResult(
                content=f"Error: {str(exc)}",
                messages=list(self._messages),
                state=self._state,
                success=False,
                error=str(exc),
                total_steps=self._state.step_count,
                total_time=elapsed,
                tokens_used=self._total_tokens,
            )
            self._fire_hook("post_run", result=result)
            return result

    def _should_respond(self) -> bool:
        """Determine if the agent should generate a final response.

        Returns
        -------
        bool
            True if the agent has gathered enough information to respond.
        """
        if self._state.pending_tool_calls:
            return False
        if self._state.step_count >= self._config.max_steps:
            return True
        if not self._state.tool_results and self._state.step_count < 2:
            return False
        if self._config.planning_enabled:
            if self._state.current_step in self._state.plan:
                step_idx = self._state.plan.index(self._state.current_step)
                if step_idx < len(self._state.plan) - 1:
                    return False
        return True

    def _process_pending_tool_calls(self) -> None:
        """Execute all pending tool calls and store results."""
        tool_calls = list(self._state.pending_tool_calls)
        self._state.pending_tool_calls.clear()
        for tool_call in tool_calls:
            result = self.execute_tool(
                tool_call.name,
                tool_call.arguments,
            )
            processed = self.handle_tool_result(result)
            tool_result_msg = Message(
                role=MessageRole.TOOL.value,
                content=processed,
                tool_results=[result],
                parent_id=self._messages[-1].id if self._messages else None,
            )
            self._messages.append(tool_result_msg)
            self.update_memory(tool_result_msg)
            if result.is_error:
                self._state.error_count += 1
                self._state.last_error = f"Tool '{tool_call.name}' failed: {result.output}"

    # --- Conversation management ---

    def reset(self) -> None:
        """Reset the agent to a clean state for a new conversation.

        Clears conversation history, state, and optionally memory.
        Generates a new conversation ID.
        """
        self._messages.clear()
        self._state = AgentState()
        self._conversation_id = uuid.uuid4().hex
        self._logger.info(
            "Agent '%s' reset (new conversation: %s)",
            self._name,
            self._conversation_id[:8],
        )

    def reset_full(self) -> None:
        """Full reset including memory and token counters."""
        self.reset()
        self.clear_memory()
        self._total_tokens = 0
        self._total_cost = 0.0
        self._run_count = 0

    def load_conversation(self, messages: List[Message]) -> None:
        """Load an existing conversation history.

        Parameters
        ----------
        messages : List[Message]
            Messages to load in chronological order.
        """
        self._messages.clear()
        for msg in messages:
            self._messages.append(msg.copy())
            self.update_memory(msg)
        self._logger.info(
            "Loaded %d messages into agent '%s'",
            len(messages),
            self._name,
        )

    def get_last_message(self, role: Optional[str] = None) -> Optional[Message]:
        """Get the most recent message, optionally filtered by role.

        Parameters
        ----------
        role : str, optional
            If provided, only consider messages with this role.

        Returns
        -------
        Optional[Message]
            The most recent matching message.
        """
        for msg in reversed(self._messages):
            if role is None or msg.role == role:
                return msg
        return None

    def get_messages_by_role(self, role: str) -> List[Message]:
        """Get all messages with a specific role.

        Parameters
        ----------
        role : str
            The role to filter by.

        Returns
        -------
        List[Message]
            All messages with the given role.
        """
        return [msg for msg in self._messages if msg.role == role]

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the agent's state to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary with agent state and metadata.
        """
        return {
            "id": self._id,
            "name": self._name,
            "role": self._role,
            "config": self._config.to_dict(),
            "state": self._state.to_dict(),
            "messages": [m.to_dict() for m in self._messages],
            "conversation_id": self._conversation_id,
            "run_count": self._run_count,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "registered_tools": sorted(self._tools.keys()),
            "memory_size": self.memory_size,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get runtime statistics for this agent.

        Returns
        -------
        Dict[str, Any]
            Dictionary with agent usage statistics.
        """
        return {
            "agent_id": self._id,
            "agent_name": self._name,
            "conversation_id": self._conversation_id,
            "run_count": self._run_count,
            "total_tokens": self._total_tokens,
            "total_cost_estimate": round(self._total_cost, 6),
            "message_count": len(self._messages),
            "memory_size": self.memory_size,
            "registered_tools": len(self._tools),
            "state": {
                "step_count": self._state.step_count,
                "error_count": self._state.error_count,
                "is_complete": self._state.is_complete,
                "is_failed": self._state.is_failed,
            },
        }


# ---------------------------------------------------------------------------
# ConversationalAgent
# ---------------------------------------------------------------------------

class ConversationalAgent(BaseAgent):
    """Chat-focused agent optimized for dialogue and conversation.

    Maintains rich dialogue state including topic tracking, sentiment
    awareness, and contextual follow-up handling.  Best suited for
    customer support, tutoring, brainstorming, and general conversation.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the conversational agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the conversational agent.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration.
        """
        super().__init__(config)
        self._topics: List[str] = []
        self._user_preferences: Dict[str, Any] = {}
        self._sentiment_history: List[float] = []
        self._follow_up_queue: List[str] = []
        self._conversation_summary: str = ""
        self._turn_count: int = 0
        self._max_turns: int = 1000
        self._summarization_threshold: int = 20

    def think(self, user_input: str) -> Dict[str, Any]:
        """Perform conversational reasoning on the user's input.

        Analyzes the input for intent, entities, and sentiment, then
        generates a thoughtful response considering conversation history.

        Parameters
        ----------
        user_input : str
            The user's input.

        Returns
        -------
        Dict[str, Any]
            Reasoning output with ``thought`` and optional ``tool_calls``.
        """
        self._fire_hook("pre_think", user_input=user_input)
        intent = self._classify_intent(user_input)
        entities = self._extract_entities(user_input)
        sentiment = self._analyze_sentiment(user_input)
        self._sentiment_history.append(sentiment)
        self._turn_count += 1
        self._state.reasoning_trace.append(
            f"[think] Intent={intent}, Entities={entities}, "
            f"Sentiment={sentiment:.2f}"
        )
        context = self.get_context()
        recent_context = self._build_conversation_context(user_input)
        if intent == "question":
            thinking_prompt = (
                f"The user is asking a question. "
                f"Consider the conversation history and provide "
                f"a helpful, accurate response.\n\n"
                f"User question: {user_input}\n"
                f"Detected entities: {', '.join(entities) if entities else 'none'}\n"
                f"Conversation context: {recent_context}"
            )
        elif intent == "command":
            thinking_prompt = (
                f"The user is issuing a command or request. "
                f"Determine if any tools are needed and "
                f"respond accordingly.\n\n"
                f"User command: {user_input}"
            )
        elif intent == "clarification":
            thinking_prompt = (
                f"The user is asking for clarification. "
                f"Review previous context and provide "
                f"a clear explanation.\n\n"
                f"Clarification request: {user_input}"
            )
        elif intent == "farewell":
            thinking_prompt = (
                f"The user is ending the conversation. "
                f"Provide a polite closing response.\n\n"
                f"User message: {user_input}"
            )
        else:
            thinking_prompt = (
                f"The user said: {user_input}\n"
                f"Sentiment: {sentiment:.2f}\n"
                f"Conversation context: {recent_context}\n"
                f"Respond helpfully and naturally."
            )
        context.append({
            "role": "system",
            "content": thinking_prompt,
        })
        tool_calls: List[ToolCall] = []
        if intent == "command" and self._config.tools_enabled:
            tool_calls = self._identify_needed_tools(user_input, entities)
        self._fire_hook("post_think", thought=thinking_prompt)
        return {
            "thought": thinking_prompt,
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "tool_calls": tool_calls,
        }

    def plan(self) -> List[str]:
        """Generate a simple conversation plan.

        For conversational agents, the plan is typically minimal —
        acknowledge, respond, and check for follow-up.

        Returns
        -------
        List[str]
            Ordered list of conversation steps.
        """
        self._fire_hook("pre_plan")
        plan_steps: List[str] = [
            "Acknowledge the user's input",
            "Process and understand the request",
            "Generate a helpful response",
            "Check if follow-up is needed",
        ]
        if self._state.pending_tool_calls:
            tool_names = [tc.name for tc in self._state.pending_tool_calls]
            plan_steps.insert(2, f"Execute tools: {', '.join(tool_names)}")
            plan_steps.insert(3, "Process tool results")
        self._fire_hook("post_plan", plan=plan_steps)
        return plan_steps

    def act(self) -> Optional[Message]:
        """Execute the next conversational action.

        Returns
        -------
        Optional[Message]
            Action message or None if direct response is appropriate.
        """
        self._fire_hook("pre_act")
        if not self._state.pending_tool_calls:
            self._fire_hook("post_act", action=None)
            return None
        tool_call = self._state.pending_tool_calls[0]
        result = self.execute_tool(tool_call.name, tool_call.arguments)
        processed = self.handle_tool_result(result)
        self._state.current_step = f"Executed tool: {tool_call.name}"
        message = Message(
            role=MessageRole.ASSISTANT.value,
            content=f"Used tool '{tool_call.name}': {processed[:500]}",
            parent_id=self._messages[-1].id if self._messages else None,
        )
        self._fire_hook("post_act", action=message)
        return message

    def _classify_intent(self, text: str) -> str:
        """Classify the intent of the user's input.

        Uses pattern matching and keyword analysis for lightweight intent
        classification without requiring a separate model call.

        Parameters
        ----------
        text : str
            The user's input text.

        Returns
        -------
        str
            One of: ``"question"``, ``"command"``, ``"clarification"``,
            ``"farewell"``, ``"greeting"``, ``"general"``.
        """
        text_lower = text.lower().strip()
        question_patterns = [
            r"^what\b",
            r"^how\b",
            r"^why\b",
            r"^when\b",
            r"^where\b",
            r"^who\b",
            r"^which\b",
            r"^can you\b",
            r"^could you\b",
            r"^do you\b",
            r"^is it\b",
            r"^are there\b",
            r"\?$",
        ]
        for pattern in question_patterns:
            if re.search(pattern, text_lower):
                return "question"
        command_patterns = [
            r"^(please |pls )?(do|make|create|write|find|search|calculate|show|tell|give|send|delete|remove|update|set|get|list)",
            r"^i (want|need|would like|require)",
        ]
        for pattern in command_patterns:
            if re.search(pattern, text_lower):
                return "command"
        clarification_patterns = [
            r"what (do you mean|did you say|are you talking about)",
            r"can you (explain|clarify|elaborate)",
            r"i (don't understand|didn't get|am confused)",
            r"could you (repeat|rephrase|say that again)",
        ]
        for pattern in clarification_patterns:
            if re.search(pattern, text_lower):
                return "clarification"
        farewell_patterns = [
            r"^(bye|goodbye|see you|farewell|take care|good night|goodbye)",
            r"^thank(s| you)",
        ]
        for pattern in farewell_patterns:
            if re.search(pattern, text_lower):
                return "farewell"
        greeting_patterns = [
            r"^(hi|hello|hey|greetings|howdy|good morning|good afternoon|good evening)\b",
        ]
        for pattern in greeting_patterns:
            if re.search(pattern, text_lower):
                return "greeting"
        return "general"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text using pattern matching.

        Identifies dates, times, numbers, email addresses, URLs, and
        quoted strings as basic entities.

        Parameters
        ----------
        text : str
            Input text to extract entities from.

        Returns
        -------
        List[str]
            List of extracted entity strings.
        """
        entities: List[str] = []
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        entities.extend(re.findall(date_pattern, text))
        time_pattern = r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b'
        entities.extend(re.findall(time_pattern, text))
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, text)
        entities.extend(numbers[:5])
        email_pattern = r'\b[\w.-]+@[\w.-]+\.\w+\b'
        entities.extend(re.findall(email_pattern, text))
        url_pattern = r'https?://\S+'
        entities.extend(re.findall(url_pattern, text))
        quote_pattern = r'"([^"]+)"'
        entities.extend(re.findall(quote_pattern, text))
        single_quote_pattern = r"'([^']+)'"
        entities.extend(re.findall(single_quote_pattern, text))
        return list(dict.fromkeys(entities))

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze the sentiment of the input text.

        Returns a float between -1.0 (very negative) and 1.0 (very positive).

        Parameters
        ----------
        text : str
            Input text to analyze.

        Returns
        -------
        float
            Sentiment score between -1.0 and 1.0.
        """
        positive_words = {
            "good", "great", "excellent", "awesome", "wonderful", "fantastic",
            "amazing", "love", "happy", "glad", "pleased", "thankful",
            "thanks", "perfect", "beautiful", "nice", "helpful", "brilliant",
            "outstanding", "superb", "impressive", "enjoy", "appreciate",
            "agree", "correct", "yes", "sure", "absolutely", "definitely",
        }
        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "angry", "sad",
            "disappointed", "frustrated", "annoyed", "unhappy", "wrong",
            "error", "fail", "broken", "useless", "stupid", "worst",
            "poor", "ugly", "slow", "confused", "lost", "sorry", "no",
            "not", "never", "cannot", "impossible", "unfortunately",
        }
        intensifiers = {
            "very", "extremely", "incredibly", "absolutely", "really",
            "truly", "highly", "super", "totally", "completely",
        }
        negators = {
            "not", "no", "never", "neither", "nobody", "nothing",
            "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't",
        }
        words = re.findall(r'\b\w+\b', text.lower())
        score = 0.0
        negate = False
        intensify = 1.0
        for word in words:
            if word in negators:
                negate = True
                continue
            if word in intensifiers:
                intensify = 1.5
                continue
            if word in positive_words:
                modifier = -1.0 if negate else 1.0
                score += 0.2 * modifier * intensify
            elif word in negative_words:
                modifier = -1.0 if negate else 1.0
                score -= 0.2 * modifier * intensify
            negate = False
            intensify = 1.0
        max_score = max(len(words) * 0.2, 1.0)
        normalized = score / max_score
        return max(-1.0, min(1.0, normalized))

    def _identify_needed_tools(
        self, text: str, entities: List[str]
    ) -> List[ToolCall]:
        """Identify which tools might be needed for a command.

        Parameters
        ----------
        text : str
            User input text.
        entities : List[str]
            Extracted entities from the text.

        Returns
        -------
        List[ToolCall]
            List of tool calls to execute.
        """
        tool_calls: List[ToolCall] = []
        text_lower = text.lower()
        if any(
            keyword in text_lower
            for keyword in ["calculate", "compute", "math", "sum", "average"]
        ):
            expr_match = re.search(
                r'([\d\s+\-*/().^%]+)', text
            )
            if expr_match:
                tool_calls.append(
                    ToolCall(
                        name="calculator",
                        arguments={"expression": expr_match.group(1).strip()},
                    )
                )
        if any(
            keyword in text_lower
            for keyword in ["time", "date", "today", "now", "clock"]
        ):
            if "calculator" not in [tc.name for tc in tool_calls]:
                tool_calls.append(
                    ToolCall(
                        name="get_current_time",
                        arguments={},
                    )
                )
        if len(tool_calls) > self._config.max_tool_calls_per_step:
            tool_calls = tool_calls[:self._config.max_tool_calls_per_step]
        return tool_calls

    def _build_conversation_context(self, user_input: str) -> str:
        """Build a summary of recent conversation context.

        Parameters
        ----------
        user_input : str
            Current user input.

        Returns
        -------
        str
            Summary of recent conversation.
        """
        recent_messages = self._messages[-6:]
        if not recent_messages:
            return "This is the start of the conversation."
        context_parts: List[str] = []
        for msg in recent_messages[-4:]:
            role_label = "User" if msg.is_user else "Assistant"
            content_preview = msg.content[:200]
            context_parts.append(f"{role_label}: {content_preview}")
        return "\n".join(context_parts)

    def get_conversation_summary(self) -> str:
        """Get or generate a summary of the conversation.

        Returns
        -------
        str
            Conversation summary text.
        """
        if self._conversation_summary:
            return self._conversation_summary
        if not self._messages:
            return "No conversation yet."
        user_messages = self.get_messages_by_role(MessageRole.USER.value)
        assistant_messages = self.get_messages_by_role(
            MessageRole.ASSISTANT.value
        )
        summary = (
            f"Conversation with {len(user_messages)} user messages "
            f"and {len(assistant_messages)} agent responses. "
            f"Topics discussed: {', '.join(self._topics) if self._topics else 'general conversation'}. "
            f"Average sentiment: "
            f"{sum(self._sentiment_history) / max(len(self._sentiment_history), 1):.2f}."
        )
        return summary


# ---------------------------------------------------------------------------
# TaskAgent
# ---------------------------------------------------------------------------

class TaskAgent(BaseAgent):
    """Goal-oriented agent with planning and tool use capabilities.

    Designed for complex, multi-step tasks that require tool integration,
    structured planning, and iterative problem solving.  Best suited for
    research tasks, data analysis, workflow automation, and any task that
    goes beyond simple conversation.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the task agent.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the task agent.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration.
        """
        super().__init__(config)
        self._task_queue: List[Dict[str, Any]] = []
        self._completed_tasks: List[Dict[str, Any]] = []
        self._failed_tasks: List[Dict[str, Any]] = []
        self._subtasks: List[str] = []
        self._current_subtask_idx: int = 0
        self._task_results: Dict[str, Any] = {}

    def think(self, user_input: str) -> Dict[str, Any]:
        """Perform task-oriented reasoning on the user's input.

        Analyzes the task, breaks it down into subtasks, identifies
        necessary tools, and develops a reasoning approach.

        Parameters
        ----------
        user_input : str
            The task description or user input.

        Returns
        -------
        Dict[str, Any]
            Reasoning output with thought, subtasks, and tool needs.
        """
        self._fire_hook("pre_think", user_input=user_input)
        task_analysis = self._analyze_task(user_input)
        subtasks = task_analysis.get("subtasks", [])
        self._subtasks = subtasks
        self._task_results = {}
        required_tools = task_analysis.get("required_tools", [])
        complexity = task_analysis.get("complexity", "medium")
        self._state.reasoning_trace.append(
            f"[think] Task analysis: complexity={complexity}, "
            f"subtasks={len(subtasks)}, tools={required_tools}"
        )
        context = self.get_context()
        reasoning_prompt = self._build_reasoning_prompt(
            user_input, task_analysis
        )
        context.append({
            "role": "system",
            "content": reasoning_prompt,
        })
        tool_calls: List[ToolCall] = []
        if (
            self._config.tools_enabled
            and required_tools
            and not self._state.completed_tool_calls
        ):
            tool_calls = self._plan_tool_usage(
                user_input, required_tools
            )
        thought = (
            f"Analyzing task: {user_input[:100]}... "
            f"Complexity: {complexity}. "
            f"Subtasks: {len(subtasks)}. "
            f"Tools needed: {', '.join(required_tools) if required_tools else 'none'}."
        )
        self._fire_hook("post_think", thought=thought)
        return {
            "thought": thought,
            "subtasks": subtasks,
            "required_tools": required_tools,
            "complexity": complexity,
            "tool_calls": tool_calls,
        }

    def plan(self) -> List[str]:
        """Create a detailed execution plan for the current task.

        Returns
        -------
        List[str]
            Ordered list of plan steps.
        """
        self._fire_hook("pre_plan")
        plan_steps: List[str] = []
        if self._subtasks:
            plan_steps.extend(
                f"Subtask {i + 1}: {st}"
                for i, st in enumerate(self._subtasks)
            )
        else:
            plan_steps.append("Analyze the user's request")
            plan_steps.append("Gather necessary information")
            plan_steps.append("Process and synthesize results")
            plan_steps.append("Formulate the response")
        if self._state.pending_tool_calls:
            tool_names = [tc.name for tc in self._state.pending_tool_calls]
            plan_steps.insert(
                1, f"Execute tools: {', '.join(tool_names)}"
            )
        if self._config.reasoning_strategy == ReasoningStrategy.REFLECTION:
            plan_steps.append("Reflect on the approach and results")
        plan_steps.append("Generate final response")
        self._fire_hook("post_plan", plan=plan_steps)
        return plan_steps

    def act(self) -> Optional[Message]:
        """Execute the next planned action.

        Processes pending tool calls or advances to the next subtask.

        Returns
        -------
        Optional[Message]
            Action message or None if ready to respond.
        """
        self._fire_hook("pre_act")
        if self._state.pending_tool_calls:
            tool_call = self._state.pending_tool_calls[0]
            result = self.execute_tool(tool_call.name, tool_call.arguments)
            processed = self.handle_tool_result(result)
            self._task_results[tool_call.name] = {
                "output": processed,
                "success": not result.is_error,
            }
            self._state.current_step = (
                f"Executed tool: {tool_call.name}"
            )
            message = Message(
                role=MessageRole.ASSISTANT.value,
                content=f"Executed tool '{tool_call.name}': {processed[:500]}",
                parent_id=self._messages[-1].id if self._messages else None,
            )
            self._fire_hook("post_act", action=message)
            return message
        if self._current_subtask_idx < len(self._subtasks):
            subtask = self._subtasks[self._current_subtask_idx]
            self._state.current_step = (
                f"Working on subtask {self._current_subtask_idx + 1}: "
                f"{subtask}"
            )
            self._state.reasoning_trace.append(
                f"[act] Processing subtask: {subtask}"
            )
            self._current_subtask_idx += 1
            self._fire_hook("post_act", action=None)
            return None
        self._fire_hook("post_act", action=None)
        return None

    def reflect(self) -> Optional[str]:
        """Reflect on task progress and adjust approach if needed.

        Returns
        -------
        Optional[str]
            Reflection text or None.
        """
        base_reflection = super().reflect()
        reflection_parts: List[str] = []
        if base_reflection:
            reflection_parts.append(base_reflection)
        completed_subtasks = self._current_subtask_idx
        total_subtasks = len(self._subtasks)
        if total_subtasks > 0:
            progress_pct = (completed_subtasks / total_subtasks) * 100
            reflection_parts.append(
                f"Task progress: {completed_subtasks}/{total_subtasks} "
                f"subtasks ({progress_pct:.0f}%)"
            )
        failed_tools = [
            name
            for name, result in self._task_results.items()
            if not result.get("success", True)
        ]
        if failed_tools:
            reflection_parts.append(
                f"Failed tools: {', '.join(failed_tools)}. "
                f"Consider alternative approaches."
            )
        successful_results = {
            name: result["output"]
            for name, result in self._task_results.items()
            if result.get("success", True)
        }
        if successful_results:
            summary = ", ".join(
                f"{name}: {output[:100]}..."
                for name, output in successful_results.items()
            )
            reflection_parts.append(f"Results so far: {summary}")
        if not reflection_parts:
            return None
        full_reflection = "\n".join(reflection_parts)
        self._state.reasoning_trace.append(f"[reflect] {full_reflection}")
        return full_reflection

    def _analyze_task(self, task: str) -> Dict[str, Any]:
        """Analyze a task to determine complexity, subtasks, and tool needs.

        Parameters
        ----------
        task : str
            The task description.

        Returns
        -------
        Dict[str, Any]
            Task analysis with subtasks, complexity, and tool requirements.
        """
        complexity = self._estimate_complexity(task)
        subtasks = self._decompose_task(task, complexity)
        required_tools = self._identify_required_tools(task, subtasks)
        return {
            "subtasks": subtasks,
            "complexity": complexity,
            "required_tools": required_tools,
            "estimated_steps": max(len(subtasks) + 1, 3),
        }

    def _estimate_complexity(self, task: str) -> str:
        """Estimate the complexity of a task.

        Parameters
        ----------
        task : str
            Task description.

        Returns
        -------
        str
            One of ``"simple"``, ``"medium"``, ``"complex"``, ``"expert"``.
        """
        word_count = len(task.split())
        sentence_count = len(re.split(r'[.!?]+', task))
        if word_count < 20 and sentence_count <= 1:
            return "simple"
        elif word_count < 100 and sentence_count <= 3:
            return "medium"
        elif word_count < 500:
            return "complex"
        else:
            return "expert"

    def _decompose_task(
        self, task: str, complexity: str
    ) -> List[str]:
        """Decompose a task into subtasks.

        Parameters
        ----------
        task : str
            Task description.
        complexity : str
            Estimated complexity level.

        Returns
        -------
        List[str]
            List of subtask descriptions.
        """
        sentences = [
            s.strip()
            for s in re.split(r'[.!?]+', task)
            if s.strip()
        ]
        if complexity == "simple":
            return [task.strip()]
        if complexity == "medium":
            subtasks = []
            for sentence in sentences:
                if len(sentence.split()) > 3:
                    subtasks.append(sentence.strip())
            if not subtasks:
                subtasks = [task.strip()]
            return subtasks
        subtasks: List[str] = []
        connector_patterns = [
            (r'\b(and then|after that|next|finally|also)\b', None),
            (r'\b(first|secondly|thirdly|lastly)\b', None),
            (r'\b(additionally|furthermore|moreover)\b', None),
            (r'\b(however|alternatively|instead)\b', None),
        ]
        for sentence in sentences:
            split_found = False
            for pattern, _ in connector_patterns:
                parts = re.split(pattern, sentence, flags=re.IGNORECASE)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if len(part.split()) > 2:
                            subtasks.append(part)
                    split_found = True
                    break
            if not split_found:
                subtasks.append(sentence.strip())
        if not subtasks:
            subtasks = [task.strip()]
        return subtasks

    def _identify_required_tools(
        self, task: str, subtasks: List[str]
    ) -> List[str]:
        """Identify which tools are needed for the task.

        Parameters
        ----------
        task : str
            Task description.
        subtasks : List[str]
            Decomposed subtasks.

        Returns
        -------
        List[str]
            List of tool names that may be needed.
        """
        task_lower = (task + " " + " ".join(subtasks)).lower()
        needed: List[str] = []
        tool_keywords: Dict[str, List[str]] = {
            "calculator": [
                "calculate", "compute", "math", "sum", "average",
                "multiply", "divide", "percentage", "equation",
            ],
            "get_current_time": [
                "time", "date", "today", "now", "schedule",
                "calendar", "deadline",
            ],
            "web_search": [
                "search", "find", "lookup", "look up", "research",
                "information about", "latest",
            ],
            "file_read": [
                "read file", "open file", "load file", "read from",
            ],
            "file_write": [
                "write file", "save file", "create file", "write to",
            ],
        }
        for tool_name, keywords in tool_keywords.items():
            if tool_name not in self._tools:
                continue
            for keyword in keywords:
                if keyword in task_lower:
                    if tool_name not in needed:
                        needed.append(tool_name)
                    break
        return needed

    def _plan_tool_usage(
        self, task: str, required_tools: List[str]
    ) -> List[ToolCall]:
        """Plan tool calls based on the task requirements.

        Parameters
        ----------
        task : str
            Task description.
        required_tools : List[str]
            Tools identified as needed.

        Returns
        -------
        List[ToolCall]
            List of tool calls to execute.
        """
        tool_calls: List[ToolCall] = []
        for tool_name in required_tools:
            if tool_name not in self._tools:
                continue
            args: Dict[str, Any] = {}
            if tool_name == "calculator":
                expr_match = re.search(
                    r'([\d\s+\-*/().^%]+(?:[a-zA-Z]+\([\d\s+\-*/().^%]+\))?)',
                    task,
                )
                if expr_match:
                    args["expression"] = expr_match.group(1).strip()
                else:
                    continue
            elif tool_name == "get_current_time":
                args = {}
            elif tool_name == "web_search":
                query_match = re.search(
                    r'(?:search|find|lookup)\s+(?:for\s+)?["\']?([^"\'\.]+)',
                    task,
                    re.IGNORECASE,
                )
                if query_match:
                    args["query"] = query_match.group(1).strip()
                else:
                    args["query"] = task[:200]
            if args or tool_name == "get_current_time":
                tool_calls.append(ToolCall(name=tool_name, arguments=args))
            if len(tool_calls) >= self._config.max_tool_calls_per_step:
                break
        return tool_calls

    def _build_reasoning_prompt(
        self, task: str, analysis: Dict[str, Any]
    ) -> str:
        """Build a reasoning prompt for the LLM based on task analysis.

        Parameters
        ----------
        task : str
            Task description.
        analysis : Dict[str, Any]
            Task analysis results.

        Returns
        -------
        str
            System prompt for reasoning.
        """
        parts: List[str] = [
            "You are a task-oriented agent. Follow these steps:",
            f"1. Goal: {task}",
            f"2. Complexity: {analysis['complexity']}",
        ]
        if analysis.get("subtasks"):
            parts.append("3. Subtasks:")
            for i, st in enumerate(analysis["subtasks"], 1):
                parts.append(f"   {i}. {st}")
        if analysis.get("required_tools"):
            parts.append(
                f"4. Tools available: {', '.join(analysis['required_tools'])}"
            )
        parts.append(
            "Think step by step, use tools when needed, "
            "and provide a comprehensive result."
        )
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# ReflectiveAgent
# ---------------------------------------------------------------------------

class ReflectiveAgent(TaskAgent):
    """Self-reflective agent that critiques its own reasoning and corrects errors.

    Extends TaskAgent with a reflection loop that evaluates each step of
    reasoning, identifies potential flaws, and generates corrections.  This
    produces higher-quality outputs on complex tasks at the cost of
    additional inference steps.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the reflective agent.
    max_reflection_iterations : int
        Maximum number of reflection-correction cycles per step.
    """

    def __init__(
        self,
        config: AgentConfig,
        max_reflection_iterations: int = 2,
    ) -> None:
        """Initialize the reflective agent.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration.
        max_reflection_iterations : int
            Maximum reflection iterations per step.
        """
        super().__init__(config)
        self._max_reflection_iterations = max_reflection_iterations
        self._reflection_history: List[Dict[str, Any]] = []
        self._correction_count: int = 0
        self._self_critique_prompt: str = (
            "Review your previous response and reasoning. "
            "Identify any errors, logical flaws, missing information, "
            "or areas that could be improved. Be specific about what "
            "needs to change and why."
        )

    def think(self, user_input: str) -> Dict[str, Any]:
        """Perform reflective reasoning on the user's input.

        In addition to standard task-oriented reasoning, evaluates
        previous reasoning steps for errors and generates corrections.

        Parameters
        ----------
        user_input : str
            The user's input.

        Returns
        -------
        Dict[str, Any]
            Reasoning output with reflection-aware thought.
        """
        self._fire_hook("pre_think", user_input=user_input)
        base_thinking = super().think(user_input)
        if self._reflection_history and self._correction_count < self._max_reflection_iterations * 3:
            critique = self._self_critique()
            if critique:
                correction = self._generate_correction(critique, base_thinking)
                base_thinking["thought"] += f"\n\n[Correction] {correction}"
                base_thinking["correction"] = correction
                self._correction_count += 1
                self._state.reasoning_trace.append(
                    f"[reflect_think] Applied correction #{self._correction_count}"
                )
        self._fire_hook("post_think", thought=base_thinking.get("thought", ""))
        return base_thinking

    def reflect(self) -> Optional[str]:
        """Perform deep self-reflection on the agent's progress.

        Extends the base reflection with explicit self-critique and
        correction generation.

        Returns
        -------
        Optional[str]
            Detailed reflection with critique and corrections.
        """
        self._fire_hook("pre_reflect")
        base_reflection = super().reflect()
        reflection_parts: List[str] = []
        if base_reflection:
            reflection_parts.append(base_reflection)
        critique = self._self_critique()
        if critique:
            reflection_parts.append(f"[Self-Critique] {critique}")
            self._reflection_history.append({
                "step": self._state.step_count,
                "critique": critique,
                "timestamp": time.time(),
            })
            if self._correction_count < self._max_reflection_iterations * 3:
                correction = self._generate_correction(
                    critique, {"thought": str(base_reflection)}
                )
                if correction:
                    reflection_parts.append(
                        f"[Self-Correction] {correction}"
                    )
                    self._correction_count += 1
        if not reflection_parts:
            self._fire_hook("post_reflect", reflection=None)
            return None
        full_reflection = "\n".join(reflection_parts)
        self._state.reasoning_trace.append(f"[reflect] {full_reflection[:500]}")
        self._fire_hook("post_reflect", reflection=full_reflection)
        return full_reflection

    def _self_critique(self) -> Optional[str]:
        """Generate a self-critique of recent reasoning.

        Analyzes the reasoning trace for logical errors, inconsistencies,
        missed steps, and areas for improvement.

        Returns
        -------
        Optional[str]
            Critique text, or None if no issues found.
        """
        if len(self._messages) < 3:
            return None
        recent_messages = self._messages[-4:]
        recent_trace = self._state.reasoning_trace[-6:]
        critique_parts: List[str] = []
        errors_found = False
        failed_tools = [
            r for r in self._state.tool_results if not r.get("success", True)
        ]
        if failed_tools:
            tool_names = [r.get("tool", "unknown") for r in failed_tools]
            critique_parts.append(
                f"Tool failures detected: {', '.join(tool_names)}. "
                f"Previous approach using these tools may need revision."
            )
            errors_found = True
        goal = self._state.current_goal
        if goal:
            assistant_msgs = [
                m for m in recent_messages if m.is_assistant
            ]
            if assistant_msgs:
                last_response = assistant_msgs[-1].content.lower()
                question_words = re.findall(
                    r'\b(what|how|why|when|where|who)\b', goal.lower()
                )
                unanswered = [
                    q for q in question_words
                    if q not in last_response
                ]
                if unanswered and len(unanswered) <= len(question_words):
                    critique_parts.append(
                        f"The response may not fully address all aspects of "
                        f"the original question. Consider: {', '.join(unanswered)}."
                    )
                    errors_found = True
        if self._state.step_count > self._config.max_steps * 0.7:
            critique_parts.append(
                "Approaching step limit. Consider consolidating "
                "results and preparing a final response."
            )
            errors_found = True
        if not errors_found:
            return None
        return "\n".join(critique_parts)

    def _generate_correction(
        self, critique: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a correction based on the self-critique.

        Parameters
        ----------
        critique : str
            The self-critique identifying issues.
        context : Dict[str, Any]
            Current reasoning context.

        Returns
        -------
        Optional[str]
            Correction action description.
        """
        if "tool failures" in critique.lower():
            failed_tool_names = [
                r.get("tool", "unknown")
                for r in self._state.tool_results
                if not r.get("success", True)
            ]
            if failed_tool_names:
                return (
                    f"Avoid using failed tools ({', '.join(failed_tool_names)}). "
                    f"Try alternative approaches to gather the needed information."
                )
        if "step limit" in critique.lower():
            return (
                "Prioritize generating the final response with available "
                "information rather than gathering more data."
            )
        if "not fully address" in critique.lower():
            return (
                "Review the original question components and ensure "
                "each aspect receives a direct, clear answer."
            )
        return (
            f"Addressing critique: {critique[:200]}. "
            f"Adjusting approach to improve output quality."
        )

    def get_reflection_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's reflection behavior.

        Returns
        -------
        Dict[str, Any]
            Reflection statistics including correction count and history.
        """
        return {
            "total_corrections": self._correction_count,
            "max_reflection_iterations": self._max_reflection_iterations,
            "reflection_history_size": len(self._reflection_history),
            "recent_critiques": [
                {
                    "step": r["step"],
                    "critique": r["critique"][:200],
                }
                for r in self._reflection_history[-5:]
            ],
        }
