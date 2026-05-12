"""
Agent Configuration Module
===========================

Comprehensive configuration dataclasses for all agent components including
individual agent settings, multi-agent orchestration, tool configurations,
and memory subsystem parameters. Every configuration object includes
validation, serialization, and sensible defaults.

Classes
-------
- ``AgentConfig``: Primary configuration for a single agent instance
- ``MultiAgentConfig``: Configuration for multi-agent orchestration systems
- ``ToolConfig``: Configuration for individual tool definitions
- ``AgentMemoryConfig``: Configuration for the multi-tier memory subsystem
- ``ReasoningStrategy``: Enum of reasoning approaches
- ``TaskAllocationStrategy``: Enum of task distribution strategies
- ``CommunicationProtocolType``: Enum of inter-agent communication modes
- ``MemoryRetrievalStrategy``: Enum of memory retrieval algorithms
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ReasoningStrategy(Enum):
    """Enumerates the available reasoning strategies for agent inference.

    Each strategy represents a different approach to how the agent processes
    information, generates hypotheses, and reaches conclusions.

    Attributes
    ----------
    CHAIN_OF_THOUGHT : str
        Step-by-step reasoning where each thought leads to the next.
    TREE_OF_THOUGHT : str
        Branching exploration of multiple reasoning paths simultaneously.
    REACT : str
        Interleaved reasoning and action (Reason + Act pattern).
    REFLECTION : str
        Self-critique and iterative refinement of reasoning.
    PLAN_AND_EXECUTE : str
        Create a full plan before executing any actions.
    LEAST_TO_MOST : str
        Decompose complex problems into simpler sub-problems.
    SELF_ASK : str
        Agent asks itself follow-up questions to gather information.
    NONE : str
        Direct inference without explicit reasoning structure.
    """
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    REFLECTION = "reflection"
    PLAN_AND_EXECUTE = "plan_and_execute"
    LEAST_TO_MOST = "least_to_most"
    SELF_ASK = "self_ask"
    NONE = "none"


class TaskAllocationStrategy(Enum):
    """Strategy for distributing tasks among multiple agents.

    Attributes
    ----------
    ROUND_ROBIN : str
        Tasks are distributed in sequential round-robin fashion.
    CAPABILITY_BASED : str
        Tasks are assigned to agents best suited by capability.
    WORK stealing : str
        Idle agents steal tasks from busy agents.
    RANDOM : str
        Tasks are randomly assigned to available agents.
    PRIORITY_QUEUE : str
        Tasks are assigned based on priority ranking.
    LOAD_BALANCED : str
        Tasks assigned to the agent with least current load.
    SPECIALIST : str
        Each task type has a designated specialist agent.
    AUCTION : str
        Agents bid on tasks, highest confidence wins.
    """
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    WORK_STEALING = "work_stealing"
    RANDOM = "random"
    PRIORITY_QUEUE = "priority_queue"
    LOAD_BALANCED = "load_balanced"
    SPECIALIST = "specialist"
    AUCTION = "auction"


class CommunicationProtocolType(Enum):
    """Protocol type for inter-agent communication.

    Attributes
    ----------
    BROADCAST : str
        Messages sent to all agents simultaneously.
    DIRECT : str
        Point-to-point messaging between specific agents.
    BLACKBOARD : str
        Shared blackboard for asynchronous communication.
    PUBLISH_SUBSCRIBE : str
        Pub/sub pattern with topic-based routing.
    STAR : str
        Hub-and-spoke topology with central coordinator.
    MESH : str
        Full mesh where every agent can talk to every other.
    """
    BROADCAST = "broadcast"
    DIRECT = "direct"
    BLACKBOARD = "blackboard"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    STAR = "star"
    MESH = "mesh"


class MemoryRetrievalStrategy(Enum):
    """Strategy for retrieving memories from the memory subsystem.

    Attributes
    ----------
    RECENCY : str
        Return the most recently stored memories first.
    RELEVANCE : str
        Return memories most semantically relevant to the query.
    IMPORTANCE : str
        Return memories ranked by importance score.
    HYBRID : str
        Combine recency, relevance, and importance scores.
    ASSOCIATIVE : str
        Retrieve memories associated with similar contexts.
    SEQUENTIAL : str
        Retrieve memories in chronological order.
    """
    RECENCY = "recency"
    RELEVANCE = "relevance"
    IMPORTANCE = "importance"
    HYBRID = "hybrid"
    ASSOCIATIVE = "associative"
    SEQUENTIAL = "sequential"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when a configuration value fails validation.

    Parameters
    ----------
    field_name : str
        Name of the invalid configuration field.
    message : str
        Human-readable description of the validation failure.
    value : Any, optional
        The offending value that caused the validation error.
    """

    def __init__(
        self,
        field_name: str,
        message: str,
        value: Any = None,
    ) -> None:
        self.field_name = field_name
        self.message = message
        self.value = value
        super().__init__(
            f"Config validation error on field '{field_name}': {message}"
            + (f" (got: {value!r})" if value is not None else "")
        )


def _validate_string_field(
    value: Any,
    field_name: str,
    min_length: int = 1,
    max_length: int = 10000,
    pattern: Optional[str] = None,
) -> str:
    """Validate that a field is a string within length constraints.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).
    min_length : int
        Minimum allowed string length.
    max_length : int
        Maximum allowed string length.
    pattern : str, optional
        Regex pattern the string must match.

    Returns
    -------
    str
        The validated string.

    Raises
    ------
    ConfigValidationError
        If the string does not meet validation criteria.
    """
    if not isinstance(value, str):
        raise ConfigValidationError(
            field_name, f"Expected string, got {type(value).__name__}", value
        )
    if len(value) < min_length:
        raise ConfigValidationError(
            field_name,
            f"String length {len(value)} is below minimum {min_length}",
            value,
        )
    if len(value) > max_length:
        raise ConfigValidationError(
            field_name,
            f"String length {len(value)} exceeds maximum {max_length}",
            value,
        )
    if pattern is not None and not re.match(pattern, value):
        raise ConfigValidationError(
            field_name,
            f"String does not match required pattern: {pattern}",
            value,
        )
    return value


def _validate_numeric_field(
    value: Any,
    field_name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    integer_only: bool = False,
) -> Union[int, float]:
    """Validate that a field is a numeric value within constraints.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).
    min_value : float, optional
        Minimum allowed value (inclusive).
    max_value : float, optional
        Maximum allowed value (inclusive).
    integer_only : bool
        If True, the value must be an integer.

    Returns
    -------
    Union[int, float]
        The validated numeric value.

    Raises
    ------
    ConfigValidationError
        If the value does not meet validation criteria.
    """
    if integer_only:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigValidationError(
                field_name, f"Expected integer, got {type(value).__name__}", value
            )
    elif not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(
            field_name, f"Expected number, got {type(value).__name__}", value
        )
    if min_value is not None and value < min_value:
        raise ConfigValidationError(
            field_name, f"Value {value} is below minimum {min_value}", value
        )
    if max_value is not None and value > max_value:
        raise ConfigValidationError(
            field_name, f"Value {value} exceeds maximum {max_value}", value
        )
    return value


def _validate_bool_field(value: Any, field_name: str) -> bool:
    """Validate that a field is a boolean value.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).

    Returns
    -------
    bool
        The validated boolean value.

    Raises
    ------
    ConfigValidationError
        If the value is not a boolean.
    """
    if not isinstance(value, bool):
        raise ConfigValidationError(
            field_name, f"Expected bool, got {type(value).__name__}", value
        )
    return value


def _validate_list_field(
    value: Any,
    field_name: str,
    min_length: int = 0,
    max_length: int = 10000,
    element_type: Optional[type] = None,
) -> list:
    """Validate that a field is a list within length constraints.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).
    min_length : int
        Minimum allowed list length.
    max_length : int
        Maximum allowed list length.
    element_type : type, optional
        If provided, all elements must be of this type.

    Returns
    -------
    list
        The validated list.

    Raises
    ------
    ConfigValidationError
        If the list does not meet validation criteria.
    """
    if not isinstance(value, list):
        raise ConfigValidationError(
            field_name, f"Expected list, got {type(value).__name__}", value
        )
    if len(value) < min_length:
        raise ConfigValidationError(
            field_name,
            f"List length {len(value)} is below minimum {min_length}",
            value,
        )
    if len(value) > max_length:
        raise ConfigValidationError(
            field_name,
            f"List length {len(value)} exceeds maximum {max_length}",
            value,
        )
    if element_type is not None:
        for idx, element in enumerate(value):
            if not isinstance(element, element_type):
                raise ConfigValidationError(
                    f"{field_name}[{idx}]",
                    f"Expected {element_type.__name__}, "
                    f"got {type(element).__name__}",
                    element,
                )
    return value


def _validate_dict_field(
    value: Any,
    field_name: str,
    key_type: Optional[type] = None,
    value_type: Optional[type] = None,
) -> dict:
    """Validate that a field is a dictionary.

    Parameters
    ----------
    value : Any
        The value to validate.
    field_name : str
        Name of the field (used in error messages).
    key_type : type, optional
        If provided, all keys must be of this type.
    value_type : type, optional
        If provided, all values must be of this type.

    Returns
    -------
    dict
        The validated dictionary.

    Raises
    ------
    ConfigValidationError
        If the value is not a valid dictionary.
    """
    if not isinstance(value, dict):
        raise ConfigValidationError(
            field_name, f"Expected dict, got {type(value).__name__}", value
        )
    if key_type is not None:
        for k in value.keys():
            if not isinstance(k, key_type):
                raise ConfigValidationError(
                    f"{field_name}.key",
                    f"Expected key type {key_type.__name__}, "
                    f"got {type(k).__name__}",
                    k,
                )
    if value_type is not None:
        for k, v in value.items():
            if not isinstance(v, value_type):
                raise ConfigValidationError(
                    f"{field_name}['{k}']",
                    f"Expected value type {value_type.__name__}, "
                    f"got {type(v).__name__}",
                    v,
                )
    return value


# ---------------------------------------------------------------------------
# ToolConfig
# ---------------------------------------------------------------------------

@dataclass
class ToolConfig:
    """Configuration for a single tool that an agent can invoke.

    Tools extend agent capabilities by allowing them to perform external
    actions such as file I/O, web searches, code execution, and API calls.

    Parameters
    ----------
    name : str
        Unique identifier for the tool (e.g., ``"web_search"``).
    description : str
        Human-readable description of what the tool does, used by the LLM
        to decide when to invoke this tool.
    parameters : Dict[str, Any]
        JSON Schema describing the tool's expected parameters.  Keys are
        parameter names, values describe type, constraints, and defaults.
    required : List[str]
        List of parameter names that must be provided on every invocation.
    timeout : float
        Maximum execution time in seconds before the tool call is aborted.
    retry_count : int
        Number of retry attempts on transient failures.
    dangerous : bool
        If True, the tool requires explicit user confirmation before execution.

    Examples
    --------
    >>> config = ToolConfig(
    ...     name="calculator",
    ...     description="Evaluates mathematical expressions safely",
    ...     parameters={
    ...         "expression": {
    ...             "type": "string",
    ...             "description": "The math expression to evaluate"
    ...         }
    ...     },
    ...     required=["expression"],
    ...     timeout=10.0,
    ...     retry_count=2,
    ...     dangerous=False,
    ... )
    """

    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    dangerous: bool = False
    examples: List[Dict[str, Any]] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    category: str = "general"
    rate_limit_per_minute: int = 60
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self.name = _validate_string_field(
            self.name,
            "ToolConfig.name",
            min_length=1,
            max_length=256,
            pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
        )
        self.description = _validate_string_field(
            self.description,
            "ToolConfig.description",
            min_length=1,
            max_length=5000,
        )
        self.parameters = _validate_dict_field(
            self.parameters, "ToolConfig.parameters", key_type=str
        )
        self.required = _validate_list_field(
            self.required, "ToolConfig.required", element_type=str
        )
        self.timeout = _validate_numeric_field(
            self.timeout,
            "ToolConfig.timeout",
            min_value=0.1,
            max_value=3600.0,
        )
        self.retry_count = _validate_numeric_field(
            self.retry_count,
            "ToolConfig.retry_count",
            min_value=0,
            max_value=100,
            integer_only=True,
        )
        self.dangerous = _validate_bool_field(self.dangerous, "ToolConfig.dangerous")
        self.version = _validate_string_field(
            self.version, "ToolConfig.version", max_length=64
        )
        self.category = _validate_string_field(
            self.category, "ToolConfig.category", max_length=128
        )
        self.rate_limit_per_minute = _validate_numeric_field(
            self.rate_limit_per_minute,
            "ToolConfig.rate_limit_per_minute",
            min_value=1,
            max_value=100000,
            integer_only=True,
        )
        self.enabled = _validate_bool_field(self.enabled, "ToolConfig.enabled")
        self._validate_required_in_parameters()

    def _validate_required_in_parameters(self) -> None:
        """Ensure all required parameters are defined in the parameters schema."""
        parameter_names = set(self.parameters.keys())
        for req_param in self.required:
            if req_param not in parameter_names:
                raise ConfigValidationError(
                    f"ToolConfig.required['{req_param}']",
                    f"Required parameter '{req_param}' not found in "
                    f"parameters schema. Available: {parameter_names}",
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a plain dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
            "required": list(self.required),
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "dangerous": self.dangerous,
            "examples": copy.deepcopy(self.examples),
            "tags": list(self.tags),
            "version": self.version,
            "category": self.category,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "enabled": self.enabled,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolConfig:
        """Deserialize a configuration from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration values.

        Returns
        -------
        ToolConfig
            Instantiated configuration object.
        """
        safe_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "tags" in safe_data and isinstance(safe_data["tags"], list):
            safe_data["tags"] = set(safe_data["tags"])
        return cls(**safe_data)

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
            JSON-encoded configuration.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> ToolConfig:
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded configuration.

        Returns
        -------
        ToolConfig
            Instantiated configuration object.
        """
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# AgentMemoryConfig
# ---------------------------------------------------------------------------

@dataclass
class AgentMemoryConfig:
    """Configuration for the multi-tier agent memory subsystem.

    Controls the capacity and behavior of short-term, long-term, episodic,
    semantic, and working memory components.

    Parameters
    ----------
    short_term_capacity : int
        Maximum number of messages in the short-term conversation buffer.
    long_term_capacity : int
        Maximum number of entries in long-term persistent storage.
    retrieval_strategy : MemoryRetrievalStrategy
        Default strategy for retrieving memories during agent reasoning.
    decay_rate : float
        Rate at which memory importance scores decay over time (0.0-1.0).

    Examples
    --------
    >>> config = AgentMemoryConfig(
    ...     short_term_capacity=50,
    ...     long_term_capacity=10000,
    ...     retrieval_strategy=MemoryRetrievalStrategy.HYBRID,
    ...     decay_rate=0.01,
    ... )
    """

    short_term_capacity: int = 100
    long_term_capacity: int = 50000
    retrieval_strategy: MemoryRetrievalStrategy = (
        MemoryRetrievalStrategy.HYBRID
    )
    decay_rate: float = 0.005
    working_memory_capacity: int = 10
    episodic_capacity: int = 1000
    semantic_capacity: int = 50000
    consolidation_threshold: float = 0.7
    consolidation_interval: int = 20
    retrieval_top_k: int = 10
    relevance_threshold: float = 0.3
    max_retrieval_time_ms: float = 100.0
    enable_persistence: bool = False
    persistence_path: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 768
    cache_enabled: bool = True
    cache_size: int = 1000
    enable_compression: bool = False
    compression_level: int = 6
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all configuration fields after initialization."""
        self.short_term_capacity = _validate_numeric_field(
            self.short_term_capacity,
            "AgentMemoryConfig.short_term_capacity",
            min_value=1,
            max_value=100000,
            integer_only=True,
        )
        self.long_term_capacity = _validate_numeric_field(
            self.long_term_capacity,
            "AgentMemoryConfig.long_term_capacity",
            min_value=1,
            max_value=10000000,
            integer_only=True,
        )
        if not isinstance(self.retrieval_strategy, MemoryRetrievalStrategy):
            valid_strategies = [s.value for s in MemoryRetrievalStrategy]
            if isinstance(self.retrieval_strategy, str):
                try:
                    self.retrieval_strategy = MemoryRetrievalStrategy(
                        self.retrieval_strategy
                    )
                except ValueError:
                    raise ConfigValidationError(
                        "AgentMemoryConfig.retrieval_strategy",
                        f"Invalid strategy '{self.retrieval_strategy}'. "
                        f"Valid: {valid_strategies}",
                    )
            else:
                raise ConfigValidationError(
                    "AgentMemoryConfig.retrieval_strategy",
                    f"Expected MemoryRetrievalStrategy, got "
                    f"{type(self.retrieval_strategy).__name__}",
                )
        self.decay_rate = _validate_numeric_field(
            self.decay_rate,
            "AgentMemoryConfig.decay_rate",
            min_value=0.0,
            max_value=1.0,
        )
        self.working_memory_capacity = _validate_numeric_field(
            self.working_memory_capacity,
            "AgentMemoryConfig.working_memory_capacity",
            min_value=1,
            max_value=10000,
            integer_only=True,
        )
        self.episodic_capacity = _validate_numeric_field(
            self.episodic_capacity,
            "AgentMemoryConfig.episodic_capacity",
            min_value=0,
            max_value=1000000,
            integer_only=True,
        )
        self.semantic_capacity = _validate_numeric_field(
            self.semantic_capacity,
            "AgentMemoryConfig.semantic_capacity",
            min_value=0,
            max_value=10000000,
            integer_only=True,
        )
        self.consolidation_threshold = _validate_numeric_field(
            self.consolidation_threshold,
            "AgentMemoryConfig.consolidation_threshold",
            min_value=0.0,
            max_value=1.0,
        )
        self.consolidation_interval = _validate_numeric_field(
            self.consolidation_interval,
            "AgentMemoryConfig.consolidation_interval",
            min_value=1,
            max_value=10000,
            integer_only=True,
        )
        self.retrieval_top_k = _validate_numeric_field(
            self.retrieval_top_k,
            "AgentMemoryConfig.retrieval_top_k",
            min_value=1,
            max_value=10000,
            integer_only=True,
        )
        self.relevance_threshold = _validate_numeric_field(
            self.relevance_threshold,
            "AgentMemoryConfig.relevance_threshold",
            min_value=0.0,
            max_value=1.0,
        )
        self.max_retrieval_time_ms = _validate_numeric_field(
            self.max_retrieval_time_ms,
            "AgentMemoryConfig.max_retrieval_time_ms",
            min_value=1.0,
            max_value=60000.0,
        )
        self.enable_persistence = _validate_bool_field(
            self.enable_persistence, "AgentMemoryConfig.enable_persistence"
        )
        self.embedding_dimension = _validate_numeric_field(
            self.embedding_dimension,
            "AgentMemoryConfig.embedding_dimension",
            min_value=1,
            max_value=32768,
            integer_only=True,
        )
        self.cache_enabled = _validate_bool_field(
            self.cache_enabled, "AgentMemoryConfig.cache_enabled"
        )
        self.cache_size = _validate_numeric_field(
            self.cache_size,
            "AgentMemoryConfig.cache_size",
            min_value=1,
            max_value=1000000,
            integer_only=True,
        )
        self.enable_compression = _validate_bool_field(
            self.enable_compression, "AgentMemoryConfig.enable_compression"
        )
        self.compression_level = _validate_numeric_field(
            self.compression_level,
            "AgentMemoryConfig.compression_level",
            min_value=1,
            max_value=9,
            integer_only=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary with all configuration fields.
        """
        return {
            "short_term_capacity": self.short_term_capacity,
            "long_term_capacity": self.long_term_capacity,
            "retrieval_strategy": self.retrieval_strategy.value,
            "decay_rate": self.decay_rate,
            "working_memory_capacity": self.working_memory_capacity,
            "episodic_capacity": self.episodic_capacity,
            "semantic_capacity": self.semantic_capacity,
            "consolidation_threshold": self.consolidation_threshold,
            "consolidation_interval": self.consolidation_interval,
            "retrieval_top_k": self.retrieval_top_k,
            "relevance_threshold": self.relevance_threshold,
            "max_retrieval_time_ms": self.max_retrieval_time_ms,
            "enable_persistence": self.enable_persistence,
            "persistence_path": self.persistence_path,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "enable_compression": self.enable_compression,
            "compression_level": self.compression_level,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentMemoryConfig:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with configuration values.

        Returns
        -------
        AgentMemoryConfig
            Instantiated configuration object.
        """
        safe_data = copy.deepcopy(data)
        if "retrieval_strategy" in safe_data:
            if isinstance(safe_data["retrieval_strategy"], str):
                safe_data["retrieval_strategy"] = MemoryRetrievalStrategy(
                    safe_data["retrieval_strategy"]
                )
        filtered = {
            k: v
            for k, v in safe_data.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**filtered)

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
            JSON-encoded configuration.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> AgentMemoryConfig:
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded configuration.

        Returns
        -------
        AgentMemoryConfig
            Instantiated configuration object.
        """
        return cls.from_dict(json.loads(json_str))


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Primary configuration for a single agent instance.

    Controls every aspect of agent behavior including identity, model
    parameters, reasoning strategy, tool access, memory settings, and
    execution constraints.

    Parameters
    ----------
    name : str
        Unique human-readable name for the agent.
    role : str
        The role or persona the agent assumes (e.g., ``"research assistant"``).
    description : str
        Detailed description of the agent's capabilities and purpose.
    model : str
        Identifier of the LLM backend to use (e.g., ``"gpt-4"``).
    temperature : float
        Sampling temperature controlling output randomness (0.0-2.0).
    max_tokens : int
        Maximum tokens the agent can generate per response.
    system_prompt : str
        System-level instructions that define the agent's behavior.
    memory_enabled : bool
        Whether the agent maintains conversation memory across turns.
    tools_enabled : bool
        Whether the agent can invoke external tools.
    max_steps : int
        Maximum number of reasoning-action cycles per run.
    planning_enabled : bool
        Whether the agent creates execution plans before acting.
    reasoning_strategy : ReasoningStrategy
        The reasoning approach used for inference.

    Examples
    --------
    >>> config = AgentConfig(
    ...     name="researcher",
    ...     role="research assistant",
    ...     model="gpt-4",
    ...     temperature=0.7,
    ...     reasoning_strategy=ReasoningStrategy.REACT,
    ...     tools_enabled=True,
    ...     planning_enabled=True,
    ... )
    """

    name: str = "agent"
    role: str = "assistant"
    description: str = "A general-purpose AI assistant"
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."
    memory_enabled: bool = True
    tools_enabled: bool = True
    max_steps: int = 10
    planning_enabled: bool = False
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.REACT
    tools: List[ToolConfig] = field(default_factory=list)
    memory_config: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)
    allowed_tool_names: Set[str] = field(default_factory=set)
    blocked_tool_names: Set[str] = field(default_factory=set)
    max_tool_calls_per_step: int = 5
    require_tool_confirmation: bool = False
    timeout_seconds: float = 300.0
    retry_on_failure: bool = True
    max_retries: int = 3
    verbose_logging: bool = False
    enable_tracing: bool = False
    max_input_tokens: int = 32768
    max_output_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all configuration fields after initialization."""
        self.name = _validate_string_field(
            self.name,
            "AgentConfig.name",
            min_length=1,
            max_length=256,
            pattern=r"^[a-zA-Z][a-zA-Z0-9_\- ]*$",
        )
        self.role = _validate_string_field(
            self.role, "AgentConfig.role", min_length=1, max_length=1000
        )
        self.description = _validate_string_field(
            self.description,
            "AgentConfig.description",
            min_length=1,
            max_length=10000,
        )
        self.model = _validate_string_field(
            self.model, "AgentConfig.model", min_length=1, max_length=256
        )
        self.system_prompt = _validate_string_field(
            self.system_prompt,
            "AgentConfig.system_prompt",
            max_length=100000,
        )
        self.temperature = _validate_numeric_field(
            self.temperature,
            "AgentConfig.temperature",
            min_value=0.0,
            max_value=2.0,
        )
        self.max_tokens = _validate_numeric_field(
            self.max_tokens,
            "AgentConfig.max_tokens",
            min_value=1,
            max_value=1000000,
            integer_only=True,
        )
        self.top_p = _validate_numeric_field(
            self.top_p, "AgentConfig.top_p", min_value=0.0, max_value=1.0
        )
        self.frequency_penalty = _validate_numeric_field(
            self.frequency_penalty,
            "AgentConfig.frequency_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
        self.presence_penalty = _validate_numeric_field(
            self.presence_penalty,
            "AgentConfig.presence_penalty",
            min_value=-2.0,
            max_value=2.0,
        )
        self.stop_sequences = _validate_list_field(
            self.stop_sequences,
            "AgentConfig.stop_sequences",
            max_length=100,
            element_type=str,
        )
        self.memory_enabled = _validate_bool_field(
            self.memory_enabled, "AgentConfig.memory_enabled"
        )
        self.tools_enabled = _validate_bool_field(
            self.tools_enabled, "AgentConfig.tools_enabled"
        )
        self.max_steps = _validate_numeric_field(
            self.max_steps,
            "AgentConfig.max_steps",
            min_value=1,
            max_value=10000,
            integer_only=True,
        )
        self.planning_enabled = _validate_bool_field(
            self.planning_enabled, "AgentConfig.planning_enabled"
        )
        if not isinstance(self.reasoning_strategy, ReasoningStrategy):
            valid_strategies = [s.value for s in ReasoningStrategy]
            if isinstance(self.reasoning_strategy, str):
                try:
                    self.reasoning_strategy = ReasoningStrategy(
                        self.reasoning_strategy
                    )
                except ValueError:
                    raise ConfigValidationError(
                        "AgentConfig.reasoning_strategy",
                        f"Invalid strategy '{self.reasoning_strategy}'. "
                        f"Valid: {valid_strategies}",
                    )
            else:
                raise ConfigValidationError(
                    "AgentConfig.reasoning_strategy",
                    f"Expected ReasoningStrategy, got "
                    f"{type(self.reasoning_strategy).__name__}",
                )
        self.tools = _validate_list_field(
            self.tools, "AgentConfig.tools", max_length=1000
        )
        for idx, tool_cfg in enumerate(self.tools):
            if not isinstance(tool_cfg, ToolConfig):
                raise ConfigValidationError(
                    f"AgentConfig.tools[{idx}]",
                    f"Expected ToolConfig, got {type(tool_cfg).__name__}",
                )
        self.max_tool_calls_per_step = _validate_numeric_field(
            self.max_tool_calls_per_step,
            "AgentConfig.max_tool_calls_per_step",
            min_value=1,
            max_value=100,
            integer_only=True,
        )
        self.require_tool_confirmation = _validate_bool_field(
            self.require_tool_confirmation,
            "AgentConfig.require_tool_confirmation",
        )
        self.timeout_seconds = _validate_numeric_field(
            self.timeout_seconds,
            "AgentConfig.timeout_seconds",
            min_value=1.0,
            max_value=86400.0,
        )
        self.retry_on_failure = _validate_bool_field(
            self.retry_on_failure, "AgentConfig.retry_on_failure"
        )
        self.max_retries = _validate_numeric_field(
            self.max_retries,
            "AgentConfig.max_retries",
            min_value=0,
            max_value=100,
            integer_only=True,
        )
        self.verbose_logging = _validate_bool_field(
            self.verbose_logging, "AgentConfig.verbose_logging"
        )
        self.enable_tracing = _validate_bool_field(
            self.enable_tracing, "AgentConfig.enable_tracing"
        )
        self.max_input_tokens = _validate_numeric_field(
            self.max_input_tokens,
            "AgentConfig.max_input_tokens",
            min_value=1,
            max_value=10000000,
            integer_only=True,
        )
        self.max_output_tokens = _validate_numeric_field(
            self.max_output_tokens,
            "AgentConfig.max_output_tokens",
            min_value=1,
            max_value=1000000,
            integer_only=True,
        )
        self._validate_tool_access_lists()

    def _validate_tool_access_lists(self) -> None:
        """Ensure allowed and blocked tool name lists are valid."""
        if not isinstance(self.allowed_tool_names, set):
            self.allowed_tool_names = set(self.allowed_tool_names)
        if not isinstance(self.blocked_tool_names, set):
            self.blocked_tool_names = set(self.blocked_tool_names)
        for name in self.allowed_tool_names:
            if not isinstance(name, str):
                raise ConfigValidationError(
                    "AgentConfig.allowed_tool_names",
                    f"Expected str, got {type(name).__name__}",
                    name,
                )
        for name in self.blocked_tool_names:
            if not isinstance(name, str):
                raise ConfigValidationError(
                    "AgentConfig.blocked_tool_names",
                    f"Expected str, got {type(name).__name__}",
                    name,
                )
        overlap = self.allowed_tool_names & self.blocked_tool_names
        if overlap:
            raise ConfigValidationError(
                "AgentConfig.tool_access",
                f"Tools cannot be both allowed and blocked: {overlap}",
            )

    @property
    def effective_tools(self) -> List[ToolConfig]:
        """Return the list of tools after applying allow/block filters.

        Returns
        -------
        List[ToolConfig]
            Filtered list of enabled tool configurations.
        """
        result = []
        for tool in self.tools:
            if not tool.enabled:
                continue
            if self.blocked_tool_names and tool.name in self.blocked_tool_names:
                continue
            if (
                self.allowed_tool_names
                and tool.name not in self.allowed_tool_names
            ):
                continue
            result.append(tool)
        return result

    @property
    def total_token_limit(self) -> int:
        """Return the combined input + output token limit.

        Returns
        -------
        int
            Sum of max_input_tokens and max_output_tokens.
        """
        return self.max_input_tokens + self.max_output_tokens

    @property
    def supports_planning(self) -> bool:
        """Check if this configuration supports planning features.

        Returns
        -------
        bool
            True if planning is enabled and the reasoning strategy is compatible.
        """
        planning_compatible = {
            ReasoningStrategy.PLAN_AND_EXECUTE,
            ReasoningStrategy.REACT,
            ReasoningStrategy.TREE_OF_THOUGHT,
            ReasoningStrategy.LEAST_TO_MOST,
        }
        return self.planning_enabled and self.reasoning_strategy in planning_compatible

    @property
    def supports_reflection(self) -> bool:
        """Check if this configuration supports reflection features.

        Returns
        -------
        bool
            True if the reasoning strategy is REFLECTION or REACT.
        """
        return self.reasoning_strategy in {
            ReasoningStrategy.REFLECTION,
            ReasoningStrategy.REACT,
        }

    def get_tool_config(self, tool_name: str) -> Optional[ToolConfig]:
        """Look up a tool configuration by name.

        Parameters
        ----------
        tool_name : str
            Name of the tool to look up.

        Returns
        -------
        Optional[ToolConfig]
            The tool configuration if found and accessible, else None.
        """
        for tool in self.effective_tools:
            if tool.name == tool_name:
                return tool
        return None

    def add_tool(self, tool_config: ToolConfig) -> None:
        """Add a tool configuration to this agent.

        Parameters
        ----------
        tool_config : ToolConfig
            The tool configuration to add.

        Raises
        ------
        ConfigValidationError
            If a tool with the same name already exists.
        """
        existing_names = {t.name for t in self.tools}
        if tool_config.name in existing_names:
            raise ConfigValidationError(
                "AgentConfig.add_tool",
                f"Tool '{tool_config.name}' already exists",
            )
        self.tools.append(tool_config)

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool configuration by name.

        Parameters
        ----------
        tool_name : str
            Name of the tool to remove.

        Returns
        -------
        bool
            True if the tool was found and removed, False otherwise.
        """
        for idx, tool in enumerate(self.tools):
            if tool.name == tool_name:
                self.tools.pop(idx)
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Complete dictionary representation.
        """
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop_sequences": list(self.stop_sequences),
            "system_prompt": self.system_prompt,
            "memory_enabled": self.memory_enabled,
            "tools_enabled": self.tools_enabled,
            "max_steps": self.max_steps,
            "planning_enabled": self.planning_enabled,
            "reasoning_strategy": self.reasoning_strategy.value,
            "tools": [t.to_dict() for t in self.tools],
            "memory_config": self.memory_config.to_dict(),
            "allowed_tool_names": list(self.allowed_tool_names),
            "blocked_tool_names": list(self.blocked_tool_names),
            "max_tool_calls_per_step": self.max_tool_calls_per_step,
            "require_tool_confirmation": self.require_tool_confirmation,
            "timeout_seconds": self.timeout_seconds,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "verbose_logging": self.verbose_logging,
            "enable_tracing": self.enable_tracing,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentConfig:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with configuration values.

        Returns
        -------
        AgentConfig
            Instantiated configuration object.
        """
        safe_data = copy.deepcopy(data)
        if "reasoning_strategy" in safe_data:
            if isinstance(safe_data["reasoning_strategy"], str):
                safe_data["reasoning_strategy"] = ReasoningStrategy(
                    safe_data["reasoning_strategy"]
                )
        if "tools" in safe_data:
            safe_data["tools"] = [
                ToolConfig.from_dict(t)
                if isinstance(t, dict)
                else t
                for t in safe_data["tools"]
            ]
        if "memory_config" in safe_data:
            if isinstance(safe_data["memory_config"], dict):
                safe_data["memory_config"] = AgentMemoryConfig.from_dict(
                    safe_data["memory_config"]
                )
        if "allowed_tool_names" in safe_data:
            safe_data["allowed_tool_names"] = set(
                safe_data["allowed_tool_names"]
            )
        if "blocked_tool_names" in safe_data:
            safe_data["blocked_tool_names"] = set(
                safe_data["blocked_tool_names"]
            )
        filtered = {
            k: v
            for k, v in safe_data.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**filtered)

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
            JSON-encoded configuration.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> AgentConfig:
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded configuration.

        Returns
        -------
        AgentConfig
            Instantiated configuration object.
        """
        return cls.from_dict(json.loads(json_str))

    def copy(self) -> AgentConfig:
        """Create a deep copy of this configuration.

        Returns
        -------
        AgentConfig
            Independent copy of the configuration.
        """
        return AgentConfig.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# MultiAgentConfig
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentConfig:
    """Configuration for multi-agent orchestration systems.

    Controls how multiple agents interact, communicate, allocate tasks,
    and share information during collaborative problem solving.

    Parameters
    ----------
    agents : List[AgentConfig]
        Configuration for each agent in the multi-agent system.
    communication_protocol : CommunicationProtocolType
        The protocol used for inter-agent communication.
    max_rounds : int
        Maximum number of communication rounds before termination.
    shared_memory : bool
        Whether agents share a common memory store.
    task_allocation_strategy : TaskAllocationStrategy
        Strategy for distributing tasks among agents.

    Examples
    --------
    >>> researcher = AgentConfig(name="researcher", role="researcher")
    >>> writer = AgentConfig(name="writer", role="writer")
    >>> config = MultiAgentConfig(
    ...     agents=[researcher, writer],
    ...     communication_protocol=CommunicationProtocolType.DIRECT,
    ...     max_rounds=5,
    ... )
    """

    agents: List[AgentConfig] = field(default_factory=list)
    communication_protocol: CommunicationProtocolType = (
        CommunicationProtocolType.BROADCAST
    )
    max_rounds: int = 10
    shared_memory: bool = False
    task_allocation_strategy: TaskAllocationStrategy = (
        TaskAllocationStrategy.CAPABILITY_BASED
    )
    timeout_seconds: float = 600.0
    max_parallel_agents: int = 5
    enable_conflict_resolution: bool = True
    enable_result_aggregation: bool = True
    consensus_threshold: float = 0.8
    voting_method: str = "majority"
    debate_rounds: int = 3
    hierarchy_levels: int = 2
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    log_inter_agent_messages: bool = False
    shared_context: str = ""
    task_priority_order: List[str] = field(default_factory=list)
    agent_priorities: Dict[str, float] = field(default_factory=dict)
    fallback_strategy: TaskAllocationStrategy = (
        TaskAllocationStrategy.RANDOM
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate all configuration fields after initialization."""
        self.agents = _validate_list_field(
            self.agents, "MultiAgentConfig.agents", max_length=1000
        )
        for idx, agent_cfg in enumerate(self.agents):
            if not isinstance(agent_cfg, AgentConfig):
                raise ConfigValidationError(
                    f"MultiAgentConfig.agents[{idx}]",
                    f"Expected AgentConfig, got {type(agent_cfg).__name__}",
                )
        self._validate_unique_agent_names()
        if not isinstance(self.communication_protocol, CommunicationProtocolType):
            valid_protocols = [p.value for p in CommunicationProtocolType]
            if isinstance(self.communication_protocol, str):
                try:
                    self.communication_protocol = CommunicationProtocolType(
                        self.communication_protocol
                    )
                except ValueError:
                    raise ConfigValidationError(
                        "MultiAgentConfig.communication_protocol",
                        f"Invalid protocol '{self.communication_protocol}'. "
                        f"Valid: {valid_protocols}",
                    )
            else:
                raise ConfigValidationError(
                    "MultiAgentConfig.communication_protocol",
                    f"Expected CommunicationProtocolType, got "
                    f"{type(self.communication_protocol).__name__}",
                )
        self.max_rounds = _validate_numeric_field(
            self.max_rounds,
            "MultiAgentConfig.max_rounds",
            min_value=1,
            max_value=10000,
            integer_only=True,
        )
        self.shared_memory = _validate_bool_field(
            self.shared_memory, "MultiAgentConfig.shared_memory"
        )
        if not isinstance(self.task_allocation_strategy, TaskAllocationStrategy):
            valid_strategies = [s.value for s in TaskAllocationStrategy]
            if isinstance(self.task_allocation_strategy, str):
                try:
                    self.task_allocation_strategy = TaskAllocationStrategy(
                        self.task_allocation_strategy
                    )
                except ValueError:
                    raise ConfigValidationError(
                        "MultiAgentConfig.task_allocation_strategy",
                        f"Invalid strategy '{self.task_allocation_strategy}'. "
                        f"Valid: {valid_strategies}",
                    )
            else:
                raise ConfigValidationError(
                    "MultiAgentConfig.task_allocation_strategy",
                    f"Expected TaskAllocationStrategy, got "
                    f"{type(self.task_allocation_strategy).__name__}",
                )
        self.timeout_seconds = _validate_numeric_field(
            self.timeout_seconds,
            "MultiAgentConfig.timeout_seconds",
            min_value=1.0,
            max_value=86400.0,
        )
        self.max_parallel_agents = _validate_numeric_field(
            self.max_parallel_agents,
            "MultiAgentConfig.max_parallel_agents",
            min_value=1,
            max_value=1000,
            integer_only=True,
        )
        self.enable_conflict_resolution = _validate_bool_field(
            self.enable_conflict_resolution,
            "MultiAgentConfig.enable_conflict_resolution",
        )
        self.enable_result_aggregation = _validate_bool_field(
            self.enable_result_aggregation,
            "MultiAgentConfig.enable_result_aggregation",
        )
        self.consensus_threshold = _validate_numeric_field(
            self.consensus_threshold,
            "MultiAgentConfig.consensus_threshold",
            min_value=0.0,
            max_value=1.0,
        )
        self.voting_method = _validate_string_field(
            self.voting_method,
            "MultiAgentConfig.voting_method",
            max_length=64,
        )
        self._validate_voting_method()
        self.debate_rounds = _validate_numeric_field(
            self.debate_rounds,
            "MultiAgentConfig.debate_rounds",
            min_value=1,
            max_value=1000,
            integer_only=True,
        )
        self.hierarchy_levels = _validate_numeric_field(
            self.hierarchy_levels,
            "MultiAgentConfig.hierarchy_levels",
            min_value=1,
            max_value=100,
            integer_only=True,
        )
        self.enable_monitoring = _validate_bool_field(
            self.enable_monitoring, "MultiAgentConfig.enable_monitoring"
        )
        self.monitoring_interval = _validate_numeric_field(
            self.monitoring_interval,
            "MultiAgentConfig.monitoring_interval",
            min_value=0.01,
            max_value=3600.0,
        )
        self.log_inter_agent_messages = _validate_bool_field(
            self.log_inter_agent_messages,
            "MultiAgentConfig.log_inter_agent_messages",
        )
        self.agent_priorities = _validate_dict_field(
            self.agent_priorities,
            "MultiAgentConfig.agent_priorities",
            key_type=str,
        )
        if not isinstance(self.fallback_strategy, TaskAllocationStrategy):
            if isinstance(self.fallback_strategy, str):
                self.fallback_strategy = TaskAllocationStrategy(
                    self.fallback_strategy
                )
            else:
                raise ConfigValidationError(
                    "MultiAgentConfig.fallback_strategy",
                    f"Expected TaskAllocationStrategy, got "
                    f"{type(self.fallback_strategy).__name__}",
                )

    def _validate_unique_agent_names(self) -> None:
        """Ensure all agent names are unique within the configuration."""
        names = [agent.name for agent in self.agents]
        seen: set = set()
        duplicates: set = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ConfigValidationError(
                "MultiAgentConfig.agents",
                f"Duplicate agent names detected: {duplicates}",
            )

    def _validate_voting_method(self) -> None:
        """Validate that the voting method is one of the known methods."""
        valid_methods = {
            "majority",
            "unanimous",
            "weighted",
            "ranked_choice",
            "approval",
            "borda_count",
            "plurality",
        }
        if self.voting_method not in valid_methods:
            raise ConfigValidationError(
                "MultiAgentConfig.voting_method",
                f"Invalid voting method '{self.voting_method}'. "
                f"Valid: {sorted(valid_methods)}",
            )

    @property
    def agent_count(self) -> int:
        """Return the number of agents in this configuration.

        Returns
        -------
        int
            Number of registered agent configurations.
        """
        return len(self.agents)

    @property
    def agent_names(self) -> List[str]:
        """Return a list of all agent names.

        Returns
        -------
        List[str]
            Agent name strings in registration order.
        """
        return [agent.name for agent in self.agents]

    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """Look up an agent configuration by name.

        Parameters
        ----------
        name : str
            Name of the agent to look up.

        Returns
        -------
        Optional[AgentConfig]
            The agent configuration if found, else None.
        """
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def add_agent(self, agent_config: AgentConfig) -> None:
        """Add an agent configuration to the multi-agent system.

        Parameters
        ----------
        agent_config : AgentConfig
            The agent configuration to add.

        Raises
        ------
        ConfigValidationError
            If an agent with the same name already exists.
        """
        existing_names = {a.name for a in self.agents}
        if agent_config.name in existing_names:
            raise ConfigValidationError(
                "MultiAgentConfig.add_agent",
                f"Agent '{agent_config.name}' already exists",
            )
        self.agents.append(agent_config)

    def remove_agent(self, agent_name: str) -> bool:
        """Remove an agent configuration by name.

        Parameters
        ----------
        agent_name : str
            Name of the agent to remove.

        Returns
        -------
        bool
            True if the agent was found and removed, False otherwise.
        """
        for idx, agent in enumerate(self.agents):
            if agent.name == agent_name:
                self.agents.pop(idx)
                self.agent_priorities.pop(agent_name, None)
                return True
        return False

    def reorder_agents(self, agent_names: List[str]) -> None:
        """Reorder agents according to a specified name sequence.

        Parameters
        ----------
        agent_names : List[str]
            Desired order of agent names. Must include all existing agents.

        Raises
        ------
        ConfigValidationError
            If the name list does not match existing agents.
        """
        existing_names = {a.name for a in self.agents}
        requested_names = set(agent_names)
        if existing_names != requested_names:
            missing = existing_names - requested_names
            extra = requested_names - existing_names
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"extra: {extra}")
            raise ConfigValidationError(
                "MultiAgentConfig.reorder_agents",
                f"Agent name mismatch: {'; '.join(parts)}",
            )
        agent_map = {a.name: a for a in self.agents}
        self.agents = [agent_map[name] for name in agent_names]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Complete dictionary representation.
        """
        return {
            "agents": [a.to_dict() for a in self.agents],
            "communication_protocol": self.communication_protocol.value,
            "max_rounds": self.max_rounds,
            "shared_memory": self.shared_memory,
            "task_allocation_strategy": self.task_allocation_strategy.value,
            "timeout_seconds": self.timeout_seconds,
            "max_parallel_agents": self.max_parallel_agents,
            "enable_conflict_resolution": self.enable_conflict_resolution,
            "enable_result_aggregation": self.enable_result_aggregation,
            "consensus_threshold": self.consensus_threshold,
            "voting_method": self.voting_method,
            "debate_rounds": self.debate_rounds,
            "hierarchy_levels": self.hierarchy_levels,
            "enable_monitoring": self.enable_monitoring,
            "monitoring_interval": self.monitoring_interval,
            "log_inter_agent_messages": self.log_inter_agent_messages,
            "shared_context": self.shared_context,
            "task_priority_order": list(self.task_priority_order),
            "agent_priorities": copy.deepcopy(self.agent_priorities),
            "fallback_strategy": self.fallback_strategy.value,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MultiAgentConfig:
        """Deserialize from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary with configuration values.

        Returns
        -------
        MultiAgentConfig
            Instantiated configuration object.
        """
        safe_data = copy.deepcopy(data)
        if "communication_protocol" in safe_data:
            if isinstance(safe_data["communication_protocol"], str):
                safe_data["communication_protocol"] = CommunicationProtocolType(
                    safe_data["communication_protocol"]
                )
        if "task_allocation_strategy" in safe_data:
            if isinstance(safe_data["task_allocation_strategy"], str):
                safe_data["task_allocation_strategy"] = TaskAllocationStrategy(
                    safe_data["task_allocation_strategy"]
                )
        if "fallback_strategy" in safe_data:
            if isinstance(safe_data["fallback_strategy"], str):
                safe_data["fallback_strategy"] = TaskAllocationStrategy(
                    safe_data["fallback_strategy"]
                )
        if "agents" in safe_data:
            safe_data["agents"] = [
                AgentConfig.from_dict(a)
                if isinstance(a, dict)
                else a
                for a in safe_data["agents"]
            ]
        filtered = {
            k: v
            for k, v in safe_data.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**filtered)

    def to_json(self) -> str:
        """Serialize to a JSON string.

        Returns
        -------
        str
            JSON-encoded configuration.
        """
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> MultiAgentConfig:
        """Deserialize from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON-encoded configuration.

        Returns
        -------
        MultiAgentConfig
            Instantiated configuration object.
        """
        return cls.from_dict(json.loads(json_str))

    def copy(self) -> MultiAgentConfig:
        """Create a deep copy of this configuration.

        Returns
        -------
        MultiAgentConfig
            Independent copy of the configuration.
        """
        return MultiAgentConfig.from_dict(self.to_dict())


# ---------------------------------------------------------------------------
# Configuration merge utilities
# ---------------------------------------------------------------------------

def merge_agent_configs(
    base: AgentConfig, override: AgentConfig
) -> AgentConfig:
    """Merge two AgentConfig objects, with override taking precedence.

    Non-default values in ``override`` replace corresponding values in
    ``base``.  Tool lists are merged (union by name).  ``None`` values in
    the override signal "keep the base value."

    Parameters
    ----------
    base : AgentConfig
        The base configuration.
    override : AgentConfig
        Override values take precedence.

    Returns
    -------
    AgentConfig
        A new merged configuration.
    """
    merged_data = base.to_dict()
    override_data = override.to_dict()

    _defaults = AgentConfig()
    default_data = _defaults.to_dict()

    for key, value in override_data.items():
        if value != default_data.get(key):
            if key == "tools":
                existing_names = {t["name"] for t in merged_data["tools"]}
                for tool in value:
                    if tool["name"] not in existing_names:
                        merged_data["tools"].append(tool)
                    else:
                        merged_data["tools"] = [
                            t if t["name"] != tool["name"] else tool
                            for t in merged_data["tools"]
                        ]
            elif key == "allowed_tool_names":
                merged_data["allowed_tool_names"] = list(
                    set(merged_data["allowed_tool_names"]) | set(value)
                )
            elif key == "blocked_tool_names":
                merged_data["blocked_tool_names"] = list(
                    set(merged_data["blocked_tool_names"]) | set(value)
                )
            else:
                merged_data[key] = value

    return AgentConfig.from_dict(merged_data)


def validate_config_compatibility(
    configs: List[AgentConfig],
) -> List[str]:
    """Validate that a list of agent configurations are compatible.

    Checks for conflicts in shared resources, tool name collisions, and
    model compatibility issues.

    Parameters
    ----------
    configs : List[AgentConfig]
        List of agent configurations to validate.

    Returns
    -------
    List[str]
        List of warning messages. An empty list means all configurations
        are fully compatible.
    """
    warnings: List[str] = []
    if not configs:
        return warnings

    all_tool_names: Dict[str, List[str]] = {}
    for config in configs:
        for tool in config.effective_tools:
            if tool.name not in all_tool_names:
                all_tool_names[tool.name] = []
            all_tool_names[tool.name].append(config.name)

    for tool_name, agent_names in all_tool_names.items():
        if len(agent_names) > 1:
            warnings.append(
                f"Tool '{tool_name}' is used by multiple agents: "
                f"{', '.join(agent_names)}"
            )

    models = {c.model for c in configs}
    if len(models) > 3:
        warnings.append(
            f"Using {len(models)} different models across agents. "
            f"This may cause inconsistent behavior."
        )

    step_counts = [c.max_steps for c in configs]
    if max(step_counts) > 10 * min(step_counts):
        warnings.append(
            "Significant variance in max_steps across agents may cause "
            "timeout mismatches in multi-agent scenarios."
        )

    return warnings
