"""Agent configuration for Nexus-LLM.

Defines the ``AgentConfig`` dataclass with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class AgentConfig:
    """Configuration for an :class:`~nexus_llm.agents.agent.Agent`.

    Attributes:
        name: Human-readable agent name.
        max_iterations: Maximum ReAct loop iterations.
        tools: List of tool names the agent can use.
        temperature: LLM sampling temperature.
        max_tokens: Maximum tokens in generated responses.
        verbose: Whether to log detailed agent reasoning.
        system_prompt: Optional system prompt override.
        retry_attempts: Number of retries on tool execution failure.
        retry_delay_seconds: Delay between retries in seconds.
    """

    name: str = "default-agent"
    max_iterations: int = 10
    tools: List[str] = field(default_factory=lambda: [
        "calculator", "web_search", "file_read", "file_write",
    ])
    temperature: float = 0.7
    max_tokens: int = 1024
    verbose: bool = False
    system_prompt: Optional[str] = None
    retry_attempts: int = 2
    retry_delay_seconds: float = 1.0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """Create an ``AgentConfig`` from a dictionary.

        Unknown keys are silently ignored so that config files can be
        forward-compatible.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
