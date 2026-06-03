"""Nexus-LLM Response Composer.

Provides the Composer class for building structured responses from LLM
outputs, including formatting, metadata injection, streaming assembly,
and multi-part composition.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.enums import ChatRole
from nexus_llm.types import Message

logger = logging.getLogger(__name__)


@dataclass
class CompositionResult:
    """Result from composing a response.

    Attributes:
        id: Unique response identifier.
        content: Final composed text content.
        messages: Ordered list of Message objects in the composition.
        metadata: Response-level metadata.
        total_tokens: Estimated total token count.
        duration_ms: Composition time in milliseconds.
        parts: Number of parts composed.
    """

    id: str = ""
    content: str = ""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    duration_ms: float = 0.0
    parts: int = 0


class Composer:
    """Response composer for building structured LLM responses.

    The Composer accumulates response parts (text chunks, metadata,
    tool calls, etc.) and assembles them into a unified CompositionResult.

    Attributes:
        part_count: Number of parts currently accumulated.
    """

    def __init__(self, model_name: str = "") -> None:
        self._model_name = model_name
        self._parts: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {"model": model_name}
        self._system_prompt: Optional[str] = None
        logger.debug("Composer initialized for model: %s", model_name or "unknown")

    @property
    def part_count(self) -> int:
        """Number of parts currently accumulated."""
        return len(self._parts)

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the composition.

        Args:
            prompt: System prompt text.
        """
        self._system_prompt = prompt

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a text part to the composition.

        Args:
            text: Text content.
            metadata: Optional metadata for this part.
        """
        self._parts.append({
            "type": "text",
            "content": text,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def add_code_block(self, code: str, language: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a code block part to the composition.

        Args:
            code: Source code content.
            language: Programming language identifier.
            metadata: Optional metadata for this part.
        """
        self._parts.append({
            "type": "code",
            "content": code,
            "language": language,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

    def add_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any = None) -> None:
        """Add a tool call part to the composition.

        Args:
            tool_name: Name of the tool called.
            arguments: Arguments passed to the tool.
            result: Optional result from the tool execution.
        """
        self._parts.append({
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "timestamp": time.time(),
        })

    def add_thinking(self, text: str) -> None:
        """Add a thinking/reasoning part to the composition.

        Args:
            text: Reasoning or chain-of-thought text.
        """
        self._parts.append({
            "type": "thinking",
            "content": text,
            "timestamp": time.time(),
        })

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the composition.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    def compose(self) -> CompositionResult:
        """Assemble all accumulated parts into a CompositionResult.

        Returns:
            A CompositionResult with the final assembled response.
        """
        start = time.perf_counter()
        response_id = str(uuid.uuid4())

        # Build the full content by concatenating text and code parts
        content_parts: List[str] = []
        messages: List[Message] = []

        if self._system_prompt:
            messages.append(Message(role=ChatRole.SYSTEM, content=self._system_prompt))

        for part in self._parts:
            if part["type"] == "text":
                content_parts.append(part["content"])
                messages.append(Message(role=ChatRole.ASSISTANT, content=part["content"]))
            elif part["type"] == "code":
                formatted = f"```{part.get('language', '')}\n{part['content']}\n```"
                content_parts.append(formatted)
                messages.append(Message(role=ChatRole.ASSISTANT, content=formatted))
            elif part["type"] == "tool_call":
                content_parts.append(f"[Tool: {part['tool_name']}]")
            elif part["type"] == "thinking":
                content_parts.append(f"<thinking>\n{part['content']}\n</thinking>")

        full_content = "\n".join(content_parts)
        total_tokens = sum(len(p.get("content", "").split()) for p in self._parts)
        duration_ms = (time.perf_counter() - start) * 1000

        result = CompositionResult(
            id=response_id,
            content=full_content,
            messages=messages,
            metadata=dict(self._metadata),
            total_tokens=total_tokens,
            duration_ms=duration_ms,
            parts=len(self._parts),
        )

        logger.debug(
            "Composed response %s: %d parts, %d tokens, %.1fms",
            response_id,
            len(self._parts),
            total_tokens,
            duration_ms,
        )
        return result

    def compose_streaming(self) -> CompositionResult:
        """Compose parts in streaming mode (yields empty result, use for API compat).

        Returns:
            A CompositionResult marked for streaming.
        """
        result = self.compose()
        result.metadata["streaming"] = True
        return result

    def reset(self) -> None:
        """Clear all accumulated parts and metadata."""
        self._parts.clear()
        self._metadata = {"model": self._model_name}
        self._system_prompt = None
        logger.debug("Composer reset")

    def health_check(self) -> Dict[str, Any]:
        """Return health status of the composer."""
        return {
            "status": "healthy",
            "model": self._model_name,
            "part_count": len(self._parts),
        }
