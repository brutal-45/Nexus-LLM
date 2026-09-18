"""MemoryManager — factory and registry for memory instances."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.memory.conversation import ConversationMemory
from nexus_llm.memory.summary import SummaryMemory
from nexus_llm.memory.working import WorkingMemory

logger = logging.getLogger(__name__)

# Type alias for any memory backend
Memory = ConversationMemory | SummaryMemory | WorkingMemory  # type: ignore[name-defined]


class MemoryNotFoundError(KeyError):
    """Raised when a requested memory instance does not exist."""


class MemoryManager:
    """Central factory and registry for memory instances.

    Example
    -------
    >>> mgr = MemoryManager()
    >>> conv = mgr.create_memory("conversation", {"max_messages": 50})
    >>> conv.add_message("user", "Hello!")
    >>> mgr.list_memories()
    ['conversation_0']
    """

    _TYPE_MAP = {
        "conversation": ConversationMemory,
        "summary": SummaryMemory,
        "working": WorkingMemory,
    }

    def __init__(self) -> None:
        self._memories: Dict[str, Memory] = {}
        self._counter: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def create_memory(
        self,
        type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Create a new memory instance and register it.

        Parameters
        ----------
        type:
            One of ``"conversation"``, ``"summary"``, or ``"working"``.
        config:
            Keyword arguments forwarded to the memory constructor.

        Returns
        -------
        The newly created memory instance.

        Raises
        ------
        ValueError
            If *type* is not recognized.
        """
        if type not in self._TYPE_MAP:
            raise ValueError(
                f"Unknown memory type {type!r}; expected one of {list(self._TYPE_MAP)}"
            )

        config = config or {}
        cls = self._TYPE_MAP[type]
        memory = cls(**config)

        # Generate a stable id for lookup.
        self._counter.setdefault(type, 0)
        mem_id = f"{type}_{self._counter[type]}"
        self._counter[type] += 1
        memory.id = mem_id

        self._memories[mem_id] = memory
        logger.info("Created %s memory %r", type, mem_id)
        return memory

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_memory(self, id: str) -> Memory:
        """Return the memory instance registered under *id*.

        Raises
        ------
        MemoryNotFoundError
            If *id* is not registered.
        """
        if id not in self._memories:
            raise MemoryNotFoundError(id)
        return self._memories[id]

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_memories(self) -> List[str]:
        """Return a sorted list of registered memory IDs."""
        return sorted(self._memories.keys())

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def remove_memory(self, id: str) -> None:
        """Unregister a memory instance."""
        if id not in self._memories:
            raise MemoryNotFoundError(id)
        del self._memories[id]

    def clear(self) -> None:
        """Remove all registered memories."""
        self._memories.clear()
        self._counter.clear()
