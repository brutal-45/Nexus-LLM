"""Abstract storage backend for Nexus-LLM.

Defines the interface that all storage backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional


class StorageBackend(ABC):
    """Abstract base class for key-value storage backends.

    All backends must support the core CRUD operations: save, load,
    delete, exists, and list_keys.  Implementations may add additional
    capabilities (transactions, queries, etc.).
    """

    @abstractmethod
    def save(self, key: str, value: Any) -> None:
        """Persist *value* under *key*.

        Args:
            key: Unique identifier.
            value: Any serialisable value.

        Raises:
            StorageError: If the value cannot be persisted.
        """

    @abstractmethod
    def load(self, key: str) -> Any:
        """Retrieve the value stored under *key*.

        Args:
            key: Unique identifier.

        Returns:
            The stored value.

        Raises:
            KeyError: If *key* does not exist.
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove the entry for *key*.

        Args:
            key: Unique identifier.

        Returns:
            True if the entry existed and was deleted, False otherwise.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True if *key* exists in the store."""

    @abstractmethod
    def list_keys(self) -> List[str]:
        """Return all keys currently stored."""

    # ------------------------------------------------------------------
    # Optional convenience methods with default implementations
    # ------------------------------------------------------------------

    def save_many(self, items: dict) -> None:
        """Persist multiple key-value pairs.

        The default implementation calls :meth:`save` in a loop.
        Subclasses should override this for atomic batch writes.
        """
        for k, v in items.items():
            self.save(k, v)

    def load_many(self, keys: List[str]) -> dict:
        """Retrieve multiple values as a dict.

        Missing keys are silently omitted.
        """
        result = {}
        for k in keys:
            if self.exists(k):
                result[k] = self.load(k)
        return result

    def clear(self) -> None:
        """Delete all stored entries."""
        for key in self.list_keys():
            self.delete(key)
