"""Base Migration class for Nexus-LLM.

Defines the interface that every migration must implement: ``up()``
to apply and ``down()`` to roll back.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Migration(ABC):
    """Abstract base class for versioned migrations.

    Each migration has a version string (timestamp-based), a
    human-readable description, and ``up``/``down`` methods.

    Subclasses must override ``up()`` and ``down()``.

    Example::

        class V1_CreateUsers(Migration):
            version = "20250101_000001"
            description = "Create users table"

            def up(self, context):
                context["users"] = []

            def down(self, context):
                context.pop("users", None)
    """

    # Subclasses must set these as class attributes
    version: str = ""
    description: str = ""

    def __init__(self) -> None:
        if not self.version:
            # Auto-generate a timestamp-based version if not provided
            self.version = time.strftime("%Y%m%d_%H%M%S")
        self._id: str = f"{self.version}_{uuid.uuid4().hex[:6]}"

    @property
    def migration_id(self) -> str:
        """Unique identifier for this migration (version + short hash)."""
        return self._id

    @abstractmethod
    def up(self, context: Dict[str, Any]) -> None:
        """Apply the migration.

        Args:
            context: A mutable dict representing the current state
                     (config, schema, etc.) that the migration can modify.
        """

    @abstractmethod
    def down(self, context: Dict[str, Any]) -> None:
        """Roll back the migration.

        Args:
            context: The same mutable dict that was passed to ``up()``.
                     The implementation should undo the changes made in ``up()``.
        """

    def __repr__(self) -> str:
        return (
            f"<Migration version={self.version!r} "
            f"description={self.description!r}>"
        )

    def __lt__(self, other: "Migration") -> bool:
        """Sort migrations by version string."""
        if not isinstance(other, Migration):
            return NotImplemented
        return self.version < other.version

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Migration):
            return NotImplemented
        return self.version == other.version

    def __hash__(self) -> int:
        return hash(self.version)
