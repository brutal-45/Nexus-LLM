"""Base Chain class for multi-step workflow composition."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)


class Chain(ABC):
    """Abstract base class for all chain types.

    A Chain represents a composable multi-step workflow where each step is a
    callable that receives input data and produces output.  Subclasses define
    *how* steps are executed (sequentially, in parallel, conditionally, etc.).
    """

    def __init__(self, name: str, steps: Optional[List[Callable]] = None) -> None:
        self.name: str = name
        self._steps: List[Callable] = list(steps) if steps else []
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> List[Callable]:
        """Return a shallow copy of the step list."""
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, step: Callable) -> "Chain":
        """Append a step to the chain and return *self* for fluent usage.

        Parameters
        ----------
        step:
            A callable that accepts one positional argument (the input data)
            and returns a result.

        Raises
        ------
        TypeError
            If *step* is not callable.
        """
        if not callable(step):
            raise TypeError(f"Step must be callable, got {type(step)!r}")
        self._steps.append(step)
        logger.debug("Added step %r to chain %r", getattr(step, "__name__", step), self.name)
        return self

    def remove_step(self, index: int) -> None:
        """Remove the step at *index*.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0 or index >= len(self._steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self._steps) - 1})")
        removed = self._steps.pop(index)
        logger.debug("Removed step %r from chain %r", getattr(removed, "__name__", removed), self.name)

    def insert_step(self, index: int, step: Callable) -> None:
        """Insert *step* at *index*."""
        if not callable(step):
            raise TypeError(f"Step must be callable, got {type(step)!r}")
        self._steps.insert(index, step)
        logger.debug("Inserted step %r at index %d in chain %r", getattr(step, "__name__", step), index, self.name)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Return ``True`` if the chain is in a valid state.

        A chain is considered valid when:
        * It has at least one step.
        * Every step is callable.
        """
        if not self._steps:
            logger.warning("Chain %r has no steps", self.name)
            return False
        for i, step in enumerate(self._steps):
            if not callable(step):
                logger.warning("Step %d in chain %r is not callable", i, self.name)
                return False
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the chain with *input_data* and return the result."""
        ...

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, steps={self.step_count})"

    def __len__(self) -> int:
        return self.step_count
