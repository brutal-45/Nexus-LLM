"""ConditionalChain — dispatches to a sub-chain based on predicate evaluation."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

from nexus_llm.chains.chain import Chain

logger = logging.getLogger(__name__)


class ConditionalChain(Chain):
    """Evaluate conditions in order and execute the first matching sub-chain.

    Conditions are ``(predicate, chain)`` pairs.  When :meth:`run` is called,
    each predicate is evaluated against *input_data*.  The first predicate
    that returns ``True`` causes its associated chain to be executed.

    If no predicate matches and a *default_chain* is set, the default is used.
    Otherwise a :class:`ValueError` is raised.

    Parameters
    ----------
    name:
        Human-readable name for the chain.
    default_chain:
        Optional fallback chain when no condition matches.
    """

    def __init__(
        self,
        name: str,
        default_chain: Optional[Chain] = None,
    ) -> None:
        super().__init__(name=name)
        self._conditions: List[Tuple[Callable[[Any], bool], Chain]] = []
        self.default_chain = default_chain

    # ------------------------------------------------------------------
    # Condition management
    # ------------------------------------------------------------------

    def add_condition(self, condition_fn: Callable[[Any], bool], chain: Chain) -> "ConditionalChain":
        """Register a condition → chain pair.

        Parameters
        ----------
        condition_fn:
            A callable that accepts *input_data* and returns ``True`` when the
            associated *chain* should be executed.
        chain:
            The :class:`Chain` to run when *condition_fn* matches.

        Raises
        ------
        TypeError
            If *condition_fn* is not callable or *chain* is not a :class:`Chain`.
        """
        if not callable(condition_fn):
            raise TypeError(f"condition_fn must be callable, got {type(condition_fn)!r}")
        if not isinstance(chain, Chain):
            raise TypeError(f"chain must be a Chain instance, got {type(chain)!r}")
        self._conditions.append((condition_fn, chain))
        logger.debug(
            "Added condition to chain %r (total conditions: %d)",
            self.name, len(self._conditions),
        )
        return self

    def remove_condition(self, index: int) -> None:
        """Remove the condition at *index*."""
        if index < 0 or index >= len(self._conditions):
            raise IndexError(f"Condition index {index} out of range")
        self._conditions.pop(index)

    @property
    def conditions(self) -> List[Tuple[Callable[[Any], bool], Chain]]:
        """Return a shallow copy of the condition list."""
        return list(self._conditions)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Return ``True`` if at least one condition or a default chain exists."""
        if self._conditions:
            return True
        if self.default_chain is not None and self.default_chain.validate():
            return True
        return False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: Any = None) -> Any:
        """Evaluate conditions and run the first matching chain.

        Raises
        ------
        ValueError
            If no condition matches and no default chain is set.
        """
        for idx, (condition_fn, chain) in enumerate(self._conditions):
            try:
                matches = bool(condition_fn(input_data))
            except Exception as exc:
                logger.warning(
                    "Chain %r condition %d raised %s — skipping",
                    self.name, idx, exc,
                )
                continue

            if matches:
                logger.debug("Chain %r matched condition %d — executing sub-chain", self.name, idx)
                return chain.run(input_data)

        # No condition matched — try the default.
        if self.default_chain is not None:
            logger.debug("Chain %r — no condition matched, running default chain", self.name)
            return self.default_chain.run(input_data)

        raise ValueError(f"No condition matched for chain {self.name!r} and no default chain set")
