"""ConditionalChain — routes input to the first sub-chain whose predicate matches.

Useful for LLM workflows that need branching, e.g. sending code-related prompts
to a code-tuned model and everything else to a general chat model, or applying
a summarisation chain only when the input exceeds a length threshold.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nexus_llm.chains.chain import Chain

logger = logging.getLogger(__name__)


@dataclass
class Condition:
    """A predicate paired with the chain to run when it holds.

    Attributes:
        predicate: Callable receiving the chain input, returning truthy to match.
        chain: The :class:`Chain` executed when *predicate* matches.
        name: Optional label used in log messages and :meth:`describe`.
    """

    predicate: Callable[[Any], bool]
    chain: Chain
    name: str = ""

    def matches(self, input_data: Any) -> bool:
        """Evaluate the predicate, treating an error as "no match".

        A faulty predicate must not abort the whole workflow: the condition is
        skipped and the failure is logged, which lets a later (or default)
        condition still produce an answer.
        """
        try:
            return bool(self.predicate(input_data))
        except Exception as exc:  # noqa: BLE001 - predicates are user-supplied
            logger.warning(
                "Condition %r raised %s: %s; skipping",
                self.name or self.chain.name,
                type(exc).__name__,
                exc,
            )
            return False

    def describe(self) -> dict[str, Any]:
        """Return a serialisable description of this condition."""
        return {
            "name": self.name or self.chain.name,
            "chain": repr(self.chain),
            "predicate": getattr(self.predicate, "__name__", "lambda"),
        }


class NoMatchError(ValueError):
    """Raised when no condition matches and no default chain is configured."""


class ConditionalChain(Chain):
    """Run the first registered sub-chain whose predicate accepts the input.

    Conditions are evaluated in registration order and the *first* match wins,
    so more specific predicates should be registered before general ones.  When
    nothing matches, *default_chain* runs if provided, otherwise a
    :class:`NoMatchError` (a :class:`ValueError`) is raised.

    Parameters
    ----------
    name:
        Human-readable name for the chain.
    steps:
        Optional callables run (in order) on the input *before* routing.
    default_chain:
        Fallback chain used when no condition matches.
    conditions:
        Optional initial list of :class:`Condition` objects.

    Example::

        small = SequentialChain(name="short").add_step(summarise)
        large = SequentialChain(name="long").add_step(chunk_then_summarise)
        router = ConditionalChain(name="router", default_chain=small)
        router.add_condition(lambda text: len(text) > 4000, large)
        result = router.run(prompt)
    """

    def __init__(
        self,
        name: str,
        steps: list[Callable] | None = None,
        default_chain: Chain | None = None,
        conditions: list[Condition] | None = None,
    ) -> None:
        super().__init__(name=name, steps=steps)
        self._conditions: list[Condition] = list(conditions) if conditions else []
        self.default_chain: Chain | None = default_chain
        self._matched: Condition | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def conditions(self) -> list[Condition]:
        """A shallow copy of the registered conditions."""
        return list(self._conditions)

    @property
    def condition_count(self) -> int:
        """Number of registered conditions."""
        return len(self._conditions)

    @property
    def last_matched(self) -> Condition | None:
        """The condition selected by the most recent :meth:`run` call."""
        return self._matched

    # ------------------------------------------------------------------
    # Condition management
    # ------------------------------------------------------------------

    def add_condition(
        self,
        predicate: Callable[[Any], bool],
        chain: Chain,
        name: str = "",
    ) -> ConditionalChain:
        """Register *chain* to run when *predicate(input)* is truthy.

        Parameters
        ----------
        predicate:
            Callable taking the chain input and returning a truthy value to match.
        chain:
            The :class:`Chain` to execute on a match.
        name:
            Optional label for logs.

        Returns
        -------
        This chain, so calls can be chained fluently.

        Raises
        ------
        TypeError
            If *predicate* is not callable or *chain* is not a :class:`Chain`.
        """
        if not callable(predicate):
            raise TypeError(f"Condition predicate must be callable, got {type(predicate)!r}")
        if not isinstance(chain, Chain):
            raise TypeError(f"Condition chain must be a Chain instance, got {type(chain)!r}")
        self._conditions.append(Condition(predicate=predicate, chain=chain, name=name))
        logger.debug("Added condition %r to chain %r", name or chain.name, self.name)
        return self

    def set_default(self, chain: Chain | None) -> None:
        """Set (or clear with ``None``) the fallback chain.

        Raises
        ------
        TypeError
            If *chain* is neither ``None`` nor a :class:`Chain`.
        """
        if chain is not None and not isinstance(chain, Chain):
            raise TypeError(f"Default chain must be a Chain instance, got {type(chain)!r}")
        self.default_chain = chain

    def remove_condition(self, index: int) -> Condition:
        """Remove and return the condition at *index*.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if not self._conditions:
            raise IndexError("ConditionalChain has no conditions to remove")
        try:
            return self._conditions.pop(index)
        except IndexError:
            raise IndexError(
                f"Condition index {index} out of range (0-{len(self._conditions) - 1})"
            ) from None

    def clear_conditions(self) -> None:
        """Drop every registered condition (the default chain is kept)."""
        self._conditions.clear()

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def matching_condition(self, input_data: Any) -> Condition | None:
        """Return the first condition whose predicate accepts *input_data*."""
        for condition in self._conditions:
            if condition.matches(input_data):
                return condition
        return None

    def validate(self) -> bool:
        """Return ``True`` when the chain can route input.

        A conditional chain is valid when it has at least one condition, or a
        default chain to fall back on; every condition must carry a callable
        predicate and a valid sub-chain.
        """
        if not self._conditions and self.default_chain is None:
            logger.warning("Chain %r has no conditions and no default chain", self.name)
            return False
        for condition in self._conditions:
            if not callable(condition.predicate):
                logger.warning("Condition %r in chain %r has a non-callable predicate", condition.name, self.name)
                return False
            if not condition.chain.validate():
                logger.warning(
                    "Condition %r in chain %r targets an invalid chain %r",
                    condition.name or condition.chain.name,
                    self.name,
                    condition.chain.name,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: Any = None) -> Any:
        """Route *input_data* through the matching sub-chain.

        Any steps registered on this chain are applied first, so shared
        pre-processing (trimming, token counting) does not need repeating in
        every branch.

        Returns
        -------
        Whatever the selected sub-chain returns.

        Raises
        ------
        NoMatchError
            If no condition matched and no default chain is configured.
        """
        data = input_data
        for step in self._steps:
            data = step(data)

        condition = self.matching_condition(data)
        if condition is None:
            if self.default_chain is None:
                raise NoMatchError(
                    f"No condition matched for input in chain {self.name!r} "
                    f"and no default_chain is configured"
                )
            logger.debug("No condition matched in %r; using default chain", self.name)
            self._matched = None
            return self.default_chain.run(data)

        self._matched = condition
        logger.debug(
            "Chain %r routed to %r", self.name, condition.name or condition.chain.name
        )
        return condition.chain.run(data)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Return a summary of the routing table (handy for debugging)."""
        return {
            "name": self.name,
            "conditions": [condition.describe() for condition in self._conditions],
            "default_chain": repr(self.default_chain) if self.default_chain else None,
        }

    def __repr__(self) -> str:
        default = self.default_chain.name if self.default_chain else None
        return (
            f"ConditionalChain(name={self.name!r}, conditions={self.condition_count}, "
            f"default={default!r})"
        )

    def __len__(self) -> int:
        return self.condition_count
