"""SequentialChain — runs steps one after another, piping outputs forward."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional

from nexus_llm.chains.chain import Chain

logger = logging.getLogger(__name__)


class StepError(Exception):
    """Raised when a step in a SequentialChain fails after all retries."""


class SequentialChain(Chain):
    """Execute steps sequentially where the output of step *N* feeds into step *N+1*.

    Parameters
    ----------
    name:
        Human-readable name for the chain.
    steps:
        Optional initial list of callables.
    max_retries:
        Maximum number of retry attempts per step on failure (default 0).
    retry_delay:
        Seconds to wait between retries (default 0).
    """

    def __init__(
        self,
        name: str,
        steps: Optional[List[Callable]] = None,
        max_retries: int = 0,
        retry_delay: float = 0.0,
    ) -> None:
        super().__init__(name=name, steps=steps)
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: Any = None) -> Any:
        """Run all steps in order, piping the output of each step to the next.

        Returns
        -------
        The output of the final step.

        Raises
        ------
        StepError
            If a step fails and retries are exhausted.
        """
        if not self.validate():
            raise StepError(f"Chain {self.name!r} is not valid (no steps or non-callable step)")

        result = input_data
        for idx, step in enumerate(self._steps):
            step_name = getattr(step, "__name__", f"step_{idx}")
            result = self._run_step_with_retries(step, step_name, idx, result)
            logger.debug(
                "Chain %r step %d (%s) completed → %s",
                self.name, idx, step_name,
                type(result).__name__,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_step_with_retries(
        self,
        step: Callable,
        step_name: str,
        step_index: int,
        input_data: Any,
    ) -> Any:
        """Execute a single step with optional retry logic."""
        last_error: Optional[Exception] = None
        attempts = 1 + self.max_retries

        for attempt in range(attempts):
            try:
                return step(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    logger.warning(
                        "Chain %r step %d (%s) failed on attempt %d/%d: %s — retrying",
                        self.name, step_index, step_name,
                        attempt + 1, attempts, exc,
                    )
                    if self.retry_delay > 0:
                        time.sleep(self.retry_delay)
                else:
                    logger.error(
                        "Chain %r step %d (%s) failed after %d attempt(s): %s",
                        self.name, step_index, step_name, attempts, exc,
                    )

        raise StepError(
            f"Step {step_index} ({step_name}) in chain {self.name!r} "
            f"failed after {attempts} attempt(s): {last_error}"
        ) from last_error
