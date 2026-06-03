"""ParallelChain — runs all steps concurrently via ThreadPoolExecutor."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

from nexus_llm.chains.chain import Chain

logger = logging.getLogger(__name__)


class ParallelChain(Chain):
    """Execute all steps concurrently and collect results.

    Each step receives the **same** *input_data* and produces an independent
    result.  The order of the returned list matches the order in which steps
    were added to the chain.

    Parameters
    ----------
    name:
        Human-readable name for the chain.
    steps:
        Optional initial list of callables.
    max_workers:
        Maximum number of threads used by the ``ThreadPoolExecutor``.
        Defaults to ``None`` (delegated to the executor).
    timeout:
        Maximum seconds to wait for each future.  ``None`` means no limit.
    """

    def __init__(
        self,
        name: str,
        steps: Optional[List[Callable]] = None,
        max_workers: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, steps=steps)
        self.max_workers = max_workers
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: Any = None) -> List[Any]:
        """Run all steps concurrently and return a list of results.

        The returned list is ordered to match the step insertion order.

        Raises
        ------
        RuntimeError
            If the chain is invalid or a step fails.
        """
        if not self.validate():
            raise RuntimeError(f"Chain {self.name!r} is not valid (no steps or non-callable step)")

        results: List[Any] = [None] * len(self._steps)
        errors: List[Optional[Exception]] = [None] * len(self._steps)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all steps, keeping track of index for ordered results.
            future_to_index = {
                executor.submit(self._safe_call, step, input_data): idx
                for idx, step in enumerate(self._steps)
            }

            for future in as_completed(future_to_index, timeout=self.timeout):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result(timeout=self.timeout)
                    logger.debug(
                        "Chain %r parallel step %d completed",
                        self.name, idx,
                    )
                except Exception as exc:
                    errors[idx] = exc
                    logger.error(
                        "Chain %r parallel step %d failed: %s",
                        self.name, idx, exc,
                    )

        # If any step failed, raise the first error encountered.
        for idx, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(
                    f"Parallel step {idx} in chain {self.name!r} failed: {err}"
                ) from err

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_call(step: Callable, input_data: Any) -> Any:
        """Invoke a single step — separated so the executor can trace it."""
        return step(input_data)
