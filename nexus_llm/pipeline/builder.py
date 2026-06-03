"""PipelineBuilder — fluent API for constructing Pipeline instances."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.pipeline.pipeline import Pipeline
from nexus_llm.pipeline.step import ErrorPolicy, PipelineStep

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Fluent builder for :class:`Pipeline` instances.

    Provides convenience methods for common step patterns including
    conditional branching, parallel fan-out, and retry wrapping.

    Example
    -------
    >>> pipe = (
    ...     PipelineBuilder("my-pipe")
    ...     .step("double", lambda x: x * 2)
    ...     .step("add_one", lambda x: x + 1)
    ...     .build()
    ... )
    >>> pipe.run(3)
    7
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._steps: List[PipelineStep] = []

    # ------------------------------------------------------------------
    # Basic step
    # ------------------------------------------------------------------

    def step(
        self,
        name: str,
        func: Callable,
        *,
        config: Optional[Dict[str, Any]] = None,
        on_error: str = "fail",
    ) -> "PipelineBuilder":
        """Add a standard step.

        Parameters
        ----------
        name:
            Step name (must be unique within the pipeline).
        func:
            Callable to execute.
        config:
            Extra keyword arguments forwarded to *func*.
        on_error:
            ``"skip"``, ``"retry"``, or ``"fail"``.
        """
        self._steps.append(PipelineStep(
            name=name,
            func=func,
            config=config or {},
            on_error=ErrorPolicy(on_error),
        ))
        return self

    # ------------------------------------------------------------------
    # Conditional step
    # ------------------------------------------------------------------

    def conditional(
        self,
        name: str,
        condition: Callable[[Any], bool],
        if_step: Callable,
        else_step: Optional[Callable] = None,
    ) -> "PipelineBuilder":
        """Add a conditional branch step.

        At runtime *condition* is evaluated with the current data; if it
        returns ``True`` then *if_step* is executed, otherwise *else_step*.

        Parameters
        ----------
        name:
            Step name.
        condition:
            Predicate callable.
        if_step:
            Callable for the true branch.
        else_step:
            Optional callable for the false branch (passes data through
            unchanged if ``None``).
        """

        def _conditional_func(data: Any) -> Any:
            if condition(data):
                return if_step(data)
            if else_step is not None:
                return else_step(data)
            return data

        self._steps.append(PipelineStep(
            name=name,
            func=_conditional_func,
            on_error=ErrorPolicy.FAIL,
        ))
        return self

    # ------------------------------------------------------------------
    # Parallel step
    # ------------------------------------------------------------------

    def parallel(
        self,
        name: str,
        steps: List[Callable],
        *,
        max_workers: Optional[int] = None,
    ) -> "PipelineBuilder":
        """Add a step that runs multiple callables concurrently.

        Each sub-step receives the **same** input data and produces an
        independent result.  The step returns a list of results in the
        same order as *steps*.

        Parameters
        ----------
        name:
            Step name.
        steps:
            List of callables to execute in parallel.
        max_workers:
            Thread pool size (default ``None`` → let the executor decide).
        """

        def _parallel_func(data: Any) -> List[Any]:
            results: List[Any] = [None] * len(steps)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(fn, data): idx
                    for idx, fn in enumerate(steps)
                }
                for future in futures:
                    idx = futures[future]
                    results[idx] = future.result()
            return results

        self._steps.append(PipelineStep(
            name=name,
            func=_parallel_func,
            on_error=ErrorPolicy.FAIL,
        ))
        return self

    # ------------------------------------------------------------------
    # Retry step
    # ------------------------------------------------------------------

    def retry(
        self,
        name: str,
        func: Callable,
        max_retries: int = 3,
    ) -> "PipelineBuilder":
        """Add a step with automatic retry on failure.

        Parameters
        ----------
        name:
            Step name.
        func:
            Callable to execute.
        max_retries:
            Number of retry attempts after the first failure.
        """
        self._steps.append(PipelineStep(
            name=name,
            func=func,
            on_error=ErrorPolicy.RETRY,
            retry_count=max_retries,
        ))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Pipeline:
        """Construct and return a :class:`Pipeline` from the accumulated steps.

        Raises
        ------
        ValueError
            If no steps have been added.
        """
        if not self._steps:
            raise ValueError("Cannot build a pipeline with no steps")

        pipeline = Pipeline(name=self._name)
        for step in self._steps:
            pipeline.add_step(step)

        logger.info("Built pipeline %r with %d step(s)", self._name, len(self._steps))
        return pipeline
