"""Pipeline — ordered sequence of PipelineSteps with validation and control."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from nexus_llm.pipeline.step import PipelineStep

logger = logging.getLogger(__name__)


class PipelineValidationError(ValueError):
    """Raised when a pipeline fails validation."""


class Pipeline:
    """Ordered sequence of :class:`PipelineStep` instances.

    Steps are executed in order; the output of step *N* feeds into step *N+1*,
    similar to :class:`SequentialChain` but with richer per-step error
    handling, timeouts, and configuration.

    Example
    -------
    >>> p = Pipeline("my-pipeline")
    >>> p.add_step(PipelineStep(name="double", func=lambda x: x * 2))
    >>> p.add_step(PipelineStep(name="add_one", func=lambda x: x + 1))
    >>> p.run(3)
    7
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: List[PipelineStep] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps(self) -> List[PipelineStep]:
        return list(self._steps)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    # ------------------------------------------------------------------
    # Step management
    # ------------------------------------------------------------------

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Append *step* to the pipeline."""
        if not isinstance(step, PipelineStep):
            raise TypeError(f"Expected PipelineStep, got {type(step)!r}")
        self._steps.append(step)
        logger.debug("Pipeline %r: added step %r", self.name, step.name)
        return self

    def insert_step(self, index: int, step: PipelineStep) -> None:
        """Insert *step* at *index*."""
        if not isinstance(step, PipelineStep):
            raise TypeError(f"Expected PipelineStep, got {type(step)!r}")
        self._steps.insert(index, step)
        logger.debug("Pipeline %r: inserted step %r at index %d", self.name, step.name, index)

    def remove_step(self, index: int) -> PipelineStep:
        """Remove and return the step at *index*.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0 or index >= len(self._steps):
            raise IndexError(f"Step index {index} out of range (0-{len(self._steps) - 1})")
        step = self._steps.pop(index)
        logger.debug("Pipeline %r: removed step %r at index %d", self.name, step.name, index)
        return step

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Return ``True`` if the pipeline is valid.

        A pipeline is valid when:
        * It has at least one step.
        * Every step has a callable ``func``.
        * Step names are unique.
        """
        if not self._steps:
            logger.warning("Pipeline %r has no steps", self.name)
            return False

        names: set[str] = set()
        for i, step in enumerate(self._steps):
            if not callable(step.func):
                logger.warning(
                    "Pipeline %r step %d (%r) has a non-callable func",
                    self.name, i, step.name,
                )
                return False
            if step.name in names:
                logger.warning(
                    "Pipeline %r has duplicate step name %r", self.name, step.name,
                )
                return False
            names.add(step.name)

        return True

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, input_data: Any = None) -> Any:
        """Execute all steps in order, piping outputs forward.

        Raises
        ------
        PipelineValidationError
            If the pipeline fails validation.
        RuntimeError
            Propagated from failing steps with ``FAIL`` error policy.
        """
        if not self.validate():
            raise PipelineValidationError(
                f"Pipeline {self.name!r} is not valid"
            )

        result = input_data
        for i, step in enumerate(self._steps):
            logger.debug(
                "Pipeline %r: running step %d/%d %r",
                self.name, i + 1, self.step_count, step.name,
            )
            result = step.execute(result)
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        step_names = [s.name for s in self._steps]
        return f"Pipeline(name={self.name!r}, steps={step_names})"

    def __len__(self) -> int:
        return self.step_count
