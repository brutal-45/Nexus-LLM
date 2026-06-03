"""PipelineStep — a single step within a pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ErrorPolicy(str, Enum):
    """How a step should behave when it encounters an error."""

    SKIP = "skip"
    RETRY = "retry"
    FAIL = "fail"


@dataclass
class PipelineStep:
    """Represents one step in a :class:`Pipeline`.

    Attributes
    ----------
    name:
        Human-readable step name.
    func:
        The callable to execute.
    config:
        Arbitrary configuration dictionary forwarded alongside *func*.
    on_error:
        Error handling policy (skip, retry, or fail).
    retry_count:
        Number of retry attempts (only relevant when ``on_error == "retry"``).
    timeout:
        Maximum seconds the step is allowed to run.  ``None`` means no limit
        (note: actual timeout enforcement requires cooperative checking or
        threading — this serves as a hint / metadata).
    """

    name: str
    func: Callable
    config: Dict[str, Any] = field(default_factory=dict)
    on_error: ErrorPolicy = ErrorPolicy.FAIL
    retry_count: int = 0
    timeout: Optional[float] = None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, input_data: Any) -> Any:
        """Run the step with *input_data*, applying error policy.

        Returns
        -------
        The output of ``func(input_data)``.

        Raises
        ------
        RuntimeError
            If the step fails and the error policy is ``FAIL``.
        """
        if self.on_error == ErrorPolicy.RETRY:
            return self._execute_with_retries(input_data)

        try:
            return self._call(input_data)
        except Exception as exc:
            if self.on_error == ErrorPolicy.SKIP:
                logger.warning(
                    "PipelineStep %r failed (skipping): %s", self.name, exc,
                )
                return input_data  # Pass through the original data
            # FAIL
            raise RuntimeError(
                f"PipelineStep {self.name!r} failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_with_retries(self, input_data: Any) -> Any:
        """Execute with retry logic."""
        attempts = 1 + max(0, self.retry_count)
        last_error: Optional[Exception] = None

        for attempt in range(attempts):
            try:
                return self._call(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_count:
                    logger.warning(
                        "PipelineStep %r failed on attempt %d/%d: %s — retrying",
                        self.name, attempt + 1, attempts, exc,
                    )

        # All retries exhausted
        if self.on_error == ErrorPolicy.SKIP:
            logger.warning(
                "PipelineStep %r failed after %d attempts (skipping): %s",
                self.name, attempts, last_error,
            )
            return input_data

        raise RuntimeError(
            f"PipelineStep {self.name!r} failed after {attempts} attempt(s): {last_error}"
        ) from last_error

    def _call(self, input_data: Any) -> Any:
        """Invoke the wrapped function."""
        start = time.monotonic()
        result = self.func(input_data, **self.config) if self.config else self.func(input_data)
        elapsed = time.monotonic() - start
        logger.debug(
            "PipelineStep %r completed in %.3fs", self.name, elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PipelineStep(name={self.name!r}, on_error={self.on_error.value!r}, "
            f"retry_count={self.retry_count})"
        )
