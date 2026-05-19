"""Nexus-LLM Pipeline Builder.

Provides a fluent API for constructing data processing pipelines
from individual steps, with support for conditional branching,
error handling, and parallel execution.
"""

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.pipeline.preprocess import PreprocessPipeline
from nexus_llm.pipeline.postprocess import PostprocessPipeline
from nexus_llm.pipeline.validation import PipelineValidator

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of pipeline steps."""

    PREPROCESS = "preprocess"
    PROCESS = "process"
    POSTPROCESS = "postprocess"
    VALIDATE = "validate"
    TRANSFORM = "transform"


@dataclass
class PipelineStep:
    """A single step in a pipeline.

    Attributes:
        name: Step name.
        step_type: Type of step.
        fn: The function to execute.
        condition: Optional condition for conditional execution.
        on_error: Error handling strategy ('skip', 'raise', 'default').
        default_output: Default output if step is skipped or fails.
    """

    name: str
    step_type: StepType = StepType.PROCESS
    fn: Optional[Callable] = None
    condition: Optional[Callable] = None
    on_error: str = "raise"
    default_output: Any = None


class PipelineBuilder:
    """Fluent builder for constructing processing pipelines.

    Example::

        pipeline = (
            PipelineBuilder("my_pipeline")
            .preprocess(strip_whitespace)
            .process(analyze_text)
            .postprocess(format_output)
            .validate(validate_result)
            .build()
        )
        result = pipeline.run(input_data)
    """

    def __init__(self, name: str = "") -> None:
        self._name = name or f"pipeline-{uuid.uuid4().hex[:8]}"
        self._steps: List[PipelineStep] = []
        self._config: Dict[str, Any] = {}
        logger.debug("PipelineBuilder created: %s", self._name)

    @property
    def name(self) -> str:
        """Pipeline name."""
        return self._name

    @property
    def step_count(self) -> int:
        """Number of steps in the pipeline."""
        return len(self._steps)

    def preprocess(self, fn: Callable, name: str = "") -> "PipelineBuilder":
        """Add a preprocessing step.

        Args:
            fn: Preprocessing function.
            name: Optional step name.

        Returns:
            Self for method chaining.
        """
        self._steps.append(PipelineStep(
            name=name or f"preprocess_{len(self._steps)}",
            step_type=StepType.PREPROCESS,
            fn=fn,
        ))
        return self

    def process(self, fn: Callable, name: str = "") -> "PipelineBuilder":
        """Add a processing step.

        Args:
            fn: Processing function.
            name: Optional step name.

        Returns:
            Self for method chaining.
        """
        self._steps.append(PipelineStep(
            name=name or f"process_{len(self._steps)}",
            step_type=StepType.PROCESS,
            fn=fn,
        ))
        return self

    def postprocess(self, fn: Callable, name: str = "") -> "PipelineBuilder":
        """Add a postprocessing step.

        Args:
            fn: Postprocessing function.
            name: Optional step name.

        Returns:
            Self for method chaining.
        """
        self._steps.append(PipelineStep(
            name=name or f"postprocess_{len(self._steps)}",
            step_type=StepType.POSTPROCESS,
            fn=fn,
        ))
        return self

    def validate(self, fn: Callable, name: str = "") -> "PipelineBuilder":
        """Add a validation step.

        Args:
            fn: Validation function.
            name: Optional step name.

        Returns:
            Self for method chaining.
        """
        self._steps.append(PipelineStep(
            name=name or f"validate_{len(self._steps)}",
            step_type=StepType.VALIDATE,
            fn=fn,
        ))
        return self

    def transform(self, fn: Callable, name: str = "") -> "PipelineBuilder":
        """Add a transformation step.

        Args:
            fn: Transform function.
            name: Optional step name.

        Returns:
            Self for method chaining.
        """
        self._steps.append(PipelineStep(
            name=name or f"transform_{len(self._steps)}",
            step_type=StepType.TRANSFORM,
            fn=fn,
        ))
        return self

    def with_config(self, **kwargs: Any) -> "PipelineBuilder":
        """Add configuration parameters.

        Returns:
            Self for method chaining.
        """
        self._config.update(kwargs)
        return self

    def on_error(self, strategy: str = "skip") -> "PipelineBuilder":
        """Set error handling for the last added step.

        Args:
            strategy: Error handling strategy ('skip', 'raise', 'default').

        Returns:
            Self for method chaining.
        """
        if self._steps:
            self._steps[-1].on_error = strategy
        return self

    def with_condition(self, condition: Callable) -> "PipelineBuilder":
        """Add a condition for the last added step.

        Args:
            condition: Condition function that receives the current data.

        Returns:
            Self for method chaining.
        """
        if self._steps:
            self._steps[-1].condition = condition
        return self

    def build(self) -> "BuiltPipeline":
        """Build and return the pipeline.

        Returns:
            A BuiltPipeline ready for execution.
        """
        logger.info("Building pipeline '%s' with %d steps", self._name, len(self._steps))
        return BuiltPipeline(
            name=self._name,
            steps=list(self._steps),
            config=dict(self._config),
        )


class BuiltPipeline:
    """A constructed pipeline ready for execution.

    Attributes:
        name: Pipeline name.
        config: Pipeline configuration.
    """

    def __init__(
        self,
        name: str,
        steps: List[PipelineStep],
        config: Dict[str, Any],
    ) -> None:
        self._name = name
        self._steps = steps
        self._config = config

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> List[PipelineStep]:
        return list(self._steps)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    def run(self, data: Any) -> Any:
        """Execute the pipeline on the given data.

        Args:
            data: Input data.

        Returns:
            Processed output data.
        """
        current = data
        for step in self._steps:
            # Check condition
            if step.condition is not None and not step.condition(current):
                logger.debug("Step '%s' skipped (condition not met)", step.name)
                continue

            if step.fn is None:
                continue

            try:
                result = step.fn(current)
                current = result
                logger.debug("Step '%s' completed", step.name)
            except Exception as exc:
                if step.on_error == "raise":
                    raise
                elif step.on_error == "skip":
                    logger.warning("Step '%s' failed, skipping: %s", step.name, exc)
                    continue
                elif step.on_error == "default":
                    logger.warning("Step '%s' failed, using default: %s", step.name, exc)
                    current = step.default_output
        return current

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the pipeline to a dictionary."""
        return {
            "name": self._name,
            "steps": [
                {"name": s.name, "type": s.step_type.value, "on_error": s.on_error}
                for s in self._steps
            ],
            "config": self._config,
        }
