"""Nexus-LLM Workflow Templates.

Provides pre-built workflow templates for common use cases,
including sequential processing, parallel fan-out/fan-in,
and conditional branching patterns.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.workflow.engine import WorkflowEngine, WorkflowConfig
from nexus_llm.workflow.nodes import WorkflowNode, NodeType

logger = logging.getLogger(__name__)


class WorkflowTemplate:
    """Base class for workflow templates.

    Templates provide pre-configured workflow patterns that can be
    customized with user-provided functions.
    """

    def __init__(self, name: str = "", description: str = "") -> None:
        self._name = name
        self._description = description

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def build(self, **kwargs: Any) -> WorkflowEngine:
        """Build a workflow engine from this template.

        Args:
            **kwargs: Template-specific parameters.

        Returns:
            A configured WorkflowEngine.
        """
        raise NotImplementedError("Subclasses must implement build()")


class SequentialTemplate(WorkflowTemplate):
    """Template for a simple sequential workflow.

    Processes data through a series of steps in order:
    START -> step1 -> step2 -> ... -> stepN -> END
    """

    def __init__(self) -> None:
        super().__init__(name="sequential", description="Sequential processing pipeline")

    def build(self, steps: Optional[List[Callable]] = None, **kwargs: Any) -> WorkflowEngine:
        """Build a sequential workflow.

        Args:
            steps: List of processing functions.

        Returns:
            A WorkflowEngine with sequential node layout.
        """
        steps = steps or []
        engine = WorkflowEngine(WorkflowConfig(name=self._name))

        # Add start node
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))

        # Add processing nodes
        prev_id = "start"
        for i, step_fn in enumerate(steps):
            step_id = f"step_{i}"
            engine.add_node(WorkflowNode(id=step_id, type=NodeType.PROCESS, fn=step_fn))
            engine.add_edge(prev_id, step_id)
            prev_id = step_id

        # Add end node
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge(prev_id, "end")

        return engine


class FanOutFanInTemplate(WorkflowTemplate):
    """Template for parallel fan-out/fan-in processing.

    Splits work across parallel branches then merges results:
    START -> [branch1, branch2, ...] -> MERGE -> END
    """

    def __init__(self) -> None:
        super().__init__(name="fan_out_fan_in", description="Parallel fan-out/fan-in pattern")

    def build(self, branches: Optional[List[Callable]] = None, merge_fn: Optional[Callable] = None, **kwargs: Any) -> WorkflowEngine:
        """Build a fan-out/fan-in workflow.

        Args:
            branches: List of branch processing functions.
            merge_fn: Function to merge branch results.

        Returns:
            A WorkflowEngine with parallel branches.
        """
        branches = branches or []
        engine = WorkflowEngine(WorkflowConfig(name=self._name))

        # Add start node
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))

        # Add branch nodes
        for i, branch_fn in enumerate(branches):
            branch_id = f"branch_{i}"
            engine.add_node(WorkflowNode(id=branch_id, type=NodeType.PROCESS, fn=branch_fn))
            engine.add_edge("start", branch_id)

        # Add merge node
        merge_id = "merge"
        engine.add_node(WorkflowNode(
            id=merge_id, type=NodeType.PROCESS,
            fn=merge_fn or (lambda *args: args),
        ))
        for i in range(len(branches)):
            engine.add_edge(f"branch_{i}", merge_id)

        # Add end node
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge(merge_id, "end")

        return engine


class ConditionalBranchTemplate(WorkflowTemplate):
    """Template for conditional branching workflows.

    Routes data based on a condition:
    START -> DECISION -> [path_a | path_b] -> END
    """

    def __init__(self) -> None:
        super().__init__(name="conditional_branch", description="Conditional branching pattern")

    def build(
        self,
        condition_fn: Optional[Callable] = None,
        true_fn: Optional[Callable] = None,
        false_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> WorkflowEngine:
        """Build a conditional branch workflow.

        Args:
            condition_fn: Function that returns True/False.
            true_fn: Function for the true branch.
            false_fn: Function for the false branch.

        Returns:
            A WorkflowEngine with conditional branching.
        """
        engine = WorkflowEngine(WorkflowConfig(name=self._name))

        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=condition_fn))
        engine.add_node(WorkflowNode(id="true_path", type=NodeType.PROCESS, fn=true_fn or (lambda x: x)))
        engine.add_node(WorkflowNode(id="false_path", type=NodeType.PROCESS, fn=false_fn or (lambda x: x)))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))

        engine.add_edge("start", "true_path")
        engine.add_edge("start", "false_path")
        engine.add_edge("true_path", "end")
        engine.add_edge("false_path", "end")

        return engine


class RetryTemplate(WorkflowTemplate):
    """Template for a retry-able processing step.

    Wraps a processing function with automatic retry logic.
    """

    def __init__(self) -> None:
        super().__init__(name="retry", description="Retry-able processing pattern")

    def build(
        self,
        process_fn: Optional[Callable] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> WorkflowEngine:
        """Build a retry workflow.

        Args:
            process_fn: The function to execute with retries.
            max_retries: Maximum number of retries.

        Returns:
            A WorkflowEngine with retry logic.
        """
        config = WorkflowConfig(name=self._name, max_retries=max_retries)
        engine = WorkflowEngine(config=config)

        def retry_wrapper(data: Any) -> Any:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    if process_fn:
                        return process_fn(data)
                    return data
                except Exception as exc:
                    last_error = exc
                    logger.warning("Attempt %d failed: %s", attempt + 1, exc)
            raise last_error  # type: ignore

        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: None))
        engine.add_node(WorkflowNode(id="process", type=NodeType.PROCESS, fn=retry_wrapper))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "process")
        engine.add_edge("process", "end")

        return engine


# Registry of built-in templates
TEMPLATES: Dict[str, WorkflowTemplate] = {
    "sequential": SequentialTemplate(),
    "fan_out_fan_in": FanOutFanInTemplate(),
    "conditional_branch": ConditionalBranchTemplate(),
    "retry": RetryTemplate(),
}


def get_template(name: str) -> Optional[WorkflowTemplate]:
    """Get a workflow template by name.

    Args:
        name: Template name.

    Returns:
        The template instance, or None if not found.
    """
    return TEMPLATES.get(name)


def list_templates() -> List[str]:
    """List available template names.

    Returns:
        List of template name strings.
    """
    return list(TEMPLATES.keys())
