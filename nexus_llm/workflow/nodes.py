"""Nexus-LLM Workflow Nodes.

Defines the WorkflowNode class and supporting types for representing
individual steps in a workflow graph.
"""

import enum
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeType(enum.Enum):
    """Types of workflow nodes."""

    START = "start"
    END = "end"
    TASK = "task"
    DECISION = "decision"
    PARALLEL = "parallel"
    SUBPROCESS = "subprocess"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    TRANSFORM = "transform"
    WAIT = "wait"


class NodeStatus(enum.Enum):
    """Possible states of a workflow node."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class NodeResult:
    """Result from executing a workflow node.

    Attributes:
        node_id: The node that produced this result.
        success: Whether execution was successful.
        output: The output data.
        error: Error message if failed.
        duration_ms: Execution duration in milliseconds.
        metadata: Additional metadata.
    """

    node_id: str = ""
    success: bool = True
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode:
    """A single node in a workflow graph.

    Each node represents a discrete step in the workflow and
    contains an execution function, configuration, and state.

    Attributes:
        id: Unique node identifier.
        name: Human-readable node name.
        type: Node type.
        fn: Execution function (receives input data, returns output).
        config: Node-specific configuration.
        status: Current execution status.
        result: Latest execution result.
        retries: Number of times this node has been retried.
        max_retries: Maximum retry attempts.
        timeout_seconds: Node-level timeout.
        on_success: Callback on successful execution.
        on_failure: Callback on failed execution.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: NodeType = NodeType.TASK
    fn: Optional[Callable[[Any], Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[NodeResult] = None
    retries: int = 0
    max_retries: int = 0
    timeout_seconds: float = 0.0
    on_success: Optional[Callable[[Any], None]] = None
    on_failure: Optional[Callable[[str], None]] = None

    def execute(self, input_data: Any = None) -> NodeResult:
        """Execute this node with the given input data.

        Args:
            input_data: Input from predecessor nodes.

        Returns:
            A NodeResult with execution output.
        """
        self.status = NodeStatus.RUNNING
        start = time.perf_counter()

        try:
            if self.fn is None:
                output = input_data
            else:
                # Handle both fns that take arguments and fns that don't
                try:
                    sig = inspect.signature(self.fn)
                    param_count = sum(
                        1 for p in sig.parameters.values()
                        if p.default is inspect.Parameter.empty
                        and p.kind not in (
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                        )
                    )
                    if param_count == 0:
                        output = self.fn()
                    else:
                        output = self.fn(input_data)
                except (ValueError, TypeError):
                    output = self.fn(input_data)

            duration_ms = (time.perf_counter() - start) * 1000
            self.result = NodeResult(
                node_id=self.id,
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
            self.status = NodeStatus.COMPLETED

            if self.on_success:
                try:
                    self.on_success(output)
                except Exception as exc:
                    logger.warning("on_success callback failed for node %s: %s", self.id, exc)

            logger.debug("Node %s completed in %.1fms", self.id, duration_ms)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            self.result = NodeResult(
                node_id=self.id,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )
            self.status = NodeStatus.FAILED
            self.retries += 1

            if self.on_failure:
                try:
                    self.on_failure(str(exc))
                except Exception as cb_exc:
                    logger.warning("on_failure callback failed for node %s: %s", self.id, cb_exc)

            logger.error("Node %s failed: %s", self.id, exc)

        return self.result

    def reset(self) -> None:
        """Reset the node to pending state."""
        self.status = NodeStatus.PENDING
        self.result = None
        self.retries = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "config": self.config,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
        }
