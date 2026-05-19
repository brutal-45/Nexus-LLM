"""Nexus-LLM Workflow Edges.

Defines the WorkflowEdge class and supporting types for representing
connections between workflow nodes, including conditional routing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EdgeCondition:
    """A condition that must be satisfied for an edge to be traversed.

    Attributes:
        expression: A string expression or Python expression to evaluate.
        fn: A callable that receives the source node output and returns bool.
        description: Human-readable description of the condition.
        negated: Whether to negate the condition result.
    """

    expression: str = ""
    fn: Optional[Callable[[Any], bool]] = None
    description: str = ""
    negated: bool = False

    def evaluate(self, data: Any) -> bool:
        """Evaluate the condition against the given data.

        Args:
            data: Output from the source node.

        Returns:
            True if the condition is satisfied.
        """
        if self.fn is not None:
            try:
                result = self.fn(data)
            except Exception as exc:
                logger.warning("Edge condition evaluation failed: %s", exc)
                result = False
        elif self.expression:
            try:
                result = bool(eval(self.expression, {"data": data, "output": data}))  # noqa: S307
            except Exception as exc:
                logger.warning("Edge condition expression failed: %s", exc)
                result = False
        else:
            # No condition means always traverse
            result = True

        return not result if self.negated else result


@dataclass
class WorkflowEdge:
    """A directed edge connecting two workflow nodes.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        condition: Optional condition that must be true to traverse.
        label: Human-readable label for the edge.
        priority: Priority when multiple edges are available (lower = higher priority).
        metadata: Additional edge metadata.
    """

    source: str = ""
    target: str = ""
    condition: Optional[EdgeCondition] = None
    label: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_traverse(self, data: Any) -> bool:
        """Check whether this edge can be traversed with the given data.

        Args:
            data: Output from the source node.

        Returns:
            True if the edge can be traversed.
        """
        if self.condition is None:
            return True
        return self.condition.evaluate(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "priority": self.priority,
        }
        if self.condition:
            result["condition"] = {
                "expression": self.condition.expression,
                "description": self.condition.description,
                "negated": self.condition.negated,
            }
        return result


def create_simple_edge(source: str, target: str, label: str = "") -> WorkflowEdge:
    """Create a simple unconditional edge.

    Args:
        source: Source node ID.
        target: Target node ID.
        label: Optional edge label.

    Returns:
        A WorkflowEdge without conditions.
    """
    return WorkflowEdge(source=source, target=target, label=label)


def create_conditional_edge(
    source: str,
    target: str,
    condition_fn: Callable[[Any], bool],
    label: str = "",
    description: str = "",
) -> WorkflowEdge:
    """Create a conditional edge.

    Args:
        source: Source node ID.
        target: Target node ID.
        condition_fn: Function that evaluates whether to traverse.
        label: Optional edge label.
        description: Condition description.

    Returns:
        A WorkflowEdge with a condition.
    """
    condition = EdgeCondition(fn=condition_fn, description=description)
    return WorkflowEdge(source=source, target=target, condition=condition, label=label)


def create_default_edge(source: str, target: str, label: str = "default") -> WorkflowEdge:
    """Create a default/fallback edge (lowest priority).

    Args:
        source: Source node ID.
        target: Target node ID.
        label: Edge label.

    Returns:
        A WorkflowEdge with low priority (used as fallback).
    """
    return WorkflowEdge(source=source, target=target, label=label, priority=999)
