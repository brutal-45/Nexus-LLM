"""Nexus-LLM Workflow Validators.

Provides validation functions for workflow definitions, including
structural validation, type checking, and custom constraint validation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set

from nexus_llm.workflow.engine import WorkflowEngine
from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus
from nexus_llm.workflow.edges import WorkflowEdge

logger = logging.getLogger(__name__)


class ValidationError:
    """Represents a workflow validation error.

    Attributes:
        path: Location of the error in the workflow definition.
        message: Human-readable error description.
        severity: Error severity (error, warning).
    """

    def __init__(self, path: str, message: str, severity: str = "error") -> None:
        self.path = path
        self.message = message
        self.severity = severity

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.path}: {self.message}"

    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path, "message": self.message, "severity": self.severity}


class WorkflowValidator:
    """Validates workflow definitions for correctness and completeness.

    Example::

        validator = WorkflowValidator()
        errors = validator.validate(engine)
        if errors:
            for error in errors:
                print(error)
    """

    def __init__(self, custom_rules: Optional[List[Callable]] = None) -> None:
        self._custom_rules = custom_rules or []
        logger.debug("WorkflowValidator initialized with %d custom rules", len(self._custom_rules))

    def validate(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Validate a workflow engine definition.

        Args:
            engine: The workflow engine to validate.

        Returns:
            List of ValidationError objects (empty if valid).
        """
        errors: List[ValidationError] = []

        # Structural checks
        errors.extend(self._check_structure(engine))
        errors.extend(self._check_nodes(engine))
        errors.extend(self._check_edges(engine))
        errors.extend(self._check_connectivity(engine))
        errors.extend(self._check_cycles(engine))

        # Custom rules
        for rule in self._custom_rules:
            try:
                rule_errors = rule(engine)
                if rule_errors:
                    errors.extend(rule_errors)
            except Exception as exc:
                errors.append(ValidationError(
                    path="custom_rule",
                    message=f"Custom validation rule failed: {exc}",
                    severity="warning",
                ))

        return errors

    def _check_structure(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Check basic structural requirements."""
        errors: List[ValidationError] = []
        if engine.node_count == 0:
            errors.append(ValidationError("workflow", "Workflow has no nodes"))
        return errors

    def _check_nodes(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Check node-specific constraints."""
        errors: List[ValidationError] = []

        start_nodes = engine.get_start_nodes()
        end_nodes = engine.get_end_nodes()

        if not start_nodes:
            errors.append(ValidationError("nodes", "Workflow must have at least one START node"))
        if len(start_nodes) > 1:
            errors.append(ValidationError("nodes", "Multiple START nodes detected", severity="warning"))
        if not end_nodes:
            errors.append(ValidationError("nodes", "Workflow must have at least one END node"))

        for nid, node in engine._nodes.items():
            if node.fn is None and node.type not in (NodeType.START, NodeType.END):
                errors.append(ValidationError(
                    f"nodes.{nid}", f"Node '{nid}' has no execution function"
                ))

        return errors

    def _check_edges(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Check edge-specific constraints."""
        errors: List[ValidationError] = []
        # Duplicate edges
        seen: Set[tuple] = set()
        for edge in engine._edges:
            key = (edge.source, edge.target)
            if key in seen:
                errors.append(ValidationError(
                    f"edges.{edge.source}->{edge.target}",
                    "Duplicate edge detected",
                    severity="warning",
                ))
            seen.add(key)
        return errors

    def _check_connectivity(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Check that all nodes are reachable."""
        errors: List[ValidationError] = []
        if engine.node_count <= 1:
            return errors

        connected: Set[str] = set()
        for edge in engine._edges:
            connected.add(edge.source)
            connected.add(edge.target)

        disconnected = set(engine._nodes.keys()) - connected
        if disconnected:
            errors.append(ValidationError(
                "nodes",
                f"Disconnected nodes: {disconnected}",
                severity="warning",
            ))
        return errors

    def _check_cycles(self, engine: WorkflowEngine) -> List[ValidationError]:
        """Check for cycles in the workflow graph."""
        errors: List[ValidationError] = []
        try:
            engine.topological_sort()
        except ValueError:
            errors.append(ValidationError("workflow", "Workflow contains a cycle"))
        return errors

    def is_valid(self, engine: WorkflowEngine) -> bool:
        """Quick check if a workflow is valid.

        Args:
            engine: The workflow engine to validate.

        Returns:
            True if the workflow has no errors (warnings are ok).
        """
        errors = self.validate(engine)
        return not any(e.severity == "error" for e in errors)
