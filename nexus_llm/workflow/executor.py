"""Nexus-LLM Workflow Executor.

Provides the WorkflowExecutor for running workflow graphs, tracking
execution state, and collecting results.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.workflow.engine import WorkflowEngine
from nexus_llm.workflow.nodes import NodeResult, NodeStatus, WorkflowNode
from nexus_llm.workflow.edges import WorkflowEdge

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result from executing a workflow.

    Attributes:
        workflow_id: ID of the executed workflow.
        success: Whether all nodes completed successfully.
        node_results: Results from each node, keyed by node ID.
        start_node: ID of the starting node.
        end_node: ID of the final node reached.
        total_duration_ms: Total execution time.
        nodes_completed: Number of nodes that completed.
        nodes_failed: Number of nodes that failed.
        error: Error message if the workflow failed.
    """

    workflow_id: str = ""
    success: bool = True
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    start_node: str = ""
    end_node: str = ""
    total_duration_ms: float = 0.0
    nodes_completed: int = 0
    nodes_failed: int = 0
    error: Optional[str] = None


class WorkflowExecutor:
    """Executes workflow graphs defined by a WorkflowEngine.

    The executor traverses the workflow graph starting from START nodes,
    follows edges based on conditions, and executes each node's function.
    It supports retry logic, timeout enforcement, and result collection.

    Example::

        engine = WorkflowEngine()
        # ... add nodes and edges ...
        executor = WorkflowExecutor(engine)
        result = executor.execute()
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        max_retries: Optional[int] = None,
        continue_on_error: Optional[bool] = None,
    ) -> None:
        self._engine = engine
        self._max_retries = max_retries if max_retries is not None else engine.config.max_retries
        self._continue_on_error = continue_on_error if continue_on_error is not None else engine.config.continue_on_error
        self._node_outputs: Dict[str, Any] = {}
        self._visited: set = set()
        logger.debug("WorkflowExecutor created for workflow: %s", engine.name or engine.id)

    def execute(self, initial_data: Any = None) -> ExecutionResult:
        """Execute the workflow from start to finish.

        Args:
            initial_data: Data to pass to the first node.

        Returns:
            An ExecutionResult with comprehensive workflow results.
        """
        start_time = time.perf_counter()
        result = ExecutionResult(workflow_id=self._engine.id)

        # Validate before execution
        errors = self._engine.validate()
        if errors:
            result.success = False
            result.error = f"Workflow validation failed: {'; '.join(errors)}"
            return result

        # Get start nodes
        start_nodes = self._engine.get_start_nodes()
        if not start_nodes:
            result.success = False
            result.error = "No start nodes found"
            return result

        # Reset all nodes
        for node in self._engine._nodes.values():
            node.reset()

        # Execute from start nodes
        self._node_outputs = {}
        self._visited = set()

        current_nodes = list(start_nodes)
        current_data = initial_data

        while current_nodes:
            next_nodes: List[str] = []

            for node_id in current_nodes:
                if node_id in self._visited:
                    continue

                node = self._engine.get_node(node_id)
                if node is None:
                    continue

                # Get input data for this node
                input_data = self._gather_input(node_id, current_data)

                # Execute with retries
                node_result = self._execute_with_retries(node, input_data)
                result.node_results[node_id] = node_result
                self._visited.add(node_id)

                if node_result.success:
                    result.nodes_completed += 1
                    self._node_outputs[node_id] = node_result.output
                else:
                    result.nodes_failed += 1
                    if not self._continue_on_error:
                        result.success = False
                        result.error = f"Node '{node_id}' failed: {node_result.error}"
                        break

                # Find next nodes
                edges = self._engine.get_edges_from(node_id)
                # Sort by priority (lower = higher priority)
                edges.sort(key=lambda e: e.priority)

                for edge in edges:
                    if edge.can_traverse(node_result.output):
                        if edge.target not in self._visited:
                            next_nodes.append(edge.target)

            current_nodes = next_nodes

        result.total_duration_ms = (time.perf_counter() - start_time) * 1000
        result.end_node = max(self._visited) if self._visited else ""
        result.start_node = start_nodes[0] if start_nodes else ""

        logger.info(
            "Workflow %s executed: %d completed, %d failed, %.1fms",
            self._engine.id,
            result.nodes_completed,
            result.nodes_failed,
            result.total_duration_ms,
        )
        return result

    def _execute_with_retries(self, node: WorkflowNode, input_data: Any) -> NodeResult:
        """Execute a node with retry logic.

        Args:
            node: The node to execute.
            input_data: Input data.

        Returns:
            The final NodeResult after retries.
        """
        max_attempts = max(self._max_retries, node.max_retries) + 1

        for attempt in range(max_attempts):
            result = node.execute(input_data)
            if result.success:
                return result

            if attempt < max_attempts - 1:
                logger.debug("Retrying node %s (attempt %d/%d)", node.id, attempt + 2, max_attempts)
                node.reset()

        return result

    def _gather_input(self, node_id: str, default_data: Any) -> Any:
        """Gather input data for a node from its predecessors.

        If a node has multiple predecessors, their outputs are
        collected into a dictionary keyed by predecessor ID.

        Args:
            node_id: The target node ID.
            default_data: Default data if no predecessors.

        Returns:
            The gathered input data.
        """
        predecessors = self._engine.get_predecessors(node_id)
        if not predecessors:
            return default_data

        if len(predecessors) == 1:
            return self._node_outputs.get(predecessors[0], default_data)

        # Multiple predecessors: collect into a dict
        return {pid: self._node_outputs.get(pid) for pid in predecessors}
