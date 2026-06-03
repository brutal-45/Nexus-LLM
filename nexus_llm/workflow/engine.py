"""Nexus-LLM Workflow Engine.

Provides the WorkflowEngine class for defining, validating, and
managing workflow graphs composed of nodes and edges.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from nexus_llm.workflow.nodes import WorkflowNode, NodeType, NodeStatus
from nexus_llm.workflow.edges import WorkflowEdge, EdgeCondition

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """Configuration for a workflow.

    Attributes:
        name: Workflow name.
        description: Workflow description.
        max_retries: Maximum retries for failed nodes.
        retry_delay_seconds: Delay between retries.
        timeout_seconds: Global workflow timeout.
        continue_on_error: Whether to continue if a node fails.
    """

    name: str = ""
    description: str = ""
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 3600.0
    continue_on_error: bool = False


class WorkflowEngine:
    """Engine for defining and managing workflow graphs.

    The WorkflowEngine maintains a directed graph of nodes and edges,
    supports adding/removing nodes, connecting them, and validating
    the graph structure before execution.

    Example::

        engine = WorkflowEngine(name="my_workflow")
        engine.add_node(WorkflowNode(id="start", type=NodeType.START, fn=lambda: "hello"))
        engine.add_node(WorkflowNode(id="end", type=NodeType.END, fn=lambda x: x))
        engine.add_edge("start", "end")
        engine.validate()
    """

    def __init__(self, config: Optional[WorkflowConfig] = None) -> None:
        self._config = config or WorkflowConfig()
        self._nodes: Dict[str, WorkflowNode] = {}
        self._edges: List[WorkflowEdge] = []
        self._id = str(uuid.uuid4())
        logger.debug("WorkflowEngine created: %s", self._config.name or self._id)

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> WorkflowConfig:
        return self._config

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_node(self, node: WorkflowNode) -> None:
        """Add a node to the workflow.

        Args:
            node: The WorkflowNode to add.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        logger.debug("Added node: %s (%s)", node.id, node.type.value)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its connected edges.

        Args:
            node_id: The node ID to remove.

        Returns:
            True if the node was found and removed.
        """
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._edges = [
            e for e in self._edges
            if e.source != node_id and e.target != node_id
        ]
        logger.debug("Removed node: %s", node_id)
        return True

    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[EdgeCondition] = None,
    ) -> None:
        """Add an edge between two nodes.

        Args:
            source: Source node ID.
            target: Target node ID.
            condition: Optional edge condition.

        Raises:
            ValueError: If source or target node does not exist.
        """
        if source not in self._nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self._nodes:
            raise ValueError(f"Target node '{target}' not found")

        edge = WorkflowEdge(source=source, target=target, condition=condition)
        self._edges.append(edge)
        logger.debug("Added edge: %s -> %s", source, target)

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge between two nodes.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            True if the edge was found and removed.
        """
        for i, edge in enumerate(self._edges):
            if edge.source == source and edge.target == target:
                self._edges.pop(i)
                return True
        return False

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Retrieve a node by ID."""
        return self._nodes.get(node_id)

    def get_edges_from(self, node_id: str) -> List[WorkflowEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self._edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[WorkflowEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self._edges if e.target == node_id]

    def get_successors(self, node_id: str) -> List[str]:
        """Get IDs of all direct successor nodes."""
        return [e.target for e in self.get_edges_from(node_id)]

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get IDs of all direct predecessor nodes."""
        return [e.source for e in self.get_edges_to(node_id)]

    def get_start_nodes(self) -> List[str]:
        """Get IDs of all START nodes."""
        return [nid for nid, n in self._nodes.items() if n.type == NodeType.START]

    def get_end_nodes(self) -> List[str]:
        """Get IDs of all END nodes."""
        return [nid for nid, n in self._nodes.items() if n.type == NodeType.END]

    def validate(self) -> List[str]:
        """Validate the workflow graph structure.

        Returns:
            List of validation errors (empty if valid).
        """
        errors: List[str] = []

        # Check for start nodes
        start_nodes = self.get_start_nodes()
        if not start_nodes:
            errors.append("Workflow must have at least one START node")

        # Check for end nodes
        if not self.get_end_nodes():
            errors.append("Workflow must have at least one END node")

        # Check for cycles
        if self._has_cycle():
            errors.append("Workflow contains a cycle")

        # Check for disconnected nodes
        connected: Set[str] = set()
        for edge in self._edges:
            connected.add(edge.source)
            connected.add(edge.target)
        disconnected = set(self._nodes.keys()) - connected
        # Start/end nodes may be disconnected in single-node workflows
        if len(self._nodes) > 1 and disconnected:
            errors.append(f"Disconnected nodes: {disconnected}")

        # Check each node has required function
        for nid, node in self._nodes.items():
            if node.fn is None and node.type not in (NodeType.START, NodeType.END):
                errors.append(f"Node '{nid}' has no execution function")

        return errors

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order.

        Returns:
            List of node IDs in topological order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        in_degree: Dict[str, int] = {nid: 0 for nid in self._nodes}
        for edge in self._edges:
            in_degree[edge.target] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            for successor in self.get_successors(node_id):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(result) != len(self._nodes):
            raise ValueError("Graph contains a cycle; cannot topologically sort")

        return result

    def _has_cycle(self) -> bool:
        """Check whether the graph contains a cycle."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the workflow to a dictionary."""
        return {
            "id": self._id,
            "name": self._config.name,
            "description": self._config.description,
            "nodes": {nid: {"type": n.type.value, "name": n.name} for nid, n in self._nodes.items()},
            "edges": [{"source": e.source, "target": e.target} for e in self._edges],
        }
