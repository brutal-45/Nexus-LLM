"""KnowledgeGraph — directed graph with typed edges for knowledge traversal."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NodeNotFoundError(KeyError):
    """Raised when a referenced node does not exist."""


class KnowledgeGraph:
    """In-memory directed knowledge graph with typed relations.

    Nodes store arbitrary data dictionaries; edges carry a ``relation``
    label describing the relationship between source and target.

    Example
    -------
    >>> kg = KnowledgeGraph()
    >>> kg.add_node("python", {"type": "language"})
    >>> kg.add_node("guido", {"type": "person"})
    >>> kg.add_edge("guido", "python", "created")
    >>> kg.get_neighbors("guido")
    [('python', 'created')]
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # Adjacency: source_id -> [(target_id, relation)]
        self._outgoing: Dict[str, List[Tuple[str, str]]] = {}
        self._incoming: Dict[str, List[Tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, id: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Create or update a node.

        If *id* already exists its data is merged with *data*.

        Parameters
        ----------
        id:
            Unique node identifier.
        data:
            Arbitrary metadata dictionary.
        """
        data = data or {}
        if id in self._nodes:
            self._nodes[id].update(data)
            logger.debug("KnowledgeGraph: updated node %r", id)
        else:
            self._nodes[id] = dict(data)
            self._outgoing.setdefault(id, [])
            self._incoming.setdefault(id, [])
            logger.debug("KnowledgeGraph: added node %r", id)

    def get_node(self, id: str) -> Dict[str, Any]:
        """Return the data dict for node *id*.

        Raises
        ------
        NodeNotFoundError
            If *id* does not exist.
        """
        if id not in self._nodes:
            raise NodeNotFoundError(id)
        return dict(self._nodes[id])

    def remove_node(self, id: str) -> None:
        """Remove a node and all its edges."""
        if id not in self._nodes:
            raise NodeNotFoundError(id)
        # Remove edges pointing to/from this node.
        for target, rel in list(self._outgoing.get(id, [])):
            self._incoming[target] = [
                (src, r) for src, r in self._incoming.get(target, []) if src != id
            ]
        for source, rel in list(self._incoming.get(id, [])):
            self._outgoing[source] = [
                (tgt, r) for tgt, r in self._outgoing.get(source, []) if tgt != id
            ]
        del self._nodes[id]
        self._outgoing.pop(id, None)
        self._incoming.pop(id, None)
        logger.debug("KnowledgeGraph: removed node %r", id)

    def list_nodes(self) -> List[str]:
        """Return all node IDs."""
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, from_id: str, to_id: str, relation: str) -> None:
        """Create a directed edge from *from_id* to *to_id* with label *relation*.

        Both nodes must already exist.

        Raises
        ------
        NodeNotFoundError
            If either endpoint does not exist.
        """
        if from_id not in self._nodes:
            raise NodeNotFoundError(from_id)
        if to_id not in self._nodes:
            raise NodeNotFoundError(to_id)
        self._outgoing[from_id].append((to_id, relation))
        self._incoming[to_id].append((from_id, relation))
        logger.debug(
            "KnowledgeGraph: added edge %r --[%s]--> %r",
            from_id, relation, to_id,
        )

    def get_neighbors(self, id: str) -> List[Tuple[str, str]]:
        """Return outgoing neighbours of *id* as ``[(target_id, relation), ...]``.

        Raises
        ------
        NodeNotFoundError
            If *id* does not exist.
        """
        if id not in self._nodes:
            raise NodeNotFoundError(id)
        return list(self._outgoing.get(id, []))

    def get_incoming(self, id: str) -> List[Tuple[str, str]]:
        """Return incoming neighbours of *id* as ``[(source_id, relation), ...]``."""
        if id not in self._nodes:
            raise NodeNotFoundError(id)
        return list(self._incoming.get(id, []))

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def find_path(
        self,
        from_id: str,
        to_id: str,
        max_depth: int = 20,
    ) -> List[Tuple[str, str]]:
        """BFS shortest-path from *from_id* to *to_id*.

        Returns
        -------
        A list of ``(node_id, relation)`` pairs describing the path
        (including the start node with relation ``None``).
        Empty list if no path exists.
        """
        if from_id not in self._nodes or to_id not in self._nodes:
            return []

        if from_id == to_id:
            return [(from_id, "")]

        visited: Set[str] = {from_id}
        queue: deque[Tuple[str, List[Tuple[str, str]]]] = deque()
        queue.append((from_id, [(from_id, "")]))

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for neighbor, relation in self._outgoing.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [(neighbor, relation)]
                if neighbor == to_id:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return []

    def get_subgraph(self, center_id: str, depth: int = 1) -> Dict[str, Any]:
        """Return a subgraph centred on *center_id* out to *depth* hops.

        Returns
        -------
        A dictionary with ``"nodes"`` and ``"edges"`` keys::

            {
                "nodes": {id: data, ...},
                "edges": [{"from": id, "to": id, "relation": str}, ...]
            }
        """
        if center_id not in self._nodes:
            raise NodeNotFoundError(center_id)

        collected_nodes: Set[str] = {center_id}
        frontier: Set[str] = {center_id}

        for _ in range(depth):
            next_frontier: Set[str] = set()
            for node_id in frontier:
                for target, _ in self._outgoing.get(node_id, []):
                    if target not in collected_nodes:
                        next_frontier.add(target)
                for source, _ in self._incoming.get(node_id, []):
                    if source not in collected_nodes:
                        next_frontier.add(source)
            collected_nodes |= next_frontier
            frontier = next_frontier

        nodes = {nid: dict(self._nodes[nid]) for nid in collected_nodes}
        edges = [
            {"from": src, "to": tgt, "relation": rel}
            for src in collected_nodes
            for tgt, rel in self._outgoing.get(src, [])
            if tgt in collected_nodes
        ]

        return {"nodes": nodes, "edges": edges}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._outgoing.values())

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={self.node_count()}, edges={self.edge_count()})"
