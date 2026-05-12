"""
Tree-of-Thought Reasoning Module
====================================

Implements tree-of-thought (ToT) reasoning, a framework that explores multiple
reasoning paths simultaneously using tree search algorithms inspired by
game-playing AI.

The key innovation of ToT is treating reasoning as a search problem where
each node represents a partial reasoning state, and edges represent reasoning
steps. The search explores the space of possible reasoning paths to find the
most promising one.

This module supports multiple search algorithms:
- Monte Carlo Tree Search (MCTS): Balances exploration and exploitation
- Beam Search: Maintains top-k candidates at each depth level
- BFS (Breadth-First Search): Systematic exploration level by level
- DFS (Depth-First Search): Deep exploration with backtracking

References:
    - Yao et al. (2023) "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    - Besta et al. (2023) "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
"""

from __future__ import annotations

import math
import heapq
import random
import copy
import json
import time
import hashlib
import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque, OrderedDict
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ExplorationStrategy(Enum):
    """Strategies for exploring the thought tree.

    Attributes:
        MCTS: Monte Carlo Tree Search with UCB selection.
        BEAM: Beam search keeping top-k candidates per level.
        BFS: Breadth-first search exploring all nodes at each level.
        DFS: Depth-first search with backtracking.
        ASTAR: A* search with heuristic-based node prioritization.
    """
    MCTS = "mcts"
    BEAM = "beam"
    BFS = "bfs"
    DFS = "dfs"
    ASTAR = "astar"


class SearchAlgorithm(Enum):
    """Available search algorithms for tree exploration.

    Attributes:
        MONTE_CARLO: Monte Carlo Tree Search.
        BEAM_SEARCH: Beam search variant.
        BREADTH_FIRST: Breadth-first search.
        DEPTH_FIRST: Depth-first search with backtracking.
        BEST_FIRST: Best-first search using node value.
    """
    MONTE_CARLO = "monte_carlo"
    BEAM_SEARCH = "beam_search"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"


class ThoughtStatus(Enum):
    """Status of a thought node in the search tree.

    Attributes:
        ACTIVE: Node is available for expansion.
        EXPANDING: Node is currently being expanded.
        EXPANDED: Node has been expanded with children.
        TERMINAL: Node represents a complete solution.
        PRUNED: Node was pruned from the tree.
        VISITING: Node is currently being visited during search.
    """
    ACTIVE = "active"
    EXPANDING = "expanding"
    EXPANDED = "expanded"
    TERMINAL = "terminal"
    PRUNED = "pruned"
    VISITING = "visiting"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ThoughtScore:
    """Detailed scoring breakdown for a thought node.

    Attributes:
        overall: Overall quality score (0.0 to 1.0).
        coherence: How logically coherent the thought is.
        relevance: How relevant the thought is to the goal.
        feasibility: How feasible the thought is to pursue.
        progress: How much progress toward the goal.
        novelty: How novel/original the thought is.
        confidence: Confidence in the score.
    """
    overall: float = 0.0
    coherence: float = 0.5
    relevance: float = 0.5
    feasibility: float = 0.5
    progress: float = 0.5
    novelty: float = 0.5
    confidence: float = 0.5

    def weighted_average(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute a weighted average of all score components.

        Args:
            weights: Optional custom weights for each component.

        Returns:
            Weighted average score between 0.0 and 1.0.
        """
        default_weights = {
            "coherence": 0.25,
            "relevance": 0.25,
            "feasibility": 0.20,
            "progress": 0.20,
            "novelty": 0.10,
        }
        w = weights or default_weights

        components = {
            "coherence": self.coherence,
            "relevance": self.relevance,
            "feasibility": self.feasibility,
            "progress": self.progress,
            "novelty": self.novelty,
        }

        total_weight = 0.0
        weighted_sum = 0.0
        for key, value in components.items():
            weight = w.get(key, 0.0)
            weighted_sum += value * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Serialize the thought score to a dictionary.

        Returns:
            Dictionary of score components.
        """
        return {
            "overall": self.overall,
            "coherence": self.coherence,
            "relevance": self.relevance,
            "feasibility": self.feasibility,
            "progress": self.progress,
            "novelty": self.novelty,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> ThoughtScore:
        """Deserialize a thought score from a dictionary.

        Args:
            data: Dictionary of score components.

        Returns:
            A ThoughtScore instance.
        """
        return cls(
            overall=data.get("overall", 0.0),
            coherence=data.get("coherence", 0.5),
            relevance=data.get("relevance", 0.5),
            feasibility=data.get("feasibility", 0.5),
            progress=data.get("progress", 0.5),
            novelty=data.get("novelty", 0.5),
            confidence=data.get("confidence", 0.5),
        )


class ThoughtNode:
    """A single node in the thought tree.

    Each node represents a reasoning state (thought) with its evaluation
    metrics and tree connectivity information.

    Attributes:
        content: The text content of this thought.
        parent: Reference to the parent node (None for root).
        children: List of child nodes.
        score: Detailed scoring of this thought.
        depth: Depth of this node in the tree (root = 0).
        visits: Number of times this node has been visited during search.
        value: Accumulated value from backpropagation.
        status: Current status of this node.
        node_id: Unique identifier for this node.
        metadata: Additional metadata about the thought.
        creation_time: Timestamp when this node was created.
        is_solution: Whether this node represents a complete solution.
        solution_text: The solution text if this is a terminal node.
    """

    _next_id: int = 0

    def __init__(
        self,
        content: str = "",
        parent: Optional[ThoughtNode] = None,
        score: Optional[ThoughtScore] = None,
        depth: int = 0,
    ) -> None:
        """Initialize a thought node.

        Args:
            content: Text content of the thought.
            parent: Parent node reference.
            score: Thought quality score.
            depth: Depth in the tree.
        """
        self.content = content
        self.parent = parent
        self.children: List[ThoughtNode] = []
        self.score = score or ThoughtScore()
        self.depth = depth
        self.visits: int = 0
        self.value: float = 0.0
        self.status: ThoughtStatus = ThoughtStatus.ACTIVE
        self.node_id: int = ThoughtNode._next_id
        ThoughtNode._next_id += 1
        self.metadata: Dict[str, Any] = {}
        self.creation_time: float = time.time()
        self.is_solution: bool = False
        self.solution_text: str = ""

        if parent is not None:
            parent.children.append(self)

    @property
    def average_value(self) -> float:
        """Get the average value per visit for this node.

        Returns:
            Average value, or 0.0 if never visited.
        """
        return self.value / self.visits if self.visits > 0 else 0.0

    @property
    def is_leaf(self) -> bool:
        """Check if this node has no children.

        Returns:
            True if this node is a leaf node.
        """
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """Check if this node is the root of the tree.

        Returns:
            True if this node has no parent.
        """
        return self.parent is None

    @property
    def path_from_root(self) -> List[ThoughtNode]:
        """Get the path from the root to this node.

        Returns:
            List of nodes from root to this node (inclusive).
        """
        path: List[ThoughtNode] = []
        current: Optional[ThoughtNode] = self
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    @property
    def path_content(self) -> str:
        """Get the concatenated content along the path from root.

        Returns:
            String of all thoughts from root to this node.
        """
        path = self.path_from_root
        return " -> ".join(node.content for node in path if node.content)

    def add_child(self, content: str, score: Optional[ThoughtScore] = None) -> ThoughtNode:
        """Add a child thought node to this node.

        Args:
            content: Content for the child node.
            score: Optional score for the child node.

        Returns:
            The newly created child node.
        """
        child = ThoughtNode(
            content=content,
            parent=self,
            score=score,
            depth=self.depth + 1,
        )
        return child

    def remove_child(self, child: ThoughtNode) -> bool:
        """Remove a specific child node.

        Args:
            child: The child node to remove.

        Returns:
            True if the child was found and removed.
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False

    def prune(self) -> None:
        """Mark this node and all descendants as pruned."""
        self.status = ThoughtStatus.PRUNED
        for child in self.children:
            child.prune()

    def clone(self) -> ThoughtNode:
        """Create a deep copy of this node and its subtree.

        Returns:
            A new ThoughtNode that is a deep copy of this subtree.
        """
        new_node = ThoughtNode(
            content=self.content,
            parent=None,
            score=ThoughtScore.from_dict(self.score.to_dict()),
            depth=self.depth,
        )
        new_node.visits = self.visits
        new_node.value = self.value
        new_node.status = self.status
        new_node.node_id = ThoughtNode._next_id
        ThoughtNode._next_id += 1
        new_node.metadata = copy.deepcopy(self.metadata)
        new_node.creation_time = self.creation_time
        new_node.is_solution = self.is_solution
        new_node.solution_text = self.solution_text

        for child in self.children:
            child_clone = child.clone()
            child_clone.parent = new_node
            new_node.children.append(child_clone)

        return new_node

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this node to a dictionary.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "content": self.content,
            "depth": self.depth,
            "score": self.score.to_dict(),
            "visits": self.visits,
            "value": self.value,
            "status": self.status.value,
            "is_solution": self.is_solution,
            "solution_text": self.solution_text,
            "num_children": len(self.children),
            "parent_id": self.parent.node_id if self.parent else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"ThoughtNode(id={self.node_id}, depth={self.depth}, "
            f"score={self.score.overall:.2f}, visits={self.visits}, "
            f"children={len(self.children)})"
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ThoughtNode):
            return NotImplemented
        return self.score.overall < other.score.overall

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ThoughtNode):
            return NotImplemented
        return self.node_id == other.node_id


# =============================================================================
# ThoughtTree
# =============================================================================

class ThoughtTree:
    """Tree structure for organizing and managing thought nodes.

    Provides operations for adding, searching, pruning, and visualizing
    the thought tree. Manages the root node and provides efficient access
    to tree elements.

    Attributes:
        root: The root node of the tree.
        nodes: Dictionary mapping node IDs to node references.
        size: Total number of nodes in the tree.
        max_depth: Maximum depth of any node in the tree.
    """

    def __init__(self, root_content: str = "", root_score: Optional[ThoughtScore] = None) -> None:
        """Initialize the thought tree.

        Args:
            root_content: Initial content for the root node.
            root_score: Score for the root node.
        """
        self.root = ThoughtNode(content=root_content, score=root_score, depth=0)
        self._nodes: Dict[int, ThoughtNode] = {self.root.node_id: self.root}
        self._search_cache: Dict[str, List[ThoughtNode]] = {}
        self._last_modified: float = time.time()

    @property
    def size(self) -> int:
        """Get the total number of nodes in the tree.

        Returns:
            Number of nodes.
        """
        return len(self._nodes)

    @property
    def max_depth(self) -> int:
        """Get the maximum depth of the tree.

        Returns:
            Maximum depth value, or 0 if tree is empty.
        """
        if not self._nodes:
            return 0
        return max(node.depth for node in self._nodes.values())

    def add_node(
        self,
        parent_id: int,
        content: str,
        score: Optional[ThoughtScore] = None,
    ) -> Optional[ThoughtNode]:
        """Add a new thought node as a child of an existing node.

        Args:
            parent_id: ID of the parent node.
            content: Content for the new node.
            score: Optional score for the new node.

        Returns:
            The newly created node, or None if parent was not found.
        """
        parent = self._nodes.get(parent_id)
        if parent is None:
            return None

        new_node = ThoughtNode(
            content=content,
            parent=parent,
            score=score,
            depth=parent.depth + 1,
        )
        self._nodes[new_node.node_id] = new_node
        self._last_modified = time.time()
        self._search_cache.clear()
        return new_node

    def get_node(self, node_id: int) -> Optional[ThoughtNode]:
        """Retrieve a node by its ID.

        Args:
            node_id: The unique node identifier.

        Returns:
            The ThoughtNode, or None if not found.
        """
        return self._nodes.get(node_id)

    def remove_node(self, node_id: int) -> bool:
        """Remove a node and all its descendants from the tree.

        Cannot remove the root node.

        Args:
            node_id: ID of the node to remove.

        Returns:
            True if the node was successfully removed.
        """
        node = self._nodes.get(node_id)
        if node is None or node.is_root:
            return False

        descendants = self._get_all_descendants(node)
        for descendant in descendants:
            self._nodes.pop(descendant.node_id, None)

        if node.parent is not None:
            node.parent.children = [
                c for c in node.parent.children if c.node_id != node_id
            ]

        self._nodes.pop(node_id, None)
        self._last_modified = time.time()
        self._search_cache.clear()
        return True

    def _get_all_descendants(self, node: ThoughtNode) -> List[ThoughtNode]:
        """Get all descendant nodes of a given node.

        Args:
            node: The starting node.

        Returns:
            List of all descendant nodes (not including the starting node).
        """
        descendants: List[ThoughtNode] = []
        queue = deque(node.children)
        while queue:
            current = queue.popleft()
            descendants.append(current)
            queue.extend(current.children)
        return descendants

    def search(
        self,
        query: Optional[str] = None,
        min_depth: int = 0,
        max_depth: int = -1,
        min_score: float = 0.0,
        status: Optional[ThoughtStatus] = None,
    ) -> List[ThoughtNode]:
        """Search for nodes matching the given criteria.

        Args:
            query: Optional text query to match against node content.
            min_depth: Minimum depth filter.
            max_depth: Maximum depth filter (-1 for no limit).
            min_score: Minimum score filter.
            status: Status filter.

        Returns:
            List of matching nodes.
        """
        cache_key = f"{query}:{min_depth}:{max_depth}:{min_score}:{status}"
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        results: List[ThoughtNode] = []
        for node in self._nodes.values():
            if node.depth < min_depth:
                continue
            if max_depth >= 0 and node.depth > max_depth:
                continue
            if node.score.overall < min_score:
                continue
            if status is not None and node.status != status:
                continue
            if query is not None:
                if query.lower() not in node.content.lower():
                    continue
            results.append(node)

        self._search_cache[cache_key] = results
        return results

    def get_best_leaf(self) -> Optional[ThoughtNode]:
        """Get the leaf node with the highest score.

        Returns:
            The best leaf node, or None if no leaves exist.
        """
        leaves = [n for n in self._nodes.values() if n.is_leaf]
        if not leaves:
            return None
        return max(leaves, key=lambda n: n.score.overall)

    def get_best_path(self) -> List[ThoughtNode]:
        """Get the path from root to the best leaf node.

        Returns:
            List of nodes forming the best reasoning path.
        """
        best_leaf = self.get_best_leaf()
        if best_leaf is None:
            return [self.root]
        return best_leaf.path_from_root

    def prune_below_threshold(self, threshold: float) -> int:
        """Prune all nodes with scores below a threshold.

        Args:
            threshold: Minimum score to keep.

        Returns:
            Number of nodes pruned.
        """
        to_prune: List[int] = []
        for node_id, node in self._nodes.items():
            if not node.is_root and node.score.overall < threshold:
                to_prune.append(node_id)

        for node_id in to_prune:
            self.remove_node(node_id)

        return len(to_prune)

    def prune_by_depth(self, max_depth: int) -> int:
        """Prune all nodes deeper than a specified depth.

        Args:
            max_depth: Maximum allowed depth.

        Returns:
            Number of nodes pruned.
        """
        to_prune: List[int] = []
        for node_id, node in self._nodes.items():
            if node.depth > max_depth:
                to_prune.append(node_id)

        for node_id in to_prune:
            self.remove_node(node_id)

        return len(to_prune)

    def prune_low_visit(self, min_visits: int) -> int:
        """Prune nodes that have been visited fewer than min_visits times.

        Args:
            min_visits: Minimum required visits.

        Returns:
            Number of nodes pruned.
        """
        to_prune: List[int] = []
        for node_id, node in self._nodes.items():
            if not node.is_root and node.visits < min_visits:
                to_prune.append(node_id)

        for node_id in to_prune:
            self.remove_node(node_id)

        return len(to_prune)

    def get_nodes_at_depth(self, depth: int) -> List[ThoughtNode]:
        """Get all nodes at a specific depth level.

        Args:
            depth: Target depth level.

        Returns:
            List of nodes at the specified depth.
        """
        return [n for n in self._nodes.values() if n.depth == depth]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the thought tree.

        Returns:
            Dictionary of tree statistics.
        """
        if not self._nodes:
            return {
                "size": 0,
                "max_depth": 0,
                "avg_score": 0.0,
                "total_visits": 0,
                "branching_factor": 0.0,
                "solution_count": 0,
            }

        scores = [n.score.overall for n in self._nodes.values()]
        visits = sum(n.visits for n in self._nodes.values())
        solutions = sum(1 for n in self._nodes.values() if n.is_solution)

        internal_nodes = [n for n in self._nodes.values() if not n.is_leaf and not n.is_root]
        if internal_nodes:
            branching = statistics.mean(len(n.children) for n in internal_nodes)
        else:
            branching = 0.0

        depth_counts = Counter(n.depth for n in self._nodes.values())
        depth_distribution = {str(k): v for k, v in sorted(depth_counts.items())}

        return {
            "size": self.size,
            "max_depth": self.max_depth,
            "avg_score": statistics.mean(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "total_visits": visits,
            "branching_factor": branching,
            "solution_count": solutions,
            "depth_distribution": depth_distribution,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tree to a dictionary.

        Returns:
            Dictionary representation of the tree structure.
        """
        return {
            "root_id": self.root.node_id,
            "size": self.size,
            "max_depth": self.max_depth,
            "nodes": {str(nid): n.to_dict() for nid, n in self._nodes.items()},
            "statistics": self.get_statistics(),
        }

    def clear(self) -> None:
        """Remove all nodes except the root."""
        for node_id in list(self._nodes.keys()):
            if node_id != self.root.node_id:
                del self._nodes[node_id]
        self.root.children.clear()
        self.root.visits = 0
        self.root.value = 0.0
        self._search_cache.clear()
        self._last_modified = time.time()


# =============================================================================
# ThoughtEvaluator
# =============================================================================

class ThoughtEvaluator:
    """Evaluates thought quality across multiple dimensions.

    Provides scoring for thought coherence, relevance, feasibility,
    progress, and novelty.

    Attributes:
        coherence_weight: Weight for coherence in overall score.
        relevance_weight: Weight for relevance in overall score.
        feasibility_weight: Weight for feasibility in overall score.
        progress_weight: Weight for progress in overall score.
        novelty_weight: Weight for novelty in overall score.
    """

    def __init__(
        self,
        coherence_weight: float = 0.25,
        relevance_weight: float = 0.25,
        feasibility_weight: float = 0.20,
        progress_weight: float = 0.20,
        novelty_weight: float = 0.10,
    ) -> None:
        """Initialize the thought evaluator.

        Args:
            coherence_weight: Weight for logical coherence.
            relevance_weight: Weight for goal relevance.
            feasibility_weight: Weight for thought feasibility.
            progress_weight: Weight for progress toward solution.
            novelty_weight: Weight for thought novelty.
        """
        total = coherence_weight + relevance_weight + feasibility_weight + progress_weight + novelty_weight
        self.coherence_weight = coherence_weight / total
        self.relevance_weight = relevance_weight / total
        self.feasibility_weight = feasibility_weight / total
        self.progress_weight = progress_weight / total
        self.novelty_weight = novelty_weight / total

    def evaluate(
        self,
        thought: str,
        context: str = "",
        goal: str = "",
        sibling_thoughts: Optional[List[str]] = None,
    ) -> ThoughtScore:
        """Evaluate a thought across all dimensions.

        Args:
            thought: The thought text to evaluate.
            context: Current reasoning context.
            goal: The overall goal or question.
            sibling_thoughts: Thoughts at the same level for novelty comparison.

        Returns:
            A ThoughtScore with detailed scoring.
        """
        coherence = self._evaluate_coherence(thought, context)
        relevance = self._evaluate_relevance(thought, goal, context)
        feasibility = self._evaluate_feasibility(thought, context)
        progress = self._evaluate_progress(thought, context, goal)
        novelty = self._evaluate_novelty(thought, sibling_thoughts)

        overall = (
            coherence * self.coherence_weight
            + relevance * self.relevance_weight
            + feasibility * self.feasibility_weight
            + progress * self.progress_weight
            + novelty * self.novelty_weight
        )

        confidence = self._compute_confidence(coherence, relevance, feasibility, progress, novelty)

        return ThoughtScore(
            overall=min(1.0, max(0.0, overall)),
            coherence=coherence,
            relevance=relevance,
            feasibility=feasibility,
            progress=progress,
            novelty=novelty,
            confidence=confidence,
        )

    def _evaluate_coherence(self, thought: str, context: str) -> float:
        """Evaluate the logical coherence of a thought.

        Checks for logical connectives, consistent terminology,
        and well-formed reasoning structure.

        Args:
            thought: The thought to evaluate.
            context: Previous reasoning context.

        Returns:
            Coherence score between 0.0 and 1.0.
        """
        if not thought.strip():
            return 0.0

        score = 0.3

        logical_connectives = [
            "because", "therefore", "thus", "since", "implies", "means",
            "shows", "indicates", "leads to", "results in", "follows",
            "consequently", "hence", "so", "then", "given that",
        ]
        connective_count = sum(1 for c in logical_connectives if c in thought.lower())
        score += min(0.2, connective_count * 0.05)

        thought_words = set(thought.lower().split())
        if context:
            context_words = set(context.lower().split())
            if thought_words and context_words:
                overlap = len(thought_words & context_words) / len(thought_words)
                score += min(0.2, overlap * 0.3)

        sentences = re.split(r'(?<=[.!?])\s+', thought)
        if len(sentences) > 1:
            score += 0.1

        has_subject_verb = any(
            word in thought.lower()
            for word in ["is", "are", "was", "were", "has", "have",
                         "can", "will", "does", "did", "should"]
        )
        if has_subject_verb:
            score += 0.1

        length = len(thought)
        if 20 <= length <= 500:
            score += 0.1
        elif length > 500:
            score += 0.05

        return min(1.0, max(0.0, score))

    def _evaluate_relevance(self, thought: str, goal: str, context: str) -> float:
        """Evaluate how relevant a thought is to the goal.

        Args:
            thought: The thought to evaluate.
            goal: The target goal or question.
            context: Previous reasoning context.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not thought.strip():
            return 0.0
        if not goal.strip():
            return 0.5

        goal_words = set(goal.lower().split())
        thought_words = set(thought.lower().split())

        stop_words = frozenset({
            "the", "a", "an", "is", "are", "to", "of", "in", "for",
            "on", "with", "and", "but", "or", "not", "this", "that",
        })
        goal_content = goal_words - stop_words
        thought_content = thought_words - stop_words

        if not goal_content or not thought_content:
            return 0.3

        overlap = len(goal_content & thought_content)
        coverage = overlap / len(goal_content) if goal_content else 0.0
        precision = overlap / len(thought_content) if thought_content else 0.0

        f1 = 2 * coverage * precision / (coverage + precision) if (coverage + precision) > 0 else 0.0
        score = f1 * 0.7 + 0.3

        goal_numbers = set(re.findall(r'\d+\.?\d*', goal.lower()))
        thought_numbers = set(re.findall(r'\d+\.?\d*', thought.lower()))
        if goal_numbers and thought_numbers:
            number_overlap = len(goal_numbers & thought_numbers) / len(goal_numbers)
            score = score * 0.7 + number_overlap * 0.3

        return min(1.0, max(0.0, score))

    def _evaluate_feasibility(self, thought: str, context: str) -> float:
        """Evaluate how feasible a thought is to pursue.

        Checks whether the thought contains actionable reasoning
        rather than speculation or unsupported claims.

        Args:
            thought: The thought to evaluate.
            context: Previous reasoning context.

        Returns:
            Feasibility score between 0.0 and 1.0.
        """
        if not thought.strip():
            return 0.0

        score = 0.4

        speculative_words = ["maybe", "perhaps", "possibly", "might", "could be", "I guess"]
        speculation_count = sum(1 for w in speculative_words if w in thought.lower())
        score -= speculation_count * 0.1

        concrete_indicators = [
            "calculate", "compute", "determine", "find", "solve",
            "equals", "is", "results", "gives", "produces",
        ]
        concrete_count = sum(1 for c in concrete_indicators if c in thought.lower())
        score += min(0.3, concrete_count * 0.1)

        has_numbers = bool(re.search(r'\d+', thought))
        if has_numbers:
            score += 0.1

        has_operations = bool(re.search(r'[+\-*/=<>]', thought))
        if has_operations:
            score += 0.1

        uncertainty_markers = ["I think", "I believe", "I'm not sure", "unclear"]
        uncertainty_count = sum(1 for m in uncertainty_markers if m in thought.lower())
        score -= uncertainty_count * 0.15

        return min(1.0, max(0.0, score))

    def _evaluate_progress(self, thought: str, context: str, goal: str) -> float:
        """Evaluate how much progress a thought makes toward the goal.

        Args:
            thought: The thought to evaluate.
            context: Previous reasoning context.
            goal: The target goal.

        Returns:
            Progress score between 0.0 and 1.0.
        """
        if not thought.strip():
            return 0.0

        score = 0.3

        progress_markers = [
            "therefore", "thus", "so", "hence", "consequently",
            "we can conclude", "this shows", "the answer is",
            "the result is", "which means", "it follows",
        ]
        has_progress = any(m in thought.lower() for m in progress_markers)
        if has_progress:
            score += 0.3

        conclusion_markers = ["finally", "in conclusion", "overall", "summing up"]
        has_conclusion = any(m in thought.lower() for m in conclusion_markers)
        if has_conclusion:
            score += 0.2

        if goal:
            goal_keywords = set(goal.lower().split()) - frozenset({
                "the", "a", "an", "is", "are", "what", "how", "why",
            })
            thought_keywords = set(thought.lower().split()) & goal_keywords
            if goal_keywords:
                keyword_coverage = len(thought_keywords) / len(goal_keywords)
                score += min(0.2, keyword_coverage * 0.3)

        step_indicators = ["first", "next", "then", "after that", "finally"]
        has_step = any(s in thought.lower() for s in step_indicators)
        if has_step:
            score += 0.05

        if context:
            context_numbers = set(re.findall(r'\d+\.?\d*', context.lower()))
            thought_numbers = set(re.findall(r'\d+\.?\d*', thought.lower()))
            new_numbers = thought_numbers - context_numbers
            if new_numbers:
                score += 0.1

        return min(1.0, max(0.0, score))

    def _evaluate_novelty(
        self,
        thought: str,
        sibling_thoughts: Optional[List[str]] = None,
    ) -> float:
        """Evaluate how novel a thought is compared to siblings.

        Args:
            thought: The thought to evaluate.
            sibling_thoughts: Thoughts at the same tree level.

        Returns:
            Novelty score between 0.0 and 1.0.
        """
        if not sibling_thoughts:
            return 0.7

        thought_normalized = thought.lower().strip()
        thought_words = set(thought_normalized.split())
        if not thought_words:
            return 0.5

        min_similarities: List[float] = []
        for sibling in sibling_thoughts:
            sibling_normalized = sibling.lower().strip()
            if thought_normalized == sibling_normalized:
                min_similarities.append(1.0)
                continue

            sibling_words = set(sibling_normalized.split())
            if not sibling_words:
                continue

            intersection = len(thought_words & sibling_words)
            union = len(thought_words | sibling_words)
            jaccard = intersection / union if union > 0 else 0.0
            min_similarities.append(jaccard)

        if not min_similarities:
            return 0.8

        max_similarity = max(min_similarities)
        novelty = 1.0 - max_similarity
        return min(1.0, max(0.0, novelty))

    def _compute_confidence(
        self,
        coherence: float,
        relevance: float,
        feasibility: float,
        progress: float,
        novelty: float,
    ) -> float:
        """Compute confidence in the evaluation score.

        Higher agreement between components means higher confidence.

        Args:
            coherence: Coherence score.
            relevance: Relevance score.
            feasibility: Feasibility score.
            progress: Progress score.
            novelty: Novelty score.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        scores = [coherence, relevance, feasibility, progress, novelty]
        mean_score = statistics.mean(scores)
        if len(scores) > 1:
            std_score = statistics.stdev(scores)
            confidence = max(0.0, 1.0 - std_score * 2)
        else:
            confidence = 0.5
        return min(1.0, max(0.0, confidence))


# =============================================================================
# TreeVisualizer
# =============================================================================

class TreeVisualizer:
    """Text-based tree visualization for debugging and analysis.

    Generates human-readable representations of the thought tree
    structure, including ASCII art and formatted text outputs.
    """

    INDENT = "  "
    BRANCH = "├── "
    LAST_BRANCH = "└── "
    VERTICAL = "│   "
    CONTINUE = "    "

    def __init__(self, max_depth: int = -1, max_width: int = 120) -> None:
        """Initialize the tree visualizer.

        Args:
            max_depth: Maximum depth to visualize (-1 for unlimited).
            max_width: Maximum line width for text output.
        """
        self.max_depth = max_depth
        self.max_width = max_width

    def visualize(self, tree: ThoughtTree) -> str:
        """Generate a text-based visualization of the thought tree.

        Args:
            tree: The thought tree to visualize.

        Returns:
            Multi-line string representation of the tree.
        """
        lines: List[str] = []
        lines.append(f"Thought Tree (size={tree.size}, depth={tree.max_depth})")
        lines.append("=" * 60)
        self._visualize_node(tree.root, "", True, lines)
        return "\n".join(lines)

    def _visualize_node(
        self,
        node: ThoughtNode,
        prefix: str,
        is_last: bool,
        lines: List[str],
    ) -> None:
        """Recursively visualize a node and its children.

        Args:
            node: The current node to visualize.
            prefix: Current indentation prefix.
            is_last: Whether this node is the last child of its parent.
            lines: Output line accumulator.
        """
        if self.max_depth >= 0 and node.depth > self.max_depth:
            return

        connector = self.LAST_BRANCH if is_last else self.BRANCH
        status_icon = self._status_icon(node)
        score_str = f"[{node.score.overall:.2f}]"
        visits_str = f"(v={node.visits})" if node.visits > 0 else ""

        content = node.content[:self.max_width - 40] if node.content else "(empty)"
        line = f"{prefix}{connector}{status_icon} {score_str} {visits_str} {content}"
        lines.append(line)

        if node.is_solution:
            lines.append(f"{prefix}{'    ' if is_last else self.VERTICAL}    ★ Solution: {node.solution_text[:80]}")

        extension = self.CONTINUE if is_last else self.VERTICAL
        for i, child in enumerate(node.children):
            is_child_last = (i == len(node.children) - 1)
            self._visualize_node(child, prefix + extension, is_child_last, lines)

    def _status_icon(self, node: ThoughtNode) -> str:
        """Get a status icon for a node.

        Args:
            node: The thought node.

        Returns:
            Unicode icon representing node status.
        """
        status_icons = {
            ThoughtStatus.ACTIVE: "○",
            ThoughtStatus.EXPANDING: "◉",
            ThoughtStatus.EXPANDED: "●",
            ThoughtStatus.TERMINAL: "★",
            ThoughtStatus.PRUNED: "✕",
            ThoughtStatus.VISITING: "◐",
        }
        return status_icons.get(node.status, "○")

    def visualize_best_path(self, tree: ThoughtTree) -> str:
        """Visualize the best reasoning path through the tree.

        Args:
            tree: The thought tree.

        Returns:
            Formatted string showing the best path.
        """
        path = tree.get_best_path()
        lines: List[str] = []
        lines.append("Best Reasoning Path")
        lines.append("-" * 60)
        total_score = 0.0

        for i, node in enumerate(path):
            arrow = "→" if i < len(path) - 1 else "✓"
            content = node.content[:80] if node.content else "(empty)"
            lines.append(f"  Step {i}: {arrow} [{node.score.overall:.2f}] {content}")
            total_score += node.score.overall

        avg_score = total_score / len(path) if path else 0.0
        lines.append(f"\n  Average path score: {avg_score:.3f}")
        lines.append(f"  Path length: {len(path)} nodes")
        return "\n".join(lines)

    def visualize_depth_map(self, tree: ThoughtTree) -> str:
        """Visualize the tree as a depth-level map.

        Args:
            tree: The thought tree.

        Returns:
            Formatted string showing nodes at each depth level.
        """
        lines: List[str] = []
        lines.append("Depth Map")
        lines.append("-" * 60)

        for depth in range(tree.max_depth + 1):
            nodes_at_depth = tree.get_nodes_at_depth(depth)
            if not nodes_at_depth:
                continue
            scores = [n.score.overall for n in nodes_at_depth]
            avg = statistics.mean(scores)
            best = max(scores)
            content_snippets = [n.content[:30] + "..." if len(n.content) > 30 else n.content for n in nodes_at_depth[:5]]
            lines.append(f"  Depth {depth}: {len(nodes_at_depth)} nodes, avg={avg:.2f}, best={best:.2f}")
            for snippet in content_snippets:
                lines.append(f"    - {snippet}")

        return "\n".join(lines)

    def generate_mermaid(self, tree: ThoughtTree) -> str:
        """Generate a Mermaid.js diagram definition for the tree.

        Args:
            tree: The thought tree.

        Returns:
            Mermaid.js graph definition string.
        """
        lines: List[str] = ["graph TD"]

        for node_id, node in tree._nodes.items():
            content = node.content[:30].replace('"', "'")
            color = "green" if node.is_solution else "lightblue"
            lines.append(f'  N{node_id}["{content}"]')

            lines.append(f'  style N{node_id} fill:{color},stroke:#333,stroke-width:1px')

        for node_id, node in tree._nodes.items():
            if node.parent is not None:
                parent_id = node.parent.node_id
                score = node.score.overall
                lines.append(f'  N{parent_id} -->|{score:.2f}| N{node_id}')

        return "\n".join(lines)


# =============================================================================
# ToTReasoner - Main Tree-of-Thought Engine
# =============================================================================

class ToTReasoner:
    """Tree-of-thought reasoning engine.

    Manages the complete ToT reasoning process including thought generation,
    evaluation, selection, expansion, and search.

    Attributes:
        config: TreeOfThoughtConfig instance.
        model: Language model interface.
        evaluator: Thought quality evaluator.
        visualizer: Tree visualization utility.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[Any] = None,
        evaluator: Optional[ThoughtEvaluator] = None,
    ) -> None:
        """Initialize the ToT reasoner.

        Args:
            config: TreeOfThoughtConfig instance.
            model: Language model interface.
            evaluator: Optional custom thought evaluator.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import TreeOfThoughtConfig
            config = TreeOfThoughtConfig()
        self.config = config
        self.model = model or self._default_model()
        self.evaluator = evaluator or ThoughtEvaluator()
        self.visualizer = TreeVisualizer()
        self._tree: Optional[ThoughtTree] = None
        self._total_thoughts_generated: int = 0
        self._total_evaluations: int = 0

    def _default_model(self) -> Any:
        """Create a default mock model.

        Returns:
            A mock model interface instance.
        """
        from nexus.reasoning.chain_of_thought import MockModel
        return MockModel()

    def generate_thoughts(
        self,
        state: str,
        num_thoughts: Optional[int] = None,
    ) -> List[str]:
        """Generate multiple candidate thoughts from the current state.

        Uses the language model to generate diverse reasoning thoughts
        that could follow from the current state.

        Args:
            state: Current reasoning state/context.
            num_thoughts: Number of thoughts to generate.

        Returns:
            List of candidate thought strings.
        """
        n = num_thoughts or self.config.num_thoughts
        prompt = self._build_generation_prompt(state, n)

        response = self.model.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.breadth_limit * 200,
        )

        thoughts = self._parse_generated_thoughts(response)
        thoughts = thoughts[:n]

        self._total_thoughts_generated += len(thoughts)
        return thoughts

    def _build_generation_prompt(self, state: str, num_thoughts: int) -> str:
        """Build the prompt for thought generation.

        Args:
            state: Current reasoning state.
            num_thoughts: Number of thoughts to generate.

        Returns:
            The constructed prompt string.
        """
        return (
            f"Given the following reasoning state:\n{state}\n\n"
            f"Generate {num_thoughts} distinct, creative reasoning thoughts "
            f"that could logically follow. Each thought should explore a "
            f"different approach or perspective.\n\n"
            f"Format your response with numbered thoughts:"
        )

    def _parse_generated_thoughts(self, response: str) -> List[str]:
        """Parse generated thoughts from model response.

        Extracts individual thoughts from the model's output, handling
        various formatting patterns.

        Args:
            response: Raw model output.

        Returns:
            List of individual thought strings.
        """
        thoughts: List[str] = []

        number_pattern = r'(?:^|\n)\s*(?:\d+[\.\)]|thought\s*\d+[:.])\s*(.+?)(?=(?:\n\s*(?:\d+[\.\)]|thought\s*\d+[:.]))|$)'
        matches = re.findall(number_pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            for match in matches:
                thought = match.strip()
                if len(thought) >= 10:
                    thoughts.append(thought)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 10:
                    thoughts.append(sentence)

        return thoughts

    def evaluate_thoughts(
        self,
        thoughts: List[str],
        state: str,
        goal: str = "",
    ) -> List[ThoughtScore]:
        """Evaluate a list of candidate thoughts.

        Args:
            thoughts: List of candidate thought strings.
            state: Current reasoning state.
            goal: The overall goal.

        Returns:
            List of ThoughtScore objects, one per thought.
        """
        scores: List[ThoughtScore] = []
        for thought in thoughts:
            score = self.evaluator.evaluate(
                thought=thought,
                context=state,
                goal=goal,
                sibling_thoughts=thoughts,
            )
            scores.append(score)
            self._total_evaluations += 1

        return scores

    def select_thought(
        self,
        thoughts: List[str],
        scores: List[ThoughtScore],
        tree: Optional[ThoughtTree] = None,
    ) -> int:
        """Select the best thought from candidates based on scores.

        Args:
            thoughts: List of candidate thoughts.
            scores: List of corresponding scores.
            tree: Optional tree for UCB computation.

        Returns:
            Index of the selected thought.
        """
        if not thoughts or not scores:
            return 0
        if len(thoughts) != len(scores):
            return 0

        strategy = self.config.selection_strategy

        if strategy.value == "greedy":
            return self._select_greedy(scores)

        if strategy.value == "ucb" and tree is not None:
            return self._select_ucb(scores, tree)

        if strategy.value == "softmax":
            return self._select_softmax(scores)

        if strategy.value == "epsilon_greedy":
            return self._select_epsilon_greedy(scores)

        return self._select_greedy(scores)

    def _select_greedy(self, scores: List[ThoughtScore]) -> int:
        """Select the thought with the highest score.

        Args:
            scores: List of thought scores.

        Returns:
            Index of the best-scoring thought.
        """
        return max(range(len(scores)), key=lambda i: scores[i].overall)

    def _select_ucb(
        self,
        scores: List[ThoughtScore],
        tree: ThoughtTree,
    ) -> int:
        """Select using Upper Confidence Bound.

        Args:
            scores: List of thought scores.
            tree: The thought tree for parent visit information.

        Returns:
            Index of the selected thought.
        """
        parent_visits = tree.root.visits
        best_idx = 0
        best_ucb = float("-inf")

        for i, score in enumerate(scores):
            node_visits = max(1, int(score.confidence * 10))
            ucb = self.config.compute_ucb_score(
                score.overall, node_visits, max(parent_visits, 1)
            )
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        return best_idx

    def _select_softmax(self, scores: List[ThoughtScore]) -> int:
        """Select a thought using softmax probability distribution.

        Args:
            scores: List of thought scores.

        Returns:
            Index of the selected thought.
        """
        temperature = max(0.1, self.config.temperature)
        exp_scores = [math.exp(s.overall / temperature) for s in scores]
        total = sum(exp_scores)
        if total == 0:
            return 0
        probabilities = [e / total for e in exp_scores]

        rand = random.random()
        cumulative = 0.0
        for i, prob in enumerate(probabilities):
            cumulative += prob
            if rand <= cumulative:
                return i
        return len(probabilities) - 1

    def _select_epsilon_greedy(self, scores: List[ThoughtScore]) -> int:
        """Select using epsilon-greedy strategy.

        Args:
            scores: List of thought scores.

        Returns:
            Index of the selected thought.
        """
        epsilon = 0.1
        if random.random() < epsilon:
            return random.randint(0, len(scores) - 1)
        return self._select_greedy(scores)

    def expand(
        self,
        node: ThoughtNode,
        num_children: Optional[int] = None,
        goal: str = "",
    ) -> List[ThoughtNode]:
        """Expand a node by generating child thoughts.

        Args:
            node: The node to expand.
            num_children: Number of children to generate.
            goal: The overall goal.

        Returns:
            List of newly created child nodes.
        """
        n = num_children or self.config.breadth_limit
        if node.depth >= self.config.depth_limit:
            return []

        state = node.path_content
        thoughts = self.generate_thoughts(state, n)
        if not thoughts:
            return []

        scores = self.evaluate_thoughts(thoughts, state, goal)

        new_children: List[ThoughtNode] = []
        for thought, score in zip(thoughts, scores):
            if self.config.should_prune(score.overall, [s.overall for s in scores]):
                continue
            child = node.add_child(content=thought, score=score)
            child.metadata["raw_thought"] = thought
            new_children.append(child)

        node.status = ThoughtStatus.EXPANDED
        return new_children

    def simulate(self, node: ThoughtNode, depth: int = 5) -> float:
        """Simulate a random rollout from a node to estimate its value.

        Args:
            node: The node to simulate from.
            depth: Number of simulation steps.

        Returns:
            Estimated value of the node.
        """
        current_content = node.content
        total_value = node.score.overall

        for step in range(depth):
            thoughts = self.generate_thoughts(current_content, 2)
            if not thoughts:
                break

            scores = self.evaluate_thoughts(thoughts, current_content)
            best_idx = self._select_greedy(scores)

            total_value += scores[best_idx].overall * self.config.effective_discount(step + 1)
            current_content = f"{current_content} -> {thoughts[best_idx]}"

        return total_value / (depth + 1)

    def backpropagate(self, node: ThoughtNode, value: float) -> None:
        """Backpropagate a value up the tree to all ancestors.

        Updates visit counts and accumulated values for all nodes
        along the path from the given node to the root.

        Args:
            node: The node to start backpropagation from.
            value: The value to backpropagate.
        """
        current: Optional[ThoughtNode] = node
        depth = 0
        while current is not None:
            discount = self.config.effective_discount(depth)
            current.visits += 1
            current.value += value * discount
            current = current.parent
            depth += 1

    def best_path(self, tree: ThoughtTree) -> List[ThoughtNode]:
        """Extract the best reasoning path from root to leaf.

        Selects the best child at each level based on average value.

        Args:
            tree: The thought tree.

        Returns:
            List of nodes forming the best path.
        """
        path: List[ThoughtNode] = []
        current = tree.root
        path.append(current)

        while current.children:
            best_child = max(current.children, key=lambda c: c.average_value)
            path.append(best_child)
            current = best_child

        return path

    def mcts_search(
        self,
        initial_state: str,
        goal: str = "",
        num_simulations: Optional[int] = None,
    ) -> ThoughtTree:
        """Perform Monte Carlo Tree Search for reasoning.

        The MCTS algorithm consists of four phases repeated for each simulation:
        1. Selection: Traverse the tree using UCB to select the most promising node.
        2. Expansion: Expand the selected node with new child thoughts.
        3. Simulation: Perform a random rollout from the new node.
        4. Backpropagation: Update all nodes on the path with the rollout result.

        Args:
            initial_state: The initial problem state or question.
            goal: The overall goal or correct answer.
            num_simulations: Number of MCTS simulations to run.

        Returns:
            The populated ThoughtTree with search results.
        """
        n_sims = num_simulations or self.config.num_simulations
        self._tree = ThoughtTree(root_content=initial_state)
        root_score = self.evaluator.evaluate(initial_state, goal=goal)
        self._tree.root.score = root_score
        self._tree.root.visits = 1

        for sim in range(n_sims):
            node = self._select_node_mcts(self._tree.root)
            if node.depth < self.config.depth_limit:
                children = self.expand(node, goal=goal)
                if children:
                    node_to_evaluate = children[0]
                else:
                    node_to_evaluate = node
            else:
                node_to_evaluate = node

            value = self.simulate(
                node_to_evaluate,
                depth=min(self.config.rollout_depth, self.config.depth_limit - node_to_evaluate.depth),
            )

            if self._is_solution(node_to_evaluate, goal):
                node_to_evaluate.is_solution = True
                node_to_evaluate.solution_text = node_to_evaluate.content
                value = 1.0

            self.backpropagate(node_to_evaluate, value)

            if self._tree.size >= self.config.max_tree_size:
                break

        self._prune_tree()
        return self._tree

    def _select_node_mcts(self, root: ThoughtNode) -> ThoughtNode:
        """Select the most promising node using UCB.

        Traverses the tree from root, selecting the child with the
        highest UCB score at each level until a leaf is reached.

        Args:
            root: The root node of the tree.

        Returns:
            The selected node for expansion.
        """
        current = root
        while current.children and current.status == ThoughtStatus.EXPANDED:
            best_child = None
            best_ucb = float("-inf")

            for child in current.children:
                ucb = self.config.compute_ucb_score(
                    child.average_value,
                    child.visits,
                    current.visits,
                )
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            if best_child is None:
                break
            current = best_child

        return current

    def beam_search(
        self,
        initial_state: str,
        goal: str = "",
        beam_width: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> ThoughtTree:
        """Perform beam search for reasoning.

        Maintains a fixed-width beam of the best candidates at each
        depth level, expanding and pruning to stay within the beam width.

        Args:
            initial_state: The initial problem state.
            goal: The overall goal.
            beam_width: Width of the beam.
            depth: Maximum search depth.

        Returns:
            The populated ThoughtTree.
        """
        width = beam_width or self.config.beam_width
        max_depth = depth or self.config.depth_limit
        self._tree = ThoughtTree(root_content=initial_state)
        root_score = self.evaluator.evaluate(initial_state, goal=goal)
        self._tree.root.score = root_score

        current_beam: List[ThoughtNode] = [self._tree.root]

        for level in range(max_depth):
            all_candidates: List[ThoughtNode] = []

            for node in current_beam:
                children = self.expand(node, goal=goal)
                all_candidates.extend(children)

            if not all_candidates:
                break

            all_candidates.sort(key=lambda n: n.score.overall, reverse=True)
            current_beam = all_candidates[:width]

            for node in all_candidates[width:]:
                node.status = ThoughtStatus.PRUNED

            if any(n.is_solution for n in current_beam):
                break

            if self._tree.size >= self.config.max_tree_size:
                break

        return self._tree

    def bfs_explore(
        self,
        initial_state: str,
        goal: str = "",
        breadth: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> ThoughtTree:
        """Perform breadth-first search exploration.

        Explores all nodes at each depth level before moving to the next,
        pruning to stay within the breadth limit.

        Args:
            initial_state: The initial problem state.
            goal: The overall goal.
            breadth: Maximum breadth at each level.
            depth: Maximum search depth.

        Returns:
            The populated ThoughtTree.
        """
        b = breadth or self.config.breadth_limit
        max_depth = depth or self.config.depth_limit
        self._tree = ThoughtTree(root_content=initial_state)
        root_score = self.evaluator.evaluate(initial_state, goal=goal)
        self._tree.root.score = root_score

        current_level: List[ThoughtNode] = [self._tree.root]

        for level in range(max_depth):
            next_level: List[ThoughtNode] = []

            for node in current_level:
                children = self.expand(node, goal=goal)
                next_level.extend(children)

            if not next_level:
                break

            next_level.sort(key=lambda n: n.score.overall, reverse=True)
            pruned = next_level[b:]
            for node in pruned:
                node.status = ThoughtStatus.PRUNED
            current_level = next_level[:b]

            if any(n.is_solution for n in current_level):
                break

            if self._tree.size >= self.config.max_tree_size:
                break

        return self._tree

    def dfs_explore(
        self,
        initial_state: str,
        goal: str = "",
        depth_limit: Optional[int] = None,
    ) -> ThoughtTree:
        """Perform depth-first search with backtracking.

        Explores reasoning paths deeply before backtracking,
        pursuing the most promising path first.

        Args:
            initial_state: The initial problem state.
            goal: The overall goal.
            depth_limit: Maximum search depth.

        Returns:
            The populated ThoughtTree.
        """
        max_depth = depth_limit or self.config.depth_limit
        self._tree = ThoughtTree(root_content=initial_state)
        root_score = self.evaluator.evaluate(initial_state, goal=goal)
        self._tree.root.score = root_score

        self._dfs_recursive(self._tree.root, goal, max_depth, 0)
        return self._tree

    def _dfs_recursive(
        self,
        node: ThoughtNode,
        goal: str,
        max_depth: int,
        depth: int,
    ) -> bool:
        """Recursive DFS implementation with backtracking.

        Args:
            node: Current node being explored.
            goal: The overall goal.
            max_depth: Maximum allowed depth.
            depth: Current depth.

        Returns:
            True if a solution was found.
        """
        if depth >= max_depth:
            return False

        if self._is_solution(node, goal):
            node.is_solution = True
            node.solution_text = node.content
            return True

        children = self.expand(node, goal=goal)
        children.sort(key=lambda n: n.score.overall, reverse=True)

        for child in children:
            found = self._dfs_recursive(child, goal, max_depth, depth + 1)
            if found:
                return True

            if self._tree.size >= self.config.max_tree_size:
                return False

        return False

    def _is_solution(self, node: ThoughtNode, goal: str) -> bool:
        """Check if a node represents a solution to the goal.

        Evaluates whether the node's content constitutes a valid
        answer to the problem.

        Args:
            node: The thought node to check.
            goal: The goal or question.

        Returns:
            True if the node appears to be a solution.
        """
        if node.score.progress > 0.8 and node.score.overall > 0.7:
            return True

        conclusion_markers = [
            "the answer is", "therefore", "in conclusion",
            "the result is", "we conclude", "final answer",
        ]
        content_lower = node.content.lower()
        has_conclusion = any(m in content_lower for m in conclusion_markers)

        if has_conclusion and node.score.relevance > 0.6:
            return True

        if goal and node.score.relevance > 0.9:
            return True

        return False

    def _prune_tree(self) -> None:
        """Apply pruning strategies to the tree."""
        if self._tree is None:
            return

        self._tree.prune_low_visit(1)

        if self._tree.size > self.config.max_tree_size * 0.8:
            leaf_nodes = [n for n in self._tree._nodes.values() if n.is_leaf and not n.is_solution]
            leaf_nodes.sort(key=lambda n: n.score.overall)
            to_remove = int(len(leaf_nodes) * 0.3)
            for node in leaf_nodes[:to_remove]:
                self._tree.remove_node(node.node_id)

    def reason(
        self,
        question: str,
        search_algorithm: Optional[str] = None,
    ) -> Tuple[ThoughtTree, str]:
        """Perform tree-of-thought reasoning on a question.

        Uses the configured search algorithm to explore reasoning
        paths and return the best solution.

        Args:
            question: The question or problem to reason about.
            search_algorithm: Optional algorithm override.

        Returns:
            Tuple of (thought tree, best solution string).
        """
        algorithm = search_algorithm or self.config.search_algorithm

        if algorithm == "mcts":
            tree = self.mcts_search(question, goal=question)
        elif algorithm == "beam":
            tree = self.beam_search(question, goal=question)
        elif algorithm == "bfs":
            tree = self.bfs_explore(question, goal=question)
        elif algorithm == "dfs":
            tree = self.dfs_explore(question, goal=question)
        else:
            tree = self.mcts_search(question, goal=question)

        best_path = self.best_path(tree)
        if best_path:
            solution = best_path[-1].content
        else:
            solution = tree.root.content

        return tree, solution

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the reasoner's performance.

        Returns:
            Dictionary of performance statistics.
        """
        tree_stats = self._tree.get_statistics() if self._tree else {}
        return {
            "total_thoughts_generated": self._total_thoughts_generated,
            "total_evaluations": self._total_evaluations,
            "search_algorithm": self.config.search_algorithm,
            "tree_statistics": tree_stats,
        }

    def reset(self) -> None:
        """Reset the reasoner's state."""
        self._tree = None
        self._total_thoughts_generated = 0
        self._total_evaluations = 0
