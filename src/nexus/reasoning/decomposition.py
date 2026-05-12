"""
Task Decomposition Module
==========================

Implements various strategies for decomposing complex tasks into simpler
sub-tasks. Decomposition is a fundamental capability for multi-step reasoning,
enabling the system to handle complex goals by breaking them into manageable parts.

Decomposition Strategies:
- SequentialDecomposer: Ordered sequential steps
- ParallelDecomposer: Independent sub-tasks for parallel execution
- HierarchicalDecomposer: Recursive decomposition into task trees
- TemplateDecomposer: Template-based decomposition for common patterns

Each decomposer produces a DecomposedTask structure that can be scored for quality
using the DecompositionScorer.
"""

from __future__ import annotations

import math
import re
import copy
import json
import time
import hashlib
import logging
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TaskNode:
    """A node in the task decomposition tree.

    Represents either the root task or a sub-task with its children.

    Attributes:
        description: Human-readable description of the task.
        task_id: Unique identifier for this task node.
        parent_id: ID of the parent task (None for root).
        children: List of child TaskNodes.
        depth: Depth in the decomposition tree (root = 0).
        is_atomic: Whether this task is atomic (cannot be further decomposed).
        estimated_effort: Estimated effort to complete this task (1-10).
        dependencies: Set of task IDs that must complete before this task.
        metadata: Additional metadata.
        priority: Execution priority (higher = more important).
        task_type: Category of the task (e.g., 'analysis', 'computation', 'retrieval').
    """

    def __init__(
        self,
        description: str = "",
        task_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        depth: int = 0,
    ) -> None:
        self.description = description
        self.task_id = task_id or hashlib.md5(
            f"{description}_{time.time()}".encode()
        ).hexdigest()[:10]
        self.parent_id = parent_id
        self.children: List[TaskNode] = []
        self.depth = depth
        self.is_atomic: bool = False
        self.estimated_effort: int = 5
        self.dependencies: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        self.priority: float = 0.5
        self.task_type: str = "general"

    def add_child(self, description: str) -> TaskNode:
        """Add a child sub-task to this task node.

        Args:
            description: Description of the child task.

        Returns:
            The newly created child TaskNode.
        """
        child = TaskNode(
            description=description,
            parent_id=self.task_id,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child

    def get_all_descendants(self) -> List[TaskNode]:
        """Get all descendant task nodes.

        Returns:
            Flat list of all descendant nodes (not including self).
        """
        descendants: List[TaskNode] = []
        queue = deque(self.children)
        while queue:
            node = queue.popleft()
            descendants.append(node)
            queue.extend(node.children)
        return descendants

    def get_leaf_tasks(self) -> List[TaskNode]:
        """Get all leaf (atomic) task nodes.

        Returns:
            List of leaf nodes.
        """
        leaves: List[TaskNode] = []
        if not self.children:
            leaves.append(self)
        else:
            for child in self.children:
                leaves.extend(child.get_leaf_tasks())
        return leaves

    def count_leaves(self) -> int:
        """Count the number of leaf tasks.

        Returns:
            Number of leaf nodes.
        """
        return len(self.get_leaf_tasks())

    def total_tasks(self) -> int:
        """Get the total number of tasks in this subtree.

        Returns:
            Total task count including self and all descendants.
        """
        return 1 + sum(child.total_tasks() for child in self.children)

    def max_depth(self) -> int:
        """Get the maximum depth of this subtree.

        Returns:
            Maximum depth value.
        """
        if not self.children:
            return self.depth
        return max(child.max_depth() for child in self.children)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "task_id": self.task_id,
            "description": self.description,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "is_atomic": self.is_atomic,
            "estimated_effort": self.estimated_effort,
            "dependencies": list(self.dependencies),
            "priority": self.priority,
            "task_type": self.task_type,
            "num_children": len(self.children),
            "children": [c.to_dict() for c in self.children],
        }

    def clone(self) -> TaskNode:
        """Create a deep copy of this task node and its subtree.

        Returns:
            A new TaskNode that is a deep copy.
        """
        new_node = TaskNode(
            description=self.description,
            parent_id=self.parent_id,
            depth=self.depth,
        )
        new_node.task_id = self.task_id
        new_node.is_atomic = self.is_atomic
        new_node.estimated_effort = self.estimated_effort
        new_node.dependencies = set(self.dependencies)
        new_node.metadata = copy.deepcopy(self.metadata)
        new_node.priority = self.priority
        new_node.task_type = self.task_type
        for child in self.children:
            child_clone = child.clone()
            new_node.children.append(child_clone)
        return new_node

    def __repr__(self) -> str:
        return f"TaskNode(id={self.task_id}, depth={self.depth}, children={len(self.children)})"


@dataclass
class DecomposedTask:
    """Result of a task decomposition.

    Contains the root task node and metadata about the decomposition.

    Attributes:
        root: Root of the task decomposition tree.
        strategy: Name of the decomposition strategy used.
        original_task: The original task description.
        num_subtasks: Number of direct sub-tasks.
        total_tasks: Total tasks in the decomposition tree.
        max_depth: Maximum depth of the decomposition.
        quality_score: Overall quality score of the decomposition.
        metadata: Additional metadata.
    """

    def __init__(
        self,
        root: Optional[TaskNode] = None,
        strategy: str = "unknown",
        original_task: str = "",
    ) -> None:
        self.root = root or TaskNode()
        self.strategy = strategy
        self.original_task = original_task
        self.num_subtasks: int = len(self.root.children)
        self.total_tasks: int = self.root.total_tasks()
        self.max_depth: int = self.root.max_depth()
        self.quality_score: float = 0.5
        self.metadata: Dict[str, Any] = {}

    def get_execution_order(self) -> List[TaskNode]:
        """Get the recommended execution order for all tasks.

        Uses topological sort based on dependencies.

        Returns:
            List of TaskNodes in execution order.
        """
        all_tasks = [self.root] + self.root.get_all_descendants()
        task_map: Dict[str, TaskNode] = {t.task_id: t for t in all_tasks}

        in_degree: Dict[str, int] = {t.task_id: 0 for t in all_tasks}
        for t in all_tasks:
            for dep in t.dependencies:
                if dep in in_degree:
                    in_degree[t.task_id] += 1

        queue: deque = deque()
        for tid, deg in in_degree.items():
            if deg == 0:
                queue.append(tid)

        order: List[TaskNode] = []
        while queue:
            tid = queue.popleft()
            if tid in task_map:
                order.append(task_map[tid])
            for t in all_tasks:
                if tid in t.dependencies:
                    in_degree[t.task_id] -= 1
                    if in_degree[t.task_id] == 0:
                        queue.append(t.task_id)

        for t in all_tasks:
            if t not in order:
                order.append(t)

        return order

    def get_flat_subtasks(self) -> List[TaskNode]:
        """Get a flat list of all sub-tasks (excluding root).

        Returns:
            List of all sub-task nodes.
        """
        return self.root.get_all_descendants()

    def get_atomic_tasks(self) -> List[TaskNode]:
        """Get all atomic (leaf) tasks.

        Returns:
            List of atomic task nodes.
        """
        return self.root.get_leaf_tasks()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "strategy": self.strategy,
            "original_task": self.original_task,
            "num_subtasks": self.num_subtasks,
            "total_tasks": self.total_tasks,
            "max_depth": self.max_depth,
            "quality_score": self.quality_score,
            "root": self.root.to_dict(),
        }

    def clone(self) -> DecomposedTask:
        """Create a deep copy of this decomposition.

        Returns:
            A new DecomposedTask with identical data.
        """
        new_dt = DecomposedTask(
            root=self.root.clone(),
            strategy=self.strategy,
            original_task=self.original_task,
        )
        new_dt.quality_score = self.quality_score
        new_dt.metadata = copy.deepcopy(self.metadata)
        return new_dt


@dataclass
class DecompositionQuality:
    """Quality assessment of a task decomposition.

    Attributes:
        completeness: How completely the decomposition covers the original task.
        atomicity: How atomic (fine-grained) the sub-tasks are.
        ordering: How well-ordered the sub-tasks are.
        clarity: How clear and understandable the sub-tasks are.
        redundancy: How much redundancy exists (lower is better).
        overall: Overall quality score.
    """
    completeness: float = 0.5
    atomicity: float = 0.5
    ordering: float = 0.5
    clarity: float = 0.5
    redundancy: float = 0.0

    @property
    def overall(self) -> float:
        """Compute overall quality score.

        Returns:
            Weighted average of quality dimensions.
        """
        return (
            self.completeness * 0.30
            + self.atomicity * 0.25
            + self.ordering * 0.20
            + self.clarity * 0.15
            + (1.0 - self.redundancy) * 0.10
        )

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a dictionary.

        Returns:
            Dictionary of quality scores.
        """
        return {
            "completeness": self.completeness,
            "atomicity": self.atomicity,
            "ordering": self.ordering,
            "clarity": self.clarity,
            "redundancy": self.redundancy,
            "overall": self.overall,
        }


# =============================================================================
# Base Task Decomposer
# =============================================================================

class TaskDecomposer(ABC):
    """Abstract base class for task decomposers.

    All decomposition strategies inherit from this class and implement
    the decompose method.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        """Initialize the task decomposer.

        Args:
            model: Optional language model interface.
        """
        self.model = model or self._default_model()
        self._decomposition_history: List[DecomposedTask] = []

    def _default_model(self) -> Any:
        """Create a default mock model.

        Returns:
            A mock model interface.
        """
        from nexus.reasoning.chain_of_thought import MockModel
        return MockModel()

    @abstractmethod
    def decompose(self, task: str, **kwargs: Any) -> DecomposedTask:
        """Decompose a task into sub-tasks.

        Args:
            task: The task description to decompose.
            **kwargs: Additional strategy-specific parameters.

        Returns:
            A DecomposedTask with the decomposition result.
        """
        raise NotImplementedError

    def _generate_subtasks(self, task: str, prompt: str) -> List[str]:
        """Use the model to generate sub-task descriptions.

        Args:
            task: The original task.
            prompt: The decomposition prompt.

        Returns:
            List of sub-task description strings.
        """
        response = self.model.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
        )
        return self._parse_subtask_list(response)

    def _parse_subtask_list(self, response: str) -> List[str]:
        """Parse a list of sub-tasks from model response.

        Handles various formatting patterns like numbered lists,
        bullet points, and plain text.

        Args:
            response: Raw model output.

        Returns:
            List of sub-task description strings.
        """
        subtasks: List[str] = []
        lines = response.strip().split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            number_match = re.match(
                r'^\s*(?:\d+[\.\)]|step\s*\d+[:.\)]?|sub-?task\s*\d+[:.\)]?)\s*(.+)',
                stripped, re.IGNORECASE,
            )
            bullet_match = re.match(r'^\s*[-•*]\s*(.+)', stripped)
            alpha_match = re.match(r'^\s*([a-zA-Z])[\.\)]\s*(.+)', stripped)

            if number_match:
                subtasks.append(number_match.group(1).strip())
            elif bullet_match:
                subtasks.append(bullet_match.group(1).strip())
            elif alpha_match:
                subtasks.append(alpha_match.group(2).strip())
            elif len(stripped) > 10 and stripped[0].isupper():
                if subtasks and len(stripped) < 200:
                    subtasks.append(stripped)

        return [s for s in subtasks if len(s) >= 5]


# =============================================================================
# Sequential Decomposer
# =============================================================================

class SequentialDecomposer(TaskDecomposer):
    """Breaks a task into ordered sequential steps.

    Each sub-task must be completed before the next one begins.
    Useful for tasks with clear sequential dependencies like
    recipes, algorithms, or multi-step procedures.
    """

    def __init__(self, model: Optional[Any] = None, max_steps: int = 10) -> None:
        """Initialize the sequential decomposer.

        Args:
            model: Language model interface.
            max_steps: Maximum number of sequential steps.
        """
        super().__init__(model=model)
        self.max_steps = max_steps

    def decompose(self, task: str, **kwargs: Any) -> DecomposedTask:
        """Decompose a task into sequential steps.

        Args:
            task: The task to decompose.
            **kwargs: Optional 'max_steps' override.

        Returns:
            A DecomposedTask with sequentially ordered sub-tasks.
        """
        max_s = kwargs.get("max_steps", self.max_steps)
        prompt = (
            f"Break down the following task into sequential steps. "
            f"Each step should be a clear, discrete action that must be "
            f"completed before the next step can begin. Maximum {max_s} steps.\n\n"
            f"Task: {task}\n\n"
            f"List the steps in order:"
        )

        subtask_descriptions = self._generate_subtasks(task, prompt)
        subtask_descriptions = subtask_descriptions[:max_s]

        root = TaskNode(description=task, depth=0)
        prev_id: Optional[str] = None

        for i, desc in enumerate(subtask_descriptions):
            child = root.add_child(desc)
            child.depth = 1
            child.is_atomic = True
            child.estimated_effort = self._estimate_effort(desc)
            child.priority = 1.0 - (i * 0.05)
            child.task_type = self._classify_task_type(desc)

            if prev_id is not None:
                child.dependencies.add(prev_id)
            prev_id = child.task_id

        decomposed = DecomposedTask(
            root=root,
            strategy="sequential",
            original_task=task,
        )
        self._decomposition_history.append(decomposed)
        return decomposed

    def _estimate_effort(self, description: str) -> int:
        """Estimate effort for a sub-task based on its description.

        Args:
            description: The sub-task description.

        Returns:
            Estimated effort from 1 to 10.
        """
        effort = 3
        complex_keywords = {
            "analyze": 2, "evaluate": 2, "compare": 2,
            "compute": 3, "calculate": 3, "implement": 3,
            "optimize": 3, "search": 2, "verify": 1,
        }
        for keyword, bonus in complex_keywords.items():
            if keyword in description.lower():
                effort += bonus

        effort += min(3, len(description) // 50)
        return min(10, max(1, effort))

    def _classify_task_type(self, description: str) -> str:
        """Classify a task based on its description.

        Args:
            description: The task description.

        Returns:
            Task type string.
        """
        desc_lower = description.lower()
        type_keywords = {
            "computation": ["calculate", "compute", "math", "number", "equation"],
            "analysis": ["analyze", "evaluate", "assess", "compare", "review"],
            "retrieval": ["find", "search", "look up", "retrieve", "gather"],
            "generation": ["write", "create", "generate", "produce", "compose"],
            "verification": ["verify", "check", "validate", "confirm", "test"],
        }
        for task_type, keywords in type_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                return task_type
        return "general"


# =============================================================================
# Parallel Decomposer
# =============================================================================

class ParallelDecomposer(TaskDecomposer):
    """Identifies independent sub-tasks that can be executed in parallel.

    Analyzes the task to find aspects that don't depend on each other,
    enabling concurrent execution for improved efficiency.
    """

    def __init__(self, model: Optional[Any] = None, max_groups: int = 5) -> None:
        """Initialize the parallel decomposer.

        Args:
            model: Language model interface.
            max_groups: Maximum number of parallel groups.
        """
        super().__init__(model=model)
        self.max_groups = max_groups

    def decompose(self, task: str, **kwargs: Any) -> DecomposedTask:
        """Decompose a task into independent parallel sub-tasks.

        Args:
            task: The task to decompose.
            **kwargs: Optional 'max_groups' override.

        Returns:
            A DecomposedTask with independent sub-task groups.
        """
        max_g = kwargs.get("max_groups", self.max_groups)
        prompt = (
            f"Identify independent aspects of the following task that can be "
            f"worked on in parallel (simultaneously). Each aspect should be "
            f"self-contained and not depend on the others. Maximum {max_g} aspects.\n\n"
            f"Task: {task}\n\n"
            f"List the independent aspects:"
        )

        subtask_descriptions = self._generate_subtasks(task, prompt)
        subtask_descriptions = subtask_descriptions[:max_g]

        root = TaskNode(description=task, depth=0)

        for i, desc in enumerate(subtask_descriptions):
            child = root.add_child(desc)
            child.depth = 1
            child.is_atomic = True
            child.estimated_effort = 5
            child.priority = 0.8
            child.task_type = "parallel"

        decomposed = DecomposedTask(
            root=root,
            strategy="parallel",
            original_task=task,
        )
        self._decomposition_history.append(decomposed)
        return decomposed

    def compute_independence_score(
        self,
        task1: str,
        task2: str,
    ) -> float:
        """Compute the independence score between two tasks.

        Higher scores indicate greater independence (suitable for
        parallel execution).

        Args:
            task1: First task description.
            task2: Second task description.

        Returns:
            Independence score between 0.0 and 1.0.
        """
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())

        stop_words = frozenset({
            "the", "a", "an", "is", "are", "to", "of", "in", "for",
            "and", "or", "but", "with", "on", "at", "by",
        })
        content1 = words1 - stop_words
        content2 = words2 - stop_words

        if not content1 or not content2:
            return 0.5

        overlap = len(content1 & content2)
        union = len(content1 | content2)
        jaccard = overlap / union if union > 0 else 0.0

        independence = 1.0 - jaccard

        dependency_keywords = [
            "depends", "requires", "after", "before", "using",
            "based on", "from", "result of",
        ]
        combined = f"{task1} {task2}".lower()
        dep_count = sum(1 for kw in dependency_keywords if kw in combined)
        independence -= dep_count * 0.1

        return max(0.0, min(1.0, independence))


# =============================================================================
# Hierarchical Decomposer
# =============================================================================

class HierarchicalDecomposer(TaskDecomposer):
    """Recursively decomposes tasks into a hierarchy of sub-tasks.

    Starts with a high-level decomposition, then recursively decomposes
    each sub-task until a minimum granularity is reached.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        max_depth: int = 3,
        min_task_size: int = 5,
        max_children: int = 5,
    ) -> None:
        """Initialize the hierarchical decomposer.

        Args:
            model: Language model interface.
            max_depth: Maximum recursion depth.
            min_task_size: Minimum task size to stop decomposing.
            max_children: Maximum children per node.
        """
        super().__init__(model=model)
        self.max_depth = max_depth
        self.min_task_size = min_task_size
        self.max_children = max_children

    def decompose(self, task: str, **kwargs: Any) -> DecomposedTask:
        """Recursively decompose a task into a hierarchy.

        Args:
            task: The task to decompose.
            **kwargs: Optional depth override.

        Returns:
            A DecomposedTask with a hierarchical structure.
        """
        root = self._recursive_decompose(task, depth=0)
        decomposed = DecomposedTask(
            root=root,
            strategy="hierarchical",
            original_task=task,
        )
        self._decomposition_history.append(decomposed)
        return decomposed

    def _recursive_decompose(self, task: str, depth: int) -> TaskNode:
        """Recursively decompose a task.

        Args:
            task: The task to decompose.
            depth: Current recursion depth.

        Returns:
            A TaskNode with potentially recursive children.
        """
        node = TaskNode(description=task, depth=depth)

        if depth >= self.max_depth:
            node.is_atomic = True
            return node

        if len(task.split()) < self.min_task_size * 2:
            node.is_atomic = True
            return node

        prompt = (
            f"Break down the following task into {self.max_children} or fewer "
            f"sub-tasks. Each sub-task should be a meaningful part of the overall task.\n\n"
            f"Task: {task}\n\n"
            f"List the sub-tasks:"
        )

        subtask_descriptions = self._generate_subtasks(task, prompt)
        subtask_descriptions = subtask_descriptions[:self.max_children]

        if not subtask_descriptions:
            node.is_atomic = True
            return node

        if len(subtask_descriptions) == 1 and depth > 0:
            node.is_atomic = True
            return node

        for desc in subtask_descriptions:
            child = self._recursive_decompose(desc, depth + 1)
            child.parent_id = node.task_id
            node.children.append(child)

        return node

    def flatten(self, decomposed: DecomposedTask) -> List[str]:
        """Flatten a hierarchical decomposition into a linear list.

        Args:
            decomposed: The hierarchical decomposition.

        Returns:
            List of task descriptions in depth-first order.
        """
        tasks: List[str] = []

        def traverse(node: TaskNode) -> None:
            if node.children:
                for child in node.children:
                    traverse(child)
            else:
                tasks.append(node.description)

        traverse(decomposed.root)
        return tasks


# =============================================================================
# Template Decomposer
# =============================================================================

class TemplateDecomposer(TaskDecomposer):
    """Uses predefined task templates for common decomposition patterns.

    Maintains a library of templates for common task types like
    mathematical problems, writing tasks, analysis tasks, etc.
    Templates provide fast, consistent decompositions without
    requiring model calls.
    """

    def __init__(self, model: Optional[Any] = None) -> None:
        """Initialize the template decomposer.

        Args:
            model: Language model interface (used for untemplated tasks).
        """
        super().__init__(model=model)
        self._templates = self._build_templates()

    def _build_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build the template library.

        Returns:
            Dictionary of task type templates.
        """
        return {
            "math_problem": {
                "keywords": ["calculate", "compute", "solve", "how many", "math", "equation", "formula"],
                "steps": [
                    "Identify the given quantities and the unknown",
                    "Determine the relevant formula or method",
                    "Substitute the known values",
                    "Perform the calculation",
                    "Verify the result",
                ],
                "task_types": ["retrieval", "computation", "verification"],
            },
            "comparison": {
                "keywords": ["compare", "difference", "similar", "versus", "vs", "contrast", "distinguish"],
                "steps": [
                    "Identify the items or concepts to compare",
                    "List key characteristics of each item",
                    "Identify similarities between the items",
                    "Identify differences between the items",
                    "Synthesize the comparison into conclusions",
                ],
                "task_types": ["analysis", "analysis", "analysis", "analysis", "generation"],
            },
            "explanation": {
                "keywords": ["explain", "why", "how does", "what causes", "reason"],
                "steps": [
                    "Identify the phenomenon or question to explain",
                    "Gather relevant facts and evidence",
                    "Identify the key cause or mechanism",
                    "Explain the process step by step",
                    "Provide examples or illustrations",
                    "Summarize the explanation",
                ],
                "task_types": ["retrieval", "retrieval", "analysis", "generation", "generation", "generation"],
            },
            "writing": {
                "keywords": ["write", "compose", "draft", "create", "essay", "article", "story", "report"],
                "steps": [
                    "Determine the topic, purpose, and audience",
                    "Brainstorm key points and arguments",
                    "Organize ideas into a structure (outline)",
                    "Draft the content section by section",
                    "Review and refine the content",
                    "Finalize the output",
                ],
                "task_types": ["analysis", "generation", "analysis", "generation", "analysis", "generation"],
            },
            "analysis": {
                "keywords": ["analyze", "evaluate", "assess", "examine", "review", "critique"],
                "steps": [
                    "Define the scope and criteria for analysis",
                    "Gather relevant data and information",
                    "Identify patterns and key findings",
                    "Evaluate findings against criteria",
                    "Draw conclusions and recommendations",
                ],
                "task_types": ["analysis", "retrieval", "analysis", "analysis", "generation"],
            },
            "decision": {
                "keywords": ["decide", "choose", "recommend", "select", "which", "best"],
                "steps": [
                    "Define the decision to be made",
                    "Identify the available options",
                    "List criteria for evaluation",
                    "Evaluate each option against criteria",
                    "Compare options and make a recommendation",
                ],
                "task_types": ["analysis", "retrieval", "analysis", "analysis", "generation"],
            },
            "troubleshooting": {
                "keywords": ["fix", "debug", "troubleshoot", "error", "problem", "issue", "broken"],
                "steps": [
                    "Identify and describe the problem",
                    "Gather information about the problem context",
                    "Formulate hypotheses about the cause",
                    "Test each hypothesis systematically",
                    "Implement the solution",
                    "Verify the fix works",
                ],
                "task_types": ["analysis", "retrieval", "analysis", "computation", "generation", "verification"],
            },
            "classification": {
                "keywords": ["classify", "categorize", "group", "sort", "organize", "type"],
                "steps": [
                    "Define the classification criteria",
                    "Examine each item to be classified",
                    "Compare items against classification criteria",
                    "Assign items to appropriate categories",
                    "Verify the classification is consistent",
                ],
                "task_types": ["analysis", "retrieval", "analysis", "generation", "verification"],
            },
        }

    def add_template(
        self,
        name: str,
        keywords: List[str],
        steps: List[str],
        task_types: Optional[List[str]] = None,
    ) -> None:
        """Add a new decomposition template.

        Args:
            name: Name for the template.
            keywords: Keywords that trigger this template.
            steps: Ordered list of decomposition steps.
            task_types: Task types for each step.
        """
        self._templates[name] = {
            "keywords": keywords,
            "steps": steps,
            "task_types": task_types or ["general"] * len(steps),
        }

    def decompose(self, task: str, **kwargs: Any) -> DecomposedTask:
        """Decompose a task using the best matching template.

        Falls back to sequential decomposition if no template matches.

        Args:
            task: The task to decompose.
            **kwargs: Optional 'force_template' to use a specific template.

        Returns:
            A DecomposedTask with template-based decomposition.
        """
        force_template = kwargs.get("force_template")
        if force_template and force_template in self._templates:
            template_name = force_template
        else:
            template_name = self._find_best_template(task)

        if template_name:
            return self._apply_template(task, template_name)
        else:
            sequential = SequentialDecomposer(model=self.model)
            result = sequential.decompose(task)
            result.strategy = "template_fallback"
            return result

    def _find_best_template(self, task: str) -> Optional[str]:
        """Find the best matching template for a task.

        Args:
            task: The task description.

        Returns:
            Template name, or None if no match.
        """
        task_lower = task.lower()
        best_name: Optional[str] = None
        best_score = 0.0

        for name, template in self._templates.items():
            keywords = template["keywords"]
            match_count = sum(1 for kw in keywords if kw in task_lower)
            score = match_count / len(keywords) if keywords else 0.0
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.2:
            return best_name
        return None

    def _apply_template(self, task: str, template_name: str) -> DecomposedTask:
        """Apply a template to decompose a task.

        Args:
            task: The task description.
            template_name: Name of the template to apply.

        Returns:
            A DecomposedTask using the template.
        """
        template = self._templates[template_name]
        steps = template["steps"]
        task_types = template.get("task_types", ["general"] * len(steps))

        root = TaskNode(description=task, depth=0)

        prev_id: Optional[str] = None
        for i, (step, t_type) in enumerate(zip(steps, task_types)):
            child = root.add_child(step)
            child.is_atomic = True
            child.estimated_effort = 5
            child.priority = 1.0 - (i * 0.05)
            child.task_type = t_type
            if prev_id is not None:
                child.dependencies.add(prev_id)
            prev_id = child.task_id

        decomposed = DecomposedTask(
            root=root,
            strategy=f"template_{template_name}",
            original_task=task,
        )
        self._decomposition_history.append(decomposed)
        return decomposed

    def list_templates(self) -> List[str]:
        """List all available template names.

        Returns:
            List of template name strings.
        """
        return list(self._templates.keys())


# =============================================================================
# Decomposition Scorer
# =============================================================================

class DecompositionScorer:
    """Scores the quality of task decompositions.

    Evaluates decompositions across multiple dimensions including
    completeness, atomicity, ordering, clarity, and redundancy.
    """

    def __init__(self) -> None:
        """Initialize the decomposition scorer."""
        self._scoring_history: List[Tuple[str, DecompositionQuality]] = []

    def score(self, decomposed: DecomposedTask) -> DecompositionQuality:
        """Score a task decomposition across all quality dimensions.

        Args:
            decomposed: The decomposition to score.

        Returns:
            A DecompositionQuality with detailed scores.
        """
        quality = DecompositionQuality()
        quality.completeness = self._score_completeness(decomposed)
        quality.atomicity = self._score_atomicity(decomposed)
        quality.ordering = self._score_ordering(decomposed)
        quality.clarity = self._score_clarity(decomposed)
        quality.redundancy = self._score_redundancy(decomposed)
        decomposed.quality_score = quality.overall
        self._scoring_history.append((decomposed.strategy, quality))
        return quality

    def _score_completeness(self, decomposed: DecomposedTask) -> float:
        """Score how completely the decomposition covers the original task.

        Args:
            decomposed: The decomposition to evaluate.

        Returns:
            Completeness score between 0.0 and 1.0.
        """
        if not decomposed.original_task.strip():
            return 0.0
        if not decomposed.root.children:
            return 0.1

        task_words = set(decomposed.original_task.lower().split())
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "to", "of", "in", "for",
            "and", "or", "but", "with", "on", "at", "by", "how", "what",
            "why", "when", "where", "which", "who", "this", "that",
        })
        task_content = task_words - stop_words
        if not task_content:
            return 0.7

        all_subtask_text = " ".join(
            node.description.lower()
            for node in decomposed.get_flat_subtasks()
        )
        covered_words = task_content & set(all_subtask_text.split())
        coverage = len(covered_words) / len(task_content)

        score = coverage * 0.6 + 0.4

        has_conclusion = any(
            "conclusion" in node.description.lower()
            or "final" in node.description.lower()
            or "verify" in node.description.lower()
            for node in decomposed.get_flat_subtasks()
        )
        if has_conclusion:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _score_atomicity(self, decomposed: DecomposedTask) -> float:
        """Score how well atomic the sub-tasks are.

        Good atomicity means each sub-task is a single, focused action.

        Args:
            decomposed: The decomposition to evaluate.

        Returns:
            Atomicity score between 0.0 and 1.0.
        """
        subtasks = decomposed.get_flat_subtasks()
        if not subtasks:
            return 0.3

        atomicity_scores = []
        for task in subtasks:
            desc = task.description.lower()
            sentences = re.split(r'(?<=[.!?])\s+', desc)
            num_clauses = sum(1 for s in sentences if len(s.strip()) > 5)

            conjunctions = desc.count(" and ") + desc.count(", and ")
            multi_action_penalty = conjunctions * 0.15

            length = len(desc.split())
            length_score = 1.0
            if length > 30:
                length_score -= (length - 30) * 0.02
            if length < 3:
                length_score -= 0.3

            score = max(0.0, length_score - multi_action_penalty)
            atomicity_scores.append(score)

        return statistics.mean(atomicity_scores)

    def _score_ordering(self, decomposed: DecomposedTask) -> float:
        """Score how well-ordered the sub-tasks are.

        Checks for logical temporal ordering and dependency consistency.

        Args:
            decomposed: The decomposition to evaluate.

        Returns:
            Ordering score between 0.0 and 1.0.
        """
        subtasks = decomposed.get_flat_subtasks()
        if not subtasks:
            return 0.5
        if len(subtasks) <= 1:
            return 0.8

        temporal_markers = [
            "first", "initially", "start", "begin",
            "then", "next", "after that", "subsequently",
            "finally", "last", "conclude", "end",
        ]

        ordering_score = 0.5
        expected_order = {marker: i for i, marker in enumerate(temporal_markers)}

        for i, task in enumerate(subtasks):
            desc_lower = task.description.lower()
            for marker in temporal_markers:
                if marker in desc_lower:
                    expected_pos = expected_order[marker] / len(temporal_markers)
                    actual_pos = i / len(subtasks)
                    deviation = abs(expected_pos - actual_pos)
                    ordering_score -= deviation * 0.1

        dep_consistency = 0.0
        dep_count = 0
        for task in subtasks:
            for dep_id in task.dependencies:
                dep_task = None
                for other in subtasks:
                    if other.task_id == dep_id:
                        dep_task = other
                        break
                if dep_task is not None:
                    dep_index = subtasks.index(dep_task)
                    task_index = subtasks.index(task)
                    if dep_index < task_index:
                        dep_consistency += 1.0
                    dep_count += 1

        if dep_count > 0:
            dep_score = dep_consistency / dep_count
            ordering_score = ordering_score * 0.5 + dep_score * 0.5

        return min(1.0, max(0.0, ordering_score))

    def _score_clarity(self, decomposed: DecomposedTask) -> float:
        """Score the clarity and specificity of sub-task descriptions.

        Args:
            decomposed: The decomposition to evaluate.

        Returns:
            Clarity score between 0.0 and 1.0.
        """
        subtasks = decomposed.get_flat_subtasks()
        if not subtasks:
            return 0.3

        clarity_scores = []
        for task in subtasks:
            desc = task.description
            score = 0.5

            length = len(desc.split())
            if 5 <= length <= 25:
                score += 0.2
            elif 3 <= length <= 40:
                score += 0.1

            if desc[0:1].isupper():
                score += 0.1

            verbs = [
                "calculate", "compute", "analyze", "evaluate", "determine",
                "identify", "find", "compare", "verify", "create", "write",
                "generate", "list", "describe", "explain", "assess",
            ]
            if any(verb in desc.lower() for verb in verbs):
                score += 0.15

            vague_words = ["something", "things", "stuff", "etc", "somehow"]
            vagueness = sum(1 for w in vague_words if w in desc.lower())
            score -= vagueness * 0.2

            clarity_scores.append(max(0.0, min(1.0, score)))

        return statistics.mean(clarity_scores)

    def _score_redundancy(self, decomposed: DecomposedTask) -> float:
        """Score how much redundancy exists in the decomposition.

        Lower redundancy scores are better.

        Args:
            decomposed: The decomposition to evaluate.

        Returns:
            Redundancy score between 0.0 and 1.0.
        """
        subtasks = decomposed.get_flat_subtasks()
        if len(subtasks) <= 1:
            return 0.0

        redundancy_count = 0
        comparisons = 0

        for i, task1 in enumerate(subtasks):
            for j, task2 in enumerate(subtasks):
                if i >= j:
                    continue
                comparisons += 1

                words1 = set(task1.description.lower().split())
                words2 = set(task2.description.lower().split())

                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    if overlap > 0.8:
                        redundancy_count += 1

        return redundancy_count / comparisons if comparisons > 0 else 0.0

    def compare_decompositions(
        self,
        decompositions: List[DecomposedTask],
    ) -> Dict[str, DecompositionQuality]:
        """Score and compare multiple decompositions.

        Args:
            decompositions: List of decompositions to compare.

        Returns:
            Dictionary mapping strategy names to quality scores.
        """
        results: Dict[str, DecompositionQuality] = {}
        for d in decompositions:
            quality = self.score(d)
            results[d.strategy] = quality
        return results

    def best_decomposition(
        self,
        decompositions: List[DecomposedTask],
    ) -> Optional[DecomposedTask]:
        """Select the best decomposition from a list.

        Args:
            decompositions: List of decompositions to evaluate.

        Returns:
            The decomposition with the highest quality score.
        """
        if not decompositions:
            return None

        scored = [(self.score(d).overall, d) for d in decompositions]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
