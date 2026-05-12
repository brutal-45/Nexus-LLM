"""
Planning Reasoning Module
===========================

Implements goal-oriented planning for the Nexus LLM framework. The planning
module breaks complex goals into ordered subgoals, manages dependencies,
executes plans step by step, monitors progress, and supports dynamic replanning.

Key components:
- Plan / SubGoal data structures with dependency tracking
- Planner: Core planning engine with decomposition, ordering, and execution
- HierarchicalPlanner: Multi-level planning (high-level → detailed sub-plans)
- ReactivePlanner: Real-time plan adaptation based on observations
- PlanVerifier: Plan quality validation before execution

The planning approach follows the Plan-Execute-Observe-Replan loop common in
classical AI planning, adapted for LLM-based reasoning.

References:
    - Yao et al. (2023) "ReAct: Synergizing Reasoning and Acting in Language Models"
    - Sun et al. (2023) "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency"
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
from collections import defaultdict, Counter, deque, OrderedDict
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
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

class PlanStatus(Enum):
    """Status of an overall plan.

    Attributes:
        DRAFT: Plan is being constructed.
        VALIDATED: Plan has been validated and is ready for execution.
        EXECUTING: Plan is currently being executed.
        PAUSED: Plan execution has been paused.
        COMPLETED: Plan execution is complete and all subgoals achieved.
        FAILED: Plan execution has failed and cannot recover.
        REPLANNING: Plan is being revised due to new information.
        CANCELLED: Plan was cancelled by the user or system.
    """
    DRAFT = "draft"
    VALIDATED = "validated"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"
    CANCELLED = "cancelled"


class SubGoalStatus(Enum):
    """Status of an individual subgoal.

    Attributes:
        PENDING: Subgoal is waiting to be executed.
        READY: Subgoal's dependencies are met and it can be executed.
        IN_PROGRESS: Subgoal is currently being executed.
        COMPLETED: Subgoal has been completed successfully.
        FAILED: Subgoal execution has failed.
        SKIPPED: Subgoal was skipped (e.g., due to irrelevance).
        BLOCKED: Subgoal is blocked by a failed dependency.
    """
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SubGoal:
    """A subgoal within a planning hierarchy.

    Each subgoal represents a discrete, achievable objective that contributes
    to the overall plan goal.

    Attributes:
        description: Human-readable description of the subgoal.
        requirements: Prerequisites that must be met before execution.
        status: Current execution status.
        result: Output/result of executing this subgoal.
        attempts: Number of execution attempts.
        max_attempts: Maximum allowed execution attempts.
        dependencies: Set of subgoal IDs that must complete first.
        estimated_cost: Estimated resource cost in tokens.
        actual_cost: Actual resource cost after execution.
        priority: Execution priority (higher = more important).
        metadata: Additional metadata about the subgoal.
        subgoal_id: Unique identifier for this subgoal.
        subplan: Optional nested plan for hierarchical decomposition.
        confidence: Confidence that this subgoal can be completed.
        tags: Tags for categorizing this subgoal.
        created_at: Timestamp of creation.
        started_at: Timestamp when execution started.
        completed_at: Timestamp when execution completed.
        error_message: Error message if execution failed.
    """

    def __init__(
        self,
        description: str = "",
        requirements: Optional[List[str]] = None,
        dependencies: Optional[Set[int]] = None,
        estimated_cost: int = 100,
        priority: float = 0.5,
        subgoal_id: Optional[int] = None,
    ) -> None:
        self.description = description
        self.requirements = requirements or []
        self.status = SubGoalStatus.PENDING
        self.result: str = ""
        self.attempts: int = 0
        self.max_attempts: int = 3
        self.dependencies = dependencies or set()
        self.estimated_cost = estimated_cost
        self.actual_cost: int = 0
        self.priority = priority
        self.metadata: Dict[str, Any] = {}
        self.subgoal_id = subgoal_id or id(self)
        self.subplan: Optional[Plan] = None
        self.confidence: float = 0.5
        self.tags: Set[str] = set()
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the subgoal to a dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "subgoal_id": self.subgoal_id,
            "description": self.description,
            "requirements": self.requirements,
            "status": self.status.value,
            "result": self.result,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "dependencies": list(self.dependencies),
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "priority": self.priority,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "error_message": self.error_message,
            "has_subplan": self.subplan is not None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SubGoal:
        """Deserialize a subgoal from a dictionary.

        Args:
            data: Dictionary containing subgoal data.

        Returns:
            A SubGoal instance.
        """
        status_str = data.get("status", "pending")
        status = SubGoalStatus(status_str)
        subgoal = cls(
            description=data.get("description", ""),
            requirements=data.get("requirements", []),
            dependencies=set(data.get("dependencies", [])),
            estimated_cost=data.get("estimated_cost", 100),
            priority=data.get("priority", 0.5),
            subgoal_id=data.get("subgoal_id"),
        )
        subgoal.status = status
        subgoal.result = data.get("result", "")
        subgoal.attempts = data.get("attempts", 0)
        subgoal.max_attempts = data.get("max_attempts", 3)
        subgoal.actual_cost = data.get("actual_cost", 0)
        subgoal.confidence = data.get("confidence", 0.5)
        subgoal.tags = set(data.get("tags", []))
        subgoal.error_message = data.get("error_message", "")
        subgoal.metadata = data.get("metadata", {})
        return subgoal

    def mark_ready(self) -> None:
        """Mark this subgoal as ready for execution."""
        self.status = SubGoalStatus.READY

    def mark_in_progress(self) -> None:
        """Mark this subgoal as currently executing."""
        self.status = SubGoalStatus.IN_PROGRESS
        self.started_at = time.time()
        self.attempts += 1

    def mark_completed(self, result: str = "") -> None:
        """Mark this subgoal as successfully completed.

        Args:
            result: The output/result of executing this subgoal.
        """
        self.status = SubGoalStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def mark_failed(self, error: str = "") -> None:
        """Mark this subgoal as failed.

        Args:
            error: Error message describing the failure.
        """
        self.status = SubGoalStatus.FAILED
        self.error_message = error
        self.completed_at = time.time()

    def mark_blocked(self) -> None:
        """Mark this subgoal as blocked by a failed dependency."""
        self.status = SubGoalStatus.BLOCKED

    def mark_skipped(self) -> None:
        """Mark this subgoal as skipped."""
        self.status = SubGoalStatus.SKIPPED
        self.completed_at = time.time()

    def can_execute(self) -> bool:
        """Check if this subgoal is eligible for execution.

        Returns:
            True if the subgoal can be executed.
        """
        if self.status == SubGoalStatus.COMPLETED:
            return False
        if self.status == SubGoalStatus.SKIPPED:
            return False
        if self.attempts >= self.max_attempts:
            return False
        return True

    def duration(self) -> float:
        """Get the execution duration in seconds.

        Returns:
            Duration from start to completion, or 0 if not executed.
        """
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    def cost_efficiency(self) -> float:
        """Compute cost efficiency (estimated vs actual cost).

        Returns:
            Ratio of estimated to actual cost. Higher is more efficient.
        """
        if self.actual_cost <= 0:
            return 1.0
        return self.estimated_cost / self.actual_cost

    def clone(self) -> SubGoal:
        """Create a deep copy of this subgoal.

        Returns:
            A new SubGoal with identical data.
        """
        new_sg = SubGoal(
            description=self.description,
            requirements=list(self.requirements),
            dependencies=set(self.dependencies),
            estimated_cost=self.estimated_cost,
            priority=self.priority,
        )
        new_sg.status = self.status
        new_sg.result = self.result
        new_sg.attempts = self.attempts
        new_sg.max_attempts = self.max_attempts
        new_sg.actual_cost = self.actual_cost
        new_sg.metadata = copy.deepcopy(self.metadata)
        new_sg.confidence = self.confidence
        new_sg.tags = set(self.tags)
        new_sg.error_message = self.error_message
        new_sg.started_at = self.started_at
        new_sg.completed_at = self.completed_at
        new_sg.created_at = self.created_at
        return new_sg


@dataclass
class Plan:
    """A complete execution plan with subgoals and dependencies.

    The Plan represents a structured approach to achieving a goal through
    ordered, dependent subgoals.

    Attributes:
        goal: The overall goal this plan aims to achieve.
        subgoals: Ordered list of subgoals.
        dependencies: Dependency graph between subgoals.
        status: Current execution status of the plan.
        progress: Fraction of subgoals completed (0.0 to 1.0).
        plan_id: Unique identifier for this plan.
        created_at: Timestamp when the plan was created.
        updated_at: Timestamp of the last update.
        execution_order: Computed optimal execution order.
        total_estimated_cost: Sum of estimated subgoal costs.
        total_actual_cost: Sum of actual subgoal costs.
        metadata: Additional plan metadata.
        parent_plan_id: ID of parent plan if this is a sub-plan.
        replan_count: Number of times this plan has been replanned.
        version: Version number of the plan (incremented on replan).
    """

    def __init__(self, goal: str = "") -> None:
        self.goal = goal
        self.subgoals: List[SubGoal] = []
        self.status = PlanStatus.DRAFT
        self.progress: float = 0.0
        self.plan_id: str = hashlib.md5(
            f"{goal}_{time.time()}".encode()
        ).hexdigest()[:12]
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.execution_order: List[int] = []
        self.total_estimated_cost: int = 0
        self.total_actual_cost: int = 0
        self.metadata: Dict[str, Any] = {}
        self.parent_plan_id: Optional[str] = None
        self.replan_count: int = 0
        self.version: int = 1

    def add_subgoal(self, subgoal: SubGoal) -> int:
        """Add a subgoal to the plan.

        Args:
            subgoal: The subgoal to add.

        Returns:
            The index of the added subgoal.
        """
        self.subgoals.append(subgoal)
        self.total_estimated_cost += subgoal.estimated_cost
        self.updated_at = time.time()
        return len(self.subgoals) - 1

    def remove_subgoal(self, subgoal_id: int) -> bool:
        """Remove a subgoal from the plan by ID.

        Args:
            subgoal_id: ID of the subgoal to remove.

        Returns:
            True if the subgoal was found and removed.
        """
        for i, sg in enumerate(self.subgoals):
            if sg.subgoal_id == subgoal_id:
                self.total_estimated_cost -= sg.estimated_cost
                self.subgoals.pop(i)
                for other in self.subgoals:
                    other.dependencies.discard(subgoal_id)
                self.updated_at = time.time()
                return True
        return False

    def get_subgoal(self, subgoal_id: int) -> Optional[SubGoal]:
        """Retrieve a subgoal by its ID.

        Args:
            subgoal_id: The unique subgoal identifier.

        Returns:
            The SubGoal, or None if not found.
        """
        for sg in self.subgoals:
            if sg.subgoal_id == subgoal_id:
                return sg
        return None

    def get_ready_subgoals(self) -> List[SubGoal]:
        """Get all subgoals that are ready for execution.

        A subgoal is ready if all its dependencies are completed
        and it has not yet been executed.

        Returns:
            List of ready SubGoal instances.
        """
        completed_ids = {
            sg.subgoal_id for sg in self.subgoals
            if sg.status == SubGoalStatus.COMPLETED
        }
        failed_ids = {
            sg.subgoal_id for sg in self.subgoals
            if sg.status == SubGoalStatus.FAILED
        }

        ready: List[SubGoal] = []
        for sg in self.subgoals:
            if not sg.can_execute():
                continue
            unmet_deps = sg.dependencies - completed_ids
            blocked_deps = sg.dependencies & failed_ids
            if blocked_deps:
                sg.mark_blocked()
                continue
            if not unmet_deps:
                sg.mark_ready()
                ready.append(sg)

        ready.sort(key=lambda s: s.priority, reverse=True)
        return ready

    def update_progress(self) -> float:
        """Recalculate the plan's overall progress.

        Returns:
            Progress fraction between 0.0 and 1.0.
        """
        if not self.subgoals:
            self.progress = 0.0
            return 0.0

        completed = sum(
            1 for sg in self.subgoals
            if sg.status == SubGoalStatus.COMPLETED
        )
        total = len(self.subgoals)
        self.progress = completed / total
        self.updated_at = time.time()
        return self.progress

    def mark_executing(self) -> None:
        """Mark the plan as currently executing."""
        self.status = PlanStatus.EXECUTING
        self.updated_at = time.time()

    def mark_completed(self) -> None:
        """Mark the plan as completed."""
        self.status = PlanStatus.COMPLETED
        self.update_progress()
        self.updated_at = time.time()

    def mark_failed(self) -> None:
        """Mark the plan as failed."""
        self.status = PlanStatus.FAILED
        self.updated_at = time.time()

    def mark_replanning(self) -> None:
        """Mark the plan as being replanned."""
        self.status = PlanStatus.REPLANNING
        self.replan_count += 1
        self.version += 1
        self.updated_at = time.time()

    def is_complete(self) -> bool:
        """Check if the plan is fully completed.

        Returns:
            True if all subgoals are completed.
        """
        return all(
            sg.status == SubGoalStatus.COMPLETED or sg.status == SubGoalStatus.SKIPPED
            for sg in self.subgoals
        )

    def has_failures(self) -> bool:
        """Check if any subgoals have failed.

        Returns:
            True if any subgoal has a failed status.
        """
        return any(
            sg.status == SubGoalStatus.FAILED
            for sg in self.subgoals
        )

    def total_duration(self) -> float:
        """Get the total execution duration.

        Returns:
            Total time from creation to now or completion.
        """
        if self.status == PlanStatus.COMPLETED:
            return self.updated_at - self.created_at
        return time.time() - self.created_at

    def clone(self) -> Plan:
        """Create a deep copy of this plan.

        Returns:
            A new Plan with identical data.
        """
        new_plan = Plan(goal=self.goal)
        for sg in self.subgoals:
            new_plan.add_subgoal(sg.clone())
        new_plan.status = self.status
        new_plan.progress = self.progress
        new_plan.execution_order = list(self.execution_order)
        new_plan.total_estimated_cost = self.total_estimated_cost
        new_plan.total_actual_cost = self.total_actual_cost
        new_plan.metadata = copy.deepcopy(self.metadata)
        new_plan.parent_plan_id = self.parent_plan_id
        new_plan.replan_count = self.replan_count
        new_plan.version = self.version
        return new_plan

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the plan to a dictionary.

        Returns:
            Dictionary representation of the plan.
        """
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "status": self.status.value,
            "progress": self.progress,
            "subgoals": [sg.to_dict() for sg in self.subgoals],
            "execution_order": self.execution_order,
            "total_estimated_cost": self.total_estimated_cost,
            "total_actual_cost": self.total_actual_cost,
            "replan_count": self.replan_count,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Plan:
        """Deserialize a plan from a dictionary.

        Args:
            data: Dictionary containing plan data.

        Returns:
            A Plan instance.
        """
        status_str = data.get("status", "draft")
        status = PlanStatus(status_str)
        plan = cls(goal=data.get("goal", ""))
        plan.plan_id = data.get("plan_id", plan.plan_id)
        plan.status = status
        plan.progress = data.get("progress", 0.0)
        plan.execution_order = data.get("execution_order", [])
        plan.total_estimated_cost = data.get("total_estimated_cost", 0)
        plan.total_actual_cost = data.get("total_actual_cost", 0)
        plan.replan_count = data.get("replan_count", 0)
        plan.version = data.get("version", 1)
        plan.created_at = data.get("created_at", time.time())
        plan.updated_at = data.get("updated_at", time.time())
        for sg_data in data.get("subgoals", []):
            sg = SubGoal.from_dict(sg_data)
            plan.subgoals.append(sg)
        return plan


# =============================================================================
# Dependency Graph
# =============================================================================

class DependencyGraph:
    """Directed acyclic graph for managing subgoal dependencies.

    Provides topological sorting, cycle detection, and dependency
    analysis capabilities.
    """

    def __init__(self) -> None:
        """Initialize an empty dependency graph."""
        self._edges: Dict[int, Set[int]] = defaultdict(set)
        self._reverse_edges: Dict[int, Set[int]] = defaultdict(set)
        self._nodes: Set[int] = set()

    def add_node(self, node_id: int) -> None:
        """Add a node to the graph.

        Args:
            node_id: Identifier for the node.
        """
        self._nodes.add(node_id)

    def add_edge(self, from_id: int, to_id: int) -> bool:
        """Add a directed dependency edge.

        Args:
            from_id: Source node (dependency).
            to_id: Target node (dependent).

        Returns:
            True if the edge was added, False if it would create a cycle.
        """
        if from_id == to_id:
            return False

        self._nodes.add(from_id)
        self._nodes.add(to_id)
        self._edges[from_id].add(to_id)
        self._reverse_edges[to_id].add(from_id)

        if self._has_cycle():
            self._edges[from_id].discard(to_id)
            self._reverse_edges[to_id].discard(from_id)
            return False

        return True

    def remove_edge(self, from_id: int, to_id: int) -> None:
        """Remove a directed edge.

        Args:
            from_id: Source node.
            to_id: Target node.
        """
        self._edges[from_id].discard(to_id)
        self._reverse_edges[to_id].discard(from_id)

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all its edges.

        Args:
            node_id: Node to remove.
        """
        self._nodes.discard(node_id)
        if node_id in self._edges:
            for target in self._edges[node_id]:
                self._reverse_edges[target].discard(node_id)
            del self._edges[node_id]
        if node_id in self._reverse_edges:
            for source in self._reverse_edges[node_id]:
                self._edges[source].discard(node_id)
            del self._reverse_edges[node_id]

    def topological_sort(self) -> List[int]:
        """Compute a topological ordering of nodes.

        Returns:
            List of node IDs in topological order.
        """
        in_degree: Dict[int, int] = {n: 0 for n in self._nodes}
        for source, targets in self._edges.items():
            for target in targets:
                if target in in_degree:
                    in_degree[target] = in_degree.get(target, 0) + 1

        queue: deque = deque()
        for node, degree in in_degree.items():
            if degree == 0:
                queue.append(node)

        order: List[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self._edges.get(node, set()):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return order

    def _has_cycle(self) -> bool:
        """Detect if the graph contains a cycle using DFS.

        Returns:
            True if a cycle exists.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[int, int] = {n: WHITE for n in self._nodes}

        def dfs(node: int) -> bool:
            color[node] = GRAY
            for neighbor in self._edges.get(node, set()):
                if neighbor in color:
                    if color[neighbor] == GRAY:
                        return True
                    if color[neighbor] == WHITE and dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def get_dependents(self, node_id: int) -> Set[int]:
        """Get all nodes that depend on the given node.

        Args:
            node_id: The node to query.

        Returns:
            Set of dependent node IDs.
        """
        return set(self._edges.get(node_id, set()))

    def get_dependencies(self, node_id: int) -> Set[int]:
        """Get all nodes that the given node depends on.

        Args:
            node_id: The node to query.

        Returns:
            Set of dependency node IDs.
        """
        return set(self._reverse_edges.get(node_id, set()))

    def get_roots(self) -> Set[int]:
        """Get all root nodes (nodes with no dependencies).

        Returns:
            Set of root node IDs.
        """
        return {
            n for n in self._nodes
            if not self._reverse_edges.get(n)
        }

    def get_leaves(self) -> Set[int]:
        """Get all leaf nodes (nodes with no dependents).

        Returns:
            Set of leaf node IDs.
        """
        return {
            n for n in self._nodes
            if not self._edges.get(n)
        }

    def find_independent_sets(self) -> List[Set[int]]:
        """Find groups of nodes that can be executed in parallel.

        Returns:
            List of sets, where each set contains independent nodes.
        """
        sorted_nodes = self.topological_sort()
        completed: Set[int] = set()
        levels: List[Set[int]] = []

        remaining = set(sorted_nodes)
        while remaining:
            ready: Set[int] = set()
            for node in remaining:
                deps = self.get_dependencies(node)
                if deps <= completed:
                    ready.add(node)
            if not ready:
                break
            levels.append(ready)
            completed |= ready
            remaining -= ready

        return levels

    def critical_path(self) -> List[int]:
        """Find the longest path through the dependency graph.

        The critical path determines the minimum execution time
        when parallel execution is available.

        Returns:
            List of node IDs forming the critical path.
        """
        sorted_nodes = self.topological_sort()
        dist: Dict[int, int] = {n: 0 for n in sorted_nodes}
        parent: Dict[int, Optional[int]] = {n: None for n in sorted_nodes}

        for node in sorted_nodes:
            for dep in self.get_dependencies(node):
                if dist[dep] + 1 > dist[node]:
                    dist[node] = dist[dep] + 1
                    parent[node] = dep

        if not sorted_nodes:
            return []

        end_node = max(sorted_nodes, key=lambda n: dist[n])
        path: List[int] = []
        current: Optional[int] = end_node
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path

    def __len__(self) -> int:
        return len(self._nodes)


# =============================================================================
# Plan Verification
# =============================================================================

@dataclass
class PlanVerificationResult:
    """Result of plan verification.

    Attributes:
        is_valid: Whether the plan passes verification.
        completeness_score: How complete the plan is (0.0 to 1.0).
        feasibility_score: How feasible the plan is (0.0 to 1.0).
        consistency_score: How internally consistent the plan is (0.0 to 1.0).
        issues: List of identified issues.
        warnings: List of warning messages.
        suggestions: List of improvement suggestions.
    """
    is_valid: bool = True
    completeness_score: float = 1.0
    feasibility_score: float = 1.0
    consistency_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Compute the overall verification score.

        Returns:
            Weighted average of all score components.
        """
        return (
            self.completeness_score * 0.3
            + self.feasibility_score * 0.3
            + self.consistency_score * 0.2
            + (1.0 - len(self.issues) * 0.1) * 0.2
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the verification result.

        Returns:
            Dictionary representation.
        """
        return {
            "is_valid": self.is_valid,
            "completeness_score": self.completeness_score,
            "feasibility_score": self.feasibility_score,
            "consistency_score": self.consistency_score,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
        }


class PlanVerifier:
    """Verifies plan quality before execution.

    Checks plans for completeness, feasibility, and internal consistency.
    """

    def __init__(self) -> None:
        """Initialize the plan verifier."""
        self._verification_history: List[PlanVerificationResult] = []

    def verify(self, plan: Plan) -> PlanVerificationResult:
        """Verify a plan across multiple quality dimensions.

        Args:
            plan: The plan to verify.

        Returns:
            A PlanVerificationResult with detailed analysis.
        """
        result = PlanVerificationResult()
        result.completeness_score = self._check_completeness(plan, result)
        result.feasibility_score = self._check_feasibility(plan, result)
        result.consistency_score = self._check_consistency(plan, result)
        result.is_valid = (
            result.completeness_score >= 0.5
            and result.feasibility_score >= 0.5
            and result.consistency_score >= 0.5
            and len(result.issues) == 0
        )
        self._verification_history.append(result)
        return result

    def _check_completeness(
        self,
        plan: Plan,
        result: PlanVerificationResult,
    ) -> float:
        """Check that the plan addresses all aspects of the goal.

        Args:
            plan: The plan to check.
            result: Result object to append issues to.

        Returns:
            Completeness score between 0.0 and 1.0.
        """
        if not plan.goal.strip():
            result.issues.append("Plan has no defined goal.")
            return 0.0

        if not plan.subgoals:
            result.issues.append("Plan has no subgoals.")
            return 0.0

        goal_words = set(plan.goal.lower().split())
        stop_words = frozenset({
            "the", "a", "an", "is", "are", "to", "of", "in", "for",
            "on", "with", "and", "but", "or", "not", "this", "that",
            "how", "what", "why", "when", "where",
        })
        goal_content = goal_words - stop_words

        if not goal_content:
            return 0.7

        covered_aspects: Set[str] = set()
        for sg in plan.subgoals:
            sg_words = set(sg.description.lower().split()) & goal_content
            covered_aspects |= sg_words

        coverage = len(covered_aspects) / len(goal_content) if goal_content else 0.0
        score = coverage * 0.6 + 0.4

        if coverage < 0.5:
            uncovered = goal_content - covered_aspects
            result.warnings.append(
                f"Plan may not fully address goal aspects: {', '.join(list(uncovered)[:5])}"
            )

        has_conclusion = any(
            "conclusion" in sg.description.lower() or "final" in sg.description.lower()
            or "verify" in sg.description.lower()
            for sg in plan.subgoals
        )
        if not has_conclusion:
            result.warnings.append("Plan lacks a verification or conclusion subgoal.")
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _check_feasibility(
        self,
        plan: Plan,
        result: PlanVerificationResult,
    ) -> float:
        """Check that the plan is feasible to execute.

        Args:
            plan: The plan to check.
            result: Result object to append issues to.

        Returns:
            Feasibility score between 0.0 and 1.0.
        """
        score = 0.8

        for sg in plan.subgoals:
            if not sg.description.strip():
                result.issues.append(f"Subgoal {sg.subgoal_id} has no description.")
                score -= 0.2

        dep_graph = DependencyGraph()
        for sg in plan.subgoals:
            dep_graph.add_node(sg.subgoal_id)
            for dep_id in sg.dependencies:
                dep_exists = any(s.subgoal_id == dep_id for s in plan.subgoals)
                if not dep_exists:
                    result.issues.append(
                        f"Subgoal {sg.subgoal_id} depends on non-existent subgoal {dep_id}."
                    )
                    score -= 0.15

        total_cost = plan.total_estimated_cost
        if total_cost > 50000:
            result.warnings.append(f"Plan has very high estimated cost: {total_cost} tokens.")
            score -= 0.1

        if len(plan.subgoals) > 20:
            result.warnings.append("Plan has many subgoals which may be difficult to manage.")
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _check_consistency(
        self,
        plan: Plan,
        result: PlanVerificationResult,
    ) -> float:
        """Check the internal consistency of the plan.

        Verifies dependency relationships and logical ordering.

        Args:
            plan: The plan to check.
            result: Result object to append issues to.

        Returns:
            Consistency score between 0.0 and 1.0.
        """
        score = 0.9

        dep_graph = DependencyGraph()
        for sg in plan.subgoals:
            dep_graph.add_node(sg.subgoal_id)
            for dep_id in sg.dependencies:
                if not dep_graph.add_edge(dep_id, sg.subgoal_id):
                    result.issues.append(
                        f"Circular dependency detected involving subgoal {sg.subgoal_id}."
                    )
                    score -= 0.3

        subgoal_ids = {sg.subgoal_id for sg in plan.subgoals}
        for sg in plan.subgoals:
            for dep_id in sg.dependencies:
                if dep_id not in subgoal_ids:
                    continue
                dep_sg = plan.get_subgoal(dep_id)
                if dep_sg and dep_sg.priority < sg.priority - 0.5:
                    result.warnings.append(
                        f"Subgoal {sg.subgoal_id} depends on lower-priority subgoal {dep_id}."
                    )
                    score -= 0.05

        descriptions = [sg.description.lower() for sg in plan.subgoals]
        for i, desc in enumerate(descriptions):
            for j, other_desc in enumerate(descriptions):
                if i != j and desc == other_desc:
                    result.warnings.append(
                        f"Duplicate subgoal descriptions detected at indices {i} and {j}."
                    )
                    score -= 0.1

        return max(0.0, min(1.0, score))


# =============================================================================
# Planner - Main Planning Engine
# =============================================================================

class Planner:
    """Core planning engine for goal-oriented reasoning.

    Decomposes goals into subgoals, orders them based on dependencies,
    estimates costs, executes plans step by step, monitors progress,
    and supports dynamic replanning.

    Attributes:
        config: PlanningConfig instance.
        model: Language model interface.
        dep_graph: Dependency graph for subgoal ordering.
        verifier: Plan verification utility.
    """

    def __init__(self, config: Optional[Any] = None, model: Optional[Any] = None) -> None:
        """Initialize the planner.

        Args:
            config: PlanningConfig instance.
            model: Language model interface.
        """
        if config is None:
            from nexus.reasoning.reasoning_config import PlanningConfig
            config = PlanningConfig()
        self.config = config
        self.model = model or self._default_model()
        self.dep_graph = DependencyGraph()
        self.verifier = PlanVerifier()
        self._plan_history: List[Plan] = []
        self._tokens_used: int = 0

    def _default_model(self) -> Any:
        """Create a default mock model.

        Returns:
            A mock model interface.
        """
        from nexus.reasoning.chain_of_thought import MockModel
        return MockModel()

    def decompose_goal(self, goal: str) -> Plan:
        """Break a complex goal into ordered subgoals.

        Uses the language model to analyze the goal and generate
        a structured plan with dependencies.

        Args:
            goal: The goal or question to plan for.

        Returns:
            A Plan with decomposed subgoals.
        """
        plan = Plan(goal=goal)

        prompt = (
            f"Break down the following goal into specific, actionable subgoals. "
            f"For each subgoal, provide:\n"
            f"1. A clear description\n"
            f"2. Dependencies on other subgoals (if any)\n"
            f"3. Estimated complexity (low/medium/high)\n\n"
            f"Goal: {goal}\n\n"
            f"Provide the subgoals as a numbered list."
        )

        response = self.model.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000,
        )

        subgoals = self._parse_subgoals(response)
        subgoal_id_map: Dict[str, int] = {}
        for i, sg_data in enumerate(subgoals):
            sg = SubGoal(
                description=sg_data["description"],
                dependencies=set(),
                estimated_cost=self._estimate_complexity_cost(sg_data.get("complexity", "medium")),
                priority=1.0 - (i * 0.05),
            )
            plan.add_subgoal(sg)
            subgoal_id_map[sg_data["description"][:30]] = sg.subgoal_id

        deps = self._parse_dependencies(response, subgoal_id_map)
        for sg in plan.subgoals:
            key = sg.description[:30]
            if key in deps:
                sg.dependencies = deps[key]

        if self.config.plan_validation:
            result = self.verifier.verify(plan)
            if not result.is_valid:
                for issue in result.issues:
                    logger.warning("Plan verification issue: %s", issue)

        plan.status = PlanStatus.VALIDATED
        plan.execution_order = self.order_subgoals(plan.subgoals)
        self._plan_history.append(plan)
        return plan

    def _parse_subgoals(self, response: str) -> List[Dict[str, str]]:
        """Parse subgoals from model response.

        Args:
            response: Raw model output.

        Returns:
            List of subgoal data dictionaries.
        """
        subgoals: List[Dict[str, str]] = []
        lines = response.strip().split("\n")

        current_desc: List[str] = []
        current_complexity = "medium"

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            number_match = re.match(r'^\s*(\d+[\.\)]|subgoal\s*\d+[:.]?)\s*(.+)', stripped, re.IGNORECASE)
            if number_match:
                if current_desc:
                    subgoals.append({
                        "description": " ".join(current_desc).strip(),
                        "complexity": current_complexity,
                    })
                    current_desc = []
                    current_complexity = "medium"

                content = number_match.group(2).strip()
                current_desc.append(content)

                complexity_match = re.search(
                    r'(?:complexity|difficulty)[:\s]*(low|medium|high)',
                    stripped, re.IGNORECASE,
                )
                if complexity_match:
                    current_complexity = complexity_match.group(1).lower()
            else:
                complexity_match = re.search(
                    r'(?:complexity|difficulty)[:\s]*(low|medium|high)',
                    stripped, re.IGNORECASE,
                )
                if complexity_match:
                    current_complexity = complexity_match.group(1).lower()
                else:
                    current_desc.append(stripped)

        if current_desc:
            subgoals.append({
                "description": " ".join(current_desc).strip(),
                "complexity": current_complexity,
            })

        return subgoals

    def _parse_dependencies(
        self,
        response: str,
        id_map: Dict[str, int],
    ) -> Dict[str, Set[int]]:
        """Parse dependency relationships from model response.

        Args:
            response: Raw model output.
            id_map: Mapping from subgoal descriptions to IDs.

        Returns:
            Dictionary mapping subgoal keys to dependency ID sets.
        """
        deps: Dict[str, Set[int]] = defaultdict(set)
        dep_patterns = [
            r'(?:depends?\s+on|requires?|needs?|after|before|follows?|precedes?)\s+(?:subgoal\s*)?(\d+)',
            r'(?:step\s*)?(\d+)\s+(?:must\s+be\s+done\s+)?before\s+(?:step\s*)?(\d+)',
        ]

        lines = response.split("\n")
        for line in lines:
            for pattern in dep_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) >= 2:
                            try:
                                dep_idx = int(match[0]) - 1
                                target_idx = int(match[1]) - 1
                                keys = list(id_map.keys())
                                if dep_idx < len(keys) and target_idx < len(keys):
                                    deps[keys[target_idx]].add(id_map[keys[dep_idx]])
                            except (ValueError, IndexError):
                                continue
                    else:
                        try:
                            dep_idx = int(match) - 1
                            keys = list(id_map.keys())
                            if 0 <= dep_idx < len(keys):
                                current_key = keys[-1] if keys else ""
                                deps[current_key].add(id_map[keys[dep_idx]])
                        except (ValueError, IndexError):
                            continue

        return dict(deps)

    def _estimate_complexity_cost(self, complexity: str) -> int:
        """Estimate token cost based on complexity level.

        Args:
            complexity: Complexity string (low, medium, high).

        Returns:
            Estimated cost in tokens.
        """
        costs = {"low": 100, "medium": 250, "high": 500}
        return costs.get(complexity.lower(), 250)

    def order_subgoals(self, subgoals: List[SubGoal]) -> List[int]:
        """Order subgoals using topological sort based on dependencies.

        Args:
            subgoals: List of subgoals to order.

        Returns:
            List of subgoal IDs in execution order.
        """
        graph = DependencyGraph()
        for sg in subgoals:
            graph.add_node(sg.subgoal_id)
            for dep_id in sg.dependencies:
                graph.add_edge(dep_id, sg.subgoal_id)

        return graph.topological_sort()

    def estimate_cost(self, subgoal: SubGoal) -> int:
        """Estimate the resource cost of executing a subgoal.

        Args:
            subgoal: The subgoal to estimate cost for.

        Returns:
            Estimated cost in tokens.
        """
        base_cost = subgoal.estimated_cost

        desc_length = len(subgoal.description)
        length_cost = desc_length * 2

        num_deps = len(subgoal.dependencies)
        dep_cost = num_deps * 50

        complexity_keywords = {
            "analyze": 100, "evaluate": 100, "compare": 80,
            "compute": 120, "calculate": 120, "verify": 80,
            "search": 150, "optimize": 200, "implement": 300,
        }
        keyword_cost = 0
        for keyword, cost in complexity_keywords.items():
            if keyword in subgoal.description.lower():
                keyword_cost += cost

        total = base_cost + length_cost + dep_cost + keyword_cost
        return total

    def execute_plan(self, plan: Plan) -> Plan:
        """Execute a plan step by step.

        Iterates through ready subgoals, executes each one using
        the model, and handles failures with recovery strategies.

        Args:
            plan: The plan to execute.

        Returns:
            The executed Plan with results.
        """
        plan.mark_executing()
        consecutive_failures = 0

        while not plan.is_complete() and plan.status == PlanStatus.EXECUTING:
            if self.config.is_over_budget(self._tokens_used):
                plan.mark_failed()
                break

            ready = plan.get_ready_subgoals()
            if not ready:
                if plan.has_failures():
                    plan.mark_failed()
                else:
                    plan.mark_completed()
                break

            for subgoal in ready:
                if self.config.is_over_budget(self._tokens_used):
                    plan.mark_failed()
                    break

                result = self._execute_subgoal(plan, subgoal)
                if result:
                    subgoal.mark_completed(result)
                    consecutive_failures = 0
                    self._tokens_used += subgoal.actual_cost
                else:
                    recovery = self.config.get_recovery_strategy(subgoal.attempts - 1)
                    if recovery == "retry" and subgoal.can_execute():
                        consecutive_failures += 1
                        continue
                    elif recovery == "skip":
                        subgoal.mark_skipped()
                        consecutive_failures = 0
                    elif recovery == "replan" and self.config.replanning_enabled:
                        plan.mark_replanning()
                        break
                    else:
                        subgoal.mark_failed("Execution failed after all retries")
                        consecutive_failures += 1

            plan.update_progress()

            if consecutive_failures >= 3:
                if self.config.replanning_enabled:
                    plan.mark_replanning()
                    break
                else:
                    plan.mark_failed()
                    break

        if plan.is_complete():
            plan.mark_completed()

        return plan

    def _execute_subgoal(self, plan: Plan, subgoal: SubGoal) -> str:
        """Execute a single subgoal using the model.

        Args:
            plan: The parent plan.
            subgoal: The subgoal to execute.

        Returns:
            Execution result string, or empty string on failure.
        """
        subgoal.mark_in_progress()
        context = self._build_execution_context(plan, subgoal)

        response = self.model.generate(
            prompt=context,
            temperature=0.3,
            max_tokens=500,
        )

        result = response.strip()
        if result:
            subgoal.actual_cost = max(50, len(result) // 4)
            return result
        return ""

    def _build_execution_context(self, plan: Plan, subgoal: SubGoal) -> str:
        """Build the context for executing a subgoal.

        Args:
            plan: The parent plan.
            subgoal: The subgoal being executed.

        Returns:
            The execution prompt.
        """
        parts = [
            f"Overall Goal: {plan.goal}",
            f"Current Subgoal: {subgoal.description}",
        ]

        if subgoal.requirements:
            parts.append(f"Requirements: {', '.join(subgoal.requirements)}")

        completed = [
            sg for sg in plan.subgoals
            if sg.status == SubGoalStatus.COMPLETED and sg.result
        ]
        if completed:
            parts.append("\nCompleted subgoals:")
            for sg in completed[-3:]:
                parts.append(f"  - {sg.description}: {sg.result[:100]}")

        parts.append("\nExecute the current subgoal:")
        return "\n".join(parts)

    def monitor_progress(self, plan: Plan) -> Dict[str, Any]:
        """Track and report plan execution progress.

        Args:
            plan: The plan to monitor.

        Returns:
            Dictionary with progress information.
        """
        status_counts = Counter(sg.status for sg in plan.subgoals)
        ready = plan.get_ready_subgoals()

        return {
            "plan_id": plan.plan_id,
            "status": plan.status.value,
            "progress": plan.update_progress(),
            "total_subgoals": len(plan.subgoals),
            "completed": status_counts.get(SubGoalStatus.COMPLETED, 0),
            "failed": status_counts.get(SubGoalStatus.FAILED, 0),
            "in_progress": status_counts.get(SubGoalStatus.IN_PROGRESS, 0),
            "ready": len(ready),
            "blocked": status_counts.get(SubGoalStatus.BLOCKED, 0),
            "tokens_used": self._tokens_used,
            "budget_remaining": self.config.budget_remaining(self._tokens_used),
            "duration": plan.total_duration(),
            "cost_efficiency": (
                plan.total_estimated_cost / plan.total_actual_cost
                if plan.total_actual_cost > 0 else 1.0
            ),
        }

    def replan(self, plan: Plan, feedback: str = "") -> Plan:
        """Adjust the plan based on execution feedback.

        Analyzes what went wrong, modifies the remaining subgoals,
        and creates an updated plan.

        Args:
            plan: The plan to revise.
            feedback: Feedback about what went wrong.

        Returns:
            A revised Plan.
        """
        plan.mark_replanning()

        completed_subgoals = [
            sg for sg in plan.subgoals
            if sg.status == SubGoalStatus.COMPLETED
        ]
        failed_subgoals = [
            sg for sg in plan.subgoals
            if sg.status == SubGoalStatus.FAILED
        ]
        pending_subgoals = [
            sg for sg in plan.subgoals
            if sg.status in (
                SubGoalStatus.PENDING,
                SubGoalStatus.READY,
                SubGoalStatus.BLOCKED,
            )
        ]

        new_plan = Plan(goal=plan.goal)
        new_plan.parent_plan_id = plan.plan_id
        new_plan.replan_count = plan.replan_count
        new_plan.version = plan.version

        for sg in completed_subgoals:
            new_plan.add_subgoal(sg.clone())

        prompt = (
            f"The following subgoals failed in a plan for '{plan.goal}':\n"
            f"{chr(10).join('- ' + sg.description + ' (Error: ' + sg.error_message + ')' for sg in failed_subgoals)}\n\n"
            f"Remaining subgoals:\n"
            f"{chr(10).join('- ' + sg.description for sg in pending_subgoals)}\n\n"
            f"Feedback: {feedback}\n\n"
            f"Generate revised subgoals to complete the goal, considering the failures."
        )

        response = self.model.generate(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1000,
        )

        new_subgoals = self._parse_subgoals(response)
        existing_ids = {sg.subgoal_id for sg in new_plan.subgoals}

        for sg_data in new_subgoals:
            sg = SubGoal(
                description=sg_data["description"],
                estimated_cost=self._estimate_complexity_cost(sg_data.get("complexity", "medium")),
                priority=0.8,
            )
            for completed_sg in completed_subgoals:
                sg.dependencies.add(completed_sg.subgoal_id)
            new_plan.add_subgoal(sg)

        new_plan.execution_order = self.order_subgoals(new_plan.subgoals)
        new_plan.status = PlanStatus.VALIDATED

        self._plan_history.append(new_plan)
        return new_plan

    def identify_dependencies(self, subgoals: List[SubGoal]) -> Dict[int, Set[int]]:
        """Find ordering constraints between subgoals.

        Analyzes subgoal descriptions to identify implicit dependencies.

        Args:
            subgoals: List of subgoals to analyze.

        Returns:
            Dictionary mapping subgoal IDs to their dependency sets.
        """
        deps: Dict[int, Set[int]] = {}

        for sg in subgoals:
            deps[sg.subgoal_id] = set(sg.dependencies)

        temporal_markers = ["first", "then", "after", "before", "next", "finally"]
        for i, sg in enumerate(subgoals):
            desc_lower = sg.description.lower()
            for marker in temporal_markers:
                if marker in desc_lower:
                    if marker in ("after", "then", "next", "finally") and i > 0:
                        prev_sg = subgoals[i - 1]
                        deps[sg.subgoal_id].add(prev_sg.subgoal_id)
                    elif marker in ("first", "before") and i < len(subgoals) - 1:
                        next_sg = subgoals[i + 1]
                        deps[next_sg.subgoal_id].add(sg.subgoal_id)

        return deps

    def parallelize_subgoals(self, plan: Plan) -> List[List[SubGoal]]:
        """Identify independent subgoals for parallel execution.

        Groups subgoals that have no dependencies on each other
        into parallel execution batches.

        Args:
            plan: The plan to parallelize.

        Returns:
            List of batches, where each batch is a list of parallel subgoals.
        """
        graph = DependencyGraph()
        for sg in plan.subgoals:
            if sg.status in (SubGoalStatus.PENDING, SubGoalStatus.READY):
                graph.add_node(sg.subgoal_id)
                for dep_id in sg.dependencies:
                    graph.add_edge(dep_id, sg.subgoal_id)

        independent_sets = graph.find_independent_sets()

        batches: List[List[SubGoal]] = []
        for node_set in independent_sets:
            batch = []
            for sg in plan.subgoals:
                if sg.subgoal_id in node_set:
                    batch.append(sg)
            if batch:
                batches.append(batch)

        return batches

    def merge_plans(self, plans: List[Plan]) -> Plan:
        """Merge multiple plans into a single consolidated plan.

        Combines subgoals from all plans, removing duplicates and
        resolving conflicts.

        Args:
            plans: List of plans to merge.

        Returns:
            A merged Plan.
        """
        if not plans:
            return Plan()
        if len(plans) == 1:
            return plans[0].clone()

        merged_goal = plans[0].goal
        merged_plan = Plan(goal=merged_goal)

        seen_descriptions: Set[str] = set()
        for plan in plans:
            for sg in plan.subgoals:
                normalized = sg.description.lower().strip()
                if normalized not in seen_descriptions:
                    seen_descriptions.add(normalized)
                    new_sg = sg.clone()
                    new_sg.status = SubGoalStatus.PENDING
                    new_sg.result = ""
                    new_sg.attempts = 0
                    merged_plan.add_subgoal(new_sg)

        merged_plan.execution_order = self.order_subgoals(merged_plan.subgoals)
        merged_plan.status = PlanStatus.DRAFT
        return merged_plan

    def plan_to_prompt(self, plan: Plan) -> str:
        """Format a plan for inclusion in a model prompt.

        Args:
            plan: The plan to format.

        Returns:
            Formatted string representation of the plan.
        """
        parts = [
            f"Plan for: {plan.goal}",
            f"Status: {plan.status.value} | Progress: {plan.progress:.0%}",
            "=" * 50,
        ]

        for i, sg in enumerate(plan.subgoals):
            status_icon = {
                SubGoalStatus.PENDING: "⬜",
                SubGoalStatus.READY: "🟦",
                SubGoalStatus.IN_PROGRESS: "🟧",
                SubGoalStatus.COMPLETED: "🟩",
                SubGoalStatus.FAILED: "🟥",
                SubGoalStatus.SKIPPED: "⬛",
                SubGoalStatus.BLOCKED: "⛔",
            }.get(sg.status, "⬜")

            deps_str = ""
            if sg.dependencies:
                deps_str = f" (depends on: {', '.join(str(d) for d in sg.dependencies)})"

            result_str = ""
            if sg.result:
                result_str = f" → {sg.result[:80]}"

            parts.append(
                f"  {status_icon} {i+1}. {sg.description}{deps_str}{result_str}"
            )

        return "\n".join(parts)


# =============================================================================
# Hierarchical Planner
# =============================================================================

class HierarchicalPlanner:
    """Multi-level planning with hierarchical decomposition.

    Creates a high-level plan first, then generates detailed sub-plans
    for each high-level subgoal. This approach handles complex goals
    by managing them at multiple levels of abstraction.

    Attributes:
        planner: The base planner for sub-plan generation.
        max_depth: Maximum nesting depth for sub-plans.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[Any] = None,
        max_depth: int = 3,
    ) -> None:
        """Initialize the hierarchical planner.

        Args:
            config: PlanningConfig instance.
            model: Language model interface.
            max_depth: Maximum nesting depth.
        """
        self.planner = Planner(config=config, model=model)
        self.max_depth = max_depth

    def create_plan(self, goal: str, depth: int = 0) -> Plan:
        """Create a hierarchical plan with nested sub-plans.

        At the top level, generates high-level subgoals. Then, for each
        subgoal, recursively creates detailed sub-plans.

        Args:
            goal: The overall goal.
            depth: Current recursion depth.

        Returns:
            A Plan with nested sub-plans.
        """
        plan = self.planner.decompose_goal(goal)

        if depth < self.max_depth:
            for sg in plan.subgoals:
                if sg.estimated_cost > 300 or "complex" in sg.description.lower():
                    subplan = self.create_plan(sg.description, depth + 1)
                    subplan.parent_plan_id = plan.plan_id
                    sg.subplan = subplan

        return plan

    def execute_plan(self, plan: Plan) -> Plan:
        """Execute a hierarchical plan.

        Executes each subgoal, including nested sub-plans.

        Args:
            plan: The plan to execute.

        Returns:
            The executed plan with results.
        """
        for sg in plan.subgoals:
            if sg.subplan is not None:
                sg.subplan = self.execute_plan(sg.subplan)
                sg.result = self._summarize_subplan(sg.subplan)
            else:
                context = f"Goal: {plan.goal}\nSubgoal: {sg.description}"
                response = self.planner.model.generate(
                    prompt=context,
                    temperature=0.3,
                    max_tokens=500,
                )
                sg.mark_completed(response.strip())

        plan.update_progress()
        plan.status = PlanStatus.COMPLETED
        return plan

    def _summarize_subplan(self, subplan: Plan) -> str:
        """Summarize a completed sub-plan into a concise result.

        Args:
            subplan: The completed sub-plan.

        Returns:
            Summary string of the sub-plan results.
        """
        completed = [sg for sg in subplan.subgoals if sg.status == SubGoalStatus.COMPLETED]
        if not completed:
            return f"Sub-plan completed with {len(subplan.subgoals)} subgoals."

        results = [sg.result for sg in completed if sg.result]
        summary = "; ".join(results[:3])
        if len(results) > 3:
            summary += f" (and {len(results) - 3} more)"
        return summary


# =============================================================================
# Reactive Planner
# =============================================================================

class ReactivePlanner:
    """Adaptive planner that adjusts plans in real-time based on observations.

    Monitors execution and triggers replanning when observations deviate
    from expectations, allowing the system to handle unexpected situations.

    Attributes:
        planner: The base planner.
        observation_buffer: Buffer of recent observations.
        replan_threshold: Threshold for triggering replanning.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        model: Optional[Any] = None,
        replan_threshold: float = 0.3,
    ) -> None:
        """Initialize the reactive planner.

        Args:
            config: PlanningConfig instance.
            model: Language model interface.
            replan_threshold: Confidence threshold below which to replan.
        """
        self.planner = Planner(config=config, model=model)
        self.replan_threshold = replan_threshold
        self.observation_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 10

    def create_plan(self, goal: str) -> Plan:
        """Create an initial plan.

        Args:
            goal: The goal to plan for.

        Returns:
            A Plan ready for reactive execution.
        """
        return self.planner.decompose_goal(goal)

    def observe(self, observation: str, confidence: float = 0.8) -> None:
        """Record an observation from the environment.

        Args:
            observation: The observed event or result.
            confidence: Confidence in the observation accuracy.
        """
        self.observation_buffer.append({
            "observation": observation,
            "confidence": confidence,
            "timestamp": time.time(),
        })
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer.pop(0)

    def should_replan(self, plan: Plan) -> Tuple[bool, str]:
        """Determine if replanning is needed based on observations.

        Args:
            plan: The current plan.

        Returns:
            Tuple of (should_replan, reason).
        """
        if not self.observation_buffer:
            return False, ""

        avg_confidence = statistics.mean(
            obs["confidence"] for obs in self.observation_buffer
        )
        if avg_confidence < self.replan_threshold:
            return True, f"Low average observation confidence: {avg_confidence:.2f}"

        if plan.has_failures():
            failed_count = sum(
                1 for sg in plan.subgoals if sg.status == SubGoalStatus.FAILED
            )
            failure_rate = failed_count / len(plan.subgoals)
            if failure_rate > 0.3:
                return True, f"High failure rate: {failure_rate:.0%}"

        recent = self.observation_buffer[-3:]
        for obs in recent:
            obs_lower = obs["observation"].lower()
            unexpected_keywords = [
                "error", "failed", "unexpected", "wrong", "impossible",
                "cannot", "blocked", "denied",
            ]
            if any(kw in obs_lower for kw in unexpected_keywords):
                return True, f"Unexpected observation: {obs['observation'][:100]}"

        return False, ""

    def adaptive_execute(self, plan: Plan) -> Plan:
        """Execute the plan with reactive replanning.

        Monitors observations during execution and triggers replanning
        when necessary.

        Args:
            plan: The plan to execute.

        Returns:
            The executed (and possibly replanned) Plan.
        """
        max_replan_attempts = self.planner.config.max_replan_attempts
        replan_count = 0

        while not plan.is_complete() and replan_count <= max_replan_attempts:
            should_replan, reason = self.should_replan(plan)
            if should_replan and self.planner.config.replanning_enabled:
                feedback = "; ".join(
                    obs["observation"] for obs in self.observation_buffer[-3:]
                )
                plan = self.planner.replan(plan, feedback)
                replan_count += 1
                self.observation_buffer.clear()
                continue

            plan = self.planner.execute_plan(plan)
            break

        return plan
