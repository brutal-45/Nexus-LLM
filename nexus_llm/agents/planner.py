"""Task planner for decomposing goals into executable plans.

Provides task decomposition, execution plan creation, subtask
management, and plan tracking with dependency resolution.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """A single step in an execution plan."""

    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    action: str = ""  # Description of what to do
    tool: Optional[str] = None  # Tool to use
    tool_args: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0  # Higher = more important
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Check if the step is ready to execute (all dependencies met)."""
        return self.status == StepStatus.PENDING and len(self.dependencies) == 0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "tool": self.tool,
            "tool_args": self.tool_args,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "priority": self.priority,
        }


@dataclass
class Plan:
    """An execution plan composed of ordered steps."""

    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    steps: List[Step] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    created_at: float = field(default_factory=lambda: __import__("time").time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        if not self.steps:
            return 0.0
        return self.completed_steps / self.total_steps

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)

    def get_ready_steps(self) -> List[Step]:
        """Get all steps that are ready to execute."""
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in step.dependencies):
                    ready.append(step)
        ready.sort(key=lambda s: s.priority, reverse=True)
        return ready

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def add_step(self, step: Step) -> None:
        """Add a step to the plan."""
        self.steps.append(step)

    def mark_step_completed(self, step_id: str, result: str = "") -> bool:
        """Mark a step as completed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.COMPLETED
            step.result = result
            return True
        return False

    def mark_step_failed(self, step_id: str, error: str = "") -> bool:
        """Mark a step as failed."""
        step = self.get_step(step_id)
        if step:
            step.status = StepStatus.FAILED
            step.error = error
            return True
        return False

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "progress": self.progress,
            "metadata": self.metadata,
        }


class TaskPlanner:
    """Decomposes tasks into executable plans with dependency resolution.

    Uses LLM-based or rule-based decomposition to break down complex
    tasks into ordered steps with dependencies, then tracks execution
    progress through the plan.
    """

    # Common task decomposition patterns
    DECOMPOSITION_PATTERNS = {
        "research": [
            "Identify the research topic and scope",
            "Search for relevant sources",
            "Analyze and evaluate sources",
            "Synthesize findings",
            "Present conclusions with citations",
        ],
        "code": [
            "Understand the requirements",
            "Design the solution",
            "Implement the code",
            "Test the implementation",
            "Review and refine",
        ],
        "write": [
            "Understand the topic and audience",
            "Create an outline",
            "Draft the content",
            "Review and revise",
            "Finalize the output",
        ],
        "analyze": [
            "Gather relevant data",
            "Clean and preprocess data",
            "Perform analysis",
            "Interpret results",
            "Present findings",
        ],
        "debug": [
            "Reproduce the issue",
            "Identify the root cause",
            "Develop a fix",
            "Test the fix",
            "Verify the solution",
        ],
    }

    def __init__(self, llm_fn: Optional[Callable] = None):
        """Initialize the task planner.

        Args:
            llm_fn: Optional LLM function for decomposition.
        """
        self.llm_fn = llm_fn
        self._plans: Dict[str, Plan] = {}

    def create_plan(self, goal: str, strategy: Optional[str] = None) -> Plan:
        """Create an execution plan for a goal.

        Args:
            goal: The goal or task to plan for.
            strategy: Optional decomposition strategy name.

        Returns:
            A Plan with ordered steps.
        """
        # Try LLM-based decomposition first
        if self.llm_fn:
            try:
                plan = self._llm_decompose(goal)
                if plan and plan.steps:
                    self._plans[plan.plan_id] = plan
                    return plan
            except Exception as e:
                logger.warning("LLM decomposition failed: %s. Falling back to rule-based.", e)

        # Rule-based decomposition
        plan = self._rule_decompose(goal, strategy)
        self._plans[plan.plan_id] = plan
        return plan

    def _rule_decompose(self, goal: str, strategy: Optional[str] = None) -> Plan:
        """Decompose a goal using rule-based patterns."""
        plan = Plan(goal=goal)

        # Find matching strategy
        goal_lower = goal.lower()
        matched_strategy = strategy

        if not matched_strategy:
            for key in self.DECOMPOSITION_PATTERNS:
                if key in goal_lower:
                    matched_strategy = key
                    break

        if matched_strategy and matched_strategy in self.DECOMPOSITION_PATTERNS:
            steps_text = self.DECOMPOSITION_PATTERNS[matched_strategy]
        else:
            # Generic decomposition
            steps_text = [
                "Understand the task requirements",
                "Gather necessary information",
                "Formulate an approach",
                "Execute the approach",
                "Verify the results",
            ]

        # Create steps with sequential dependencies
        prev_id = None
        for i, step_text in enumerate(steps_text):
            deps = [prev_id] if prev_id else []
            step = Step(
                name=f"Step {i + 1}",
                description=step_text,
                action=step_text,
                dependencies=deps,
                priority=len(steps_text) - i,  # Earlier steps have higher priority
            )
            plan.add_step(step)
            prev_id = step.step_id

        logger.info("Created plan '%s' with %d steps for goal: %s", plan.plan_id, len(plan.steps), goal[:50])
        return plan

    def _llm_decompose(self, goal: str) -> Optional[Plan]:
        """Decompose a goal using LLM."""
        prompt = (
            f"Break down the following task into 3-7 concrete, actionable steps.\n"
            f"Each step should have a name and description.\n"
            f"Indicate dependencies between steps.\n\n"
            f"Task: {goal}\n\n"
            f"Format each step as:\n"
            f"STEP: <name>\n"
            f"DESC: <description>\n"
            f"DEPS: <comma-separated step names this depends on, or 'none'>\n"
        )

        response = self.llm_fn(prompt)
        return self._parse_llm_response(goal, response)

    def _parse_llm_response(self, goal: str, response: str) -> Plan:
        """Parse LLM response into a Plan."""
        plan = Plan(goal=goal)
        step_map: Dict[str, Step] = {}
        dep_map: Dict[str, List[str]] = {}

        lines = response.strip().split("\n")
        current_name = None

        for line in lines:
            line = line.strip()
            if line.startswith("STEP:"):
                current_name = line.replace("STEP:", "").strip()
                step = Step(name=current_name, description="", action=current_name)
                step_map[current_name] = step
                dep_map[current_name] = []
            elif line.startswith("DESC:") and current_name:
                step_map[current_name].description = line.replace("DESC:", "").strip()
                step_map[current_name].action = step_map[current_name].description
            elif line.startswith("DEPS:") and current_name:
                deps_text = line.replace("DEPS:", "").strip()
                if deps_text.lower() != "none":
                    dep_map[current_name] = [d.strip() for d in deps_text.split(",") if d.strip()]

        # Resolve dependencies
        for name, step in step_map.items():
            for dep_name in dep_map.get(name, []):
                if dep_name in step_map:
                    step.dependencies.append(step_map[dep_name].step_id)
            plan.add_step(step)

        return plan

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def list_plans(self) -> List[Plan]:
        """List all plans."""
        return list(self._plans.values())

    def get_next_step(self, plan: Plan) -> Optional[Step]:
        """Get the next step to execute in a plan."""
        ready = plan.get_ready_steps()
        return ready[0] if ready else None

    def update_step(self, plan: Plan, step_id: str, status: StepStatus, result: str = "", error: str = "") -> None:
        """Update a step's status in a plan."""
        step = plan.get_step(step_id)
        if step:
            step.status = status
            if result:
                step.result = result
            if error:
                step.error = error

    def replan_from_failure(self, plan: Plan, failed_step_id: str, alternative_action: str = "") -> Plan:
        """Create a revised plan after a step failure.

        Args:
            plan: The current plan.
            failed_step_id: The step that failed.
            alternative_action: Alternative action for the failed step.

        Returns:
            A new Plan with revised steps.
        """
        new_plan = Plan(
            goal=f"{plan.goal} (revised after failure)",
            steps=[],
        )

        for step in plan.steps:
            if step.step_id == failed_step_id:
                new_step = Step(
                    name=f"{step.name} (retry)",
                    description=alternative_action or step.description,
                    action=alternative_action or step.action,
                    tool=step.tool,
                    tool_args=step.tool_args,
                    dependencies=step.dependencies,
                    priority=step.priority + 1,
                )
                new_plan.add_step(new_step)
            elif step.status == StepStatus.PENDING:
                new_plan.add_step(Step(
                    name=step.name,
                    description=step.description,
                    action=step.action,
                    tool=step.tool,
                    tool_args=step.tool_args,
                    dependencies=step.dependencies,
                    priority=step.priority,
                ))

        self._plans[new_plan.plan_id] = new_plan
        logger.info("Created revised plan '%s' after failure at step '%s'.", new_plan.plan_id, failed_step_id)
        return new_plan
