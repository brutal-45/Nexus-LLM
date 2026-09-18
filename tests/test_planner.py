"""Test task planner for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TaskStep:
    id: int
    description: str
    dependencies: List[int] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None

    def is_ready(self, completed_steps: set) -> bool:
        return all(dep in completed_steps for dep in self.dependencies)

    def complete(self, result=None):
        self.status = "completed"
        self.result = result

    def fail(self, error: str = ""):
        self.status = "failed"
        self.result = error


@dataclass
class Plan:
    goal: str
    steps: List[TaskStep] = field(default_factory=list)

    def get_ready_steps(self) -> List[TaskStep]:
        completed = {s.id for s in self.steps if s.status == "completed"}
        return [s for s in self.steps if s.status == "pending" and s.is_ready(completed)]

    def get_completed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == "completed"]

    def get_pending_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == "pending"]

    @property
    def is_complete(self) -> bool:
        return all(s.status == "completed" for s in self.steps) if self.steps else False

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return len(self.get_completed_steps()) / len(self.steps)


class TaskPlanner:
    def __init__(self):
        self._plans: Dict[str, Plan] = {}

    def create_plan(self, goal: str, step_descriptions: List[str] = None) -> Plan:
        steps = []
        if step_descriptions:
            for i, desc in enumerate(step_descriptions):
                steps.append(TaskStep(id=i, description=desc))
        else:
            steps.append(TaskStep(id=0, description=f"Execute: {goal}"))

        plan = Plan(goal=goal, steps=steps)
        self._plans[goal] = plan
        return plan

    def create_plan_with_dependencies(self, goal: str, steps_spec: List[Dict]) -> Plan:
        steps = []
        for i, spec in enumerate(steps_spec):
            steps.append(TaskStep(
                id=i,
                description=spec.get("description", f"Step {i}"),
                dependencies=spec.get("dependencies", []),
            ))
        plan = Plan(goal=goal, steps=steps)
        self._plans[goal] = plan
        return plan

    def get_plan(self, goal: str) -> Optional[Plan]:
        return self._plans.get(goal)

    def execute_step(self, goal: str, step_id: int, result=None) -> bool:
        plan = self._plans.get(goal)
        if plan is None:
            return False
        for step in plan.steps:
            if step.id == step_id:
                step.complete(result)
                return True
        return False

    def fail_step(self, goal: str, step_id: int, error: str = "") -> bool:
        plan = self._plans.get(goal)
        if plan is None:
            return False
        for step in plan.steps:
            if step.id == step_id:
                step.fail(error)
                return True
        return False


class TestTaskStep:
    def test_creation(self):
        step = TaskStep(id=0, description="Do something")
        assert step.status == "pending"
        assert step.dependencies == []

    def test_is_ready_no_deps(self):
        step = TaskStep(id=0, description="test")
        assert step.is_ready(set()) is True

    def test_is_ready_with_deps(self):
        step = TaskStep(id=1, description="test", dependencies=[0])
        assert step.is_ready(set()) is False
        assert step.is_ready({0}) is True

    def test_complete(self):
        step = TaskStep(id=0, description="test")
        step.complete("done")
        assert step.status == "completed"
        assert step.result == "done"

    def test_fail(self):
        step = TaskStep(id=0, description="test")
        step.fail("error msg")
        assert step.status == "failed"
        assert step.result == "error msg"


class TestPlan:
    def test_creation(self):
        plan = Plan(goal="test goal")
        assert plan.goal == "test goal"
        assert plan.is_complete is False

    def test_get_ready_steps(self):
        plan = Plan(
            goal="test",
            steps=[
                TaskStep(id=0, description="step0"),
                TaskStep(id=1, description="step1", dependencies=[0]),
            ],
        )
        ready = plan.get_ready_steps()
        assert len(ready) == 1
        assert ready[0].id == 0

    def test_progress(self):
        plan = Plan(
            goal="test",
            steps=[
                TaskStep(id=0, description="step0"),
                TaskStep(id=1, description="step1"),
            ],
        )
        assert plan.progress == 0.0
        plan.steps[0].complete()
        assert plan.progress == 0.5
        plan.steps[1].complete()
        assert plan.progress == 1.0

    def test_is_complete(self):
        plan = Plan(
            goal="test",
            steps=[TaskStep(id=0, description="step0")],
        )
        assert plan.is_complete is False
        plan.steps[0].complete()
        assert plan.is_complete is True

    def test_empty_plan(self):
        plan = Plan(goal="empty")
        assert plan.is_complete is False
        assert plan.progress == 0.0


class TestTaskPlanner:
    def test_create_plan(self):
        planner = TaskPlanner()
        plan = planner.create_plan("research AI", ["search", "analyze", "report"])
        assert len(plan.steps) == 3

    def test_create_plan_default(self):
        planner = TaskPlanner()
        plan = planner.create_plan("simple task")
        assert len(plan.steps) == 1

    def test_create_plan_with_dependencies(self):
        planner = TaskPlanner()
        plan = planner.create_plan_with_dependencies("complex", [
            {"description": "step1"},
            {"description": "step2", "dependencies": [0]},
            {"description": "step3", "dependencies": [0, 1]},
        ])
        assert len(plan.steps) == 3
        assert plan.steps[2].dependencies == [0, 1]

    def test_get_plan(self):
        planner = TaskPlanner()
        planner.create_plan("test")
        assert planner.get_plan("test") is not None
        assert planner.get_plan("nonexistent") is None

    def test_execute_step(self):
        planner = TaskPlanner()
        planner.create_plan("test", ["step1"])
        result = planner.execute_step("test", 0, result="done")
        assert result is True
        plan = planner.get_plan("test")
        assert plan.steps[0].status == "completed"

    def test_execute_step_nonexistent_plan(self):
        planner = TaskPlanner()
        assert planner.execute_step("nonexistent", 0) is False

    def test_fail_step(self):
        planner = TaskPlanner()
        planner.create_plan("test", ["step1"])
        planner.fail_step("test", 0, "error")
        plan = planner.get_plan("test")
        assert plan.steps[0].status == "failed"
