"""Agent for Nexus-LLM.

Implements a ReAct-style (Reason + Act) agent loop with tool access,
memory, and planning capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.agents.config import AgentConfig
from nexus_llm.agents.tool_registry import ToolRegistry
from nexus_llm.agents.planner import Planner, Plan, Step
from nexus_llm.agents.executor import Executor, ExecutionResult
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent memory
# ---------------------------------------------------------------------------

@dataclass
class AgentMemory:
    """Simple conversation / action memory for the agent.

    Attributes:
        observations: List of observation strings from executed steps.
        actions: List of action descriptions taken.
    """

    observations: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def add_observation(self, obs: str) -> None:
        self.observations.append(obs)

    def add_action(self, action: str) -> None:
        self.actions.append(action)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()

    @property
    def last_observation(self) -> str:
        return self.observations[-1] if self.observations else ""

    @property
    def last_action(self) -> str:
        return self.actions[-1] if self.actions else ""


# ---------------------------------------------------------------------------
# Agent result
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """The final result of an agent run.

    Attributes:
        task: The original task description.
        answer: The agent's final answer.
        success: Whether the agent completed the task successfully.
        steps_taken: Number of steps executed.
        memory: The agent's memory at the end of the run.
    """

    task: str
    answer: str
    success: bool
    steps_taken: int
    memory: AgentResult = None  # type: ignore[assignment]

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"[{status}] steps={self.steps_taken} answer={self.answer[:100]}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """ReAct-style agent with tool access, memory, and planning.

    The agent loop follows:
    1. **Reason** – analyse the current state and decide the next action.
    2. **Act** – execute a tool or reasoning step.
    3. **Observe** – record the result.
    4. Repeat until the task is complete or ``max_iterations`` is reached.

    Args:
        config: Agent configuration.  Defaults are used if not provided.
        tool_registry: Registry of available tools.  A fresh one with
            built-in tools is created if not provided.
        planner: The planner for decomposing tasks.
        executor: The executor for running plan steps.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.tools = tool_registry or ToolRegistry()
        self.planner = planner or Planner(
            available_tools=[t.name for t in self.tools.list_tools()],
        )
        self.executor = executor or Executor(tool_registry=self.tools)
        self.memory = AgentMemory()

        logger.info(
            "Agent '%s' initialised (max_iter=%d, tools=%d)",
            self.config.name,
            self.config.max_iterations,
            len(self.tools.list_tools()),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Run the full ReAct agent loop for *task*.

        The agent:
        1. Plans the task into steps.
        2. Executes each step.
        3. Observes results and decides whether to continue.
        4. Returns the final answer.

        Args:
            task: The task to accomplish.

        Returns:
            An :class:`AgentResult` with the final answer.
        """
        logger.info("Agent '%s' starting task: %s", self.config.name, task[:100])
        self.memory.clear()

        # Plan
        plan = self.planner.plan(task)
        if self.config.verbose:
            logger.info("Plan:\n%s", plan)

        # Execute
        exec_result = self.executor.execute_plan(plan)

        # Record observations
        for step_result in exec_result.step_results:
            self.memory.add_action(f"Step {step_result.step_id}")
            self.memory.add_observation(step_result.output)

        # Iterative refinement loop
        iterations = len(exec_result.step_results)
        final_output = exec_result.final_output

        if not exec_result.success and iterations < self.config.max_iterations:
            # Try a simplified re-plan
            refined_plan = self.planner.plan(task)
            refined_exec = self.executor.execute_plan(refined_plan)
            for step_result in refined_exec.step_results:
                self.memory.add_action(f"Step {step_result.step_id} (retry)")
                self.memory.add_observation(step_result.output)
            iterations += len(refined_exec.step_results)
            if refined_exec.final_output:
                final_output = refined_exec.final_output

        success = exec_result.success
        result = AgentResult(
            task=task,
            answer=final_output,
            success=success,
            steps_taken=iterations,
        )
        logger.info("Agent '%s' completed: %s", self.config.name, result.summary())
        return result

    def step(self, task: str) -> tuple:
        """Execute a single step: plan → act → observe.

        Returns:
            A tuple of ``(action_description, observation)``.
        """
        plan = self.planner.plan(task)
        if not plan.steps:
            return ("no_action", "No steps planned")

        step = plan.steps[0]
        step_result = self.executor.execute_step(step)

        action = step.description
        observation = step_result.output

        self.memory.add_action(action)
        self.memory.add_observation(observation)

        return (action, observation)
