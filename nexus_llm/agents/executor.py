"""Executor for Nexus-LLM agents.

Executes plans step-by-step with error handling and retries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nexus_llm.agents.planner import Plan, Step
from nexus_llm.agents.tool_registry import ToolRegistry
from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing a single step.

    Attributes:
        step_id: The step number.
        success: Whether the step completed without errors.
        output: The step's output string.
        error: Error message if the step failed.
        duration_seconds: Wall-clock execution time.
    """

    step_id: int
    success: bool
    output: str
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class ExecutionResult:
    """Aggregate result of executing a plan.

    Attributes:
        plan: The plan that was executed.
        step_results: Results for each step.
        success: Whether all steps succeeded.
        final_output: The output of the last successful step.
    """

    plan: Plan
    step_results: List[StepResult] = field(default_factory=list)
    success: bool = True
    final_output: str = ""

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "PARTIAL/FAILED"
        return f"ExecutionResult({status}, {len(self.step_results)} steps)"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class Executor:
    """Execute plans step-by-step with error handling and retries.

    Args:
        tool_registry: The registry of available tools.
        retry_attempts: Number of retries per failed step.
        retry_delay: Seconds to wait between retries.
    """

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        retry_attempts: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.registry = tool_registry or ToolRegistry()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        logger.info(
            "Executor initialised (retries=%d, delay=%.1fs)",
            retry_attempts, retry_delay,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_plan(self, plan: Plan) -> ExecutionResult:
        """Execute all steps in *plan* sequentially.

        Steps that depend on previous steps receive those outputs via
        ``step_outputs``.  If a step fails after all retries, execution
        continues with remaining steps (soft-fail mode).

        Returns:
            An :class:`ExecutionResult` with per-step details.
        """
        step_outputs: Dict[int, str] = {}
        results: List[StepResult] = []
        all_success = True

        for step in plan.steps:
            # Inject outputs from dependency steps into parameters
            enriched_params = self._inject_dependencies(
                step, step_outputs,
            )

            step_result = self._execute_with_retries(step, enriched_params)
            results.append(step_result)
            step_outputs[step.id] = step_result.output

            if not step_result.success:
                all_success = False

        final_output = results[-1].output if results else ""
        exec_result = ExecutionResult(
            plan=plan,
            step_results=results,
            success=all_success,
            final_output=final_output,
        )
        logger.info("Plan execution: %s", exec_result)
        return exec_result

    def execute_step(self, step: Step, **overrides: Any) -> StepResult:
        """Execute a single step in isolation.

        Extra keyword arguments override step parameters.

        Returns:
            A :class:`StepResult`.
        """
        params = {**step.parameters, **overrides}
        return self._execute_with_retries(step, params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_with_retries(
        self,
        step: Step,
        params: Dict[str, Any],
    ) -> StepResult:
        """Try executing a step up to ``retry_attempts + 1`` times."""
        last_error: Optional[str] = None

        for attempt in range(1, self.retry_attempts + 2):  # +1 for initial try
            start = time.monotonic()
            try:
                output = self._run_step(step, params)
                duration = time.monotonic() - start
                logger.debug(
                    "Step %d succeeded (attempt %d, %.2fs)",
                    step.id, attempt, duration,
                )
                return StepResult(
                    step_id=step.id,
                    success=True,
                    output=output,
                    duration_seconds=round(duration, 4),
                )
            except Exception as exc:
                duration = time.monotonic() - start
                last_error = str(exc)
                logger.warning(
                    "Step %d failed (attempt %d/%d): %s",
                    step.id, attempt, self.retry_attempts + 1, exc,
                )
                if attempt <= self.retry_attempts:
                    time.sleep(self.retry_delay)

        return StepResult(
            step_id=step.id,
            success=False,
            output="",
            error=last_error,
        )

    def _run_step(self, step: Step, params: Dict[str, Any]) -> str:
        """Execute a single step exactly once.

        Dependency outputs (``prev_output_N``) are filtered out of the
        parameters passed to the tool if the tool function does not
        accept them, preventing ``TypeError`` on unexpected kwargs.
        """
        if step.tool is None:
            # Reasoning-only step
            return f"Reasoning step: {step.description}"

        if not self.registry.has_tool(step.tool):
            raise KeyError(f"Tool not found: {step.tool}")

        # Strip dependency-injected keys that the tool cannot accept
        filtered = self._filter_params_for_tool(step.tool, params)
        return self.registry.execute(step.tool, **filtered)

    def _filter_params_for_tool(
        self, tool_name: str, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Remove keys the tool function doesn't accept.

        Dependency-injected keys (``prev_output_N``) are kept only if
        the underlying function accepts ``**kwargs`` or has a matching
        parameter.
        """
        import inspect

        tool_func = self.registry.get_tool(tool_name)
        if tool_func is None:
            return params

        try:
            sig = inspect.signature(tool_func)
        except (ValueError, TypeError):
            return params

        # If the function accepts **kwargs, pass everything through
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return params

        # Otherwise, keep only parameters the function actually accepts
        accepted = set(sig.parameters.keys())
        filtered = {k: v for k, v in params.items() if k in accepted}

        return filtered

    @staticmethod
    def _inject_dependencies(
        step: Step,
        step_outputs: Dict[int, str],
    ) -> Dict[str, Any]:
        """Inject outputs from dependency steps into step parameters.

        The output of dependency step N is available as ``prev_output_N``
        in the parameters.
        """
        params = dict(step.parameters)
        for dep_id in step.depends_on:
            if dep_id in step_outputs:
                params[f"prev_output_{dep_id}"] = step_outputs[dep_id]
        return params
