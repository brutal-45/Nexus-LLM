"""Executor for Nexus-LLM agents.

Executes plans step-by-step with error handling and retries.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from nexus_llm.agents.planner import Plan, Step
from nexus_llm.agents.tools import ToolResult
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
    error: str | None = None
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
    step_results: list[StepResult] = field(default_factory=list)
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
        tool_registry: ToolRegistry | None = None,
        retry_attempts: int = 2,
        retry_delay: float = 1.0,
    ) -> None:
        self.registry = tool_registry or ToolRegistry()
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        logger.info(
            "Executor initialised (retries=%d, delay=%.1fs)",
            retry_attempts,
            retry_delay,
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
        step_outputs: dict[int, str] = {}
        results: list[StepResult] = []
        all_success = True

        for step in plan.steps:
            # Inject outputs from dependency steps into parameters
            enriched_params = self._inject_dependencies(
                step,
                step_outputs,
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
        params: dict[str, Any],
    ) -> StepResult:
        """Try executing a step up to ``retry_attempts + 1`` times."""
        last_error: str | None = None

        for attempt in range(1, self.retry_attempts + 2):  # +1 for initial try
            start = time.monotonic()
            try:
                output = self._run_step(step, params)
                duration = time.monotonic() - start
                logger.debug(
                    "Step %d succeeded (attempt %d, %.2fs)",
                    step.id,
                    attempt,
                    duration,
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
                    step.id,
                    attempt,
                    self.retry_attempts + 1,
                    exc,
                )
                if attempt <= self.retry_attempts:
                    time.sleep(self.retry_delay)

        return StepResult(
            step_id=step.id,
            success=False,
            output="",
            error=last_error,
        )

    def _run_step(self, step: Step, params: dict[str, Any]) -> str:
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
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
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
        step_outputs: dict[int, str],
    ) -> dict[str, Any]:
        """Inject outputs from dependency steps into step parameters.

        The output of dependency step N is available as ``prev_output_N``
        in the parameters.
        """
        params = dict(step.parameters)
        for dep_id in step.depends_on:
            if dep_id in step_outputs:
                params[f"prev_output_{dep_id}"] = step_outputs[dep_id]
        return params


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------


class UnknownActionError(KeyError):
    """Raised when an action references a tool that is not registered."""


class ActionExecutor:
    """Execute single named tool actions, with retries and argument checking.

    While :class:`Executor` runs a whole :class:`~nexus_llm.agents.planner.Plan`
    of interdependent steps, :class:`ActionExecutor` serves the ReAct-style
    loop used by the agents: the model emits *one* tool call, the executor runs
    it and hands the :class:`~nexus_llm.agents.tools.ToolResult` back so the
    agent can observe and decide the next step.

    Args:
        tools: Mapping of tool name to :class:`~nexus_llm.agents.tools.Tool`.
            The agent's own ``tools`` dict can be passed straight through; the
            executor will share it so tools registered later are visible.
        retry_attempts: How many times a failing tool call is retried.
        retry_delay: Seconds to wait between retries.
        registry: Optional fallback :class:`ToolRegistry` consulted when a
            name is not present in *tools*.

    Example::

        executor = ActionExecutor(tools={"calculator": CalculatorTool()})
        result = executor.execute("calculator", expression="2 + 2")
        assert result.success and result.output == "4"
    """

    def __init__(
        self,
        tools: dict[str, Any] | None = None,
        retry_attempts: int = 2,
        retry_delay: float = 0.5,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.tools: dict[str, Any] = tools if tools is not None else {}
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay = retry_delay
        self.registry = registry
        logger.debug(
            "ActionExecutor initialised with %d tool(s), retries=%d",
            len(self.tools),
            self.retry_attempts,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_tool(self, tool: Any, name: str | None = None) -> None:
        """Register a tool instance under *name* (defaults to ``tool.name``)."""
        key = name or getattr(tool, "name", None)
        if not key:
            raise ValueError("Tool must have a 'name' attribute or an explicit name must be given")
        self.tools[key] = tool
        logger.debug("ActionExecutor: registered tool '%s'", key)

    def unregister_tool(self, name: str) -> bool:
        """Remove a tool by name.  Returns ``True`` when it was present."""
        return self.tools.pop(name, None) is not None

    def has_tool(self, name: str) -> bool:
        """Return whether *name* can be executed."""
        return name in self.tools or (self.registry is not None and self.registry.has_tool(name))

    def list_tools(self) -> list[str]:
        """Return every executable tool name."""
        names = set(self.tools)
        if self.registry is not None:
            names.update(info.name for info in self.registry.list_tools())
        return sorted(names)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Run the tool registered as *tool_name*.

        Args:
            tool_name: Name of the tool to call.
            **kwargs: Arguments forwarded to the tool.

        Returns:
            A :class:`~nexus_llm.agents.tools.ToolResult`.  Failures never
            raise: an unknown tool or a tool that blows up is reported as an
            unsuccessful result so the agent can recover and try again.
        """
        tool = self.tools.get(tool_name)
        if tool is None:
            if self.registry is not None and self.registry.has_tool(tool_name):
                return self._execute_via_registry(tool_name, kwargs)
            return ToolResult(
                success=False,
                error=f"No tool registered under '{tool_name}'. Available: {self.list_tools()}",
            )

        validate = getattr(tool, "validate_args", None)
        if callable(validate) and not validate(**kwargs):
            return ToolResult(
                success=False,
                error=f"Invalid arguments for tool '{tool_name}': {sorted(kwargs)}",
            )

        last_error = "unknown error"
        for attempt in range(1, self.retry_attempts + 1):
            started = time.perf_counter()
            try:
                result = tool.execute(**kwargs)
            except Exception as exc:  # noqa: BLE001 - tools may raise anything
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Tool '%s' failed (attempt %d/%d): %s",
                    tool_name,
                    attempt,
                    self.retry_attempts,
                    last_error,
                )
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
                continue

            result = self._normalise(result)
            if result.execution_time == 0.0:
                result.execution_time = time.perf_counter() - started
            if result.success:
                return result

            last_error = result.error or "tool reported failure"
            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay)

        return ToolResult(success=False, error=f"Tool '{tool_name}' failed: {last_error}")

    def __call__(self, tool_name: str, **kwargs: Any) -> Any:
        """Allow the executor to be used directly as a callable tool runner."""
        return self.execute(tool_name, **kwargs)

    def __contains__(self, tool_name: object) -> bool:
        return isinstance(tool_name, str) and self.has_tool(tool_name)

    def __len__(self) -> int:
        return len(self.list_tools())

    def __repr__(self) -> str:
        return f"ActionExecutor(tools={self.list_tools()!r})"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _execute_via_registry(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Execute through the fallback :class:`ToolRegistry`."""
        assert self.registry is not None
        started = time.perf_counter()
        try:
            output = self.registry.execute(tool_name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(success=True, output=str(output), execution_time=time.perf_counter() - started)

    @staticmethod
    def _normalise(result: Any) -> Any:
        """Coerce a tool's return value into a ``ToolResult``."""
        if isinstance(result, ToolResult):
            return result
        if isinstance(result, str):
            return ToolResult(success=True, output=result)
        if result is None:
            return ToolResult(success=True, output="")
        if isinstance(result, dict) and "success" in result:
            return ToolResult(
                success=bool(result.get("success")),
                output=str(result.get("output", "")),
                error=str(result.get("error", "") or ""),
                data=result.get("data"),
            )
        return ToolResult(success=True, output=str(result))
