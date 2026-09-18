"""Planner for Nexus-LLM agents.

Decomposes tasks into executable steps using rule-based heuristics
with optional LLM-assisted planning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from nexus_llm.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Plan data structures
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """A single step in an execution plan.

    Attributes:
        id: Step number (1-based).
        description: What this step accomplishes.
        tool: The tool to use (or ``None`` for reasoning-only steps).
        parameters: Key-word arguments for the tool.
        depends_on: IDs of steps that must complete first.
    """

    id: int
    description: str
    tool: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        tool_str = f" [{self.tool}]" if self.tool else ""
        deps_str = f" (after step {self.depends_on})" if self.depends_on else ""
        return f"Step {self.id}: {self.description}{tool_str}{deps_str}"


@dataclass
class Plan:
    """An execution plan composed of ordered steps.

    Attributes:
        task: The original task description.
        steps: The ordered list of steps.
    """

    task: str
    steps: list[Step] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Plan for: {self.task}"]
        for step in self.steps:
            lines.append(f"  {step}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task decomposition heuristics
# ---------------------------------------------------------------------------

_CALC_PATTERN = re.compile(
    r"(?:calculate|compute|what\s+is|evaluate|solve)\s+(.+)",
    re.IGNORECASE,
)
_SEARCH_PATTERN = re.compile(
    r"(?:search|find|look\s+up|research)\s+(.+)",
    re.IGNORECASE,
)
_READ_PATTERN = re.compile(
    r"(?:read|open|show|display)\s+(.+)",
    re.IGNORECASE,
)
_WRITE_PATTERN = re.compile(
    r"(?:write|save|store|create)\s+(.+)",
    re.IGNORECASE,
)


class Planner:
    """Decompose tasks into execution plans.

    Uses rule-based heuristics to identify required tools and break
    complex tasks into ordered steps.  When a mock LLM is available,
    it can refine the plan further.

    Args:
        available_tools: List of tool names the planner can reference.
    """

    def __init__(self, available_tools: list[str] | None = None) -> None:
        self.available_tools = set(
            available_tools
            or [
                "calculator",
                "web_search",
                "file_read",
                "file_write",
            ]
        )
        logger.info("Planner initialised with %d available tool(s)", len(self.available_tools))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, task: str) -> Plan:
        """Create an execution plan for *task*.

        The planner analyses the task text for keywords that map to
        specific tools and creates a sequence of steps.

        Args:
            task: Natural-language task description.

        Returns:
            A :class:`Plan` with ordered :class:`Step` objects.
        """
        steps = self._decompose(task)
        plan = Plan(task=task, steps=steps)
        logger.info("Created plan with %d step(s) for: %s", len(steps), task[:80])
        return plan

    def decompose(self, task: str) -> list[Step]:
        """Decompose *task* into subtasks (alias for the step-extraction
        logic used by :meth:`plan`).

        Returns:
            A list of :class:`Step` objects.
        """
        return self._decompose(task)

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _decompose(self, task: str) -> list[Step]:
        """Apply rule-based heuristics to create steps from *task*."""
        steps: list[Step] = []
        step_id = 0

        # Check for search component
        search_match = _SEARCH_PATTERN.search(task)
        if search_match and "web_search" in self.available_tools:
            step_id += 1
            steps.append(
                Step(
                    id=step_id,
                    description=f"Search for: {search_match.group(1).strip()}",
                    tool="web_search",
                    parameters={"query": search_match.group(1).strip()},
                )
            )

        # Check for calculation component
        calc_match = _CALC_PATTERN.search(task)
        if calc_match and "calculator" in self.available_tools:
            step_id += 1
            steps.append(
                Step(
                    id=step_id,
                    description=f"Calculate: {calc_match.group(1).strip()}",
                    tool="calculator",
                    parameters={"expression": calc_match.group(1).strip()},
                )
            )

        # Check for file read component
        read_match = _READ_PATTERN.search(task)
        if read_match and "file_read" in self.available_tools:
            step_id += 1
            steps.append(
                Step(
                    id=step_id,
                    description=f"Read file: {read_match.group(1).strip()}",
                    tool="file_read",
                    parameters={"path": read_match.group(1).strip()},
                )
            )

        # Check for file write component
        write_match = _WRITE_PATTERN.search(task)
        if write_match and "file_write" in self.available_tools:
            step_id += 1
            depends = [s.id for s in steps] if steps else []
            steps.append(
                Step(
                    id=step_id,
                    description=f"Write: {write_match.group(1).strip()}",
                    tool="file_write",
                    parameters={"path": write_match.group(1).strip(), "content": ""},
                    depends_on=depends,
                )
            )

        # If no specific steps detected, create a reasoning step
        if not steps:
            step_id += 1
            steps.append(
                Step(
                    id=step_id,
                    description=f"Reason about: {task}",
                    tool=None,
                    parameters={},
                )
            )

        return steps


# ---------------------------------------------------------------------------
# Strategy-driven planning
# ---------------------------------------------------------------------------


#: Steps used per strategy.  Each entry is ``(description_template, tool)`` and
#: ``{task}`` is substituted into the description.
_STRATEGY_STEPS: dict[str, list[tuple[str, str | None]]] = {
    "research": [
        ("Search for an overview of {task}", "search"),
        ("Identify open questions from the initial results", None),
        ("Search for details on each open question", "search"),
        ("Cross-check conflicting findings", "search"),
        ("Synthesize findings into a cited report", None),
    ],
    "code": [
        ("Clarify the requirements of {task}", None),
        ("Draft an implementation", "code_run"),
        ("Run tests on the implementation", "code_run"),
        ("Fix failures and re-run", "code_run"),
    ],
    "analysis": [
        ("Collect the data needed for {task}", "search"),
        ("Compute the relevant metrics", "calculator"),
        ("Interpret the results", None),
    ],
    "generic": [
        ("Understand {task}", None),
        ("Gather what is needed", "search"),
        ("Produce the answer", None),
    ],
}

DEFAULT_STRATEGY = "generic"


class TaskPlanner(Planner):
    """Strategy-aware planner used by the research and code agents.

    Where :class:`Planner` decomposes a task purely from its wording,
    :class:`TaskPlanner` first picks a *strategy* (a research workflow, a
    code-fix loop, ...) and produces the step sequence that strategy implies.
    When an ``llm_fn`` is supplied the model is asked to decompose the task and
    its answer is validated; anything unusable falls back to the built-in
    templates so planning never fails.

    Args:
        llm_fn: Optional callable ``(prompt: str) -> str`` backed by a model.
        available_tools: Tool names the planner already knows about; used for
            validation hints only (see :meth:`_usable_tool`).
        default_strategy: Strategy used when none is given.

    Example::

        planner = TaskPlanner()
        plan = planner.create_plan("why do tides happen", strategy="research")
        assert [s.tool for s in plan.steps][:2] == ["search", None]
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        available_tools: list[str] | None = None,
        default_strategy: str = DEFAULT_STRATEGY,
    ) -> None:
        super().__init__(available_tools=available_tools)
        self.llm_fn = llm_fn
        self.default_strategy = (
            default_strategy if default_strategy in _STRATEGY_STEPS else DEFAULT_STRATEGY
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def strategies() -> list[str]:
        """Return the names of the built-in planning strategies."""
        return sorted(_STRATEGY_STEPS)

    def create_plan(self, task: str, strategy: str | None = None, **kwargs: Any) -> Plan:
        """Build a :class:`Plan` for *task* using *strategy*.

        Args:
            task: The task to plan for.
            strategy: One of :meth:`strategies`; unknown or missing values fall
                back to the default strategy.
            **kwargs: Extra context forwarded to the LLM prompt (e.g. ``depth``).

        Returns:
            A :class:`Plan` with at least one step.
        """
        if not task or not str(task).strip():
            raise ValueError("create_plan requires a non-empty task description")

        name = (strategy or self.default_strategy).strip().lower()
        if name not in _STRATEGY_STEPS:
            logger.debug("Unknown strategy %r; using %r", name, self.default_strategy)
            name = self.default_strategy

        steps = self._llm_steps(str(task).strip(), name, kwargs) or self._template_steps(
            str(task).strip(), name
        )
        return Plan(task=str(task).strip(), steps=steps)

    def plan(self, task: str, strategy: str | None = None, **kwargs: Any) -> Plan:
        """Alias for :meth:`create_plan` (keeps the :class:`Planner` API working)."""
        return self.create_plan(task, strategy=strategy, **kwargs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _template_steps(self, task: str, strategy: str) -> list[Step]:
        """Expand the strategy template into concrete steps."""
        steps: list[Step] = []
        for index, (description, tool) in enumerate(_STRATEGY_STEPS[strategy], start=1):
            steps.append(
                Step(
                    id=index,
                    description=description.format(task=task),
                    tool=self._usable_tool(tool),
                    parameters={"query": task} if tool == "search" else {},
                    depends_on=[index - 1] if index > 1 else [],
                )
            )
        return steps

    def _usable_tool(self, tool: str | None) -> str | None:
        """Validate *tool* against the planner's known tools.

        Unknown names are *kept* rather than dropped: agents register their
        tools after construction, so the planner must not silently downgrade a
        step to pure reasoning just because it has not seen the tool yet.
        """
        if tool is None:
            return None
        allowed = getattr(self, "available_tools", None)
        if allowed and tool not in allowed:
            logger.debug("Tool %r is not among the planner's known tools", tool)
        return tool

    def _llm_steps(self, task: str, strategy: str, context: dict[str, Any]) -> list[Step]:
        """Ask the model for a decomposition, returning ``[]`` when unusable."""
        if self.llm_fn is None:
            return []
        prompt = (
            "Break the task into short ordered steps, one per line, optionally"
            " suffixing a tool in brackets.\n"
            f"Strategy: {strategy}\n"
            f"Context: {context or 'none'}\n"
            f"Task: {task}\n"
        )
        try:
            raw = self.llm_fn(prompt)
        except Exception as exc:
            logger.warning("LLM planning failed (%s); using template strategy", exc)
            return []
        if not isinstance(raw, str) or not raw.strip():
            return []
        return self._parse_llm_steps(raw)

    def _parse_llm_steps(self, raw: str) -> list[Step]:
        """Parse ``"do thing [tool]"`` lines into validated steps."""
        steps: list[Step] = []
        for line in raw.splitlines():
            text = line.strip().lstrip("-*0123456789.) ").strip()
            if not text:
                continue
            tool: str | None = None
            match = re.search(r"\[([a-zA-Z0-9_-]+)\]\s*$", text)
            if match:
                tool = self._usable_tool(match.group(1))
                text = text[: match.start()].strip()
            if not text:
                continue
            steps.append(
                Step(
                    id=len(steps) + 1,
                    description=text,
                    tool=tool,
                    parameters={},
                    depends_on=[len(steps)] if steps else [],
                )
            )
        return steps
