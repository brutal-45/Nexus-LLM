"""Planner for Nexus-LLM agents.

Decomposes tasks into executable steps using rule-based heuristics
with optional LLM-assisted planning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    tool: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)

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
    steps: List[Step] = field(default_factory=list)

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

    def __init__(self, available_tools: Optional[List[str]] = None) -> None:
        self.available_tools = set(available_tools or [
            "calculator", "web_search", "file_read", "file_write",
        ])
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

    def decompose(self, task: str) -> List[Step]:
        """Decompose *task* into subtasks (alias for the step-extraction
        logic used by :meth:`plan`).

        Returns:
            A list of :class:`Step` objects.
        """
        return self._decompose(task)

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    def _decompose(self, task: str) -> List[Step]:
        """Apply rule-based heuristics to create steps from *task*."""
        steps: List[Step] = []
        step_id = 0

        # Check for search component
        search_match = _SEARCH_PATTERN.search(task)
        if search_match and "web_search" in self.available_tools:
            step_id += 1
            steps.append(Step(
                id=step_id,
                description=f"Search for: {search_match.group(1).strip()}",
                tool="web_search",
                parameters={"query": search_match.group(1).strip()},
            ))

        # Check for calculation component
        calc_match = _CALC_PATTERN.search(task)
        if calc_match and "calculator" in self.available_tools:
            step_id += 1
            steps.append(Step(
                id=step_id,
                description=f"Calculate: {calc_match.group(1).strip()}",
                tool="calculator",
                parameters={"expression": calc_match.group(1).strip()},
            ))

        # Check for file read component
        read_match = _READ_PATTERN.search(task)
        if read_match and "file_read" in self.available_tools:
            step_id += 1
            steps.append(Step(
                id=step_id,
                description=f"Read file: {read_match.group(1).strip()}",
                tool="file_read",
                parameters={"path": read_match.group(1).strip()},
            ))

        # Check for file write component
        write_match = _WRITE_PATTERN.search(task)
        if write_match and "file_write" in self.available_tools:
            step_id += 1
            depends = [s.id for s in steps] if steps else []
            steps.append(Step(
                id=step_id,
                description=f"Write: {write_match.group(1).strip()}",
                tool="file_write",
                parameters={"path": write_match.group(1).strip(), "content": ""},
                depends_on=depends,
            ))

        # If no specific steps detected, create a reasoning step
        if not steps:
            step_id += 1
            steps.append(Step(
                id=step_id,
                description=f"Reason about: {task}",
                tool=None,
                parameters={},
            ))

        return steps
