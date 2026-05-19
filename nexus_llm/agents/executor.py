"""Action executor for running tools, validating results, and handling errors.

Provides robust tool execution with input validation, result checking,
error handling, retry logic, and execution tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from nexus_llm.agents.tools import Tool, ToolResult

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an action execution."""

    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    VALIDATION_FAILED = "validation_failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionRecord:
    """Record of a single action execution."""

    tool_name: str
    tool_args: Dict[str, Any]
    status: ExecutionStatus
    result: Optional[ToolResult] = None
    attempts: int = 1
    total_time: float = 0.0
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "status": self.status.value,
            "attempts": self.attempts,
            "total_time": self.total_time,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class ActionExecutor:
    """Executes agent actions with validation, retry, and error handling.

    Provides a robust execution layer between agents and tools,
    handling argument validation, result checking, retry logic
    with exponential backoff, and execution logging.
    """

    def __init__(
        self,
        tools: Optional[Dict[str, Tool]] = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout: float = 60.0,
        validators: Optional[Dict[str, Callable]] = None,
    ):
        """Initialize the action executor.

        Args:
            tools: Dictionary of available tools.
            max_retries: Maximum number of retry attempts per action.
            base_delay: Base delay for exponential backoff (seconds).
            max_delay: Maximum delay between retries (seconds).
            timeout: Maximum execution time per attempt (seconds).
            validators: Optional dict mapping tool names to validation functions.
        """
        self.tools: Dict[str, Tool] = tools or {}
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.validators = validators or {}
        self._execution_history: List[ExecutionRecord] = []

    def register_tool(self, tool: Tool) -> None:
        """Register a tool with the executor."""
        self.tools[tool.name] = tool

    def register_validator(self, tool_name: str, validator: Callable[[ToolResult], bool]) -> None:
        """Register a result validator for a tool.

        Args:
            tool_name: Name of the tool.
            validator: Function that takes a ToolResult and returns True if valid.
        """
        self.validators[tool_name] = validator

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool with retry logic and error handling.

        Args:
            tool_name: Name of the tool to execute.
            **kwargs: Arguments to pass to the tool.

        Returns:
            ToolResult from the execution.
        """
        start_time = time.time()
        record = ExecutionRecord(
            tool_name=tool_name,
            tool_args=kwargs,
            status=ExecutionStatus.FAILED,
        )

        # Check tool exists
        if tool_name not in self.tools:
            error_msg = f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"
            record.error = error_msg
            record.status = ExecutionStatus.FAILED
            self._execution_history.append(record)
            return ToolResult(success=False, error=error_msg)

        tool = self.tools[tool_name]

        # Validate tool arguments
        if not tool.validate_args(**kwargs):
            error_msg = f"Invalid arguments for tool '{tool_name}'. Expected: {tool.parameters}"
            record.error = error_msg
            record.status = ExecutionStatus.VALIDATION_FAILED
            self._execution_history.append(record)
            return ToolResult(success=False, error=error_msg)

        # Execute with retries
        result = None
        last_error = ""

        for attempt in range(1, self.max_retries + 2):  # +2 for initial + retries
            record.attempts = attempt

            try:
                result = tool.execute(**kwargs)
                record.result = result

                # Validate result if validator exists
                validator = self.validators.get(tool_name)
                if validator and not validator(result):
                    if result.success:
                        result = ToolResult(
                            success=False,
                            output=result.output,
                            error="Result failed validation check",
                            data=result.data,
                        )
                    record.status = ExecutionStatus.VALIDATION_FAILED
                    last_error = "Result failed validation"
                    if attempt <= self.max_retries:
                        self._backoff_delay(attempt)
                        continue
                    break

                # Check for success
                if result.success:
                    record.status = ExecutionStatus.SUCCESS
                    break
                else:
                    last_error = result.error or "Tool execution failed"
                    record.status = ExecutionStatus.RETRYING if attempt <= self.max_retries else ExecutionStatus.FAILED

            except Exception as e:
                last_error = str(e)
                result = ToolResult(success=False, error=f"Exception: {e}")
                record.result = result
                record.status = ExecutionStatus.RETRYING if attempt <= self.max_retries else ExecutionStatus.FAILED

            # Retry with backoff
            if record.status == ExecutionStatus.RETRYING:
                logger.info(
                    "Retrying tool '%s' (attempt %d/%d): %s",
                    tool_name, attempt, self.max_retries + 1, last_error,
                )
                self._backoff_delay(attempt)

        # Finalize record
        record.total_time = time.time() - start_time
        if record.status == ExecutionStatus.RETRYING:
            record.status = ExecutionStatus.FAILED

        record.error = last_error
        self._execution_history.append(record)

        logger.debug(
            "Executed '%s': status=%s, attempts=%d, time=%.2fs",
            tool_name, record.status.value, record.attempts, record.total_time,
        )

        return result or ToolResult(success=False, error="No result produced")

    def _backoff_delay(self, attempt: int) -> None:
        """Apply exponential backoff delay."""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        delay = min(delay + (delay * 0.1 * (attempt - 1)), self.max_delay)  # Add jitter
        time.sleep(delay)

    def get_execution_history(self, tool_name: Optional[str] = None, limit: int = 50) -> List[ExecutionRecord]:
        """Get execution history, optionally filtered by tool name.

        Args:
            tool_name: Optional tool name to filter by.
            limit: Maximum number of records to return.

        Returns:
            List of ExecutionRecord objects.
        """
        records = self._execution_history
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        return records[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {"total_executions": 0}

        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.status == ExecutionStatus.SUCCESS)
        failures = sum(1 for r in self._execution_history if r.status == ExecutionStatus.FAILED)
        avg_time = sum(r.total_time for r in self._execution_history) / total

        by_tool: Dict[str, Dict[str, int]] = {}
        for record in self._execution_history:
            if record.tool_name not in by_tool:
                by_tool[record.tool_name] = {"total": 0, "success": 0, "failed": 0}
            by_tool[record.tool_name]["total"] += 1
            if record.status == ExecutionStatus.SUCCESS:
                by_tool[record.tool_name]["success"] += 1
            elif record.status == ExecutionStatus.FAILED:
                by_tool[record.tool_name]["failed"] += 1

        return {
            "total_executions": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "avg_execution_time": avg_time,
            "by_tool": by_tool,
        }

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
