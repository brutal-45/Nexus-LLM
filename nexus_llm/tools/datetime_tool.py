"""Nexus-LLM Date/Time Tool.

Provides the DateTimeTool for date and time operations including
current time, formatting, parsing, and arithmetic.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class DateTimeTool(BaseTool):
    """Date and time tool for temporal operations.

    Supports operations: now, format, parse, add, subtract, diff, timezone_list.

    Example::

        dt = DateTimeTool()
        result = dt.execute(operation="now", format="%Y-%m-%d %H:%M:%S")
        result = dt.execute(operation="add", datetime_str="2024-01-01", days=30)
    """

    def __init__(self) -> None:
        super().__init__(name="datetime", description="Date and time operations")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="DateTime operation", required=True,
                          choices=["now", "format", "parse", "add", "subtract", "diff", "timezone_list"]),
            ToolParameter(name="datetime_str", type=ParameterType.STRING, description="Datetime string to operate on", required=False),
            ToolParameter(name="format", type=ParameterType.STRING, description=" strftime/strptime format string", required=False, default="%Y-%m-%d %H:%M:%S"),
            ToolParameter(name="days", type=ParameterType.INTEGER, description="Days to add/subtract", required=False, default=0),
            ToolParameter(name="hours", type=ParameterType.INTEGER, description="Hours to add/subtract", required=False, default=0),
            ToolParameter(name="minutes", type=ParameterType.INTEGER, description="Minutes to add/subtract", required=False, default=0),
            ToolParameter(name="seconds", type=ParameterType.INTEGER, description="Seconds to add/subtract", required=False, default=0),
            ToolParameter(name="datetime_str2", type=ParameterType.STRING, description="Second datetime for diff", required=False),
        ]

    def execute(
        self,
        operation: str = "",
        datetime_str: str = "",
        format: str = "%Y-%m-%d %H:%M:%S",
        days: int = 0,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        datetime_str2: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute a datetime operation.

        Args:
            operation: The operation to perform.
            datetime_str: Input datetime string.
            format: Format string for formatting/parsing.
            days: Days for add/subtract.
            hours: Hours for add/subtract.
            minutes: Minutes for add/subtract.
            seconds: Seconds for add/subtract.
            datetime_str2: Second datetime string for diff.

        Returns:
            ToolResult with the operation output.
        """
        try:
            if operation == "now":
                return self._now(format)
            elif operation == "format":
                return self._format(datetime_str, format)
            elif operation == "parse":
                return self._parse(datetime_str, format)
            elif operation == "add":
                return self._add(datetime_str, format, days, hours, minutes, seconds)
            elif operation == "subtract":
                return self._subtract(datetime_str, format, days, hours, minutes, seconds)
            elif operation == "diff":
                return self._diff(datetime_str, datetime_str2, format)
            elif operation == "timezone_list":
                return self._timezone_list()
            else:
                return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    def _now(self, fmt: str) -> ToolResult:
        now = datetime.now(timezone.utc)
        return ToolResult(
            tool_name=self.name,
            success=True,
            output=now.strftime(fmt),
            metadata={"iso": now.isoformat(), "timezone": "UTC"},
        )

    def _format(self, datetime_str: str, fmt: str) -> ToolResult:
        dt = datetime.fromisoformat(datetime_str)
        return ToolResult(tool_name=self.name, success=True, output=dt.strftime(fmt))

    def _parse(self, datetime_str: str, fmt: str) -> ToolResult:
        dt = datetime.strptime(datetime_str, fmt)
        return ToolResult(tool_name=self.name, success=True, output=dt.isoformat(), metadata={"parsed": True})

    def _add(self, datetime_str: str, fmt: str, days: int, hours: int, minutes: int, seconds: int) -> ToolResult:
        dt = self._parse_input(datetime_str, fmt)
        result = dt + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return ToolResult(
            tool_name=self.name, success=True, output=result.strftime(fmt),
            metadata={"operation": "add", "delta": {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}},
        )

    def _subtract(self, datetime_str: str, fmt: str, days: int, hours: int, minutes: int, seconds: int) -> ToolResult:
        dt = self._parse_input(datetime_str, fmt)
        result = dt - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return ToolResult(
            tool_name=self.name, success=True, output=result.strftime(fmt),
            metadata={"operation": "subtract", "delta": {"days": days, "hours": hours, "minutes": minutes, "seconds": seconds}},
        )

    def _diff(self, dt_str1: str, dt_str2: str, fmt: str) -> ToolResult:
        dt1 = self._parse_input(dt_str1, fmt)
        dt2 = self._parse_input(dt_str2, fmt)
        delta = dt2 - dt1
        return ToolResult(
            tool_name=self.name, success=True,
            output={
                "total_seconds": delta.total_seconds(),
                "days": delta.days,
                "seconds": delta.seconds,
            },
            metadata={"from": dt_str1, "to": dt_str2},
        )

    def _timezone_list(self) -> ToolResult:
        common_tz = [
            "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
            "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo",
            "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney",
        ]
        return ToolResult(tool_name=self.name, success=True, output=common_tz)

    @staticmethod
    def _parse_input(datetime_str: str, fmt: str) -> datetime:
        """Parse a datetime string, trying ISO format first, then the given format."""
        if not datetime_str:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(datetime_str)
        except ValueError:
            return datetime.strptime(datetime_str, fmt)
