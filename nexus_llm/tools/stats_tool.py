"""Nexus-LLM Statistics Tool.

Provides statistical analysis capabilities including descriptive
statistics, distribution analysis, and correlation calculations.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class StatsTool(BaseTool):
    """Tool for statistical analysis.

    Supports descriptive statistics, frequency analysis, and
    basic inferential statistics.

    Example::

        tool = StatsTool()
        result = tool.run(operation="describe", data=[1, 2, 3, 4, 5])
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="stats", description="Statistical analysis", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Operation to perform",
                         required=True, choices=["describe", "mean", "median", "std", "variance", "correlation", "frequency"]),
            ToolParameter(name="data", type=ParameterType.ARRAY, description="Numeric data array", required=True),
            ToolParameter(name="data2", type=ParameterType.ARRAY, description="Second data array for correlation", required=False),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        operation = kwargs.get("operation", "")
        data = kwargs.get("data", [])

        if not operation:
            return ToolResult(success=False, error="No operation specified")
        if not data:
            return ToolResult(success=False, error="No data provided")

        try:
            if operation == "describe":
                return self._describe(data)
            elif operation == "mean":
                return ToolResult(success=True, output=self._mean(data))
            elif operation == "median":
                return ToolResult(success=True, output=self._median(data))
            elif operation == "std":
                return ToolResult(success=True, output=self._std(data))
            elif operation == "variance":
                return ToolResult(success=True, output=self._variance(data))
            elif operation == "correlation":
                data2 = kwargs.get("data2", [])
                return self._correlation(data, data2)
            elif operation == "frequency":
                return self._frequency(data)
            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _mean(self, data: List) -> float:
        return sum(data) / len(data)

    def _median(self, data: List) -> float:
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        return sorted_data[n // 2]

    def _variance(self, data: List) -> float:
        m = self._mean(data)
        return sum((x - m) ** 2 for x in data) / len(data)

    def _std(self, data: List) -> float:
        return math.sqrt(self._variance(data))

    def _describe(self, data: List) -> ToolResult:
        m = self._mean(data)
        med = self._median(data)
        var = self._variance(data)
        std = math.sqrt(var)
        return ToolResult(success=True, output={
            "count": len(data),
            "mean": m,
            "median": med,
            "min": min(data),
            "max": max(data),
            "variance": var,
            "std": std,
            "range": max(data) - min(data),
            "sum": sum(data),
        })

    def _correlation(self, data1: List, data2: List) -> ToolResult:
        if not data2:
            return ToolResult(success=False, error="Second data array required for correlation")
        if len(data1) != len(data2):
            return ToolResult(success=False, error="Data arrays must have equal length")
        n = len(data1)
        m1 = self._mean(data1)
        m2 = self._mean(data2)
        cov = sum((data1[i] - m1) * (data2[i] - m2) for i in range(n)) / n
        std1 = self._std(data1)
        std2 = self._std(data2)
        if std1 == 0 or std2 == 0:
            corr = 0.0
        else:
            corr = cov / (std1 * std2)
        return ToolResult(success=True, output={"correlation": corr})

    def _frequency(self, data: List) -> ToolResult:
        freq: Dict[Any, int] = {}
        for item in data:
            freq[item] = freq.get(item, 0) + 1
        return ToolResult(success=True, output=freq)
