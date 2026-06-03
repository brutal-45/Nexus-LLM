"""Nexus-LLM Math Operations Tool.

Provides the MathTool for advanced mathematical operations beyond
basic arithmetic, including statistics, linear algebra helpers,
and number theory.
"""

import logging
import math
from typing import Any, Dict, List

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)


class MathTool(BaseTool):
    """Advanced math operations tool.

    Supports operations: stats, gcd, lcm, factorial, fibonacci,
    is_prime, prime_factors, permutations, combinations, degrees, radians,
    distance, mean, median, stddev.

    Example::

        mtool = MathTool()
        result = mtool.execute(operation="stats", numbers="1,2,3,4,5")
        result = mtool.execute(operation="fibonacci", n=10)
    """

    def __init__(self) -> None:
        super().__init__(name="math", description="Advanced mathematical operations and statistics")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="operation", type=ParameterType.STRING, description="Math operation", required=True,
                          choices=["stats", "gcd", "lcm", "factorial", "fibonacci", "is_prime", "prime_factors",
                                   "permutations", "combinations", "degrees", "radians", "distance", "mean", "median", "stddev"]),
            ToolParameter(name="numbers", type=ParameterType.STRING, description="Comma-separated numbers", required=False),
            ToolParameter(name="n", type=ParameterType.INTEGER, description="Single integer parameter", required=False),
            ToolParameter(name="k", type=ParameterType.INTEGER, description="Second integer parameter (for nCk, nPk)", required=False),
        ]

    def execute(self, operation: str = "", numbers: str = "", n: int = 0, k: int = 0, **kwargs: Any) -> ToolResult:
        """Execute a math operation.

        Args:
            operation: The operation to perform.
            numbers: Comma-separated numbers.
            n: First integer parameter.
            k: Second integer parameter.

        Returns:
            ToolResult with the computation output.
        """
        try:
            nums = self._parse_numbers(numbers) if numbers else []

            if operation == "stats":
                return self._stats(nums)
            elif operation == "gcd":
                return self._gcd_op(nums)
            elif operation == "lcm":
                return self._lcm_op(nums)
            elif operation == "factorial":
                return ToolResult(tool_name=self.name, success=True, output=math.factorial(n), metadata={"n": n})
            elif operation == "fibonacci":
                return self._fibonacci(n)
            elif operation == "is_prime":
                return ToolResult(tool_name=self.name, success=True, output=self._is_prime(n), metadata={"n": n})
            elif operation == "prime_factors":
                return ToolResult(tool_name=self.name, success=True, output=self._prime_factors(n), metadata={"n": n})
            elif operation == "permutations":
                return ToolResult(tool_name=self.name, success=True, output=math.perm(n, k), metadata={"n": n, "k": k})
            elif operation == "combinations":
                return ToolResult(tool_name=self.name, success=True, output=math.comb(n, k), metadata={"n": n, "k": k})
            elif operation == "degrees":
                return ToolResult(tool_name=self.name, success=True, output=math.degrees(n), metadata={"radians": n})
            elif operation == "radians":
                return ToolResult(tool_name=self.name, success=True, output=math.radians(n), metadata={"degrees": n})
            elif operation == "distance":
                return self._distance(nums)
            elif operation == "mean":
                m = sum(nums) / len(nums) if nums else 0
                return ToolResult(tool_name=self.name, success=True, output=m)
            elif operation == "median":
                return ToolResult(tool_name=self.name, success=True, output=self._median(nums))
            elif operation == "stddev":
                return ToolResult(tool_name=self.name, success=True, output=self._stddev(nums))
            else:
                return ToolResult(tool_name=self.name, success=False, error=f"Unknown operation: {operation}")
        except Exception as exc:
            return ToolResult(tool_name=self.name, success=False, error=str(exc))

    @staticmethod
    def _parse_numbers(numbers: str) -> List[float]:
        """Parse comma-separated numbers."""
        return [float(x.strip()) for x in numbers.split(",") if x.strip()]

    def _stats(self, nums: List[float]) -> ToolResult:
        if not nums:
            return ToolResult(tool_name=self.name, success=False, error="No numbers provided")
        result = {
            "count": len(nums),
            "mean": sum(nums) / len(nums),
            "min": min(nums),
            "max": max(nums),
            "sum": sum(nums),
            "median": self._median(nums),
            "stddev": self._stddev(nums),
        }
        return ToolResult(tool_name=self.name, success=True, output=result)

    def _gcd_op(self, nums: List[float]) -> ToolResult:
        if len(nums) < 2:
            return ToolResult(tool_name=self.name, success=False, error="At least 2 numbers required")
        result = int(nums[0])
        for num in nums[1:]:
            result = math.gcd(result, int(num))
        return ToolResult(tool_name=self.name, success=True, output=result)

    def _lcm_op(self, nums: List[float]) -> ToolResult:
        if len(nums) < 2:
            return ToolResult(tool_name=self.name, success=False, error="At least 2 numbers required")
        result = int(nums[0])
        for num in nums[1:]:
            result = abs(result * int(num)) // math.gcd(result, int(num))
        return ToolResult(tool_name=self.name, success=True, output=result)

    def _fibonacci(self, n: int) -> ToolResult:
        if n < 0:
            return ToolResult(tool_name=self.name, success=False, error="n must be non-negative")
        if n <= 1:
            return ToolResult(tool_name=self.name, success=True, output=n, metadata={"n": n})
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return ToolResult(tool_name=self.name, success=True, output=b, metadata={"n": n})

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    @staticmethod
    def _prime_factors(n: int) -> List[int]:
        factors: List[int] = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    def _distance(self, nums: List[float]) -> ToolResult:
        if len(nums) != 4:
            return ToolResult(tool_name=self.name, success=False, error="Need 4 numbers: x1,y1,x2,y2")
        x1, y1, x2, y2 = nums
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return ToolResult(tool_name=self.name, success=True, output=dist)

    @staticmethod
    def _median(nums: List[float]) -> float:
        if not nums:
            return 0.0
        sorted_nums = sorted(nums)
        mid = len(sorted_nums) // 2
        if len(sorted_nums) % 2 == 0:
            return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
        return sorted_nums[mid]

    @staticmethod
    def _stddev(nums: List[float]) -> float:
        if len(nums) < 2:
            return 0.0
        mean = sum(nums) / len(nums)
        variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
        return math.sqrt(variance)
