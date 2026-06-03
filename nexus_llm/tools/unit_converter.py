"""Nexus-LLM Unit Conversion Tool.

Provides unit conversion capabilities for length, weight, temperature,
and other common measurement categories.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

# Conversion factors (to SI base units)
_LENGTH_TO_METER = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
}

_WEIGHT_TO_KG = {
    "mg": 1e-6, "g": 0.001, "kg": 1.0, "lb": 0.453592, "oz": 0.0283495,
}

_VOLUME_TO_LITER = {
    "ml": 0.001, "l": 1.0, "gal": 3.78541, "qt": 0.946353, "cup": 0.236588,
}

_CATEGORIES = {
    "length": _LENGTH_TO_METER,
    "weight": _WEIGHT_TO_KG,
    "volume": _VOLUME_TO_LITER,
}


class UnitConverterTool(BaseTool):
    """Tool for converting between measurement units.

    Supports length, weight, volume, and temperature conversions.

    Example::

        tool = UnitConverterTool()
        result = tool.run(value=100, from_unit="km", to_unit="mi", category="length")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="unit_converter", description="Convert between measurement units", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="value", type=ParameterType.FLOAT, description="Value to convert", required=True),
            ToolParameter(name="from_unit", type=ParameterType.STRING, description="Source unit", required=True),
            ToolParameter(name="to_unit", type=ParameterType.STRING, description="Target unit", required=True),
            ToolParameter(name="category", type=ParameterType.STRING, description="Unit category",
                         required=False, default="auto", choices=["auto", "length", "weight", "volume", "temperature"]),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        value = kwargs.get("value")
        from_unit = kwargs.get("from_unit", "").lower()
        to_unit = kwargs.get("to_unit", "").lower()
        category = kwargs.get("category", "auto")

        if value is None:
            return ToolResult(success=False, error="No value provided")
        if not from_unit or not to_unit:
            return ToolResult(success=False, error="Both from_unit and to_unit are required")

        try:
            result = self._convert(float(value), from_unit, to_unit, category)
            return ToolResult(success=True, output=result)
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _convert(self, value: float, from_unit: str, to_unit: str, category: str) -> Dict[str, Any]:
        """Perform the unit conversion."""
        # Temperature is special
        if category == "temperature" or (from_unit in ("c", "f", "k") and to_unit in ("c", "f", "k")):
            converted = self._convert_temperature(value, from_unit, to_unit)
            return {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": converted,
                "category": "temperature",
            }

        # Find the category
        if category == "auto":
            for cat_name, units in _CATEGORIES.items():
                if from_unit in units and to_unit in units:
                    category = cat_name
                    break
            else:
                raise ValueError(f"Cannot determine category for {from_unit} -> {to_unit}")

        units = _CATEGORIES.get(category)
        if not units:
            raise ValueError(f"Unknown category: {category}")
        if from_unit not in units:
            raise ValueError(f"Unknown unit: {from_unit}")
        if to_unit not in units:
            raise ValueError(f"Unknown unit: {to_unit}")

        # Convert: value -> base unit -> target unit
        base_value = value * units[from_unit]
        result = base_value / units[to_unit]

        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": result,
            "category": category,
        }

    def _convert_temperature(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert between Celsius, Fahrenheit, and Kelvin."""
        # Convert to Celsius first
        if from_unit == "c":
            celsius = value
        elif from_unit == "f":
            celsius = (value - 32) * 5 / 9
        elif from_unit == "k":
            celsius = value - 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")

        # Convert from Celsius to target
        if to_unit == "c":
            return celsius
        elif to_unit == "f":
            return celsius * 9 / 5 + 32
        elif to_unit == "k":
            return celsius + 273.15
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")

    def get_supported_units(self, category: str = "all") -> Dict[str, List[str]]:
        """Get supported units by category.

        Args:
            category: Category name, or 'all' for all categories.

        Returns:
            Dictionary mapping categories to lists of unit names.
        """
        result = {}
        for cat, units in _CATEGORIES.items():
            if category == "all" or category == cat:
                result[cat] = list(units.keys())
        result["temperature"] = ["c", "f", "k"]
        return result
