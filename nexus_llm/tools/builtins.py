"""BuiltinTools — a collection of ready-made tools for common operations."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit conversion tables
# ---------------------------------------------------------------------------

_LENGTH_TO_METERS: Dict[str, float] = {
    "mm": 0.001, "millimeter": 0.001, "millimetre": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimetre": 0.01,
    "m": 1.0, "meter": 1.0, "metre": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometre": 1000.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
}

_WEIGHT_TO_KG: Dict[str, float] = {
    "mg": 1e-6, "milligram": 1e-6,
    "g": 0.001, "gram": 0.001,
    "kg": 1.0, "kilogram": 1.0,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
}

_TEMP_OFFSETS: Dict[str, Optional[float]] = {
    "c": None, "celsius": None,
    "f": None, "fahrenheit": None,
    "k": None, "kelvin": None,
}


class BuiltinTools:
    """Stateless collection of commonly-needed tool functions.

    Each method is intentionally simple and side-effect-free so it can be
    safely exposed to an LLM.
    """

    # ------------------------------------------------------------------
    # Calculator
    # ------------------------------------------------------------------

    @staticmethod
    def calculator(expression: str) -> Any:
        """Evaluate a mathematical expression safely.

        Only a whitelist of names (``math`` module functions/constants) is
        available inside the expression to prevent code injection.

        Parameters
        ----------
        expression:
            A Python math expression, e.g. ``"2 ** 10"`` or ``"math.sin(0)"``.

        Returns
        -------
        The numeric result.

        Raises
        ------
        ValueError
            If the expression is empty.
        TypeError
            If the result is not a number.
        NameError
            If a disallowed name is used.
        """
        if not expression or not expression.strip():
            raise ValueError("Expression must not be empty")

        allowed_names = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
        allowed_names["abs"] = abs
        allowed_names["round"] = round
        allowed_names["min"] = min
        allowed_names["max"] = max

        code = compile(expression, "<calculator>", "eval")

        # Verify all names are in the whitelist.
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Name {name!r} is not allowed in calculator expressions")

        result = eval(code, {"__builtins__": {}}, allowed_names)  # noqa: S307
        if not isinstance(result, (int, float)):
            raise TypeError(f"Expression did not evaluate to a number: {type(result).__name__}")
        return result

    # ------------------------------------------------------------------
    # Text transform
    # ------------------------------------------------------------------

    @staticmethod
    def text_transform(text: str, operation: str) -> str:
        """Apply a text transformation.

        Parameters
        ----------
        text:
            Input string.
        operation:
            One of ``"upper"``, ``"lower"``, ``"title"``, ``"capitalize"``,
            ``"reverse"``, ``"strip"``, ``"swapcase"``.
        """
        ops = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
            "capitalize": str.capitalize,
            "reverse": lambda s: s[::-1],
            "strip": str.strip,
            "swapcase": str.swapcase,
        }
        operation = operation.lower()
        if operation not in ops:
            raise ValueError(f"Unknown operation {operation!r}; expected one of {list(ops)}")
        return ops[operation](text)

    # ------------------------------------------------------------------
    # JSON parser
    # ------------------------------------------------------------------

    @staticmethod
    def json_parser(json_str: str, key: Optional[str] = None) -> Any:
        """Parse a JSON string and optionally extract a nested key.

        Parameters
        ----------
        json_str:
            Valid JSON string.
        key:
            Dot-separated key path, e.g. ``"data.results.0.name"``.
            If ``None``, the full parsed object is returned.
        """
        data = json.loads(json_str)
        if key is None:
            return data
        parts = key.split(".")
        current: Any = data
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            elif isinstance(current, list):
                current = current[int(part)]
            else:
                raise KeyError(f"Cannot traverse into {type(current).__name__} with key {part!r}")
        return current

    # ------------------------------------------------------------------
    # File reader
    # ------------------------------------------------------------------

    @staticmethod
    def file_reader(path: str, encoding: str = "utf-8") -> str:
        """Read a text file and return its content.

        Parameters
        ----------
        path:
            Filesystem path to the file.
        encoding:
            Text encoding (default ``"utf-8"``).
        """
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        with open(abs_path, "r", encoding=encoding) as fh:
            return fh.read()

    # ------------------------------------------------------------------
    # Table formatter
    # ------------------------------------------------------------------

    @staticmethod
    def table_formatter(data: List[Dict[str, Any]], format: str = "simple") -> str:
        """Format a list of dictionaries as a text table.

        Parameters
        ----------
        data:
            List of row dictionaries with uniform keys.
        format:
            ``"simple"`` for ASCII, ``"markdown"`` for Markdown table,
            ``"csv"`` for CSV output.
        """
        if not data:
            return ""
        headers = list(data[0].keys())
        rows = [[str(row.get(h, "")) for h in headers] for row in data]

        if format == "markdown":
            header_line = "| " + " | ".join(headers) + " |"
            separator = "| " + " | ".join("---" for _ in headers) + " |"
            body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
            return f"{header_line}\n{separator}\n{body}"

        if format == "csv":
            lines = [",".join(headers)]
            lines.extend(",".join(row) for row in rows)
            return "\n".join(lines)

        # Default: simple ASCII
        col_widths = [max(len(h), *(len(r) for r in col)) for h, col in zip(headers, zip(*rows))]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        separator = "  ".join("-" * w for w in col_widths)
        body = "\n".join("  ".join(c.ljust(w) for c, w in zip(row, col_widths)) for row in rows)
        return f"{header_line}\n{separator}\n{body}"

    # ------------------------------------------------------------------
    # Unit converter
    # ------------------------------------------------------------------

    @staticmethod
    def unit_converter(value: float, from_unit: str, to_unit: str) -> float:
        """Convert *value* between common units.

        Supported categories: length, weight/mass, temperature.

        Parameters
        ----------
        value:
            Numeric value to convert.
        from_unit:
            Source unit (case-insensitive).
        to_unit:
            Target unit (case-insensitive).

        Raises
        ------
        ValueError
            If units are unknown or incompatible.
        """
        fu = from_unit.lower().strip()
        tu = to_unit.lower().strip()

        # --- Length ---
        if fu in _LENGTH_TO_METERS and tu in _LENGTH_TO_METERS:
            meters = value * _LENGTH_TO_METERS[fu]
            return meters / _LENGTH_TO_METERS[tu]

        # --- Weight ---
        if fu in _WEIGHT_TO_KG and tu in _WEIGHT_TO_KG:
            kg = value * _WEIGHT_TO_KG[fu]
            return kg / _WEIGHT_TO_KG[tu]

        # --- Temperature ---
        temp_units = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
        if fu in temp_units and tu in temp_units:
            return BuiltinTools._convert_temperature(value, fu, tu)

        raise ValueError(f"Cannot convert from {from_unit!r} to {to_unit!r}")

    # ------------------------------------------------------------------
    # Internal: temperature conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        """Convert between Celsius, Fahrenheit, and Kelvin."""
        # Normalise to Celsius first
        fu = from_unit[0]  # 'c', 'f', or 'k'
        tu = to_unit[0]

        if fu == "c":
            celsius = value
        elif fu == "f":
            celsius = (value - 32) * 5 / 9
        elif fu == "k":
            celsius = value - 273.15
        else:
            raise ValueError(f"Unsupported temperature unit: {from_unit!r}")

        if tu == "c":
            return celsius
        elif tu == "f":
            return celsius * 9 / 5 + 32
        elif tu == "k":
            return celsius + 273.15
        else:
            raise ValueError(f"Unsupported temperature unit: {to_unit!r}")
