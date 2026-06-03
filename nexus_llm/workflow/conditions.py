"""Nexus-LLM Workflow Conditions.

Provides built-in condition functions for workflow edge evaluation,
including comparison conditions, logical combinators, and data-driven
conditions.
"""

import logging
import operator
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Comparison operators
_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def field_equals(field_name: str, value: Any) -> Callable:
    """Create a condition that checks if a context field equals a value.

    Args:
        field_name: Name of the field in the execution context.
        value: Expected value.

    Returns:
        A condition function.
    """

    def condition(context: Dict[str, Any]) -> bool:
        return context.get(field_name) == value

    condition.__doc__ = f"field_equals({field_name!r}, {value!r})"
    return condition


def field_compare(field_name: str, op: str, value: Any) -> Callable:
    """Create a condition that compares a context field against a value.

    Args:
        field_name: Name of the field in the execution context.
        op: Comparison operator string (==, !=, <, <=, >, >=).
        value: Value to compare against.

    Returns:
        A condition function.

    Raises:
        ValueError: If the operator is not supported.
    """
    if op not in _OPERATORS:
        raise ValueError(f"Unsupported operator: {op}. Supported: {list(_OPERATORS.keys())}")

    cmp_func = _OPERATORS[op]

    def condition(context: Dict[str, Any]) -> bool:
        field_value = context.get(field_name)
        if field_value is None:
            return False
        try:
            return cmp_func(field_value, value)
        except TypeError:
            return False

    condition.__doc__ = f"field_compare({field_name!r}, {op!r}, {value!r})"
    return condition


def field_in(field_name: str, values: list) -> Callable:
    """Create a condition that checks if a context field is in a list of values.

    Args:
        field_name: Name of the field.
        values: List of allowed values.

    Returns:
        A condition function.
    """

    def condition(context: Dict[str, Any]) -> bool:
        return context.get(field_name) in values

    condition.__doc__ = f"field_in({field_name!r}, {values!r})"
    return condition


def field_exists(field_name: str) -> Callable:
    """Create a condition that checks if a field exists in the context.

    Args:
        field_name: Name of the field.

    Returns:
        A condition function.
    """

    def condition(context: Dict[str, Any]) -> bool:
        return field_name in context

    condition.__doc__ = f"field_exists({field_name!r})"
    return condition


def always() -> Callable:
    """Create a condition that always evaluates to True.

    Returns:
        A condition function that always returns True.
    """

    def condition(context: Dict[str, Any]) -> bool:
        return True

    condition.__doc__ = "always()"
    return condition


def never() -> Callable:
    """Create a condition that always evaluates to False.

    Returns:
        A condition function that always returns False.
    """

    def condition(context: Dict[str, Any]) -> bool:
        return False

    condition.__doc__ = "never()"
    return condition


def not_condition(condition: Callable) -> Callable:
    """Create a condition that negates another condition.

    Args:
        condition: The condition to negate.

    Returns:
        A negated condition function.
    """

    def negated(context: Dict[str, Any]) -> bool:
        return not condition(context)

    negated.__doc__ = f"not({condition.__doc__})"
    return negated


def and_conditions(*conditions: Callable) -> Callable:
    """Create a condition that is True only if all conditions are True.

    Args:
        *conditions: Conditions to combine with AND logic.

    Returns:
        A combined condition function.
    """

    def combined(context: Dict[str, Any]) -> bool:
        return all(cond(context) for cond in conditions)

    combined.__doc__ = f"and({', '.join(getattr(c, '__doc__', '?') for c in conditions)})"
    return combined


def or_conditions(*conditions: Callable) -> Callable:
    """Create a condition that is True if any condition is True.

    Args:
        *conditions: Conditions to combine with OR logic.

    Returns:
        A combined condition function.
    """

    def combined(context: Dict[str, Any]) -> bool:
        return any(cond(context) for cond in conditions)

    combined.__doc__ = f"or({', '.join(getattr(c, '__doc__', '?') for c in conditions)})"
    return combined
