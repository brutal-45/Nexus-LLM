"""Nexus-LLM Pipeline Validation.

Provides the PipelineValidator for validating data at various stages
of the processing pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from a validation check.

    Attributes:
        is_valid: Whether the data passed validation.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
        checked_by: Name of the validator that produced this result.
        metadata: Additional validation metadata.
    """

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning without marking as invalid."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False


class PipelineValidator:
    """Validates data at pipeline stages.

    The PipelineValidator maintains a registry of named validators
    and runs them against input data, collecting all errors and
    warnings.

    Example::

        validator = PipelineValidator()
        validator.register("not_empty", lambda d: ValidationResult() if d else ValidationResult(is_valid=False, errors=["Empty input"]))
        result = validator.validate("hello", validators=["not_empty"])
    """

    def __init__(self) -> None:
        self._validators: Dict[str, Callable[[Any], ValidationResult]] = {}
        self._register_builtin_validators()
        logger.debug("PipelineValidator initialized")

    def register(self, name: str, validator: Callable[[Any], ValidationResult]) -> None:
        """Register a named validator.

        Args:
            name: Unique validator name.
            validator: Callable that accepts data and returns a ValidationResult.
        """
        self._validators[name] = validator
        logger.debug("Registered validator: %s", name)

    def unregister(self, name: str) -> bool:
        """Unregister a validator."""
        return self._validators.pop(name, None) is not None

    def validate(self, data: Any, validators: Optional[List[str]] = None) -> ValidationResult:
        """Run validators against data.

        Args:
            data: The data to validate.
            validators: List of validator names to run (None = all).

        Returns:
            A merged ValidationResult.
        """
        names = validators or list(self._validators.keys())
        result = ValidationResult()

        for name in names:
            validator = self._validators.get(name)
            if validator is None:
                result.add_warning(f"Unknown validator: {name}")
                continue

            try:
                sub_result = validator(data)
                sub_result.checked_by = name
                result.merge(sub_result)
            except Exception as exc:
                result.add_error(f"Validator '{name}' raised exception: {exc}")

        return result

    def list_validators(self) -> List[str]:
        """Return list of registered validator names."""
        return list(self._validators.keys())

    def _register_builtin_validators(self) -> None:
        """Register built-in validators."""
        self._validators["not_empty"] = self._validate_not_empty
        self._validators["is_string"] = self._validate_is_string
        self._validators["max_length"] = self._validate_max_length
        self._validators["no_null_bytes"] = self._validate_no_null_bytes

    @staticmethod
    def _validate_not_empty(data: Any) -> ValidationResult:
        result = ValidationResult()
        if data is None or (isinstance(data, str) and not data.strip()):
            result.add_error("Data is empty")
        return result

    @staticmethod
    def _validate_is_string(data: Any) -> ValidationResult:
        result = ValidationResult()
        if not isinstance(data, str):
            result.add_error(f"Expected string, got {type(data).__name__}")
        return result

    @staticmethod
    def _validate_max_length(data: Any, max_len: int = 100000) -> ValidationResult:
        result = ValidationResult()
        if isinstance(data, str) and len(data) > max_len:
            result.add_error(f"Data exceeds max length ({len(data)} > {max_len})")
        return result

    @staticmethod
    def _validate_no_null_bytes(data: Any) -> ValidationResult:
        result = ValidationResult()
        if isinstance(data, str) and "\x00" in data:
            result.add_error("Data contains null bytes")
        return result
