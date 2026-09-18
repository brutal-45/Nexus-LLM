"""Test configuration validators for Nexus-LLM."""
import pytest
from typing import Any, Optional, List


# --- Validator implementations to test ---

class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_positive_int(value: Any, name: str = "value") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative_int(value: Any, name: str = "value") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return value


def validate_float_range(value: Any, low: float, high: float, name: str = "value") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    if not (low <= value <= high):
        raise ValidationError(f"{name} must be between {low} and {high}, got {value}")
    return float(value)


def validate_string_nonempty(value: Any, name: str = "value") -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValidationError(f"{name} must not be empty")
    return value


def validate_choice(value: Any, choices: List[Any], name: str = "value") -> Any:
    if value not in choices:
        raise ValidationError(f"{name} must be one of {choices}, got {value}")
    return value


def validate_port(value: Any) -> int:
    validate_positive_int(value, "port")
    if not (1 <= value <= 65535):
        raise ValidationError(f"port must be between 1 and 65535, got {value}")
    return value


def validate_temperature(value: Any) -> float:
    return validate_float_range(value, 0.0, 2.0, "temperature")


def validate_top_p(value: Any) -> float:
    return validate_float_range(value, 0.0, 1.0, "top_p")


def validate_top_k(value: Any) -> int:
    return validate_positive_int(value, "top_k")


def validate_model_path(value: Any) -> str:
    validate_string_nonempty(value, "model_path")
    if any(c in value for c in "\0"):
        raise ValidationError("model_path contains null bytes")
    return value


def validate_device(value: Any) -> str:
    return validate_choice(value, ["auto", "cpu", "cuda", "mps"], "device")


class TestPositiveIntValidator:
    def test_valid_positive_int(self):
        assert validate_positive_int(1) == 1
        assert validate_positive_int(100) == 100
        assert validate_positive_int(999999) == 999999

    def test_zero_fails(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(0)

    def test_negative_fails(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_positive_int(-1)

    def test_float_fails(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(3.14)

    def test_string_fails(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int("5")

    def test_bool_fails(self):
        with pytest.raises(ValidationError, match="must be an integer"):
            validate_positive_int(True)

    def test_custom_name(self):
        with pytest.raises(ValidationError, match="batch_size"):
            validate_positive_int(-1, "batch_size")


class TestNonNegativeIntValidator:
    def test_zero_passes(self):
        assert validate_non_negative_int(0) == 0

    def test_positive_passes(self):
        assert validate_non_negative_int(42) == 42

    def test_negative_fails(self):
        with pytest.raises(ValidationError, match="non-negative"):
            validate_non_negative_int(-1)

    def test_float_fails(self):
        with pytest.raises(ValidationError, match="integer"):
            validate_non_negative_int(1.0)


class TestFloatRangeValidator:
    def test_valid_within_range(self):
        assert validate_float_range(0.5, 0.0, 1.0) == 0.5

    def test_boundary_low(self):
        assert validate_float_range(0.0, 0.0, 1.0) == 0.0

    def test_boundary_high(self):
        assert validate_float_range(1.0, 0.0, 1.0) == 1.0

    def test_below_range_fails(self):
        with pytest.raises(ValidationError, match="between"):
            validate_float_range(-0.1, 0.0, 1.0)

    def test_above_range_fails(self):
        with pytest.raises(ValidationError, match="between"):
            validate_float_range(1.1, 0.0, 1.0)

    def test_string_fails(self):
        with pytest.raises(ValidationError, match="number"):
            validate_float_range("0.5", 0.0, 1.0)

    def test_int_coerced_to_float(self):
        result = validate_float_range(1, 0.0, 2.0)
        assert isinstance(result, float)
        assert result == 1.0


class TestStringNonemptyValidator:
    def test_valid_string(self):
        assert validate_string_nonempty("hello") == "hello"

    def test_empty_string_fails(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_string_nonempty("")

    def test_whitespace_only_fails(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_string_nonempty("   ")

    def test_non_string_fails(self):
        with pytest.raises(ValidationError, match="string"):
            validate_string_nonempty(123)


class TestChoiceValidator:
    def test_valid_choice(self):
        assert validate_choice("a", ["a", "b", "c"]) == "a"

    def test_invalid_choice(self):
        with pytest.raises(ValidationError, match="one of"):
            validate_choice("d", ["a", "b", "c"])

    def test_numeric_choices(self):
        assert validate_choice(1, [1, 2, 3]) == 1


class TestSpecificValidators:
    def test_validate_port_valid(self):
        assert validate_port(8000) == 8000
        assert validate_port(1) == 1
        assert validate_port(65535) == 65535

    def test_validate_port_invalid(self):
        with pytest.raises(ValidationError):
            validate_port(0)
        with pytest.raises(ValidationError):
            validate_port(70000)

    def test_validate_temperature_valid(self):
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(2.0) == 2.0

    def test_validate_temperature_invalid(self):
        with pytest.raises(ValidationError):
            validate_temperature(2.5)
        with pytest.raises(ValidationError):
            validate_temperature(-0.1)

    def test_validate_top_p_valid(self):
        assert validate_top_p(0.9) == 0.9

    def test_validate_top_p_invalid(self):
        with pytest.raises(ValidationError):
            validate_top_p(1.5)

    def test_validate_device_valid(self):
        assert validate_device("auto") == "auto"
        assert validate_device("cpu") == "cpu"
        assert validate_device("cuda") == "cuda"

    def test_validate_device_invalid(self):
        with pytest.raises(ValidationError):
            validate_device("tpu")

    def test_validate_model_path_valid(self):
        assert validate_model_path("/models/test") == "/models/test"

    def test_validate_model_path_null_bytes(self):
        with pytest.raises(ValidationError, match="null"):
            validate_model_path("/models/\0test")
