"""Test input validation utilities for Nexus-LLM."""
import pytest
from typing import Any, Optional, List


class ValidationError(Exception):
    pass


def validate_string(value: Any, min_length: int = 0, max_length: int = None, name: str = "value") -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")
    if len(value) < min_length:
        raise ValidationError(f"{name} must be at least {min_length} characters, got {len(value)}")
    if max_length and len(value) > max_length:
        raise ValidationError(f"{name} must be at most {max_length} characters, got {len(value)}")
    return value


def validate_int(value: Any, min_val: int = None, max_val: int = None, name: str = "value") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return value


def validate_float(value: Any, min_val: float = None, max_val: float = None, name: str = "value") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got {value}")
    return float(value)


def validate_bool(value: Any, name: str = "value") -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be a boolean, got {type(value).__name__}")
    return value


def validate_list(value: Any, min_items: int = 0, name: str = "value") -> list:
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list, got {type(value).__name__}")
    if len(value) < min_items:
        raise ValidationError(f"{name} must have at least {min_items} items, got {len(value)}")
    return value


def validate_dict(value: Any, required_keys: List[str] = None, name: str = "value") -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a dict, got {type(value).__name__}")
    if required_keys:
        missing = [k for k in required_keys if k not in value]
        if missing:
            raise ValidationError(f"{name} missing required keys: {missing}")
    return value


def validate_email(value: Any, name: str = "email") -> str:
    validate_string(value, min_length=3, name=name)
    if "@" not in value:
        raise ValidationError(f"{name} must contain '@'")
    local, _, domain = value.partition("@")
    if not local or not domain:
        raise ValidationError(f"{name} has invalid format")
    if "." not in domain:
        raise ValidationError(f"{name} domain must contain '.'")
    return value


def validate_enum(value: Any, allowed: List[Any], name: str = "value") -> Any:
    if value not in allowed:
        raise ValidationError(f"{name} must be one of {allowed}, got {value}")
    return value


def validate_optional(validator, value: Any, **kwargs) -> Optional[Any]:
    if value is None:
        return None
    return validator(value, **kwargs)


class TestValidateString:
    def test_valid_string(self):
        assert validate_string("hello") == "hello"

    def test_non_string_fails(self):
        with pytest.raises(ValidationError):
            validate_string(123)

    def test_min_length(self):
        assert validate_string("ab", min_length=2) == "ab"
        with pytest.raises(ValidationError, match="at least"):
            validate_string("a", min_length=2)

    def test_max_length(self):
        assert validate_string("ab", max_length=2) == "ab"
        with pytest.raises(ValidationError, match="at most"):
            validate_string("abc", max_length=2)

    def test_empty_with_min(self):
        with pytest.raises(ValidationError):
            validate_string("", min_length=1)

    def test_custom_name(self):
        with pytest.raises(ValidationError, match="username"):
            validate_string("", min_length=1, name="username")


class TestValidateInt:
    def test_valid_int(self):
        assert validate_int(42) == 42

    def test_float_fails(self):
        with pytest.raises(ValidationError):
            validate_int(3.14)

    def test_bool_fails(self):
        with pytest.raises(ValidationError):
            validate_int(True)

    def test_min_val(self):
        assert validate_int(5, min_val=1) == 5
        with pytest.raises(ValidationError):
            validate_int(0, min_val=1)

    def test_max_val(self):
        assert validate_int(5, max_val=10) == 5
        with pytest.raises(ValidationError):
            validate_int(11, max_val=10)


class TestValidateFloat:
    def test_valid_float(self):
        assert validate_float(3.14) == 3.14

    def test_int_coerced(self):
        result = validate_float(5)
        assert isinstance(result, float)
        assert result == 5.0

    def test_bool_fails(self):
        with pytest.raises(ValidationError):
            validate_float(True)

    def test_min_val(self):
        with pytest.raises(ValidationError):
            validate_float(0.0, min_val=0.1)

    def test_max_val(self):
        with pytest.raises(ValidationError):
            validate_float(2.0, max_val=1.0)


class TestValidateBool:
    def test_true(self):
        assert validate_bool(True) is True

    def test_false(self):
        assert validate_bool(False) is False

    def test_int_fails(self):
        with pytest.raises(ValidationError):
            validate_bool(1)

    def test_string_fails(self):
        with pytest.raises(ValidationError):
            validate_bool("true")


class TestValidateList:
    def test_valid_list(self):
        assert validate_list([1, 2, 3]) == [1, 2, 3]

    def test_non_list_fails(self):
        with pytest.raises(ValidationError):
            validate_list("not a list")

    def test_min_items(self):
        assert validate_list([1, 2], min_items=2) == [1, 2]
        with pytest.raises(ValidationError):
            validate_list([1], min_items=2)

    def test_empty_list(self):
        assert validate_list([]) == []


class TestValidateDict:
    def test_valid_dict(self):
        assert validate_dict({"a": 1}) == {"a": 1}

    def test_non_dict_fails(self):
        with pytest.raises(ValidationError):
            validate_dict([1, 2])

    def test_required_keys(self):
        result = validate_dict({"a": 1, "b": 2}, required_keys=["a", "b"])
        assert result == {"a": 1, "b": 2}

    def test_missing_required_keys(self):
        with pytest.raises(ValidationError, match="missing required"):
            validate_dict({"a": 1}, required_keys=["a", "b"])


class TestValidateEmail:
    def test_valid_email(self):
        assert validate_email("user@example.com") == "user@example.com"

    def test_no_at_sign(self):
        with pytest.raises(ValidationError, match="@"):
            validate_email("userexample.com")

    def test_empty_local(self):
        with pytest.raises(ValidationError, match="invalid"):
            validate_email("@example.com")

    def test_no_domain_dot(self):
        with pytest.raises(ValidationError, match="domain"):
            validate_email("user@localhost")


class TestValidateEnum:
    def test_valid_choice(self):
        assert validate_enum("a", ["a", "b", "c"]) == "a"

    def test_invalid_choice(self):
        with pytest.raises(ValidationError, match="one of"):
            validate_enum("d", ["a", "b", "c"])

    def test_numeric_enum(self):
        assert validate_enum(1, [1, 2, 3]) == 1


class TestValidateOptional:
    def test_none_returns_none(self):
        result = validate_optional(validate_int, None)
        assert result is None

    def test_value_passes_through(self):
        result = validate_optional(validate_int, 42)
        assert result == 42

    def test_invalid_value_fails(self):
        with pytest.raises(ValidationError):
            validate_optional(validate_int, "not an int")
