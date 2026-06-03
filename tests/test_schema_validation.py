"""Tests for schema validation."""
import pytest


def test_valid_config_schema():
    schema = {"model": {"type": str, "required": True}, "max_length": {"type": int, "required": True}}
    config = {"model": "gpt2", "max_length": 2048}
    errors = []
    for key, rules in schema.items():
        if rules.get("required") and key not in config:
            errors.append(f"Missing: {key}")
    assert len(errors) == 0


def test_missing_required_field():
    config = {"model": "gpt2"}
    required = ["model", "max_length"]
    missing = [f for f in required if f not in config]
    assert "max_length" in missing


def test_type_validation():
    config = {"max_length": "not_a_number"}
    errors = []
    if not isinstance(config["max_length"], int):
        errors.append("max_length must be int")
    assert len(errors) == 1


def test_range_validation():
    temperature = 3.0
    assert not (0.0 <= temperature <= 2.0)


def test_nested_schema_validation():
    config = {"model": {"name": "gpt2", "size": "small"}, "training": {"lr": 0.001}}
    assert isinstance(config["model"], dict)
    assert "name" in config["model"]
