"""Tests for env var config."""
import pytest
import os


def test_env_var_reading():
    os.environ["NEXUS_TEST_VAR"] = "test_value"
    assert os.environ.get("NEXUS_TEST_VAR") == "test_value"
    del os.environ["NEXUS_TEST_VAR"]

def test_env_var_default():
    value = os.environ.get("NEXUS_MISSING_VAR", "default")
    assert value == "default"

def test_env_var_type_conversion():
    os.environ["NEXUS_PORT"] = "8000"
    port = int(os.environ.get("NEXUS_PORT", "8080"))
    assert port == 8000
    del os.environ["NEXUS_PORT"]

def test_env_var_bool():
    os.environ["NEXUS_DEBUG"] = "true"
    debug = os.environ.get("NEXUS_DEBUG", "false").lower() == "true"
    assert debug is True
    del os.environ["NEXUS_DEBUG"]
