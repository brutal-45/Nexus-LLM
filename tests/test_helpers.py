"""Tests for helper functions."""
import pytest
import os
import tempfile


def test_path_expansion():
    """Test tilde expansion in paths."""
    home = os.path.expanduser("~")
    assert "~" not in home
    assert os.path.isdir(home)


def test_file_write_read(tmp_dir):
    """Test writing and reading a file."""
    path = os.path.join(tmp_dir, "test.txt")
    content = "Hello, World!"
    with open(path, "w") as f:
        f.write(content)
    with open(path, "r") as f:
        assert f.read() == content


def test_directory_creation(tmp_dir):
    """Test creating nested directories."""
    nested = os.path.join(tmp_dir, "a", "b", "c")
    os.makedirs(nested, exist_ok=True)
    assert os.path.isdir(nested)


def test_json_roundtrip(tmp_dir):
    """Test JSON serialization round-trip."""
    import json
    data = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": True}}
    path = os.path.join(tmp_dir, "test.json")
    with open(path, "w") as f:
        json.dump(data, f)
    with open(path, "r") as f:
        loaded = json.load(f)
    assert loaded == data


def test_yaml_roundtrip(tmp_dir):
    """Test YAML serialization round-trip."""
    import yaml
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    path = os.path.join(tmp_dir, "test.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    with open(path, "r") as f:
        loaded = yaml.safe_load(f)
    assert loaded == data
