"""Tests for JSON I/O."""
import pytest
import json
from pathlib import Path


def test_json_write_read(tmp_dir):
    p = tmp_dir / "data.json"
    data = {"key": "value", "num": 42}
    p.write_text(json.dumps(data))
    loaded = json.loads(p.read_text())
    assert loaded == data

def test_json_list(tmp_dir):
    p = tmp_dir / "list.json"
    data = [1, 2, 3, "four"]
    p.write_text(json.dumps(data))
    loaded = json.loads(p.read_text())
    assert loaded == data

def test_json_nested(tmp_dir):
    p = tmp_dir / "nested.json"
    data = {"a": {"b": {"c": 1}}}
    p.write_text(json.dumps(data))
    loaded = json.loads(p.read_text())
    assert loaded["a"]["b"]["c"] == 1

def test_json_unicode(tmp_dir):
    p = tmp_dir / "unicode.json"
    data = {"greeting": "Hello \u4e16\u754c"}
    p.write_text(json.dumps(data))
    loaded = json.loads(p.read_text())
    assert "greeting" in loaded
