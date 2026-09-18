"""Tests for YAML I/O."""
import pytest
import yaml
from pathlib import Path


def test_yaml_write_read(tmp_dir):
    p = tmp_dir / "config.yaml"
    data = {"model": "gpt2", "params": {"lr": 0.001}}
    p.write_text(yaml.dump(data))
    loaded = yaml.safe_load(p.read_text())
    assert loaded == data

def test_yaml_list_io(tmp_dir):
    p = tmp_dir / "list.yaml"
    data = {"items": ["a", "b", "c"]}
    p.write_text(yaml.dump(data))
    loaded = yaml.safe_load(p.read_text())
    assert loaded["items"] == ["a", "b", "c"]

def test_yaml_multiline(tmp_dir):
    p = tmp_dir / "multi.yaml"
    text = "line1\nline2\nline3"
    data = {"content": text}
    p.write_text(yaml.dump(data, default_flow_style=False))
    loaded = yaml.safe_load(p.read_text())
    assert "line1" in loaded["content"]
