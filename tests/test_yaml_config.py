"""Tests for YAML config loading."""
import pytest
import yaml


def test_yaml_load_basic():
    config_str = "model:\n  name: gpt2\n  max_length: 2048"
    config = yaml.safe_load(config_str)
    assert config["model"]["name"] == "gpt2"

def test_yaml_load_with_lists():
    config_str = "models:\n  - gpt2\n  - llama\n  - mistral"
    config = yaml.safe_load(config_str)
    assert len(config["models"]) == 3

def test_yaml_load_nested():
    config_str = "training:\n  optimizer:\n    lr: 0.001\n    weight_decay: 0.01"
    config = yaml.safe_load(config_str)
    assert config["training"]["optimizer"]["lr"] == 0.001

def test_yaml_safe_load():
    # safe_load prevents arbitrary code execution
    config = yaml.safe_load("key: value")
    assert config["key"] == "value"
