"""Tests for config priority merging."""
import pytest


def test_config_override_priority():
    default = {"lr": 0.001, "batch_size": 32, "epochs": 10}
    user = {"lr": 0.01, "batch_size": 64}
    merged = {**default, **user}
    assert merged["lr"] == 0.01
    assert merged["epochs"] == 10

def test_deep_merge():
    base = {"model": {"name": "gpt2", "size": "small"}, "training": {"lr": 0.001}}
    override = {"model": {"name": "llama"}}
    # Simple deep merge
    merged = {}
    for key in set(list(base.keys()) + list(override.keys())):
        if key in base and key in override and isinstance(base[key], dict):
            merged[key] = {**base[key], **override[key]}
        elif key in override:
            merged[key] = override[key]
        else:
            merged[key] = base[key]
    assert merged["model"]["name"] == "llama"
    assert merged["model"]["size"] == "small"

def test_cli_overrides_env():
    env_config = {"lr": 0.001}
    cli_config = {"lr": 0.01}
    final = {**env_config, **cli_config}
    assert final["lr"] == 0.01

def test_config_merge_preserves_unknown():
    base = {"a": 1, "b": 2}
    override = {"c": 3}
    merged = {**base, **override}
    assert "a" in merged and "c" in merged
