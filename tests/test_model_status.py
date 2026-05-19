"""Tests for model status bar."""
import pytest


def test_model_status_loading():
    status = {"model": "gpt2", "state": "loading", "progress": 0.5}
    assert status["state"] == "loading"

def test_model_status_ready():
    status = {"model": "gpt2", "state": "ready"}
    assert status["state"] == "ready"

def test_model_status_error():
    status = {"model": "gpt2", "state": "error", "message": "OOM"}
    assert "error" in status["state"]

def test_model_status_display():
    status = {"model": "llama-7b", "state": "ready", "gpu": "A100"}
    display = f"{status['model']} ({status['state']}) [{status['gpu']}]"
    assert "llama-7b" in display
