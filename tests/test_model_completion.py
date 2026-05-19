"""Tests for model name completion."""
import pytest


def test_model_name_completion():
    models = ["gpt2", "gpt2-medium", "gpt2-large", "llama-7b", "llama-13b"]
    prefix = "gpt"
    matches = [m for m in models if m.startswith(prefix)]
    assert len(matches) == 3

def test_model_completion_with_org():
    models = ["openai/gpt2", "meta/llama-7b", "openai/gpt3"]
    prefix = "openai"
    matches = [m for m in models if m.startswith(prefix)]
    assert len(matches) == 2

def test_model_completion_case_insensitive():
    models = ["GPT2", "LLaMA", "BERT"]
    prefix = "gpt"
    matches = [m for m in models if m.lower().startswith(prefix.lower())]
    assert len(matches) == 1
