"""Tests for inference engine, generation, and streaming."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from nexus.inference.generator import GenerationConfig, GenerationResult, TextGenerator, SamplingStrategy


def test_generation_config_defaults():
    """Test that GenerationConfig has sensible defaults."""
    config = GenerationConfig()
    assert config.max_new_tokens == 4096
    assert config.temperature == 0.7
    assert config.top_k == 50
    assert config.top_p == 0.9
    assert config.num_beams == 1
    assert config.use_cache is True


def test_generation_config_custom():
    """Test custom GenerationConfig values."""
    config = GenerationConfig(
        max_new_tokens=100,
        temperature=0.5,
        top_k=10,
        top_p=0.8,
        num_beams=4,
    )
    assert config.max_new_tokens == 100
    assert config.temperature == 0.5
    assert config.top_k == 10
    assert config.num_beams == 4


def test_sampling_strategy_enum():
    """Test SamplingStrategy enum values."""
    assert SamplingStrategy.GREEDY.value == "greedy"
    assert SamplingStrategy.TEMPERATURE.value == "temperature"
    assert SamplingStrategy.TOP_K.value == "top_k"
    assert SamplingStrategy.TOP_P.value == "top_p"
    assert SamplingStrategy.BEAM_SEARCH.value == "beam_search"


def test_generation_result_dataclass():
    """Test GenerationResult dataclass."""
    result = GenerationResult(
        generated_ids=torch.randint(0, 100, (1, 10)),
        generated_text=["test output"],
        finish_reason=["eos"],
        num_generated_tokens=5,
    )
    assert result.num_generated_tokens == 5
    assert result.generated_text == ["test output"]
    assert result.finish_reason == ["eos"]


def test_text_generator_top_k_filtering():
    """Test top-k filtering method."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1]])
    filtered = TextGenerator._apply_top_k(logits, k=2)
    # After filtering, only top 2 should remain non-inf
    non_inf = (filtered != float("-inf")).sum().item()
    assert non_inf == 2


def test_text_generator_top_p_filtering():
    """Test top-p (nucleus) filtering method."""
    logits = torch.tensor([[0.1, 0.2, 0.3, 5.0, 0.05]])
    filtered = TextGenerator._apply_top_p(logits, p=0.9)
    # At least the highest probability token should remain
    assert filtered[0, 3] != float("-inf")


def test_text_generator_repetition_penalty():
    """Test repetition penalty application."""
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    generated = torch.tensor([[0, 2]])  # tokens 0 and 2 were generated
    penalized = TextGenerator._apply_repetition_penalty(logits.clone(), generated, penalty=2.0)
    # Penalized tokens should have modified logits (divided for positive, multiplied for negative)
    assert penalized[0, 0].item() != logits[0, 0].item()
    assert penalized[0, 2].item() != logits[0, 2].item()
    # Non-generated token should be unchanged
    assert penalized[0, 1].item() == logits[0, 1].item()
