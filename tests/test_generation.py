"""Tests for generation config."""
import pytest
from nexus.inference.generator import GenerationConfig, SamplingStrategy


def test_generation_config_default_values():
    """Test all default values of GenerationConfig."""
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 4096
    assert cfg.min_new_tokens == 0
    assert cfg.do_sample is True
    assert cfg.temperature == 0.7
    assert cfg.top_k == 50
    assert cfg.top_p == 0.9
    assert cfg.repetition_penalty == 1.1
    assert cfg.num_beams == 1
    assert cfg.length_penalty == 1.0
    assert cfg.eos_token_id == 2
    assert cfg.use_cache is True


def test_generation_config_greedy():
    """Test config for greedy decoding."""
    cfg = GenerationConfig(do_sample=False, temperature=0.0)
    assert cfg.do_sample is False


def test_generation_config_beam_search():
    """Test config for beam search."""
    cfg = GenerationConfig(num_beams=4, length_penalty=0.6, early_stopping=True)
    assert cfg.num_beams == 4
    assert cfg.length_penalty == 0.6
    assert cfg.early_stopping is True


def test_generation_config_stop_tokens():
    """Test config with stop tokens."""
    cfg = GenerationConfig(stop_token_ids=[2, 100, 200])
    assert len(cfg.stop_token_ids) == 3


def test_sampling_strategies():
    """Test all sampling strategy enum values."""
    strategies = [s.value for s in SamplingStrategy]
    assert "greedy" in strategies
    assert "temperature" in strategies
    assert "top_k" in strategies
    assert "top_p" in strategies
    assert "beam_search" in strategies
