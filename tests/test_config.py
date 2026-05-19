"""Tests for config loading."""
import pytest
import os
import yaml
from nexus.model.config import ModelConfig
from nexus.chat.config import ChatConfig, GenerationConfig as ChatGenConfig, UIConfig, HistoryConfig


def test_model_config_defaults():
    """Test ModelConfig default values."""
    cfg = ModelConfig()
    assert cfg.hidden_size == 12288
    assert cfg.num_hidden_layers == 80
    assert cfg.vocab_size == 65536
    assert cfg.rms_norm_eps == 1e-5


def test_model_config_custom():
    """Test creating a custom ModelConfig."""
    cfg = ModelConfig(
        name="Nexus-Small",
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=32000,
    )
    assert cfg.name == "Nexus-Small"
    assert cfg.hidden_size == 512
    assert cfg.head_dim == 512 // 8  # auto-computed


def test_model_config_validation():
    """Test that invalid configs raise errors."""
    with pytest.raises(AssertionError):
        ModelConfig(hidden_size=100, num_attention_heads=7)  # 100 % 7 != 0


def test_model_config_head_dim_auto():
    """Test that head_dim is auto-computed."""
    cfg = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    assert cfg.head_dim == 256 // 4  # = 64


def test_model_config_total_params():
    """Test total parameter estimation."""
    cfg = ModelConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=1024)
    params = cfg.total_params
    assert params > 0


def test_model_config_to_dict():
    """Test serialization to dict."""
    cfg = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    d = cfg.to_dict()
    assert "hidden_size" in d
    assert d["hidden_size"] == 256


def test_model_config_from_dict():
    """Test deserialization from dict."""
    d = {"hidden_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2,
         "num_hidden_layers": 4}
    cfg = ModelConfig.from_dict(d)
    assert cfg.hidden_size == 512


def test_model_config_save_load_yaml(tmp_dir):
    """Test saving and loading config from YAML."""
    cfg = ModelConfig(name="Test", hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
    path = os.path.join(tmp_dir, "test_config.yaml")
    cfg.save_yaml(path)
    assert os.path.exists(path)
    
    loaded = ModelConfig.from_yaml(path)
    assert loaded.hidden_size == 256
    assert loaded.name == "Test"


def test_chat_config_defaults():
    """Test ChatConfig defaults."""
    cfg = ChatConfig()
    assert cfg.model_name == "nexus-ai"
    assert cfg.backend == "zai"
    assert isinstance(cfg.generation, ChatGenConfig)
    assert isinstance(cfg.ui, UIConfig)
    assert isinstance(cfg.history, HistoryConfig)


def test_chat_config_from_dict():
    """Test ChatConfig from dictionary."""
    d = {
        "model_name": "test-model",
        "generation": {"temperature": 0.5, "max_tokens": 2048},
        "ui": {"theme": "light"},
    }
    cfg = ChatConfig.from_dict(d)
    assert cfg.model_name == "test-model"
    assert cfg.generation.temperature == 0.5
    assert cfg.generation.max_tokens == 2048
    assert cfg.ui.theme == "light"


def test_chat_config_apply_overrides():
    """Test applying overrides to ChatConfig."""
    cfg = ChatConfig()
    cfg.apply_overrides({"model_name": "overridden"})
    assert cfg.model_name == "overridden"


def test_chat_config_save_load(tmp_dir):
    """Test saving and loading ChatConfig."""
    cfg = ChatConfig(model_name="test-model")
    path = os.path.join(tmp_dir, "chat_config.yaml")
    cfg.save(path)
    loaded = ChatConfig.from_file(path)
    assert loaded.model_name == "test-model"
