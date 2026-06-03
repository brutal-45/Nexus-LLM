"""Tests for the config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from nexus_llm.core.config import (
    ModelSettings,
    ServerSettings,
    TerminalSettings,
    TrainingSettings,
    Settings,
    get_settings,
    reset_settings,
    PROJECT_ROOT,
    DEFAULT_CONFIG_PATH,
)


# ---------------------------------------------------------------------------
# ModelSettings
# ---------------------------------------------------------------------------

class TestModelSettings:
    """Tests for the ModelSettings dataclass."""

    def test_defaults(self):
        ms = ModelSettings()
        assert ms.name == "gpt2-medium"
        assert ms.device == "auto"
        assert ms.precision == "fp32"
        assert ms.max_length == 512
        assert ms.temperature == 0.7
        assert ms.top_p == 0.9
        assert ms.top_k == 50
        assert ms.repetition_penalty == 1.1
        assert ms.num_beams == 1
        assert ms.do_sample is True
        assert isinstance(ms.cache_dir, str)

    def test_custom_values(self):
        ms = ModelSettings(name="phi-2", device="cuda", precision="fp16", max_length=1024)
        assert ms.name == "phi-2"
        assert ms.device == "cuda"
        assert ms.precision == "fp16"
        assert ms.max_length == 1024


# ---------------------------------------------------------------------------
# ServerSettings
# ---------------------------------------------------------------------------

class TestServerSettings:
    """Tests for the ServerSettings dataclass."""

    def test_defaults(self):
        ss = ServerSettings()
        assert ss.host == "127.0.0.1"
        assert ss.port == 8000
        assert ss.workers == 1
        assert ss.cors_origins == ["*"]
        assert ss.api_key is None

    def test_custom_values(self):
        ss = ServerSettings(host="0.0.0.0", port=9000, api_key="secret")
        assert ss.host == "0.0.0.0"
        assert ss.port == 9000
        assert ss.api_key == "secret"


# ---------------------------------------------------------------------------
# TerminalSettings
# ---------------------------------------------------------------------------

class TestTerminalSettings:
    """Tests for the TerminalSettings dataclass."""

    def test_defaults(self):
        ts = TerminalSettings()
        assert ts.theme == "dark"
        assert ts.show_tokens is True
        assert ts.show_timing is True
        assert ts.streaming is True
        assert ts.markdown is True
        assert ts.syntax_highlight is True
        assert isinstance(ts.history_file, str)
        assert ts.max_history == 1000

    def test_custom_values(self):
        ts = TerminalSettings(theme="light", streaming=False, max_history=500)
        assert ts.theme == "light"
        assert ts.streaming is False
        assert ts.max_history == 500


# ---------------------------------------------------------------------------
# TrainingSettings
# ---------------------------------------------------------------------------

class TestTrainingSettings:
    """Tests for the TrainingSettings dataclass."""

    def test_defaults(self):
        ts = TrainingSettings()
        assert isinstance(ts.output_dir, str)
        assert ts.lora_r == 8
        assert ts.lora_alpha == 16
        assert ts.lora_dropout == 0.05
        assert ts.learning_rate == 2e-4
        assert ts.num_epochs == 3
        assert ts.batch_size == 4
        assert ts.gradient_accumulation_steps == 4
        assert ts.warmup_steps == 100
        assert ts.save_steps == 500
        assert ts.eval_steps == 500
        assert ts.max_seq_length == 512

    def test_custom_values(self):
        ts = TrainingSettings(lora_r=16, batch_size=8, learning_rate=1e-4)
        assert ts.lora_r == 16
        assert ts.batch_size == 8
        assert ts.learning_rate == 1e-4


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class TestSettings:
    """Tests for the main Settings dataclass."""

    def test_defaults(self):
        s = Settings()
        assert isinstance(s.model, ModelSettings)
        assert isinstance(s.server, ServerSettings)
        assert isinstance(s.terminal, TerminalSettings)
        assert isinstance(s.training, TrainingSettings)
        assert s.debug is False
        assert s.log_level == "INFO"
        assert isinstance(s.log_file, str)

    def test_nested_defaults_independent(self):
        """Each Settings instance gets its own sub-setting objects."""
        s1 = Settings()
        s2 = Settings()
        s1.model.name = "phi-2"
        assert s2.model.name == "gpt2-medium"  # s2 should be unaffected

    def test_from_yaml_nonexistent_file(self):
        """from_yaml returns default Settings when the file doesn't exist."""
        s = Settings.from_yaml("/nonexistent/path/config.yaml")
        assert isinstance(s, Settings)
        assert s.model.name == "gpt2-medium"  # default

    def test_from_yaml_valid_file(self):
        """from_yaml correctly loads values from a YAML file."""
        data = {
            "model": {"name": "phi-2", "device": "cuda", "precision": "fp16"},
            "server": {"host": "0.0.0.0", "port": 9000},
            "terminal": {"theme": "light", "streaming": False},
            "training": {"lora_r": 16, "batch_size": 8},
            "debug": True,
            "log_level": "DEBUG",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            s = Settings.from_yaml(tmp_path)
            assert s.model.name == "phi-2"
            assert s.model.device == "cuda"
            assert s.model.precision == "fp16"
            assert s.server.host == "0.0.0.0"
            assert s.server.port == 9000
            assert s.terminal.theme == "light"
            assert s.terminal.streaming is False
            assert s.training.lora_r == 16
            assert s.training.batch_size == 8
            assert s.debug is True
            assert s.log_level == "DEBUG"
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_partial_file(self):
        """from_yaml with a YAML file that only overrides some fields."""
        data = {"model": {"name": "gpt2-xl"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            s = Settings.from_yaml(tmp_path)
            assert s.model.name == "gpt2-xl"
            # Other settings remain defaults
            assert s.server.host == "127.0.0.1"
            assert s.debug is False
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_empty_file(self):
        """from_yaml with an empty YAML file returns defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            tmp_path = f.name

        try:
            s = Settings.from_yaml(tmp_path)
            assert s.model.name == "gpt2-medium"
        finally:
            os.unlink(tmp_path)

    def test_from_yaml_ignores_unknown_keys(self):
        """Unknown keys in the YAML file are silently ignored."""
        data = {"model": {"name": "phi-2", "nonexistent_key": "value"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            s = Settings.from_yaml(tmp_path)
            assert s.model.name == "phi-2"
            assert not hasattr(s.model, "nonexistent_key")
        finally:
            os.unlink(tmp_path)

    def test_to_yaml(self):
        """to_yaml writes settings that can be read back."""
        s = Settings()
        s.model.name = "phi-2"
        s.debug = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            s.to_yaml(path)

            assert os.path.exists(path)
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            assert data["model"]["name"] == "phi-2"
            assert data["debug"] is True

    def test_to_yaml_roundtrip(self):
        """Settings survive a to_yaml → from_yaml roundtrip."""
        s1 = Settings()
        s1.model.name = "phi-2"
        s1.model.device = "cuda"
        s1.server.port = 9999
        s1.debug = True
        s1.log_level = "WARNING"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            s1.to_yaml(path)
            s2 = Settings.from_yaml(path)

            assert s2.model.name == "phi-2"
            assert s2.model.device == "cuda"
            assert s2.server.port == 9999
            assert s2.debug is True
            assert s2.log_level == "WARNING"


# ---------------------------------------------------------------------------
# get_settings / reset_settings
# ---------------------------------------------------------------------------

class TestGetSettings:
    """Tests for the get_settings / reset_settings singleton."""

    def setup_method(self):
        """Reset the global settings instance before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert isinstance(s, Settings)

    def test_get_settings_singleton(self):
        """Repeated calls return the same instance."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_get_settings_with_config_path(self):
        """get_settings with a custom config path."""
        data = {"model": {"name": "gpt2-xl"}, "debug": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp_path = f.name

        try:
            s = get_settings(config_path=tmp_path)
            assert s.model.name == "gpt2-xl"
            assert s.debug is True
        finally:
            os.unlink(tmp_path)

    def test_reset_settings(self):
        """reset_settings clears the singleton so next call creates a new one."""
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        assert s1 is not s2

    def test_reset_settings_none(self):
        """After reset, the internal instance is None."""
        reset_settings()
        from nexus_llm.core import config
        assert config._settings_instance is None
