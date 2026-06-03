"""Tests for state management (StateManager, observers, persistence)."""
import json
import os
import tempfile
from pathlib import Path

import pytest

from nexus_llm.state import StateManager, StateObserver


class TestStateObserver:
    """Test StateObserver."""

    def test_matches_all_keys(self):
        obs = StateObserver(name="test", callback=lambda k, o, n: None)
        assert obs.matches("any.key") is True

    def test_matches_specific_keys(self):
        obs = StateObserver(name="test", callback=lambda k, o, n: None, keys={"model.current"})
        assert obs.matches("model.current") is True
        assert obs.matches("model.device") is False


class TestStateManager:
    """Test StateManager."""

    @pytest.fixture
    def manager(self):
        return StateManager()

    def test_defaults(self, manager):
        assert manager.get("app.initialized") is False
        assert manager.get("model.device") == "auto"
        assert manager.get("server.port") == 8000

    def test_set_and_get(self, manager):
        manager.set("model.current", "gpt2")
        assert manager.get("model.current") == "gpt2"

    def test_nested_set(self, manager):
        manager.set("custom.nested.key", "value")
        assert manager.get("custom.nested.key") == "value"

    def test_get_default(self, manager):
        assert manager.get("nonexistent", "fallback") == "fallback"

    def test_has(self, manager):
        assert manager.has("app.initialized") is True
        assert manager.has("nonexistent") is False

    def test_delete(self, manager):
        manager.set("custom.key", "val")
        assert manager.delete("custom.key") is True
        assert manager.has("custom.key") is False

    def test_delete_nonexistent(self, manager):
        assert manager.delete("nonexistent.key") is False

    def test_get_all(self, manager):
        state = manager.get_all()
        assert isinstance(state, dict)
        assert "app" in state

    def test_update(self, manager):
        manager.update({"model.current": "gpt2", "model.device": "cuda"})
        assert manager.get("model.current") == "gpt2"
        assert manager.get("model.device") == "cuda"


class TestStateObservers:
    """Test observer functionality."""

    def test_observe_changes(self):
        manager = StateManager()
        changes = []
        manager.observe("test_obs", callback=lambda k, o, n: changes.append((k, o, n)))
        manager.set("model.current", "gpt2")
        assert len(changes) == 1
        assert changes[0][0] == "model.current"
        assert changes[0][2] == "gpt2"

    def test_observe_specific_keys(self):
        manager = StateManager()
        changes = []
        manager.observe("test_obs", callback=lambda k, o, n: changes.append(k),
                         keys={"model.current"})
        manager.set("model.current", "gpt2")
        manager.set("model.device", "cuda")
        assert len(changes) == 1
        assert changes[0] == "model.current"

    def test_unobserve(self):
        manager = StateManager()
        changes = []
        obs = manager.observe("test_obs", callback=lambda k, o, n: changes.append(k))
        manager.set("model.current", "gpt2")
        manager.unobserve(obs)
        manager.set("model.current", "llama")
        assert len(changes) == 1

    def test_unobserve_by_name(self):
        manager = StateManager()
        manager.observe("obs1", callback=lambda k, o, n: None)
        manager.observe("obs1", callback=lambda k, o, n: None)
        count = manager.unobserve_by_name("obs1")
        assert count == 2
        assert manager.observer_count == 0

    def test_observer_error_does_not_crash(self):
        manager = StateManager()

        def bad_callback(k, o, n):
            raise RuntimeError("boom")

        manager.observe("bad_obs", callback=bad_callback)
        manager.set("model.current", "gpt2")  # Should not raise


class TestStateHistory:
    """Test state change history."""

    def test_history_recorded(self):
        manager = StateManager()
        manager.set("model.current", "gpt2")
        history = manager.get_history()
        assert len(history) >= 1

    def test_history_filter_by_key(self):
        manager = StateManager()
        manager.set("model.current", "gpt2")
        manager.set("model.device", "cuda")
        history = manager.get_history(key="model.current")
        assert all(h["key"] == "model.current" for h in history)

    def test_history_limit(self):
        manager = StateManager()
        for i in range(10):
            manager.set("model.current", f"model_{i}")
        history = manager.get_history(limit=3)
        assert len(history) == 3


class TestStatePersistence:
    """Test state save and load."""

    def test_save_and_load(self, tmp_dir):
        path = str(tmp_dir / "state.json")
        manager = StateManager()
        manager.set("model.current", "gpt2")
        manager.save(path)

        manager2 = StateManager()
        manager2.load(path)
        assert manager2.get("model.current") == "gpt2"

    def test_save_creates_directories(self, tmp_dir):
        path = str(tmp_dir / "nested" / "dir" / "state.json")
        manager = StateManager()
        manager.save(path)
        assert Path(path).exists()

    def test_load_nonexistent_file(self):
        manager = StateManager()
        manager.load("/nonexistent/path/state.json")  # Should not raise

    def test_reset(self):
        manager = StateManager()
        manager.set("model.current", "gpt2")
        manager.reset()
        assert manager.get("model.current") is None
        assert manager.get("app.initialized") is False
