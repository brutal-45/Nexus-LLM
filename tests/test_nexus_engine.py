"""Tests for nexus_llm.nexus.engine module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.nexus.engine import NexusEngine


class TestNexusEngine:
    """Tests for the NexusEngine class."""

    def test_init_default(self):
        engine = NexusEngine()
        assert engine is not None

    def test_init_with_config(self):
        engine = NexusEngine(config={"max_workers": 4})
        assert engine is not None

    def test_start(self):
        engine = NexusEngine()
        engine.start()
        assert engine.is_running is True

    def test_stop(self):
        engine = NexusEngine()
        engine.start()
        engine.stop()
        assert engine.is_running is False

    def test_submit_task(self):
        engine = NexusEngine()
        engine.start()
        result = engine.submit_task(lambda: 42)
        assert result == 42

    def test_submit_task_when_stopped(self):
        engine = NexusEngine()
        with pytest.raises(RuntimeError):
            engine.submit_task(lambda: 42)

    def test_health_check(self):
        engine = NexusEngine()
        engine.start()
        health = engine.health_check()
        assert "status" in health

    def test_get_metrics(self):
        engine = NexusEngine()
        engine.start()
        metrics = engine.get_metrics()
        assert isinstance(metrics, dict)
