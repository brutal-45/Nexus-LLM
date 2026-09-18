"""Tests for nexus_llm.nexus.optimizer module."""

import pytest
from unittest.mock import MagicMock
from nexus_llm.nexus.optimizer import NexusOptimizer


class TestNexusOptimizer:
    """Tests for the NexusOptimizer class."""

    def test_init_default(self):
        opt = NexusOptimizer()
        assert opt is not None

    def test_init_with_config(self):
        opt = NexusOptimizer(config={"level": "aggressive"})
        assert opt is not None

    def test_optimize(self):
        opt = NexusOptimizer()
        data = {"latency_ms": 500, "throughput": 100}
        result = opt.optimize(data)
        assert isinstance(result, dict)

    def test_get_bottlenecks(self):
        opt = NexusOptimizer()
        data = {"latency_ms": 500, "memory_mb": 8192}
        bottlenecks = opt.get_bottlenecks(data)
        assert isinstance(bottlenecks, list)

    def test_get_recommendations(self):
        opt = NexusOptimizer()
        data = {"latency_ms": 500, "memory_mb": 8192}
        recs = opt.get_recommendations(data)
        assert isinstance(recs, list)

    def test_set_level(self):
        opt = NexusOptimizer()
        opt.set_level("aggressive")
        assert opt.level == "aggressive"

    def test_get_metrics(self):
        opt = NexusOptimizer()
        metrics = opt.get_metrics()
        assert isinstance(metrics, dict)
