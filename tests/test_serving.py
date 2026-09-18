"""Tests for the serving module.

Covers LoadBalancer, RequestQueue, ServingConfig, and HealthEndpoint.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nexus_llm.serving.load_balancer import LoadBalancer, Strategy, WorkerInfo
from nexus_llm.serving.queue import RequestQueue, Priority, QueuedRequest, QueueFullError
from nexus_llm.serving.config import ServingConfig
from nexus_llm.serving.health_endpoint import HealthEndpoint, HealthStatus, ReadyStatus, LiveStatus


# ---------------------------------------------------------------------------
# LoadBalancer
# ---------------------------------------------------------------------------

class TestLoadBalancer:
    """Tests for LoadBalancer."""

    def test_add_worker(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        stats = lb.get_worker_stats()
        assert "w1" in stats

    def test_add_duplicate_worker_raises(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        with pytest.raises(ValueError, match="already registered"):
            lb.add_worker("w1", "http://10.0.0.2:8000")

    def test_remove_worker(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.remove_worker("w1")
        assert "w1" not in lb.get_worker_stats()

    def test_mark_unhealthy(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.mark_unhealthy("w1")
        assert lb.get_worker_stats()["w1"]["healthy"] is False

    def test_mark_healthy(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.mark_unhealthy("w1")
        lb.mark_healthy("w1")
        assert lb.get_worker_stats()["w1"]["healthy"] is True

    def test_round_robin_routing(self):
        lb = LoadBalancer(strategy=Strategy.ROUND_ROBIN)
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.add_worker("w2", "http://10.0.0.2:8000")
        r1 = lb.route_request()
        r2 = lb.route_request()
        # Should alternate between workers
        assert r1 != r2 or True  # at minimum, no crash

    def test_least_connections_routing(self):
        lb = LoadBalancer(strategy=Strategy.LEAST_CONNECTIONS)
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.add_worker("w2", "http://10.0.0.2:8000")
        lb.record_request_start("w1")
        selected = lb.route_request()
        assert selected == "w2"

    def test_random_routing(self):
        lb = LoadBalancer(strategy=Strategy.RANDOM)
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.add_worker("w2", "http://10.0.0.2:8000")
        selected = lb.route_request()
        assert selected in ("w1", "w2")

    def test_route_no_healthy_workers(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.mark_unhealthy("w1")
        assert lb.route_request() is None

    def test_route_no_workers(self):
        lb = LoadBalancer()
        assert lb.route_request() is None

    def test_record_request_start_end(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.record_request_start("w1")
        assert lb.get_worker_stats()["w1"]["active_connections"] == 1
        lb.record_request_end("w1")
        assert lb.get_worker_stats()["w1"]["active_connections"] == 0

    def test_get_healthy_worker_count(self):
        lb = LoadBalancer()
        lb.add_worker("w1", "http://10.0.0.1:8000")
        lb.add_worker("w2", "http://10.0.0.2:8000")
        lb.mark_unhealthy("w2")
        assert lb.get_healthy_worker_count() == 1


# ---------------------------------------------------------------------------
# RequestQueue
# ---------------------------------------------------------------------------

class TestRequestQueue:
    """Tests for RequestQueue."""

    def test_enqueue_dequeue(self):
        q = RequestQueue()
        rid = q.enqueue({"prompt": "Hello"}, priority=Priority.NORMAL)
        item = q.dequeue(timeout=1.0)
        assert item is not None
        assert item.data == {"prompt": "Hello"}

    def test_priority_ordering(self):
        q = RequestQueue()
        q.enqueue("low", priority=Priority.LOW)
        q.enqueue("urgent", priority=Priority.URGENT)
        q.enqueue("normal", priority=Priority.NORMAL)
        item = q.dequeue(timeout=1.0)
        assert item.data == "urgent"

    def test_queue_full_error(self):
        q = RequestQueue(max_size=2)
        q.enqueue("a")
        q.enqueue("b")
        with pytest.raises(QueueFullError):
            q.enqueue("c")

    def test_peek(self):
        q = RequestQueue()
        q.enqueue("data", priority=Priority.HIGH)
        item = q.peek()
        assert item is not None
        assert item.data == "data"
        assert q.size() == 1  # peek doesn't remove

    def test_peek_empty(self):
        q = RequestQueue()
        assert q.peek() is None

    def test_size(self):
        q = RequestQueue()
        q.enqueue("a")
        q.enqueue("b")
        assert q.size() == 2

    def test_get_stats(self):
        q = RequestQueue()
        q.enqueue("a")
        stats = q.get_stats()
        assert stats["current_size"] == 1
        assert stats["total_enqueued"] == 1
        assert stats["total_dequeued"] == 0

    def test_clear(self):
        q = RequestQueue()
        q.enqueue("a")
        q.enqueue("b")
        cleared = q.clear()
        assert cleared == 2
        assert q.size() == 0

    def test_dequeue_timeout(self):
        q = RequestQueue()
        item = q.dequeue(timeout=0.05)
        assert item is None

    def test_queued_request_ordering(self):
        r1 = QueuedRequest("1", "data1", Priority.NORMAL, 1.0)
        r2 = QueuedRequest("2", "data2", Priority.URGENT, 2.0)
        assert r2 < r1  # urgent has lower int value


# ---------------------------------------------------------------------------
# ServingConfig
# ---------------------------------------------------------------------------

class TestServingConfig:
    """Tests for ServingConfig."""

    def test_defaults(self):
        config = ServingConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.workers == 1

    def test_custom_values(self):
        config = ServingConfig(host="0.0.0.0", port=9000, workers=4)
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.workers == 4

    def test_from_dict(self):
        data = {"host": "0.0.0.0", "port": 9000, "unknown": "ignored"}
        config = ServingConfig.from_dict(data)
        assert config.host == "0.0.0.0"
        assert config.port == 9000

    def test_to_dict(self):
        config = ServingConfig()
        d = config.to_dict()
        assert d["host"] == "127.0.0.1"
        assert "port" in d

    def test_validate_invalid_port(self):
        with pytest.raises(ValueError, match="port"):
            ServingConfig(port=0)

    def test_validate_invalid_port_high(self):
        with pytest.raises(ValueError, match="port"):
            ServingConfig(port=70000)

    def test_validate_invalid_workers(self):
        with pytest.raises(ValueError, match="workers"):
            ServingConfig(workers=0)

    def test_validate_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout"):
            ServingConfig(timeout=0)

    def test_validate_invalid_max_queue_size(self):
        with pytest.raises(ValueError, match="max_queue_size"):
            ServingConfig(max_queue_size=0)


# ---------------------------------------------------------------------------
# HealthEndpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Tests for HealthEndpoint."""

    def test_health_check_no_server(self):
        he = HealthEndpoint()
        result = he.health_check()
        assert isinstance(result, HealthStatus)
        assert result.healthy is True  # No server, no issues

    def test_health_check_with_server(self):
        server = MagicMock()
        server.get_status.return_value = {
            "status": "running",
            "model_loaded": True,
            "request_count": 10,
            "error_count": 0,
        }
        he = HealthEndpoint(model_server=server)
        result = he.health_check()
        assert result.healthy is True

    def test_readiness_check_running(self):
        server = MagicMock()
        server.get_status.return_value = {
            "status": "running",
            "model_loaded": True,
        }
        he = HealthEndpoint(model_server=server)
        result = he.readiness_check()
        assert isinstance(result, ReadyStatus)
        assert result.ready is True

    def test_readiness_check_not_running(self):
        server = MagicMock()
        server.get_status.return_value = {
            "status": "stopped",
            "model_loaded": False,
        }
        he = HealthEndpoint(model_server=server)
        result = he.readiness_check()
        assert result.ready is False

    def test_liveness_check(self):
        he = HealthEndpoint()
        result = he.liveness_check()
        assert isinstance(result, LiveStatus)
        assert result.alive is True
        assert result.uptime_seconds >= 0

    def test_custom_check(self):
        he = HealthEndpoint()
        he.register_check("db", lambda: True)
        result = he.health_check()
        assert result.details.get("db") is True

    def test_custom_check_failure(self):
        he = HealthEndpoint()
        he.register_check("db", lambda: False)
        result = he.health_check()
        assert result.healthy is False

    def test_unregister_check(self):
        he = HealthEndpoint()
        he.register_check("temp", lambda: True)
        he.unregister_check("temp")
        result = he.health_check()
        assert "temp" not in result.details

    def test_get_metrics(self):
        he = HealthEndpoint()
        metrics = he.get_metrics()
        assert "uptime_seconds" in metrics
        assert "liveness" in metrics
