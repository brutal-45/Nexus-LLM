"""Tests for the monitoring module.

Covers MetricsCollector, PerformanceMonitor, HealthChecker, and AlertManager.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from nexus_llm.monitoring.metrics import MetricsCollector, MetricPoint
from nexus_llm.monitoring.performance import PerformanceMonitor, TimingRecord, OperationStats
from nexus_llm.monitoring.health import HealthChecker, HealthReport, SystemHealth, ModelHealth
from nexus_llm.monitoring.alerts import AlertManager, AlertRule, Alert, Severity


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_metric(self):
        mc = MetricsCollector()
        mc.record_metric("latency", 0.5)
        data = mc.get_metric("latency")
        assert len(data) == 1
        assert data[0][1] == 0.5

    def test_record_metric_with_tags(self):
        mc = MetricsCollector()
        mc.record_metric("request_count", 1.0, tags={"endpoint": "/api"})
        data = mc.get_metric("request_count")
        assert len(data) == 1

    def test_get_nonexistent_metric(self):
        mc = MetricsCollector()
        data = mc.get_metric("nonexistent")
        assert data == []

    def test_get_aggregated(self):
        mc = MetricsCollector()
        mc.record_metric("latency", 0.1)
        mc.record_metric("latency", 0.3)
        mc.record_metric("latency", 0.5)
        agg = mc.get_aggregated("latency")
        assert agg["count"] == 3
        assert agg["min"] == 0.1
        assert agg["max"] == 0.5
        assert abs(agg["avg"] - 0.3) < 0.01

    def test_get_aggregated_empty(self):
        mc = MetricsCollector()
        agg = mc.get_aggregated("nonexistent")
        assert agg["count"] == 0
        assert agg["avg"] == 0.0

    def test_get_aggregated_with_window(self):
        mc = MetricsCollector()
        mc.record_metric("latency", 0.1)
        time.sleep(0.05)
        # Use a very large window to include the point
        agg = mc.get_aggregated("latency", window=10.0)
        assert agg["count"] == 1

    def test_list_metrics(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1)
        mc.record_metric("b", 2)
        names = mc.list_metrics()
        assert "a" in names
        assert "b" in names

    def test_clear_specific_metric(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1)
        mc.record_metric("b", 2)
        mc.clear("a")
        assert mc.get_metric("a") == []
        assert len(mc.get_metric("b")) == 1

    def test_clear_all(self):
        mc = MetricsCollector()
        mc.record_metric("a", 1)
        mc.record_metric("b", 2)
        mc.clear()
        assert mc.list_metrics() == []

    def test_increment(self):
        mc = MetricsCollector()
        mc.increment("counter")
        mc.increment("counter")
        data = mc.get_metric("counter")
        assert len(data) == 2

    def test_max_points_eviction(self):
        mc = MetricsCollector(max_points_per_metric=5)
        for i in range(10):
            mc.record_metric("test", float(i))
        data = mc.get_metric("test")
        assert len(data) == 5

    def test_percentiles(self):
        mc = MetricsCollector()
        for i in range(100):
            mc.record_metric("latency", float(i))
        agg = mc.get_aggregated("latency")
        assert agg["p50"] >= 40
        assert agg["p95"] >= 90
        assert agg["p99"] >= 95


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class TestPerformanceMonitor:
    """Tests for PerformanceMonitor."""

    def test_start_and_stop_timer(self):
        pm = PerformanceMonitor()
        timer_id = pm.start_timer("test_op")
        elapsed = pm.stop_timer(timer_id)
        assert elapsed >= 0

    def test_stop_unknown_timer_raises(self):
        pm = PerformanceMonitor()
        with pytest.raises(KeyError):
            pm.stop_timer(999)

    def test_get_operation_stats(self):
        pm = PerformanceMonitor()
        for _ in range(3):
            tid = pm.start_timer("test_op")
            pm.stop_timer(tid)
        stats = pm.get_operation_stats("test_op")
        assert stats is not None
        assert stats.count == 3
        assert stats.avg_time >= 0

    def test_get_stats_nonexistent(self):
        pm = PerformanceMonitor()
        assert pm.get_operation_stats("nonexistent") is None

    def test_list_operations(self):
        pm = PerformanceMonitor()
        tid1 = pm.start_timer("op1")
        pm.stop_timer(tid1)
        tid2 = pm.start_timer("op2")
        pm.stop_timer(tid2)
        ops = pm.list_operations()
        assert "op1" in ops
        assert "op2" in ops

    def test_profile_decorator(self):
        pm = PerformanceMonitor()

        @pm.profile
        def my_func():
            return 42

        result = my_func()
        assert result == 42
        assert "my_func" in " ".join(pm.list_operations())

    def test_profile_with_custom_name(self):
        pm = PerformanceMonitor()

        @pm.profile(operation="custom_op")
        def my_func():
            return "hello"

        result = my_func()
        assert result == "hello"
        assert "custom_op" in pm.list_operations()

    def test_clear_specific(self):
        pm = PerformanceMonitor()
        tid = pm.start_timer("op1")
        pm.stop_timer(tid)
        pm.clear("op1")
        assert pm.get_operation_stats("op1") is None

    def test_clear_all(self):
        pm = PerformanceMonitor()
        tid = pm.start_timer("op1")
        pm.stop_timer(tid)
        pm.clear()
        assert pm.list_operations() == []


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------

class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_check_model_health_default(self):
        hc = HealthChecker()
        model_health = hc.check_model_health()
        assert isinstance(model_health, ModelHealth)
        assert model_health.model_loaded is False

    def test_check_system_health(self):
        hc = HealthChecker()
        system_health = hc.check_system_health()
        assert isinstance(system_health, SystemHealth)
        assert system_health.cpu_percent >= 0

    def test_check_health(self):
        hc = HealthChecker()
        report = hc.check_health()
        assert isinstance(report, HealthReport)
        assert isinstance(report.healthy, bool)

    def test_update_model_state(self):
        hc = HealthChecker()
        hc.update_model_state(loaded=True, name="gpt2")
        model_health = hc.check_model_health()
        assert model_health.model_loaded is True
        assert model_health.model_name == "gpt2"

    def test_custom_check(self):
        hc = HealthChecker()
        hc.register_check("db", lambda: True)
        report = hc.check_health()
        assert "db" in report.custom_checks
        assert report.custom_checks["db"] is True

    def test_custom_check_failure(self):
        hc = HealthChecker()
        hc.register_check("failing", lambda: False)
        report = hc.check_health()
        assert report.custom_checks["failing"] is False
        assert report.healthy is False

    def test_custom_check_exception(self):
        hc = HealthChecker()
        hc.register_check("error_check", lambda: 1 / 0)
        report = hc.check_health()
        assert report.custom_checks["error_check"] is False

    def test_unregister_check(self):
        hc = HealthChecker()
        hc.register_check("temp", lambda: True)
        hc.unregister_check("temp")
        report = hc.check_health()
        assert "temp" not in report.custom_checks

    def test_model_health_is_healthy(self):
        hc = HealthChecker()
        hc.update_model_state(loaded=True, inference_count=10, error_count=1)
        mh = hc.check_model_health()
        assert mh.is_healthy is True

    def test_system_health_is_healthy(self):
        sh = SystemHealth(
            cpu_percent=50.0, memory_total_gb=16.0, memory_used_gb=8.0,
            memory_percent=50.0, disk_total_gb=500.0, disk_used_gb=100.0,
            disk_percent=20.0, gpu_available=False, gpu_name=None,
            gpu_memory_total_mb=None, gpu_memory_used_mb=None,
            gpu_utilization_percent=None,
        )
        assert sh.is_healthy is True


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    """Tests for AlertManager."""

    def test_add_rule(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, severity=Severity.CRITICAL)
        rules = am.list_rules()
        assert len(rules) == 1
        assert rules[0].name == "high_cpu"

    def test_add_rule_invalid_condition(self):
        am = AlertManager()
        with pytest.raises(ValueError, match="Unsupported condition"):
            am.add_rule("test", "invalid_op", 50.0)

    def test_evaluate_triggers_alert(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, severity=Severity.CRITICAL, cooldown_seconds=0)
        alerts = am.evaluate("high_cpu", 95.0)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.CRITICAL

    def test_evaluate_no_trigger(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0)
        alerts = am.evaluate("high_cpu", 50.0)
        assert len(alerts) == 0

    def test_evaluate_no_matching_rule(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0)
        alerts = am.evaluate("other_metric", 95.0)
        assert len(alerts) == 0

    def test_cooldown(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=300)
        alerts1 = am.evaluate("high_cpu", 95.0)
        alerts2 = am.evaluate("high_cpu", 96.0)
        assert len(alerts1) == 1
        assert len(alerts2) == 0  # still in cooldown

    def test_get_active_alerts(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=0)
        am.evaluate("high_cpu", 95.0)
        active = am.get_active_alerts()
        assert len(active) == 1

    def test_acknowledge_alert(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=0)
        alerts = am.evaluate("high_cpu", 95.0)
        assert am.acknowledge(alerts[0].alert_id) is True
        active = am.get_active_alerts()
        assert len(active) == 0  # acknowledged, so not in active

    def test_acknowledge_nonexistent(self):
        am = AlertManager()
        assert am.acknowledge("fake_id") is False

    def test_clear_alert(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=0)
        alerts = am.evaluate("high_cpu", 95.0)
        assert am.clear_alert(alerts[0].alert_id) is True

    def test_clear_all(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=0)
        am.evaluate("high_cpu", 95.0)
        am.clear_all()
        assert am.list_rules() == []
        assert am.get_active_alerts() == []

    def test_remove_rule(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0)
        am.remove_rule("high_cpu")
        assert am.list_rules() == []

    def test_get_all_alerts_history(self):
        am = AlertManager()
        am.add_rule("high_cpu", "gt", 90.0, cooldown_seconds=0)
        am.evaluate("high_cpu", 95.0)
        history = am.get_all_alerts()
        assert len(history) == 1

    def test_all_conditions(self):
        am = AlertManager()
        am.add_rule("test_gt", "gt", 10.0, cooldown_seconds=0)
        am.add_rule("test_lt", "lt", 5.0, cooldown_seconds=0)
        am.add_rule("test_gte", "gte", 10.0, cooldown_seconds=0)
        am.add_rule("test_lte", "lte", 5.0, cooldown_seconds=0)
        am.add_rule("test_eq", "eq", 42.0, cooldown_seconds=0)

        assert len(am.evaluate("test_gt", 15.0)) == 1
        assert len(am.evaluate("test_lt", 3.0)) == 1
        assert len(am.evaluate("test_gte", 10.0)) == 1
        assert len(am.evaluate("test_lte", 5.0)) == 1
        assert len(am.evaluate("test_eq", 42.0)) == 1
