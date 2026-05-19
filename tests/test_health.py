"""Tests for health checks."""
import pytest


class HealthChecker:
    """Simple health checker for testing."""
    def __init__(self):
        self.checks = {}

    def register(self, name, check_fn):
        self.checks[name] = check_fn

    def run(self):
        results = {}
        for name, fn in self.checks.items():
            try:
                results[name] = fn()
            except Exception as e:
                results[name] = {"status": "unhealthy", "error": str(e)}
        return results

    def is_healthy(self):
        results = self.run()
        return all(r.get("status") == "healthy" for r in results.values())


@pytest.fixture
def health_checker():
    return HealthChecker()


def test_health_checker_register(health_checker):
    """Test registering a health check."""
    health_checker.register("test", lambda: {"status": "healthy"})
    assert "test" in health_checker.checks


def test_health_checker_healthy(health_checker):
    """Test all checks passing."""
    health_checker.register("check1", lambda: {"status": "healthy"})
    health_checker.register("check2", lambda: {"status": "healthy"})
    assert health_checker.is_healthy()


def test_health_checker_unhealthy(health_checker):
    """Test a failing health check."""
    health_checker.register("good", lambda: {"status": "healthy"})
    health_checker.register("bad", lambda: {"status": "unhealthy"})
    assert not health_checker.is_healthy()


def test_health_checker_exception(health_checker):
    """Test health check that raises an exception."""
    health_checker.register("error", lambda: 1/0)
    results = health_checker.run()
    assert results["error"]["status"] == "unhealthy"
