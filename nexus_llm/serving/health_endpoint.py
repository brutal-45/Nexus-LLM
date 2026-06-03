"""Health endpoint for Nexus-LLM serving."""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class HealthStatus:
    """Overall health status."""
    healthy: bool
    status: str  # "healthy", "degraded", "unhealthy"
    details: Dict[str, Any]


@dataclass
class ReadyStatus:
    """Readiness probe result."""
    ready: bool
    reason: Optional[str]
    details: Dict[str, Any]


@dataclass
class LiveStatus:
    """Liveness probe result."""
    alive: bool
    uptime_seconds: float


class HealthEndpoint:
    """Implements health, readiness, and liveness probes.

    Designed for integration with Kubernetes-style health checks.
    """

    def __init__(
        self,
        model_server: Optional[Any] = None,
        check_interval: float = 30.0,
    ) -> None:
        self._server = model_server
        self._check_interval = check_interval
        self._start_time: float = time.time()
        self._last_check: Optional[float] = None
        self._last_result: Optional[HealthStatus] = None
        self._custom_checks: Dict[str, Any] = {}

    # -- Probes ---------------------------------------------------------------

    def health_check(self) -> HealthStatus:
        """Run a comprehensive health check.

        Returns:
            ``HealthStatus`` with overall assessment.
        """
        now = time.time()
        self._last_check = now

        issues: list = []
        details: Dict[str, Any] = {
            "server": "unknown",
            "model_loaded": False,
        }

        # Check server status
        if self._server is not None:
            try:
                status = self._server.get_status()
                details["server"] = status.get("status", "unknown")
                details["model_loaded"] = status.get("model_loaded", False)
                details["request_count"] = status.get("request_count", 0)
                details["error_count"] = status.get("error_count", 0)

                if status.get("status") != "running":
                    issues.append("Server not running")
                if status.get("error_count", 0) > 100:
                    issues.append("High error count")
            except Exception as exc:
                issues.append(f"Server check failed: {exc}")
                details["server"] = "error"

        # Run custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                ok = bool(check_fn())
                details[name] = ok
                if not ok:
                    issues.append(f"Custom check '{name}' failed")
            except Exception as exc:
                details[name] = f"error: {exc}"
                issues.append(f"Custom check '{name}' raised: {exc}")

        if not issues:
            overall = "healthy"
            healthy = True
        elif len(issues) == 1 and details.get("model_loaded") is True:
            overall = "degraded"
            healthy = True
        else:
            overall = "unhealthy"
            healthy = False

        result = HealthStatus(
            healthy=healthy,
            status=overall,
            details=details,
        )
        self._last_result = result
        return result

    def readiness_check(self) -> ReadyStatus:
        """Check if the server is ready to accept traffic.

        Returns:
            ``ReadyStatus`` indicating readiness.
        """
        if self._server is not None:
            try:
                status = self._server.get_status()
                if status.get("status") != "running":
                    return ReadyStatus(
                        ready=False,
                        reason=f"Server status is '{status.get('status')}'",
                        details=status,
                    )
                if not status.get("model_loaded", False):
                    return ReadyStatus(
                        ready=False,
                        reason="No model loaded",
                        details=status,
                    )
            except Exception as exc:
                return ReadyStatus(
                    ready=False,
                    reason=str(exc),
                    details={},
                )

        return ReadyStatus(ready=True, reason=None, details={})

    def liveness_check(self) -> LiveStatus:
        """Check if the server process is alive.

        Returns:
            ``LiveStatus`` with uptime.
        """
        return LiveStatus(
            alive=True,
            uptime_seconds=round(time.time() - self._start_time, 2),
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Return server and health metrics for monitoring.

        Returns:
            Dict with ``uptime``, ``health``, ``readiness``, ``liveness``.
        """
        return {
            "uptime_seconds": round(time.time() - self._start_time, 2),
            "health": self._last_result is not None and self._last_result.healthy,
            "readiness": self.readiness_check().ready,
            "liveness": self.liveness_check().alive,
            "last_check": self._last_check,
        }

    # -- Custom checks --------------------------------------------------------

    def register_check(self, name: str, check_fn: Any) -> None:
        """Register a custom health check function.

        Args:
            name: Check name.
            check_fn: Callable returning truthy for healthy.
        """
        self._custom_checks[name] = check_fn

    def unregister_check(self, name: str) -> None:
        """Remove a custom health check."""
        self._custom_checks.pop(name, None)
