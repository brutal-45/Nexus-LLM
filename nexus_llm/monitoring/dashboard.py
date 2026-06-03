"""Dashboard for Nexus-LLM monitoring terminal display."""

import time
from typing import Any, Dict, List, Optional

from nexus_llm.monitoring.metrics import MetricsCollector
from nexus_llm.monitoring.health import HealthChecker
from nexus_llm.monitoring.alerts import AlertManager, Severity


class Dashboard:
    """Aggregates monitoring data for terminal display.

    Pulls data from ``MetricsCollector``, ``HealthChecker``, and
    ``AlertManager`` to provide a unified view of system status.
    """

    def __init__(
        self,
        metrics: Optional[MetricsCollector] = None,
        health: Optional[HealthChecker] = None,
        alerts: Optional[AlertManager] = None,
    ) -> None:
        self._metrics = metrics or MetricsCollector()
        self._health = health or HealthChecker()
        self._alerts = alerts or AlertManager()
        self._error_log: List[Dict[str, Any]] = []

    # -- Data accessors -------------------------------------------------------

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of all metrics and status.

        Returns:
            Dict with keys: ``timestamp``, ``metrics``, ``alerts``,
            ``health``, ``system``.
        """
        now = time.time()

        # Aggregate all known metrics
        metric_names = self._metrics.list_metrics()
        metrics_summary: Dict[str, Any] = {}
        for name in metric_names:
            metrics_summary[name] = self._metrics.get_aggregated(name)

        # Active alert counts by severity
        active = self._alerts.get_active_alerts()
        alert_counts = {"info": 0, "warning": 0, "critical": 0}
        for a in active:
            alert_counts[a.severity.value] += 1

        # Health check
        health_report = self._health.check_health()

        return {
            "timestamp": now,
            "metrics": metrics_summary,
            "alerts": {
                "active_count": len(active),
                "by_severity": alert_counts,
            },
            "health": {
                "healthy": health_report.healthy,
                "model_healthy": health_report.model_health.is_healthy,
                "system_healthy": health_report.system_health.is_healthy,
            },
            "system": {
                "cpu_percent": health_report.system_health.cpu_percent,
                "memory_percent": health_report.system_health.memory_percent,
                "disk_percent": health_report.system_health.disk_percent,
                "gpu_available": health_report.system_health.gpu_available,
            },
        }

    def get_recent_errors(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the *n* most recent logged errors.

        Args:
            n: Number of errors to return (default 10).

        Returns:
            List of error dicts with keys: ``timestamp``, ``message``,
            ``details``.
        """
        return list(self._error_log[-n:])

    def log_error(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an error for dashboard display.

        Args:
            message: Error message.
            details: Optional additional context.
        """
        self._error_log.append({
            "timestamp": time.time(),
            "message": message,
            "details": details or {},
        })

    def get_system_status(self) -> Dict[str, Any]:
        """Return detailed system status information.

        Returns:
            Dict with CPU, memory, disk, and GPU details.
        """
        system = self._health.check_system_health()
        return {
            "cpu_percent": system.cpu_percent,
            "memory": {
                "total_gb": system.memory_total_gb,
                "used_gb": system.memory_used_gb,
                "percent": system.memory_percent,
            },
            "disk": {
                "total_gb": system.disk_total_gb,
                "used_gb": system.disk_used_gb,
                "percent": system.disk_percent,
            },
            "gpu": {
                "available": system.gpu_available,
                "name": system.gpu_name,
                "memory_total_mb": system.gpu_memory_total_mb,
                "memory_used_mb": system.gpu_memory_used_mb,
                "utilization_percent": system.gpu_utilization_percent,
            },
        }

    # -- Terminal display -----------------------------------------------------

    def render(self) -> str:
        """Render the dashboard as a formatted string for terminal display.

        Returns:
            Multi-line string with ASCII-art dashboard.
        """
        summary = self.get_summary()
        system = self.get_system_status()
        active_alerts = self._alerts.get_active_alerts()
        lines: List[str] = []

        lines.append("=" * 60)
        lines.append("  Nexus-LLM Dashboard")
        lines.append("=" * 60)

        # Health status
        health = summary["health"]
        status_icon = "\u2705" if health["healthy"] else "\u274c"
        lines.append(f"\n  Overall Status: {status_icon} {'HEALTHY' if health['healthy'] else 'UNHEALTHY'}")
        lines.append(f"  Model:  {'OK' if health['model_healthy'] else 'ISSUE'}")
        lines.append(f"  System: {'OK' if health['system_healthy'] else 'ISSUE'}")

        # System resources
        lines.append("\n  System Resources:")
        lines.append(f"    CPU:     {system['cpu_percent']:.1f}%")
        lines.append(f"    Memory:  {system['memory']['used_gb']:.1f}/{system['memory']['total_gb']:.1f} GB ({system['memory']['percent']:.1f}%)")
        lines.append(f"    Disk:    {system['disk']['used_gb']:.1f}/{system['disk']['total_gb']:.1f} GB ({system['disk']['percent']:.1f}%)")
        if system["gpu"]["available"]:
            lines.append(f"    GPU:     {system['gpu']['name']}")
            lines.append(f"    GPU Mem: {system['gpu']['memory_used_mb']:.0f}/{system['gpu']['memory_total_mb']:.0f} MB ({system['gpu']['utilization_percent']:.1f}%)")
        else:
            lines.append("    GPU:     Not available")

        # Metrics summary
        metrics = summary["metrics"]
        if metrics:
            lines.append("\n  Metrics:")
            for name, agg in metrics.items():
                lines.append(
                    f"    {name}: count={agg['count']}, "
                    f"avg={agg['avg']:.2f}, "
                    f"p95={agg['p95']:.2f}, "
                    f"max={agg['max']:.2f}"
                )

        # Active alerts
        lines.append(f"\n  Active Alerts: {len(active_alerts)}")
        for alert in active_alerts[:5]:
            lines.append(f"    [{alert.severity.value.upper()}] {alert.message}")

        # Recent errors
        recent = self.get_recent_errors(5)
        if recent:
            lines.append("\n  Recent Errors:")
            for err in recent:
                ts = time.strftime("%H:%M:%S", time.localtime(err["timestamp"]))
                lines.append(f"    [{ts}] {err['message']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def display(self) -> None:
        """Print the dashboard to stdout."""
        print(self.render())
