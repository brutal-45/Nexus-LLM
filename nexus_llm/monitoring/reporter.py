"""Nexus-LLM Status Reporting.

Generates comprehensive status reports for the Nexus-LLM framework,
covering system health, model status, inference performance, training
progress, and resource utilization. Supports multiple output formats.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report output formats."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ComponentStatus:
    """Status of a single system component.

    Attributes:
        name: Component name.
        status: Health status.
        message: Status message.
        details: Additional details.
        latency_ms: Response latency if applicable.
    """

    name: str = ""
    status: HealthStatus = HealthStatus.HEALTHY
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "latency_ms": self.latency_ms,
        }


@dataclass
class StatusReport:
    """Complete system status report.

    Attributes:
        generated_at: When the report was generated.
        overall_status: Aggregate health status.
        components: Individual component statuses.
        system_info: System-level information.
        inference_stats: Inference performance statistics.
        training_stats: Training job statistics.
        resource_summary: Resource utilization summary.
        alerts_summary: Active alerts summary.
        uptime_seconds: System uptime in seconds.
    """

    generated_at: datetime = field(default_factory=datetime.now)
    overall_status: HealthStatus = HealthStatus.HEALTHY
    components: List[ComponentStatus] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    inference_stats: Dict[str, Any] = field(default_factory=dict)
    training_stats: Dict[str, Any] = field(default_factory=dict)
    resource_summary: Dict[str, Any] = field(default_factory=dict)
    alerts_summary: Dict[str, Any] = field(default_factory=dict)
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "components": [c.to_dict() for c in self.components],
            "system_info": self.system_info,
            "inference_stats": self.inference_stats,
            "training_stats": self.training_stats,
            "resource_summary": self.resource_summary,
            "alerts_summary": self.alerts_summary,
            "uptime_seconds": self.uptime_seconds,
        }


class StatusReporter:
    """Generates comprehensive status reports for Nexus-LLM.

    Collects information from various subsystems including the database,
    model manager, resource tracker, and alert manager to produce
    unified status reports in multiple formats.

    Attributes:
        start_time: When the reporter (and typically the application) started.
    """

    def __init__(self, start_time: Optional[datetime] = None) -> None:
        """Initialize the status reporter.

        Args:
            start_time: Application start time for uptime calculation.
        """
        self.start_time = start_time or datetime.now()
        self._custom_collectors: Dict[str, Callable[[], ComponentStatus]] = {}

    def register_collector(self, name: str, collector: Any) -> None:
        """Register a custom component status collector.

        Args:
            name: Component name.
            collector: Callable returning a ComponentStatus.
        """
        self._custom_collectors[name] = collector

    def generate_report(self) -> StatusReport:
        """Generate a comprehensive status report.

        Collects status information from all registered collectors,
        system resources, and framework components.

        Returns:
            Complete StatusReport.
        """
        report = StatusReport(
            generated_at=datetime.now(),
            uptime_seconds=(datetime.now() - self.start_time).total_seconds(),
        )

        # Collect system info
        report.system_info = self._collect_system_info()

        # Collect component statuses
        report.components = self._collect_component_statuses()

        # Collect inference stats
        report.inference_stats = self._collect_inference_stats()

        # Collect training stats
        report.training_stats = self._collect_training_stats()

        # Collect resource summary
        report.resource_summary = self._collect_resource_summary()

        # Determine overall health
        report.overall_status = self._compute_overall_status(report.components)

        return report

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect basic system information."""
        info: Dict[str, Any] = {
            "python_version": "",
            "platform": "",
            "cpu_count": 0,
            "hostname": "",
        }

        try:
            import platform
            info["python_version"] = platform.python_version()
            info["platform"] = platform.platform()
            info["cpu_count"] = os.cpu_count() or 0
            info["hostname"] = platform.node()
        except Exception as e:
            logger.debug(f"Failed to collect system info: {e}")

        try:
            import psutil
            mem = psutil.virtual_memory()
            info["total_memory_gb"] = round(mem.total / (1024 ** 3), 2)
            info["available_memory_gb"] = round(mem.available / (1024 ** 3), 2)
            info["disk_total_gb"] = round(psutil.disk_usage("/").total / (1024 ** 3), 2)
        except ImportError:
            pass

        # GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            info["gpu_count"] = gpu_count
            if gpu_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info["gpu_name"] = pynvml.nvmlDeviceGetName(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info["gpu_memory_total_mb"] = round(mem_info.total / (1024 ** 2))
            pynvml.nvmlShutdown()
        except ImportError:
            info["gpu_available"] = False
        except Exception:
            info["gpu_available"] = False

        return info

    def _collect_component_statuses(self) -> List[ComponentStatus]:
        """Collect status from all components."""
        components: List[ComponentStatus] = []

        # Database status
        db_status = ComponentStatus(name="database")
        try:
            from nexus_llm.storage.database import DatabaseManager
            db = DatabaseManager(DatabaseConfig(db_path="nexus_llm.db"))
            db.initialize()
            health = db.health_check()
            db_status.status = HealthStatus.HEALTHY if health.get("healthy") else HealthStatus.UNHEALTHY
            db_status.latency_ms = health.get("latency_ms")
            db_status.details = health
            db.close()
        except Exception as e:
            db_status.status = HealthStatus.UNHEALTHY
            db_status.message = str(e)
        components.append(db_status)

        # Custom collectors
        for name, collector in self._custom_collectors.items():
            try:
                status = collector()
                components.append(status)
            except Exception as e:
                components.append(ComponentStatus(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Collector error: {e}",
                ))

        return components

    def _collect_inference_stats(self) -> Dict[str, Any]:
        """Collect inference performance statistics."""
        stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "requests_per_second": 0.0,
        }

        try:
            from nexus_llm.storage.database import DatabaseManager
            from nexus_llm.storage.cache_store import CacheStore

            db = DatabaseManager()
            try:
                if db.table_exists("response_cache"):
                    cache = CacheStore(db)
                    cache_stats = cache.get_stats()
                    stats["cache_hit_rate"] = cache_stats.hit_rate
                    stats["cache_entries"] = cache_stats.total_entries
            except Exception:
                pass
            finally:
                db.close()
        except Exception:
            pass

        return stats

    def _collect_training_stats(self) -> Dict[str, Any]:
        """Collect training job statistics."""
        stats: Dict[str, Any] = {
            "active_jobs": 0,
            "total_jobs": 0,
        }

        try:
            from nexus_llm.storage.database import DatabaseManager
            db = DatabaseManager()
            try:
                if db.table_exists("training_jobs"):
                    total = db.fetch_value("SELECT COUNT(*) FROM training_jobs") or 0
                    active = db.fetch_value(
                        "SELECT COUNT(*) FROM training_jobs WHERE status IN ('running', 'pending')"
                    ) or 0
                    stats["total_jobs"] = total
                    stats["active_jobs"] = active
            except Exception:
                pass
            finally:
                db.close()
        except Exception:
            pass

        return stats

    def _collect_resource_summary(self) -> Dict[str, Any]:
        """Collect current resource utilization summary."""
        summary: Dict[str, Any] = {}

        try:
            import psutil
            summary["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            summary["memory_percent"] = mem.percent
            summary["memory_used_gb"] = round(mem.used / (1024 ** 3), 2)
        except ImportError:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            summary["gpu_utilization"] = util.gpu
            summary["gpu_memory_percent"] = round(mem_info.used / mem_info.total * 100, 1) if mem_info.total else 0
            summary["gpu_memory_used_mb"] = round(mem_info.used / (1024 ** 2))
            pynvml.nvmlShutdown()
        except Exception:
            pass

        return summary

    def _compute_overall_status(self, components: List[ComponentStatus]) -> HealthStatus:
        """Compute overall health from component statuses."""
        if not components:
            return HealthStatus.HEALTHY

        statuses = [c.status for c in components]

        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def format_report(self, report: StatusReport, fmt: ReportFormat = ReportFormat.TEXT) -> str:
        """Format a status report in the specified format.

        Args:
            report: The report to format.
            fmt: Output format.

        Returns:
            Formatted report string.
        """
        if fmt == ReportFormat.JSON:
            return json.dumps(report.to_dict(), indent=2)

        elif fmt == ReportFormat.MARKDOWN:
            return self._format_markdown(report)

        elif fmt == ReportFormat.HTML:
            return self._format_html(report)

        else:
            return self._format_text(report)

    def _format_text(self, report: StatusReport) -> str:
        """Format report as plain text."""
        lines = [
            "=" * 60,
            "  Nexus-LLM Status Report",
            "=" * 60,
            f"  Generated: {report.generated_at.isoformat()}",
            f"  Overall:   {report.overall_status.value.upper()}",
            f"  Uptime:    {report.uptime_seconds:.0f}s",
            "",
            "Components:",
        ]

        for comp in report.components:
            status_icon = {
                HealthStatus.HEALTHY: "[OK]",
                HealthStatus.DEGRADED: "[!!]",
                HealthStatus.UNHEALTHY: "[XX]",
                HealthStatus.CRITICAL: "[!!]",
            }.get(comp.status, "[??]")
            lines.append(f"  {status_icon} {comp.name}: {comp.message or comp.status.value}")

        if report.resource_summary:
            lines.append("")
            lines.append("Resources:")
            for key, value in report.resource_summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.1f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_markdown(self, report: StatusReport) -> str:
        """Format report as Markdown."""
        lines = [
            "# Nexus-LLM Status Report",
            "",
            f"**Generated:** {report.generated_at.isoformat()}  ",
            f"**Overall Status:** `{report.overall_status.value}`  ",
            f"**Uptime:** {report.uptime_seconds:.0f}s",
            "",
            "## Components",
            "",
            "| Component | Status | Message | Latency |",
            "|-----------|--------|---------|---------|",
        ]

        for comp in report.components:
            latency = f"{comp.latency_ms:.1f}ms" if comp.latency_ms else "-"
            lines.append(f"| {comp.name} | {comp.status.value} | {comp.message or '-'} | {latency} |")

        if report.resource_summary:
            lines.extend(["", "## Resources", ""])
            for key, value in report.resource_summary.items():
                lines.append(f"- **{key}:** {value}")

        return "\n".join(lines)

    def _format_html(self, report: StatusReport) -> str:
        """Format report as HTML."""
        status_colors = {
            HealthStatus.HEALTHY: "#10b981",
            HealthStatus.DEGRADED: "#f59e0b",
            HealthStatus.UNHEALTHY: "#ef4444",
            HealthStatus.CRITICAL: "#dc2626",
        }

        html_parts = [
            "<html><head><style>",
            "body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            ".status { padding: 2px 8px; border-radius: 4px; color: white; font-weight: bold; }",
            "</style></head><body>",
            f"<h1>Nexus-LLM Status Report</h1>",
            f"<p>Generated: {report.generated_at.isoformat()}</p>",
            f"<p>Overall Status: <span class='status' style='background:{status_colors.get(report.overall_status, '#6b7280')}'>{report.overall_status.value}</span></p>",
            f"<p>Uptime: {report.uptime_seconds:.0f}s</p>",
            "<h2>Components</h2><table><tr><th>Name</th><th>Status</th><th>Message</th></tr>",
        ]

        for comp in report.components:
            color = status_colors.get(comp.status, "#6b7280")
            html_parts.append(
                f"<tr><td>{comp.name}</td>"
                f"<td><span class='status' style='background:{color}'>{comp.status.value}</span></td>"
                f"<td>{comp.message or '-'}</td></tr>"
            )

        html_parts.append("</table></body></html>")
        return "\n".join(html_parts)
