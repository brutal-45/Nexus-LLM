"""Nexus-LLM Terminal Dashboard with Rich.

Provides a rich terminal-based dashboard for real-time monitoring of
Nexus-LLM system metrics including GPU/CPU utilization, memory usage,
inference throughput, active models, and request statistics.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard.

    Attributes:
        refresh_interval: Time between dashboard updates in seconds.
        show_gpu: Whether to display GPU metrics.
        show_cpu: Whether to display CPU metrics.
        show_memory: Whether to display memory metrics.
        show_requests: Whether to display request statistics.
        show_models: Whether to display active model information.
        max_history: Maximum number of historical data points to keep.
    """

    refresh_interval: float = 1.0
    show_gpu: bool = True
    show_cpu: bool = True
    show_memory: bool = True
    show_requests: bool = True
    show_models: bool = True
    max_history: int = 120


@dataclass
class SystemMetrics:
    """Current system metrics snapshot.

    Attributes:
        timestamp: When the metrics were collected.
        cpu_percent: CPU utilization percentage.
        cpu_count: Number of CPU cores.
        memory_total_gb: Total system memory in GB.
        memory_used_gb: Used system memory in GB.
        memory_percent: Memory utilization percentage.
        gpu_utilization: GPU utilization percentage (0 if no GPU).
        gpu_memory_total_mb: Total GPU memory in MB.
        gpu_memory_used_mb: Used GPU memory in MB.
        gpu_memory_percent: GPU memory utilization percentage.
        gpu_temperature_c: GPU temperature in Celsius.
        active_models: Number of currently loaded models.
        requests_per_second: Current inference request rate.
        avg_latency_ms: Average inference latency.
        total_requests: Total requests processed.
        total_tokens_generated: Total tokens generated.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    cpu_count: int = 0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    active_models: int = 0
    requests_per_second: float = 0.0
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    total_tokens_generated: int = 0


class Dashboard:
    """Terminal-based monitoring dashboard for Nexus-LLM.

    Renders real-time system metrics using the Rich library with
    auto-refreshing panels, sparklines, and progress bars.

    Attributes:
        config: Dashboard configuration.
    """

    def __init__(self, config: Optional[DashboardConfig] = None) -> None:
        """Initialize the dashboard.

        Args:
            config: Dashboard configuration. Uses defaults if None.
        """
        self.config = config or DashboardConfig()
        self._metrics_history: List[SystemMetrics] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._custom_panels: Dict[str, Callable[[], Any]] = {}
        self._latest_metrics = SystemMetrics()

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics.

        Gathers CPU, memory, GPU, and inference metrics from
        available system resources.

        Returns:
            SystemMetrics snapshot.
        """
        metrics = SystemMetrics()

        # CPU metrics
        try:
            import psutil
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_count = psutil.cpu_count() or 0
            mem = psutil.virtual_memory()
            metrics.memory_total_gb = mem.total / (1024 ** 3)
            metrics.memory_used_gb = mem.used / (1024 ** 3)
            metrics.memory_percent = mem.percent
        except ImportError:
            metrics.cpu_count = os.cpu_count() or 0
        except Exception as e:
            logger.debug(f"Failed to collect CPU/memory metrics: {e}")

        # GPU metrics
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            metrics.gpu_utilization = util.gpu
            metrics.gpu_memory_total_mb = mem_info.total / (1024 ** 2)
            metrics.gpu_memory_used_mb = mem_info.used / (1024 ** 2)
            metrics.gpu_memory_percent = (mem_info.used / mem_info.total * 100) if mem_info.total > 0 else 0
            metrics.gpu_temperature_c = temp
            pynvml.nvmlShutdown()
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to collect GPU metrics: {e}")

        # Record metrics
        with self._lock:
            self._latest_metrics = metrics
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self.config.max_history:
                self._metrics_history = self._metrics_history[-self.config.max_history:]

        return metrics

    def update_inference_stats(
        self,
        requests_per_second: float = 0.0,
        avg_latency_ms: float = 0.0,
        total_requests: int = 0,
        total_tokens: int = 0,
        active_models: int = 0,
    ) -> None:
        """Update inference-specific metrics.

        Args:
            requests_per_second: Current request rate.
            avg_latency_ms: Average latency.
            total_requests: Total processed requests.
            total_tokens: Total generated tokens.
            active_models: Number of active models.
        """
        with self._lock:
            self._latest_metrics.requests_per_second = requests_per_second
            self._latest_metrics.avg_latency_ms = avg_latency_ms
            self._latest_metrics.total_requests = total_requests
            self._latest_metrics.total_tokens_generated = total_tokens
            self._latest_metrics.active_models = active_models

    def register_panel(self, name: str, render_func: Callable[[], Any]) -> None:
        """Register a custom dashboard panel.

        Args:
            name: Panel name for display.
            render_func: Callable that returns a Rich renderable.
        """
        self._custom_panels[name] = render_func

    def render(self) -> None:
        """Render the dashboard to the terminal using Rich.

        Creates a live-updating terminal display with panels for
        system metrics, inference stats, and custom panels.
        """
        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.live import Live
            from rich.panel import Panel
            from rich.progress import BarColumn, Progress, TextColumn
            from rich.table import Table
            from rich.text import Text
        except ImportError:
            logger.warning("Rich library not installed. Dashboard rendering unavailable.")
            self._print_simple_dashboard()
            return

        console = Console()
        metrics = self.collect_metrics()

        # Header
        header = Panel(
            Text("Nexus-LLM Dashboard", style="bold cyan", justify="center"),
            style="bold white on blue",
        )

        # System metrics table
        sys_table = Table(title="System Resources", show_header=True, header_style="bold magenta")
        sys_table.add_column("Resource", style="cyan")
        sys_table.add_column("Usage", style="green")
        sys_table.add_column("Details", style="yellow")

        if self.config.show_cpu:
            sys_table.add_row("CPU", f"{metrics.cpu_percent:.1f}%", f"{metrics.cpu_count} cores")

        if self.config.show_memory:
            sys_table.add_row(
                "Memory",
                f"{metrics.memory_percent:.1f}%",
                f"{metrics.memory_used_gb:.1f}/{metrics.memory_total_gb:.1f} GB",
            )

        if self.config.show_gpu and metrics.gpu_utilization > 0:
            sys_table.add_row("GPU", f"{metrics.gpu_utilization:.1f}%", f"{metrics.gpu_temperature_c:.0f}°C")
            sys_table.add_row(
                "GPU Memory",
                f"{metrics.gpu_memory_percent:.1f}%",
                f"{metrics.gpu_memory_used_mb:.0f}/{metrics.gpu_memory_total_mb:.0f} MB",
            )

        # Inference stats table
        inf_table = Table(title="Inference", show_header=True, header_style="bold magenta")
        inf_table.add_column("Metric", style="cyan")
        inf_table.add_column("Value", style="green")

        inf_table.add_row("Active Models", str(metrics.active_models))
        inf_table.add_row("Requests/sec", f"{metrics.requests_per_second:.2f}")
        inf_table.add_row("Avg Latency", f"{metrics.avg_latency_ms:.1f} ms")
        inf_table.add_row("Total Requests", f"{metrics.total_requests:,}")
        inf_table.add_row("Total Tokens", f"{metrics.total_tokens_generated:,}")

        console.print(header)
        console.print(sys_table)
        if self.config.show_requests:
            console.print(inf_table)

        # Custom panels
        for name, render_func in self._custom_panels.items():
            try:
                content = render_func()
                console.print(Panel(content, title=name))
            except Exception as e:
                console.print(Panel(f"Error rendering panel: {e}", title=name))

    def start(self) -> None:
        """Start the dashboard auto-refresh loop in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        logger.info("Dashboard started")

    def stop(self) -> None:
        """Stop the dashboard auto-refresh loop."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("Dashboard stopped")

    def _refresh_loop(self) -> None:
        """Background loop that periodically collects metrics and renders."""
        while self._running:
            try:
                self.collect_metrics()
                self.render()
            except Exception as e:
                logger.error(f"Dashboard refresh error: {e}")
            time.sleep(self.config.refresh_interval)

    def _print_simple_dashboard(self) -> None:
        """Fallback dashboard rendering without Rich."""
        m = self._latest_metrics
        print("=" * 60)
        print("  Nexus-LLM Dashboard")
        print("=" * 60)
        print(f"  CPU: {m.cpu_percent:.1f}% ({m.cpu_count} cores)")
        print(f"  Memory: {m.memory_percent:.1f}% ({m.memory_used_gb:.1f}/{m.memory_total_gb:.1f} GB)")
        if m.gpu_utilization > 0:
            print(f"  GPU: {m.gpu_utilization:.1f}% ({m.gpu_temperature_c:.0f}°C)")
            print(f"  GPU Memory: {m.gpu_memory_percent:.1f}% ({m.gpu_memory_used_mb:.0f}/{m.gpu_memory_total_mb:.0f} MB)")
        print(f"  Active Models: {m.active_models}")
        print(f"  Requests/sec: {m.requests_per_second:.2f}")
        print(f"  Avg Latency: {m.avg_latency_ms:.1f} ms")
        print("=" * 60)

    def get_history(self, limit: Optional[int] = None) -> List[SystemMetrics]:
        """Get historical metrics data.

        Args:
            limit: Maximum number of data points to return.

        Returns:
            List of SystemMetrics in chronological order.
        """
        with self._lock:
            data = list(self._metrics_history)
        if limit:
            data = data[-limit:]
        return data

    def export_metrics(self) -> Dict[str, Any]:
        """Export all collected metrics as a dictionary.

        Returns:
            Dictionary with current metrics and history summary.
        """
        with self._lock:
            m = self._latest_metrics
            return {
                "current": {
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "gpu_utilization": m.gpu_utilization,
                    "gpu_memory_percent": m.gpu_memory_percent,
                    "requests_per_second": m.requests_per_second,
                    "avg_latency_ms": m.avg_latency_ms,
                    "active_models": m.active_models,
                },
                "history_length": len(self._metrics_history),
            }
