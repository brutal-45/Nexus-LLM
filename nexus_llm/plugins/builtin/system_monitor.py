"""System monitor plugin for CPU, memory, and disk usage.

A builtin plugin that provides system resource monitoring
capabilities including CPU, memory, disk, and process information.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from typing import Any, Dict, List, Optional

from nexus_llm.plugins.hook import HookManager, HookPriority

logger = logging.getLogger(__name__)


class SystemMonitorPlugin:
    """Plugin providing system resource monitoring.

    Monitors CPU usage, memory consumption, disk space, and
    system information. Uses only standard library modules
    for broad compatibility.
    """

    name = "system_monitor"
    version = "1.0.0"
    description = "Monitor system resources: CPU, memory, disk, and process info"
    dependencies: List[str] = []
    tags = ["system", "monitoring", "performance", "builtin"]

    def __init__(self, hook_manager: Optional[HookManager] = None, **kwargs):
        self.hook_manager = hook_manager
        self._active = False
        self._monitoring_history: List[Dict[str, Any]] = []
        self._max_history = 100
        self._last_cpu_times: Optional[tuple] = None
        self._last_cpu_check: float = 0.0

    def activate(self) -> None:
        """Activate the system monitor plugin."""
        if self.hook_manager:
            self.hook_manager.register(
                "tool_request",
                self._handle_tool_request,
                name="system_monitor_tool_handler",
                priority=HookPriority.NORMAL,
                owner=self.name,
            )
        self._active = True
        logger.info("System monitor plugin activated.")

    def deactivate(self) -> None:
        """Deactivate the system monitor plugin."""
        if self.hook_manager:
            self.hook_manager.unregister_by_owner(self.name)
        self._active = False
        logger.info("System monitor plugin deactivated.")

    def get_system_info(self) -> Dict[str, Any]:
        """Get general system information.

        Returns:
            Dictionary with system details.
        """
        return {
            "success": True,
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }

    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU usage information.

        Returns:
            Dictionary with CPU details.
        """
        cpu_count = os.cpu_count() or 1

        # Calculate CPU usage from /proc/stat (Linux) or using psutil
        cpu_percent = self._get_cpu_percent()

        load_avg = None
        if hasattr(os, "getloadavg"):
            try:
                load_avg = list(os.getloadavg())
            except OSError:
                pass

        return {
            "success": True,
            "cpu_count": cpu_count,
            "cpu_percent": round(cpu_percent, 1),
            "load_average": load_avg,
        }

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information.

        Returns:
            Dictionary with memory details.
        """
        mem_info = self._get_memory_info_os()

        return {
            "success": True,
            "total_bytes": mem_info.get("total", 0),
            "available_bytes": mem_info.get("available", 0),
            "used_bytes": mem_info.get("used", 0),
            "percent": mem_info.get("percent", 0.0),
            "total_gb": round(mem_info.get("total", 0) / (1024 ** 3), 2),
            "available_gb": round(mem_info.get("available", 0) / (1024 ** 3), 2),
            "used_gb": round(mem_info.get("used", 0) / (1024 ** 3), 2),
        }

    def get_disk_info(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage information.

        Args:
            path: Filesystem path to check.

        Returns:
            Dictionary with disk details.
        """
        try:
            if platform.system() == "Windows":
                path = path if path else "C:\\"
            else:
                path = path if path else "/"

            usage = os.statvfs(path) if hasattr(os, "statvfs") else None

            if usage:
                total = usage.f_blocks * usage.f_frsize
                free = usage.f_bfree * usage.f_frsize
                available = usage.f_bavail * usage.f_frsize
                used = total - free

                return {
                    "success": True,
                    "path": path,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "available_bytes": available,
                    "percent": round(used / total * 100, 1) if total > 0 else 0,
                    "total_gb": round(total / (1024 ** 3), 2),
                    "used_gb": round(used / (1024 ** 3), 2),
                    "free_gb": round(free / (1024 ** 3), 2),
                }
            else:
                # Windows fallback
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                total_bytes = ctypes.c_ulonglong(0)
                available_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    path, ctypes.pointer(available_bytes),
                    ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
                )
                total = total_bytes.value
                free = free_bytes.value
                used = total - free

                return {
                    "success": True,
                    "path": path,
                    "total_bytes": total,
                    "used_bytes": used,
                    "free_bytes": free,
                    "percent": round(used / total * 100, 1) if total > 0 else 0,
                    "total_gb": round(total / (1024 ** 3), 2),
                    "used_gb": round(used / (1024 ** 3), 2),
                    "free_gb": round(free / (1024 ** 3), 2),
                }
        except Exception as e:
            return {"success": False, "error": f"Error getting disk info: {e}"}

    def get_full_report(self) -> Dict[str, Any]:
        """Get a comprehensive system report.

        Returns:
            Dictionary with all system metrics.
        """
        report = {
            "success": True,
            "timestamp": time.time(),
            "system": self.get_system_info(),
            "cpu": self.get_cpu_info(),
            "memory": self.get_memory_info(),
            "disk": self.get_disk_info(),
        }

        # Record in history
        self._monitoring_history.append({
            "timestamp": report["timestamp"],
            "cpu_percent": report["cpu"].get("cpu_percent", 0),
            "memory_percent": report["memory"].get("percent", 0),
            "disk_percent": report["disk"].get("percent", 0),
        })

        if len(self._monitoring_history) > self._max_history:
            self._monitoring_history = self._monitoring_history[-self._max_history:]

        return report

    def format_report(self) -> str:
        """Format a full system report as a readable string."""
        report = self.get_full_report()
        sys_info = report["system"]
        cpu = report["cpu"]
        mem = report["memory"]
        disk = report["disk"]

        lines = [
            f"System Monitor Report",
            f"{'=' * 40}",
            f"System: {sys_info.get('system', 'Unknown')} {sys_info.get('release', '')}",
            f"Machine: {sys_info.get('machine', 'Unknown')}",
            f"Python: {sys_info.get('python_version', 'Unknown')}",
            f"CPUs: {cpu.get('cpu_count', 'Unknown')}",
            f"CPU Usage: {cpu.get('cpu_percent', 0)}%",
        ]

        if cpu.get("load_average"):
            loads = cpu["load_average"]
            lines.append(f"Load Average: {loads[0]:.2f}, {loads[1]:.2f}, {loads[2]:.2f}")

        lines.extend([
            f"",
            f"Memory: {mem.get('used_gb', 0):.1f} / {mem.get('total_gb', 0):.1f} GB ({mem.get('percent', 0):.1f}%)",
            f"Available: {mem.get('available_gb', 0):.1f} GB",
            f"",
            f"Disk: {disk.get('used_gb', 0):.1f} / {disk.get('total_gb', 0):.1f} GB ({disk.get('percent', 0):.1f}%)",
            f"Free: {disk.get('free_gb', 0):.1f} GB",
        ])

        return "\n".join(lines)

    def _get_cpu_percent(self) -> float:
        """Get CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            pass

        # Fallback: read /proc/stat on Linux
        if os.path.exists("/proc/stat"):
            try:
                with open("/proc/stat", "r") as f:
                    line = f.readline()
                fields = [int(x) for x in line.split()[1:]]
                idle = fields[3]
                total = sum(fields)

                now = time.time()
                if self._last_cpu_times is not None:
                    prev_idle, prev_total, _ = self._last_cpu_times
                    delta_idle = idle - prev_idle
                    delta_total = total - prev_total
                    if delta_total > 0:
                        percent = (1.0 - delta_idle / delta_total) * 100
                    else:
                        percent = 0.0
                    self._last_cpu_times = (idle, total, now)
                    return percent

                self._last_cpu_times = (idle, total, now)
                return 0.0
            except Exception:
                pass

        return 0.0

    def _get_memory_info_os(self) -> Dict[str, Any]:
        """Get memory info using OS-specific methods."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent,
            }
        except ImportError:
            pass

        # Fallback: read /proc/meminfo on Linux
        if os.path.exists("/proc/meminfo"):
            try:
                info = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(":")
                            value = int(parts[1]) * 1024  # Convert kB to bytes
                            info[key] = value

                total = info.get("MemTotal", 0)
                available = info.get("MemAvailable", info.get("MemFree", 0))
                used = total - available
                percent = (used / total * 100) if total > 0 else 0

                return {
                    "total": total,
                    "available": available,
                    "used": used,
                    "percent": percent,
                }
            except Exception:
                pass

        return {"total": 0, "available": 0, "used": 0, "percent": 0.0}

    def get_monitoring_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent monitoring history."""
        return self._monitoring_history[-limit:]

    def _handle_tool_request(self, result, *args, **kwargs):
        """Handle tool requests for system monitoring."""
        tool_name = kwargs.get("tool_name", "")
        if tool_name == "system_monitor":
            return self.format_report()
        elif tool_name == "system_info":
            return str(self.get_system_info())
        return result
