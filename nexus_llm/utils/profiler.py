"""Profiling: CPU profiler, memory profiler, function timing, performance reports."""

import os
import time
import cProfile
import pstats
import io
import logging
import functools
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class CPUProfiler:
    """CPU profiling using cProfile with convenient context manager and decorator support."""

    def __init__(
        self,
        name: str = "CPUProfiler",
        sort_by: str = "cumulative",
        max_results: int = 30,
    ):
        """Initialize the CPU profiler.

        Args:
            name: Profiler name for identification.
            sort_by: Sort key for profile results (cumulative, time, calls, etc.).
            max_results: Maximum number of results to show.
        """
        self.name = name
        self.sort_by = sort_by
        self.max_results = max_results
        self._profiler: Optional[cProfile.Profile] = None
        self._stats: Optional[pstats.Stats] = None
        self._results: Optional[str] = None

    def start(self):
        """Start profiling."""
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        logger.debug(f"CPU profiler '{self.name}' started.")

    def stop(self):
        """Stop profiling."""
        if self._profiler is not None:
            self._profiler.disable()
            self._stats = pstats.Stats(self._profiler, stream=io.StringIO())
            self._stats.sort_stats(self.sort_by)
            logger.debug(f"CPU profiler '{self.name}' stopped.")

    def get_results(self) -> str:
        """Get profiling results as a formatted string.

        Returns:
            Formatted profiling results.
        """
        if self._stats is None:
            return "No profiling data available."

        stream = io.StringIO()
        self._stats.stream = stream
        self._stats.print_stats(self.max_results)
        self._results = stream.getvalue()
        return self._results

    def get_top_functions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the top N functions by cumulative time.

        Args:
            n: Number of top functions to return.

        Returns:
            List of dictionaries with function profiling info.
        """
        if self._stats is None:
            return []

        self._stats.sort_stats("cumulative")
        stats_list = list(self._stats.stats.items())
        stats_list.sort(key=lambda x: x[1][3], reverse=True)

        results = []
        for func_info, (cc, nc, tt, ct, callers) in stats_list[:n]:
            filename, line, funcname = func_info
            results.append({
                "function": funcname,
                "file": filename,
                "line": line,
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
                "per_call": ct / nc if nc > 0 else 0,
            })

        return results

    def save_results(self, filepath: str):
        """Save profiling results to a file.

        Args:
            filepath: Output file path.
        """
        results = self.get_results()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(results)
        logger.info(f"Profiling results saved to {filepath}")

    def __enter__(self) -> "CPUProfiler":
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class MemoryProfiler:
    """Memory profiling utility for tracking memory usage."""

    def __init__(self, name: str = "MemoryProfiler"):
        self.name = name
        self._snapshots: List[Dict[str, Any]] = []
        self._baseline: Optional[float] = None

    def snapshot(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Take a memory usage snapshot.

        Args:
            label: Optional label for this snapshot.

        Returns:
            Dictionary with memory usage info.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()

            snapshot_data = {
                "label": label or f"snapshot_{len(self._snapshots)}",
                "timestamp": time.time(),
                "rss_bytes": mem_info.rss,
                "rss_mb": mem_info.rss / (1024 * 1024),
                "vms_bytes": mem_info.vms,
                "vms_mb": mem_info.vms / (1024 * 1024),
            }

            if self._baseline is None:
                self._baseline = mem_info.rss
            snapshot_data["delta_mb"] = (mem_info.rss - self._baseline) / (1024 * 1024)

        except ImportError:
            # Fallback using torch if available
            try:
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    snapshot_data = {
                        "label": label or f"snapshot_{len(self._snapshots)}",
                        "timestamp": time.time(),
                        "gpu_allocated_bytes": allocated,
                        "gpu_allocated_mb": allocated / (1024 * 1024),
                    }
                else:
                    snapshot_data = {
                        "label": label or f"snapshot_{len(self._snapshots)}",
                        "timestamp": time.time(),
                        "note": "psutil not available; limited memory info",
                    }
            except ImportError:
                snapshot_data = {
                    "label": label or f"snapshot_{len(self._snapshots)}",
                    "timestamp": time.time(),
                    "note": "Neither psutil nor torch available",
                }

        self._snapshots.append(snapshot_data)
        return snapshot_data

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of memory usage across all snapshots.

        Returns:
            Summary dictionary.
        """
        if not self._snapshots:
            return {"error": "No snapshots taken"}

        rss_values = [s.get("rss_mb", 0) for s in self._snapshots if "rss_mb" in s]

        summary = {
            "num_snapshots": len(self._snapshots),
            "labels": [s["label"] for s in self._snapshots],
        }

        if rss_values:
            summary.update({
                "min_rss_mb": min(rss_values),
                "max_rss_mb": max(rss_values),
                "current_rss_mb": rss_values[-1],
                "peak_delta_mb": max(s.get("delta_mb", 0) for s in self._snapshots if "delta_mb" in s) if any("delta_mb" in s for s in self._snapshots) else 0,
            })

        return summary

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """Get all recorded snapshots."""
        return list(self._snapshots)

    def reset(self):
        """Reset all snapshots and baseline."""
        self._snapshots = []
        self._baseline = None


class FunctionProfiler:
    """Profile individual function calls with timing and call counts."""

    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = {}

    def profile(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator to profile a function.

        Args:
            func: Function to profile.
            name: Custom name for the profile entry.

        Returns:
            Decorated function.
        """
        def decorator(fn: Callable) -> Callable:
            profile_name = name or fn.__qualname__

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = None
                    success = False
                    raise
                finally:
                    elapsed = time.perf_counter() - start
                    self._record(profile_name, elapsed, success)

                return result

            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    def _record(self, name: str, elapsed: float, success: bool):
        """Record a function call profile entry."""
        if name not in self._profiles:
            self._profiles[name] = {
                "name": name,
                "calls": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "successes": 0,
                "failures": 0,
            }

        profile = self._profiles[name]
        profile["calls"] += 1
        profile["total_time"] += elapsed
        profile["min_time"] = min(profile["min_time"], elapsed)
        profile["max_time"] = max(profile["max_time"], elapsed)

        if success:
            profile["successes"] += 1
        else:
            profile["failures"] += 1

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profile data for a specific function.

        Args:
            name: Function name.

        Returns:
            Profile dictionary, or None if not found.
        """
        profile = self._profiles.get(name)
        if profile is None:
            return None

        result = dict(profile)
        if result["calls"] > 0:
            result["avg_time"] = result["total_time"] / result["calls"]
        else:
            result["avg_time"] = 0.0

        return result

    def get_all_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get profile data for all profiled functions.

        Returns:
            Dictionary mapping function names to profile data.
        """
        return {name: self.get_profile(name) for name in self._profiles}

    def get_summary(self) -> str:
        """Get a formatted summary of all profiled functions.

        Returns:
            Formatted summary string.
        """
        if not self._profiles:
            return "No profiled functions."

        lines = []
        lines.append(f"{'Function':<40} {'Calls':>8} {'Total(s)':>10} {'Avg(s)':>10} {'Min(s)':>10} {'Max(s)':>10}")
        lines.append("-" * 88)

        sorted_profiles = sorted(
            self._profiles.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        for name, profile in sorted_profiles:
            calls = profile["calls"]
            total = profile["total_time"]
            avg = total / calls if calls > 0 else 0
            min_t = profile["min_time"] if profile["min_time"] != float("inf") else 0
            max_t = profile["max_time"]

            display_name = name[:38] + ".." if len(name) > 40 else name
            lines.append(
                f"{display_name:<40} {calls:>8} {total:>10.4f} {avg:>10.6f} {min_t:>10.6f} {max_t:>10.6f}"
            )

        return "\n".join(lines)

    def reset(self):
        """Reset all profile data."""
        self._profiles = {}


@contextmanager
def profile_block(name: str = "block", sort_by: str = "cumulative"):
    """Context manager for quick profiling of a code block.

    Args:
        name: Block name for identification.
        sort_by: Sort key for profile output.

    Yields:
        CPUProfiler instance.
    """
    profiler = CPUProfiler(name=name, sort_by=sort_by)
    with profiler:
        yield profiler
    logger.info(f"Profile results for '{name}':\n{profiler.get_results()}")
