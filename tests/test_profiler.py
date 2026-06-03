"""Test profiling utilities for Nexus-LLM."""
import time
import pytest
from contextlib import contextmanager
from collections import defaultdict


class Profiler:
    def __init__(self):
        self._records = {}
        self._call_counts = defaultdict(int)
        self._total_times = defaultdict(float)

    def record(self, name: str, duration: float):
        if name not in self._records:
            self._records[name] = []
        self._records[name].append(duration)
        self._call_counts[name] += 1
        self._total_times[name] += duration

    @contextmanager
    def profile(self, name: str):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.record(name, elapsed)

    def get_stats(self, name: str) -> dict:
        if name not in self._records:
            return {}
        records = self._records[name]
        return {
            "name": name,
            "calls": len(records),
            "total": sum(records),
            "mean": sum(records) / len(records),
            "min": min(records),
            "max": max(records),
        }

    def get_all_stats(self) -> list:
        return [self.get_stats(name) for name in self._records]

    def reset(self):
        self._records.clear()
        self._call_counts.clear()
        self._total_times.clear()

    def summary(self) -> str:
        lines = []
        for name in sorted(self._records.keys()):
            stats = self.get_stats(name)
            lines.append(f"{name}: calls={stats['calls']}, total={stats['total']:.4f}s, avg={stats['mean']:.4f}s")
        return "\n".join(lines)


class MemoryProfiler:
    def __init__(self):
        self._snapshots = []

    def snapshot(self, label: str = ""):
        import sys
        self._snapshots.append({
            "label": label,
            "timestamp": time.perf_counter(),
            "object_count": 0,
        })
        return self._snapshots[-1]

    def get_snapshots(self):
        return list(self._snapshots)

    def clear(self):
        self._snapshots.clear()


def profile_function(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return {"result": result, "elapsed": elapsed, "function": func.__name__}


class TestProfiler:
    def test_record(self):
        p = Profiler()
        p.record("test_op", 0.5)
        stats = p.get_stats("test_op")
        assert stats["calls"] == 1
        assert stats["total"] == 0.5

    def test_multiple_records(self):
        p = Profiler()
        p.record("op", 0.1)
        p.record("op", 0.2)
        p.record("op", 0.3)
        stats = p.get_stats("op")
        assert stats["calls"] == 3
        assert abs(stats["total"] - 0.6) < 0.001
        assert abs(stats["mean"] - 0.2) < 0.001

    def test_min_max(self):
        p = Profiler()
        p.record("op", 0.1)
        p.record("op", 0.5)
        p.record("op", 0.3)
        stats = p.get_stats("op")
        assert stats["min"] == 0.1
        assert stats["max"] == 0.5

    def test_profile_context_manager(self):
        p = Profiler()
        with p.profile("sleep"):
            time.sleep(0.01)
        stats = p.get_stats("sleep")
        assert stats["calls"] == 1
        assert stats["total"] >= 0.01

    def test_multiple_profiles(self):
        p = Profiler()
        with p.profile("fast"):
            pass
        with p.profile("slow"):
            time.sleep(0.01)
        all_stats = p.get_all_stats()
        assert len(all_stats) == 2

    def test_nonexistent_stats(self):
        p = Profiler()
        assert p.get_stats("nonexistent") == {}

    def test_reset(self):
        p = Profiler()
        p.record("op", 0.5)
        p.reset()
        assert p.get_stats("op") == {}

    def test_summary(self):
        p = Profiler()
        p.record("test_op", 0.5)
        summary = p.summary()
        assert "test_op" in summary
        assert "calls=1" in summary


class TestMemoryProfiler:
    def test_snapshot(self):
        mp = MemoryProfiler()
        snap = mp.snapshot("start")
        assert snap["label"] == "start"

    def test_multiple_snapshots(self):
        mp = MemoryProfiler()
        mp.snapshot("a")
        mp.snapshot("b")
        assert len(mp.get_snapshots()) == 2

    def test_clear(self):
        mp = MemoryProfiler()
        mp.snapshot("test")
        mp.clear()
        assert len(mp.get_snapshots()) == 0


class TestProfileFunction:
    def test_profiles_result(self):
        result = profile_function(lambda: 42)
        assert result["result"] == 42
        assert result["elapsed"] >= 0
        assert result["function"] == "<lambda>"

    def test_profiles_with_args(self):
        result = profile_function(lambda x, y: x + y, 3, 4)
        assert result["result"] == 7

    def test_profiles_slow_function(self):
        def slow():
            time.sleep(0.01)
            return "done"
        result = profile_function(slow)
        assert result["result"] == "done"
        assert result["elapsed"] >= 0.01
