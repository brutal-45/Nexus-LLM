"""Test timing utilities for Nexus-LLM."""
import time
import pytest
from contextlib import contextmanager


# --- Timing utility implementations to test ---

class Timer:
    def __init__(self, name: str = "timer"):
        self.name = name
        self._start = None
        self._end = None
        self._elapsed = None

    def start(self):
        self._start = time.perf_counter()
        self._end = None
        self._elapsed = None
        return self

    def stop(self):
        if self._start is None:
            raise RuntimeError("Timer not started")
        self._end = time.perf_counter()
        self._elapsed = self._end - self._start
        return self._elapsed

    @property
    def elapsed(self):
        if self._elapsed is None:
            raise RuntimeError("Timer not stopped")
        return self._elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


@contextmanager
def measure_time(label: str = ""):
    timer = Timer(label)
    timer.start()
    yield timer
    timer.stop()


class Stopwatch:
    def __init__(self):
        self._laps = []
        self._start = None

    def start(self):
        self._start = time.perf_counter()
        self._laps = []

    def lap(self):
        if self._start is None:
            raise RuntimeError("Stopwatch not started")
        current = time.perf_counter()
        elapsed = current - self._start
        self._laps.append(elapsed)
        return elapsed

    @property
    def laps(self):
        return list(self._laps)

    @property
    def lap_count(self):
        return len(self._laps)


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def estimate_remaining(elapsed: float, progress: float) -> float:
    if progress <= 0:
        return float("inf")
    if progress >= 1:
        return 0.0
    rate = elapsed / progress
    remaining_progress = 1.0 - progress
    return rate * remaining_progress


class TestTimer:
    def test_start_stop(self):
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed >= 0.01

    def test_elapsed_property(self):
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        assert timer.elapsed >= 0.01

    def test_elapsed_before_stop_raises(self):
        timer = Timer()
        timer.start()
        with pytest.raises(RuntimeError, match="not stopped"):
            _ = timer.elapsed

    def test_stop_before_start_raises(self):
        timer = Timer()
        with pytest.raises(RuntimeError, match="not started"):
            timer.stop()

    def test_context_manager(self):
        with Timer() as timer:
            time.sleep(0.01)
        assert timer.elapsed >= 0.01

    def test_timer_name(self):
        timer = Timer("test_timer")
        assert timer.name == "test_timer"

    def test_reuse_timer(self):
        timer = Timer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        first = timer.elapsed
        timer.start()
        time.sleep(0.01)
        timer.stop()
        second = timer.elapsed
        assert second >= 0.01


class TestMeasureTime:
    def test_measures_elapsed(self):
        with measure_time("test") as timer:
            time.sleep(0.01)
        assert timer.elapsed >= 0.01

    def test_label(self):
        with measure_time("my_label") as timer:
            pass
        assert timer.name == "my_label"


class TestStopwatch:
    def test_lap_timing(self):
        sw = Stopwatch()
        sw.start()
        time.sleep(0.01)
        lap1 = sw.lap()
        assert lap1 >= 0.01
        time.sleep(0.01)
        lap2 = sw.lap()
        assert lap2 >= lap1

    def test_lap_count(self):
        sw = Stopwatch()
        sw.start()
        sw.lap()
        sw.lap()
        sw.lap()
        assert sw.lap_count == 3

    def test_laps_returns_list(self):
        sw = Stopwatch()
        sw.start()
        sw.lap()
        sw.lap()
        laps = sw.laps
        assert isinstance(laps, list)
        assert len(laps) == 2

    def test_lap_before_start_raises(self):
        sw = Stopwatch()
        with pytest.raises(RuntimeError, match="not started"):
            sw.lap()


class TestFormatDuration:
    def test_microseconds(self):
        assert "μs" in format_duration(0.0001)

    def test_milliseconds(self):
        assert "ms" in format_duration(0.5)

    def test_seconds(self):
        result = format_duration(5.0)
        assert "s" in result
        assert "ms" not in result

    def test_minutes(self):
        result = format_duration(125.0)
        assert "m" in result

    def test_hours(self):
        result = format_duration(3700.0)
        assert "h" in result

    def test_zero(self):
        result = format_duration(0.0)
        assert "μs" in result or "ms" in result


class TestEstimateRemaining:
    def test_half_progress(self):
        remaining = estimate_remaining(10.0, 0.5)
        assert abs(remaining - 10.0) < 0.01

    def test_quarter_progress(self):
        remaining = estimate_remaining(5.0, 0.25)
        assert abs(remaining - 15.0) < 0.01

    def test_complete(self):
        remaining = estimate_remaining(10.0, 1.0)
        assert remaining == 0.0

    def test_zero_progress(self):
        remaining = estimate_remaining(10.0, 0.0)
        assert remaining == float("inf")

    def test_near_complete(self):
        remaining = estimate_remaining(99.0, 0.99)
        assert abs(remaining - 1.0) < 0.01
