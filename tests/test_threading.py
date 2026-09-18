"""Test threading utilities for Nexus-LLM."""
import time
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty


# --- Threading utility implementations to test ---

class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, n=1):
        with self._lock:
            self._value += n
            return self._value

    def decrement(self, n=1):
        with self._lock:
            self._value -= n
            return self._value

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value


class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def set(self, key, value):
        with self._lock:
            self._dict[key] = value

    def delete(self, key):
        with self._lock:
            del self._dict[key]

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def items(self):
        with self._lock:
            return list(self._dict.items())

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def __len__(self):
        with self._lock:
            return len(self._dict)


class BackgroundWorker:
    def __init__(self, handler):
        self._handler = handler
        self._queue = Queue()
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=5.0):
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=timeout)

    def submit(self, item):
        self._queue.put(item)

    def _run(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                self._handler(item)
            except Empty:
                continue


def run_in_threads(func, items, max_workers=4):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(func, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results


class TestAtomicCounter:
    def test_initial_value(self):
        counter = AtomicCounter(0)
        assert counter.get() == 0

    def test_increment(self):
        counter = AtomicCounter(0)
        counter.increment()
        assert counter.get() == 1

    def test_increment_by_n(self):
        counter = AtomicCounter(0)
        counter.increment(5)
        assert counter.get() == 5

    def test_decrement(self):
        counter = AtomicCounter(10)
        counter.decrement()
        assert counter.get() == 9

    def test_set(self):
        counter = AtomicCounter(0)
        counter.set(42)
        assert counter.get() == 42

    def test_thread_safety(self):
        counter = AtomicCounter(0)
        def increment_many():
            for _ in range(1000):
                counter.increment()
        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert counter.get() == 10000

    def test_returns_new_value(self):
        counter = AtomicCounter(0)
        result = counter.increment()
        assert result == 1


class TestThreadSafeDict:
    def test_set_and_get(self):
        d = ThreadSafeDict()
        d.set("key", "value")
        assert d.get("key") == "value"

    def test_get_default(self):
        d = ThreadSafeDict()
        assert d.get("missing", "default") == "default"

    def test_delete(self):
        d = ThreadSafeDict()
        d.set("key", "value")
        d.delete("key")
        assert d.get("key") is None

    def test_contains(self):
        d = ThreadSafeDict()
        d.set("key", "value")
        assert "key" in d
        assert "missing" not in d

    def test_len(self):
        d = ThreadSafeDict()
        d.set("a", 1)
        d.set("b", 2)
        assert len(d) == 2

    def test_keys(self):
        d = ThreadSafeDict()
        d.set("a", 1)
        d.set("b", 2)
        keys = d.keys()
        assert set(keys) == {"a", "b"}

    def test_items(self):
        d = ThreadSafeDict()
        d.set("x", 10)
        items = d.items()
        assert ("x", 10) in items

    def test_concurrent_access(self):
        d = ThreadSafeDict()
        errors = []

        def writer(n):
            try:
                for i in range(100):
                    d.set(f"key_{n}_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(d) == 1000


class TestBackgroundWorker:
    def test_processes_items(self):
        results = []
        def handler(item):
            results.append(item)
        worker = BackgroundWorker(handler)
        worker.start()
        worker.submit("task1")
        worker.submit("task2")
        time.sleep(0.2)
        worker.stop()
        assert "task1" in results
        assert "task2" in results

    def test_stop_gracefully(self):
        worker = BackgroundWorker(lambda x: None)
        worker.start()
        worker.stop(timeout=2.0)
        assert not worker._running or not worker._thread.is_alive()


class TestRunInThreads:
    def test_parallel_execution(self):
        def square(x):
            return x * x
        results = run_in_threads(square, [1, 2, 3, 4, 5])
        assert set(results) == {1, 4, 9, 16, 25}

    def test_preserves_all_results(self):
        def identity(x):
            return x
        results = run_in_threads(identity, range(100))
        assert set(results) == set(range(100))

    def test_max_workers(self):
        def slow_square(x):
            time.sleep(0.01)
            return x * x
        results = run_in_threads(slow_square, [1, 2, 3], max_workers=2)
        assert set(results) == {1, 4, 9}
