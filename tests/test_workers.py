"""Tests for the workers module.

Covers WorkerPool, Worker, TaskQueue, and WorkerConfig.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from nexus_llm.workers.pool import WorkerPool
from nexus_llm.workers.worker import Worker
from nexus_llm.workers.queue import TaskQueue
from nexus_llm.workers.config import WorkerConfig


# ---------------------------------------------------------------------------
# WorkerConfig
# ---------------------------------------------------------------------------

class TestWorkerConfig:
    """Tests for WorkerConfig."""

    def test_defaults(self):
        config = WorkerConfig()
        assert config is not None

    def test_custom_values(self):
        config = WorkerConfig(num_workers=4, timeout=30.0)
        assert config.num_workers == 4
        assert config.timeout == 30.0

    def test_to_dict(self):
        config = WorkerConfig()
        d = config.to_dict()
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TestWorker:
    """Tests for Worker."""

    def test_create_worker(self):
        worker = Worker(worker_id="w1")
        assert worker.worker_id == "w1"

    def test_worker_status(self):
        worker = Worker(worker_id="w1")
        status = worker.get_status()
        assert isinstance(status, dict)

    def test_worker_is_idle(self):
        worker = Worker(worker_id="w1")
        assert worker.is_idle() is True


# ---------------------------------------------------------------------------
# TaskQueue
# ---------------------------------------------------------------------------

class TestTaskQueue:
    """Tests for TaskQueue."""

    def test_create_queue(self):
        q = TaskQueue()
        assert q is not None

    def test_submit_and_get(self):
        q = TaskQueue()
        task_id = q.submit(lambda: 42)
        task = q.get(task_id)
        assert task is not None

    def test_queue_size(self):
        q = TaskQueue()
        q.submit(lambda: 1)
        q.submit(lambda: 2)
        assert q.size() == 2

    def test_queue_empty(self):
        q = TaskQueue()
        assert q.size() == 0


# ---------------------------------------------------------------------------
# WorkerPool
# ---------------------------------------------------------------------------

class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_create_pool(self):
        pool = WorkerPool(num_workers=2)
        assert pool is not None

    def test_submit_task(self):
        pool = WorkerPool(num_workers=2)
        future = pool.submit(lambda: 42)
        assert future is not None

    def test_pool_status(self):
        pool = WorkerPool(num_workers=2)
        status = pool.get_status()
        assert isinstance(status, dict)

    def test_shutdown(self):
        pool = WorkerPool(num_workers=2)
        pool.shutdown()
        # Should not crash
