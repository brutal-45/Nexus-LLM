"""Tests for request scheduler."""
import pytest
import time
from collections import deque


class SimpleRequestScheduler:
    """Simple request scheduler for testing."""
    def __init__(self, max_concurrent=10):
        self.max_concurrent = max_concurrent
        self.queue = deque()
        self.active = 0

    def submit(self, request_id):
        if self.active < self.max_concurrent:
            self.active += 1
            return True
        self.queue.append(request_id)
        return False

    def complete(self):
        self.active -= 1
        if self.queue:
            self.queue.popleft()
            self.active += 1
            return True
        return False

    @property
    def pending(self):
        return len(self.queue)


@pytest.fixture
def scheduler():
    return SimpleRequestScheduler(max_concurrent=2)


def test_scheduler_accepts_within_limit(scheduler):
    """Test that scheduler accepts requests within limit."""
    assert scheduler.submit("r1") is True
    assert scheduler.active == 1


def test_scheduler_queues_over_limit(scheduler):
    """Test that scheduler queues requests over limit."""
    scheduler.submit("r1")
    scheduler.submit("r2")
    result = scheduler.submit("r3")
    assert result is False
    assert scheduler.pending == 1


def test_scheduler_dequeue_on_complete(scheduler):
    """Test that queued requests are started when slots free."""
    scheduler.submit("r1")
    scheduler.submit("r2")
    scheduler.submit("r3")
    scheduler.complete()
    assert scheduler.pending == 0
    assert scheduler.active == 2


def test_scheduler_full_cycle(scheduler):
    """Test full submit-process-complete cycle."""
    scheduler.submit("r1")
    scheduler.submit("r2")
    scheduler.submit("r3")
    assert scheduler.pending == 1
    scheduler.complete()
    assert scheduler.pending == 0
    scheduler.complete()
    assert scheduler.active == 1
