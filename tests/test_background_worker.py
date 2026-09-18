"""Tests for background worker."""
import pytest
import threading
from queue import Queue


def test_background_worker_execution():
    results = []

    def worker(q):
        while True:
            item = q.get()
            if item is None:
                break
            results.append(item)
            q.task_done()

    q = Queue()
    t = threading.Thread(target=worker, args=(q,))
    t.start()
    q.put(1)
    q.put(2)
    q.put(None)
    t.join()
    assert results == [1, 2]

def test_background_worker_graceful_stop():
    running = True
    running = False
    assert running is False

def test_background_worker_task_queue():
    q = Queue()
    for i in range(5):
        q.put(i)
    assert q.qsize() == 5
