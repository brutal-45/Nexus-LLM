"""Tests for async streamer."""
import pytest
import asyncio


def test_async_iteration():
    async def gen():
        for t in ["Hello", " world"]:
            yield t
    async def collect():
        r = []
        async for t in gen():
            r.append(t)
        return r
    assert asyncio.get_event_loop().run_until_complete(collect()) == ["Hello", " world"]


def test_async_streamer_with_queue():
    async def test_q():
        q = asyncio.Queue()
        await q.put("t1")
        await q.put(None)
        r = []
        while True:
            item = await q.get()
            if item is None:
                break
            r.append(item)
        return r
    assert asyncio.get_event_loop().run_until_complete(test_q()) == ["t1"]


def test_async_streamer_cancellation():
    cancelled = False
    async def cancel_stream():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled = True
            raise
    async def test():
        task = asyncio.create_task(cancel_stream())
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return cancelled
    assert asyncio.get_event_loop().run_until_complete(test()) is True


def test_async_streamer_timeout():
    async def slow():
        await asyncio.sleep(5)
    async def test():
        try:
            await asyncio.wait_for(slow(), timeout=0.1)
        except asyncio.TimeoutError:
            return "timeout"
    assert asyncio.get_event_loop().run_until_complete(test()) == "timeout"
