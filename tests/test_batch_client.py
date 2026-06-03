"""Tests for nexus_llm.client.batch_client module."""

import pytest
from unittest.mock import MagicMock, patch
from nexus_llm.client.batch_client import BatchClient, BatchRequest, BatchConfig, BatchStatus


class TestBatchConfig:
    def test_default(self):
        config = BatchConfig()
        assert config.max_workers == 4
        assert config.rate_limit == 10.0

    def test_custom(self):
        config = BatchConfig(max_workers=8, max_retries=5)
        assert config.max_workers == 8
        assert config.max_retries == 5


class TestBatchRequest:
    def test_default(self):
        req = BatchRequest()
        assert req.method == "POST"
        assert req.id is not None

    def test_custom(self):
        req = BatchRequest(method="GET", endpoint="/models")
        assert req.method == "GET"
        assert req.endpoint == "/models"


class TestBatchClient:
    def test_init(self):
        client = BatchClient()
        assert client.status == BatchStatus.PENDING
        assert client.request_count == 0

    def test_add_request(self):
        client = BatchClient()
        client.add_request(BatchRequest(endpoint="/chat"))
        assert client.request_count == 1

    def test_add_multiple_requests(self):
        client = BatchClient()
        client.add_requests([
            BatchRequest(endpoint="/chat"),
            BatchRequest(endpoint="/complete"),
        ])
        assert client.request_count == 2

    def test_clear(self):
        client = BatchClient()
        client.add_request(BatchRequest(endpoint="/chat"))
        client.clear()
        assert client.request_count == 0

    def test_get_summary(self):
        client = BatchClient()
        summary = client.get_summary()
        assert "total" in summary
        assert "successes" in summary
        assert "failures" in summary

    def test_cancel(self):
        client = BatchClient()
        client.cancel()
        assert client.status == BatchStatus.CANCELLED
