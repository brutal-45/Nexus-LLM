"""Tests for nexus_llm.protocols.completion_protocol module."""

import pytest
from nexus_llm.protocols.completion_protocol import CompletionRequest, CompletionResponse, CompletionProtocol


class SampleCompletionProtocol(CompletionProtocol):
    def complete(self, request):
        return CompletionResponse(model=request.model, text="completed text")


class TestCompletionRequest:
    def test_default(self):
        req = CompletionRequest()
        assert req.temperature == 0.7
        assert req.max_tokens == 2048

    def test_with_prompt(self):
        req = CompletionRequest(prompt="Once upon a time", model="gpt-4")
        assert req.prompt == "Once upon a time"


class TestCompletionResponse:
    def test_default(self):
        resp = CompletionResponse()
        assert resp.finish_reason == "stop"

    def test_with_text(self):
        resp = CompletionResponse(text="Hello world", model="gpt-4")
        assert resp.text == "Hello world"


class TestCompletionProtocol:
    def test_validate_valid(self):
        proto = SampleCompletionProtocol()
        req = CompletionRequest(prompt="Hello", model="gpt-4")
        errors = proto.validate_request(req)
        assert errors == []

    def test_validate_no_prompt(self):
        proto = SampleCompletionProtocol()
        req = CompletionRequest(model="gpt-4")
        errors = proto.validate_request(req)
        assert len(errors) > 0

    def test_complete(self):
        proto = SampleCompletionProtocol()
        req = CompletionRequest(prompt="Hello", model="gpt-4")
        resp = proto.complete(req)
        assert resp.text == "completed text"
