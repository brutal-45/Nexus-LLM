"""Tests for nexus_llm.protocols.chat_protocol module."""

import pytest
from nexus_llm.protocols.chat_protocol import ChatMessage, ChatRequest, ChatResponse, ChatProtocol


class SampleChatProtocol(ChatProtocol):
    def chat(self, request):
        msg = ChatMessage(role="assistant", content="Response")
        return ChatResponse(model=request.model, message=msg)


class TestChatMessage:
    def test_creation(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self):
        msg = ChatMessage(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"


class TestChatRequest:
    def test_default(self):
        req = ChatRequest()
        assert req.temperature == 0.7
        assert req.max_tokens == 2048
        assert req.stream is False

    def test_to_dict(self):
        req = ChatRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Hi")])
        d = req.to_dict()
        assert d["model"] == "gpt-4"
        assert len(d["messages"]) == 1


class TestChatResponse:
    def test_default(self):
        resp = ChatResponse()
        assert resp.finish_reason == "stop"

    def test_to_dict(self):
        resp = ChatResponse(model="gpt-4", message=ChatMessage(role="assistant", content="Hi"))
        d = resp.to_dict()
        assert d["model"] == "gpt-4"


class TestChatProtocol:
    def test_validate_valid(self):
        proto = SampleChatProtocol()
        req = ChatRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Hi")])
        errors = proto.validate_request(req)
        assert errors == []

    def test_validate_empty_messages(self):
        proto = SampleChatProtocol()
        req = ChatRequest(model="gpt-4")
        errors = proto.validate_request(req)
        assert len(errors) > 0

    def test_validate_no_model(self):
        proto = SampleChatProtocol()
        req = ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        errors = proto.validate_request(req)
        assert any("Model" in e for e in errors)

    def test_chat(self):
        proto = SampleChatProtocol()
        req = ChatRequest(model="gpt-4", messages=[ChatMessage(role="user", content="Hi")])
        resp = proto.chat(req)
        assert resp.message.content == "Response"
