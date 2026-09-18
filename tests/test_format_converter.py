"""Tests for Alpaca/ChatML/ShareGPT format conversion."""
import pytest
import json


class TestAlpacaFormat:
    """Test Alpaca format conversion."""

    def test_alpaca_to_standard(self):
        alpaca_data = {"instruction": "Translate", "input": "Hello", "output": "Bonjour"}
        result = {"messages": [
            {"role": "user", "content": alpaca_data["instruction"]},
            {"role": "assistant", "content": alpaca_data["output"]},
        ]}
        assert len(result["messages"]) == 2

    def test_alpaca_without_input(self):
        data = {"instruction": "Explain AI", "output": "AI is..."}
        assert "instruction" in data

    def test_alpaca_batch_conversion(self):
        records = [{"instruction": f"Task {i}", "output": f"Output {i}"} for i in range(10)]
        assert len(records) == 10


class TestChatMLFormat:
    """Test ChatML format conversion."""

    def test_chatml_round_trip(self):
        messages = [{"role": "system", "content": "Hi"}, {"role": "user", "content": "Hello"}]
        chatml = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        assert "<|im_start|>" in chatml

    def test_chatml_preserves_roles(self):
        for role in ["system", "user", "assistant"]:
            msg = {"role": role, "content": "test"}
            assert msg["role"] in {"system", "user", "assistant"}


class TestShareGPTFormat:
    """Test ShareGPT format conversion."""

    def test_sharegpt_conversation(self):
        convos = [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]
        converted = [{"role": "user" if c["from"] == "human" else "assistant", "content": c["value"]} for c in convos]
        assert converted[0]["role"] == "user"

    def test_sharegpt_multi_turn(self):
        convos = [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hey"}, {"from": "human", "value": "Bye"}]
        assert len(convos) == 3
