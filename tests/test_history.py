"""Tests for the chat history module."""

import json
import os
import tempfile
import time

import pytest

from nexus_llm.terminal.history import Message, Conversation, ChatHistory


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

class TestMessage:
    """Tests for the Message dataclass."""

    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        msg = Message(role="assistant", content="Hi!", metadata={"model": "gpt2"})
        assert msg.metadata == {"model": "gpt2"}

    def test_message_to_dict(self):
        msg = Message(role="user", content="Hello", metadata={"key": "val"})
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert d["metadata"] == {"key": "val"}
        assert "timestamp" in d

    def test_message_from_dict(self):
        d = {"role": "user", "content": "Hello", "timestamp": 12345.0, "metadata": {"k": "v"}}
        msg = Message.from_dict(d)
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == 12345.0
        assert msg.metadata == {"k": "v"}

    def test_message_from_dict_defaults(self):
        d = {"role": "user", "content": "Hello"}
        msg = Message.from_dict(d)
        assert msg.timestamp > 0
        assert msg.metadata == {}

    def test_message_roundtrip(self):
        original = Message(role="system", content="Be helpful", metadata={"priority": "high"})
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.timestamp == original.timestamp
        assert restored.metadata == original.metadata


# ---------------------------------------------------------------------------
# Conversation dataclass
# ---------------------------------------------------------------------------

class TestConversation:
    """Tests for the Conversation dataclass."""

    def test_create_conversation(self):
        conv = Conversation(id="conv_1")
        assert conv.id == "conv_1"
        assert conv.title == "Untitled"
        assert conv.model == "unknown"
        assert conv.messages == []

    def test_create_conversation_with_fields(self):
        conv = Conversation(id="conv_1", title="Test", model="gpt2-medium")
        assert conv.title == "Test"
        assert conv.model == "gpt2-medium"

    def test_add_message(self):
        conv = Conversation(id="conv_1")
        msg = conv.add_message("user", "Hello")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello"
        assert isinstance(msg, Message)

    def test_add_message_with_metadata(self):
        conv = Conversation(id="conv_1")
        msg = conv.add_message("assistant", "Hi!", metadata={"tokens": 5})
        assert msg.metadata == {"tokens": 5}

    def test_add_multiple_messages(self):
        conv = Conversation(id="conv_1")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        conv.add_message("user", "How are you?")
        assert len(conv.messages) == 3

    def test_conversation_updated_at(self):
        conv = Conversation(id="conv_1")
        old_updated = conv.updated_at
        time.sleep(0.01)
        conv.add_message("user", "Hello")
        assert conv.updated_at >= old_updated

    def test_conversation_to_dict(self):
        conv = Conversation(id="conv_1", title="Test", model="gpt2")
        conv.add_message("user", "Hello")
        d = conv.to_dict()
        assert d["id"] == "conv_1"
        assert d["title"] == "Test"
        assert d["model"] == "gpt2"
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"

    def test_conversation_from_dict(self):
        d = {
            "id": "conv_1",
            "title": "Test",
            "model": "gpt2",
            "created_at": 12345.0,
            "updated_at": 12346.0,
            "messages": [
                {"role": "user", "content": "Hello", "timestamp": 12345.5, "metadata": {}}
            ],
        }
        conv = Conversation.from_dict(d)
        assert conv.id == "conv_1"
        assert conv.title == "Test"
        assert conv.model == "gpt2"
        assert conv.created_at == 12345.0
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello"

    def test_conversation_from_dict_defaults(self):
        d = {"id": "conv_2"}
        conv = Conversation.from_dict(d)
        assert conv.title == "Untitled"
        assert conv.model == "unknown"
        assert conv.messages == []

    def test_conversation_roundtrip(self):
        conv = Conversation(id="conv_1", title="Round Trip", model="phi-2")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        d = conv.to_dict()
        restored = Conversation.from_dict(d)
        assert restored.id == conv.id
        assert restored.title == conv.title
        assert restored.model == conv.model
        assert len(restored.messages) == 2

    def test_clear(self):
        conv = Conversation(id="conv_1")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        conv.clear()
        assert len(conv.messages) == 0


# ---------------------------------------------------------------------------
# ChatHistory
# ---------------------------------------------------------------------------

class TestChatHistory:
    """Tests for the ChatHistory class."""

    @pytest.fixture
    def history_dir(self, tmp_path):
        """Provide a temporary history directory."""
        return str(tmp_path / "history")

    @pytest.fixture
    def history(self, history_dir):
        """Create a ChatHistory with a temporary directory."""
        return ChatHistory(history_dir=history_dir, max_history=10)

    # -- Current conversation --

    def test_initial_no_current(self, history):
        assert history.current is None

    def test_new_conversation(self, history):
        conv = history.new_conversation(title="Test Chat", model="gpt2-medium")
        assert history.current is conv
        assert conv.title == "Test Chat"
        assert conv.model == "gpt2-medium"
        assert conv.id.startswith("conv_")

    def test_add_message_creates_conversation(self, history):
        history.add_message("user", "Hello")
        assert history.current is not None
        assert len(history.current.messages) == 1

    def test_add_message_to_existing(self, history):
        history.new_conversation()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi!")
        assert len(history.current.messages) == 2

    def test_get_messages_empty(self, history):
        assert history.get_messages() == []

    def test_get_messages(self, history):
        history.add_message("user", "Hello")
        msgs = history.get_messages()
        assert len(msgs) == 1
        assert msgs[0].content == "Hello"

    def test_clear_current(self, history):
        history.add_message("user", "Hello")
        history.clear_current()
        assert len(history.current.messages) == 0

    def test_clear_current_none(self, history):
        history.clear_current()  # Should not raise

    def test_get_message_dicts(self, history):
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi!")
        dicts = history.get_message_dicts()
        assert len(dicts) == 2
        assert dicts[0] == {"role": "user", "content": "Hello"}
        assert dicts[1] == {"role": "assistant", "content": "Hi!"}

    def test_get_message_dicts_no_conversation(self, history):
        assert history.get_message_dicts() == []

    # -- Save / Load --

    def test_save_and_load(self, history, history_dir):
        conv = history.new_conversation(title="Save Test", model="phi-2")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")

        path = history.save()
        assert os.path.exists(path)

        history._current = None  # Reset current
        loaded = history.load(os.path.basename(path))
        assert loaded.title == "Save Test"
        assert loaded.model == "phi-2"
        assert len(loaded.messages) == 2
        assert history.current is loaded

    def test_save_no_conversation_raises(self, history):
        with pytest.raises(ValueError, match="No conversation to save"):
            history.save()

    def test_load_nonexistent_file_raises(self, history):
        with pytest.raises(FileNotFoundError):
            history.load("nonexistent.json")

    def test_save_with_custom_filename(self, history, history_dir):
        conv = history.new_conversation(title="Custom")
        path = history.save(filename="custom.json")
        assert path.endswith("custom.json")
        assert os.path.exists(path)

    def test_list_conversations(self, history):
        conv = history.new_conversation(title="Chat 1")
        conv.add_message("user", "Hi")
        history.save()

        convs = history.list_conversations()
        assert len(convs) >= 1
        assert convs[0]["title"] == "Chat 1"

    def test_delete(self, history):
        conv = history.new_conversation(title="Delete Me")
        path = history.save()
        fname = os.path.basename(path)
        assert history.delete(fname) is True
        assert not os.path.exists(path)

    def test_delete_nonexistent(self, history):
        assert history.delete("nonexistent.json") is False

    # -- Search --

    def test_search_by_title(self, history):
        conv = history.new_conversation(title="Python Discussion")
        conv.add_message("user", "Tell me about Python")
        history.save()

        results = history.search("python")
        assert len(results) >= 1
        matched_titles = [r["title"] for r in results]
        assert "Python Discussion" in matched_titles

    def test_search_by_content(self, history):
        conv = history.new_conversation(title="General Chat")
        conv.add_message("user", "Tell me about machine learning algorithms")
        history.save()

        results = history.search("machine learning")
        assert len(results) >= 1
        assert results[0]["match"] == "message"

    def test_search_no_results(self, history):
        conv = history.new_conversation(title="Empty Chat")
        conv.add_message("user", "Hello")
        history.save()

        results = history.search("xyznonexistent")
        assert results == []

    def test_search_limit(self, history):
        for i in range(5):
            conv = history.new_conversation(title=f"Chat {i}")
            conv.add_message("user", f"Test message {i}")
            history.save()
            history._current = None

        results = history.search("Chat", limit=2)
        assert len(results) <= 2

    def test_search_case_insensitive(self, history):
        conv = history.new_conversation(title="UPPERCASE TITLE")
        conv.add_message("user", "Hello")
        history.save()

        results = history.search("uppercase")
        assert len(results) >= 1

    # -- Export --

    def test_export_markdown(self, history, history_dir):
        conv = history.new_conversation(title="MD Export Test")
        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        path = history.export_markdown()
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read()
        assert "# MD Export Test" in content
        assert "Hello" in content
        assert "Hi!" in content

    def test_export_json(self, history, history_dir):
        conv = history.new_conversation(title="JSON Export Test")
        conv.add_message("user", "Hello")
        path = history.export_json()
        assert os.path.exists(path)
        with open(path, "r") as f:
            data = json.load(f)
        assert data["title"] == "JSON Export Test"

    def test_export_markdown_no_conversation_raises(self, history):
        with pytest.raises(ValueError, match="No conversation to export"):
            history.export_markdown()

    def test_export_json_no_conversation_raises(self, history):
        with pytest.raises(ValueError, match="No conversation to export"):
            history.export_json()


# ---------------------------------------------------------------------------
# ChatHistory — Auto-trim
# ---------------------------------------------------------------------------

class TestChatHistoryAutoTrim:
    """Tests for the auto_trim functionality."""

    def test_auto_trim_no_excess(self, tmp_path):
        history_dir = str(tmp_path / "history")
        history = ChatHistory(history_dir=history_dir, max_history=10)
        conv = history.new_conversation(title="Chat")
        conv.add_message("user", "Hi")
        history.save()
        removed = history.auto_trim()
        assert removed == 0

    def test_auto_trim_removes_oldest(self, tmp_path):
        history_dir = str(tmp_path / "history")
        history = ChatHistory(history_dir=history_dir, max_history=2)

        # Create 4 conversations with unique filenames to avoid ID collisions
        for i in range(4):
            conv = Conversation(id=f"conv_trim_{i}", title=f"Chat {i}", model="test")
            conv.add_message("user", f"Message {i}")
            history.save(conversation=conv, filename=f"conv_trim_{i}.json")

        # There should be 4 files, max is 2
        files = list(history.history_dir.glob("*.json"))
        assert len(files) == 4

        removed = history.auto_trim()
        assert removed == 2

        files = list(history.history_dir.glob("*.json"))
        assert len(files) == 2

    def test_auto_trim_with_max_history_zero(self, tmp_path):
        history_dir = str(tmp_path / "history")
        history = ChatHistory(history_dir=history_dir, max_history=0)

        conv = history.new_conversation(title="Chat")
        conv.add_message("user", "Hi")
        history.save()

        removed = history.auto_trim()
        # All files should be removed since max_history is 0
        assert removed >= 1
