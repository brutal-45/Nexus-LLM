"""Tests for the storage module.

Covers FileStorage, SQLiteStorage, ConversationStore, and ModelStore.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from nexus_llm.storage.file_storage import FileStorage, StorageError
from nexus_llm.storage.sqlite_storage import SQLiteStorage
from nexus_llm.storage.conversation_store import ConversationStore, Conversation, Message
from nexus_llm.storage.model_store import ModelStore


# ---------------------------------------------------------------------------
# FileStorage
# ---------------------------------------------------------------------------

class TestFileStorage:
    """Tests for FileStorage."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            store.save("key1", {"hello": "world"})
            result = store.load("key1")
            assert result == {"hello": "world"}

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            with pytest.raises(KeyError):
                store.load("nonexistent")

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            store.save("key1", "value1")
            assert store.delete("key1") is True
            assert not store.exists("key1")

    def test_delete_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            assert store.delete("nonexistent") is False

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            assert store.exists("key1") is False
            store.save("key1", "val")
            assert store.exists("key1") is True

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            store.save("a", 1)
            store.save("b", 2)
            keys = store.list_keys()
            assert set(keys) == {"a", "b"}

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            store.save("key", "v1")
            store.save("key", "v2")
            assert store.load("key") == "v2"

    def test_base_dir_property(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileStorage(base_dir=tmpdir)
            assert store.base_dir == os.path.realpath(tmpdir)


# ---------------------------------------------------------------------------
# SQLiteStorage
# ---------------------------------------------------------------------------

class TestSQLiteStorage:
    """Tests for SQLiteStorage."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("key1", {"data": "value"})
            result = store.load("key1")
            assert result == {"data": "value"}
            store.close()

    def test_load_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            with pytest.raises(KeyError):
                store.load("nonexistent")
            store.close()

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("key1", "val")
            assert store.delete("key1") is True
            assert store.delete("key1") is False
            store.close()

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("key1", "val")
            assert store.exists("key1") is True
            assert store.exists("key2") is False
            store.close()

    def test_list_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("alpha", 1)
            store.save("beta", 2)
            keys = store.list_keys()
            assert "alpha" in keys
            assert "beta" in keys
            store.close()

    def test_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("a", 1)
            store.save("b", 2)
            assert store.count() == 2
            store.close()

    def test_query_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save("models/gpt2", {"size": 100})
            store.save("models/llama", {"size": 200})
            store.save("config", {"theme": "dark"})
            result = store.query("models/")
            assert len(result) == 2
            store.close()

    def test_save_many(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            store.save_many({"k1": "v1", "k2": "v2", "k3": "v3"})
            assert store.count() == 3
            store.close()

    def test_repr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = SQLiteStorage(db_path=db_path)
            r = repr(store)
            assert "SQLiteStorage" in r
            store.close()


# ---------------------------------------------------------------------------
# Conversation & Message
# ---------------------------------------------------------------------------

class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp  # auto-generated

    def test_message_metadata(self):
        msg = Message(role="assistant", content="Hi", metadata={"tokens": 5})
        assert msg.metadata["tokens"] == 5


class TestConversation:
    """Tests for Conversation dataclass."""

    def test_create_conversation(self):
        conv = Conversation(title="Test Chat")
        assert conv.title == "Test Chat"
        assert conv.id  # auto-generated
        assert conv.messages == []

    def test_add_message(self):
        conv = Conversation()
        msg = conv.add_message("user", "Hello")
        assert len(conv.messages) == 1
        assert msg.content == "Hello"
        assert conv.updated_at  # timestamp updated

    def test_to_dict(self):
        conv = Conversation(id="c1", title="Test")
        conv.add_message("user", "Hi")
        d = conv.to_dict()
        assert d["id"] == "c1"
        assert len(d["messages"]) == 1

    def test_from_dict(self):
        data = {
            "id": "c1",
            "title": "Test",
            "messages": [{"role": "user", "content": "Hi"}],
            "created_at": "",
            "updated_at": "",
            "metadata": {},
        }
        conv = Conversation.from_dict(data)
        assert conv.id == "c1"
        assert len(conv.messages) == 1


# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------

class TestConversationStore:
    """Tests for ConversationStore."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ConversationStore(backend)
            conv = Conversation(title="Test Chat")
            conv.add_message("user", "Hello")
            cid = store.save_conversation(conv)
            loaded = store.load_conversation(cid)
            assert loaded.title == "Test Chat"
            assert len(loaded.messages) == 1

    def test_list_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ConversationStore(backend)
            conv1 = Conversation(title="Chat 1")
            conv2 = Conversation(title="Chat 2")
            store.save_conversation(conv1)
            store.save_conversation(conv2)
            summaries = store.list_conversations()
            assert len(summaries) == 2

    def test_delete_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ConversationStore(backend)
            conv = Conversation(title="Delete Me")
            cid = store.save_conversation(conv)
            assert store.delete_conversation(cid) is True
            assert store.delete_conversation(cid) is False

    def test_search_conversations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ConversationStore(backend)
            conv = Conversation(title="Python Chat")
            conv.add_message("user", "Tell me about Python")
            store.save_conversation(conv)
            results = store.search_conversations("Python")
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------

class TestModelStore:
    """Tests for ModelStore."""

    def test_save_and_get_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"path": "/models/gpt2", "size_bytes": 1000})
            meta = store.get_model_metadata("gpt2")
            assert meta["path"] == "/models/gpt2"
            assert meta["model_id"] == "gpt2"

    def test_get_nonexistent_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            with pytest.raises(KeyError):
                store.get_model_metadata("nonexistent")

    def test_list_cached_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"format": "pytorch"})
            store.save_model_metadata("llama", {"format": "safetensors"})
            models = store.list_cached_models()
            assert len(models) == 2

    def test_delete_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"path": "/m"})
            assert store.delete_model("gpt2") is True
            assert store.delete_model("gpt2") is False

    def test_model_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            assert store.model_exists("gpt2") is False
            store.save_model_metadata("gpt2", {"path": "/m"})
            assert store.model_exists("gpt2") is True

    def test_get_model_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"size_bytes": 1500000})
            assert store.get_model_size("gpt2") == 1500000
            assert store.get_model_size("nonexistent") == 0

    def test_get_models_by_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"format": "pytorch"})
            store.save_model_metadata("llama", {"format": "safetensors"})
            store.save_model_metadata("mistral", {"format": "pytorch"})
            pytorch_models = store.get_models_by_format("pytorch")
            assert len(pytorch_models) == 2

    def test_total_cache_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"size_bytes": 1000})
            store.save_model_metadata("llama", {"size_bytes": 2000})
            assert store.total_cache_size() == 3000

    def test_metadata_merge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = FileStorage(base_dir=tmpdir)
            store = ModelStore(backend)
            store.save_model_metadata("gpt2", {"a": 1})
            store.save_model_metadata("gpt2", {"b": 2})
            meta = store.get_model_metadata("gpt2")
            assert meta["a"] == 1
            assert meta["b"] == 2
