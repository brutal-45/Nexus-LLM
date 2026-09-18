"""Tests for the export module.

Covers ExportManager, ChatExporter, ModelExporter, and DataExporter.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from nexus_llm.export.manager import ExportManager
from nexus_llm.export.chat_exporter import ChatExporter
from nexus_llm.export.model_exporter import ModelExporter
from nexus_llm.export.data_exporter import DataExporter


# ---------------------------------------------------------------------------
# ChatExporter
# ---------------------------------------------------------------------------

class TestChatExporter:
    """Tests for ChatExporter."""

    def test_create_exporter(self):
        exporter = ChatExporter()
        assert exporter is not None

    def test_export_json(self):
        exporter = ChatExporter()
        conversations = [
            {"id": "c1", "messages": [{"role": "user", "content": "Hello"}]},
        ]
        result = exporter.export_json(conversations)
        data = json.loads(result)
        assert len(data) == 1

    def test_export_markdown(self):
        exporter = ChatExporter()
        conversations = [
            {"id": "c1", "messages": [{"role": "user", "content": "Hello"}]},
        ]
        result = exporter.export_markdown(conversations)
        assert "Hello" in result


# ---------------------------------------------------------------------------
# ModelExporter
# ---------------------------------------------------------------------------

class TestModelExporter:
    """Tests for ModelExporter."""

    def test_create_exporter(self):
        exporter = ModelExporter()
        assert exporter is not None

    def test_get_supported_formats(self):
        exporter = ModelExporter()
        formats = exporter.get_supported_formats()
        assert isinstance(formats, list)


# ---------------------------------------------------------------------------
# DataExporter
# ---------------------------------------------------------------------------

class TestDataExporter:
    """Tests for DataExporter."""

    def test_create_exporter(self):
        exporter = DataExporter()
        assert exporter is not None

    def test_export_json(self):
        exporter = DataExporter()
        data = [{"key": "value"}]
        result = exporter.export_json(data)
        parsed = json.loads(result)
        assert parsed[0]["key"] == "value"

    def test_export_csv(self):
        exporter = DataExporter()
        data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        result = exporter.export_csv(data)
        assert "Alice" in result
        assert "Bob" in result

    def test_export_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = DataExporter()
            data = [{"key": "value"}]
            path = os.path.join(tmpdir, "output.json")
            exporter.export_to_file(data, path, format="json")
            assert os.path.exists(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded[0]["key"] == "value"


# ---------------------------------------------------------------------------
# ExportManager
# ---------------------------------------------------------------------------

class TestExportManager:
    """Tests for ExportManager."""

    def test_create_manager(self):
        mgr = ExportManager()
        assert mgr is not None

    def test_get_chat_exporter(self):
        mgr = ExportManager()
        exporter = mgr.get_chat_exporter()
        assert isinstance(exporter, ChatExporter)

    def test_get_model_exporter(self):
        mgr = ExportManager()
        exporter = mgr.get_model_exporter()
        assert isinstance(exporter, ModelExporter)

    def test_get_data_exporter(self):
        mgr = ExportManager()
        exporter = mgr.get_data_exporter()
        assert isinstance(exporter, DataExporter)
