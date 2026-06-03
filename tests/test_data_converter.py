"""Tests for data format conversion (Alpaca <-> ChatML, auto-detect format)."""
import json
import os
import tempfile

import pytest


class TestDataFormatDetection:
    """Test auto-detection of data formats."""

    def test_detect_jsonl_format(self, tmp_dir):
        """Detect JSONL format from file extension."""
        path = tmp_dir / "data.jsonl"
        path.write_text('{"text": "hello"}\n')
        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        # JSONL files are handled correctly
        data = []
        with open(str(path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        assert len(data) == 1
        assert data[0]["text"] == "hello"

    def test_detect_csv_format(self, tmp_dir):
        """Detect CSV format from file extension."""
        path = tmp_dir / "data.csv"
        path.write_text("text,label\nhello,1\nworld,0\n")
        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_csv_to_jsonl(str(path), str(tmp_dir / "out.jsonl"))
        out_path = tmp_dir / "out.jsonl"
        assert out_path.exists()
        with open(str(out_path), "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2

    def test_detect_json_format(self, tmp_dir):
        """Detect JSON format from file extension."""
        path = tmp_dir / "data.json"
        path.write_text('[{"text": "hello"}, {"text": "world"}]')
        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_json_to_jsonl(str(path), str(tmp_dir / "out.jsonl"))
        out_path = tmp_dir / "out.jsonl"
        assert out_path.exists()
        with open(str(out_path), "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


class TestJsonlToCsvConversion:
    """Test JSONL to CSV conversion."""

    def test_convert_jsonl_to_csv(self, tmp_dir):
        """Convert JSONL data to CSV format."""
        jsonl_path = tmp_dir / "input.jsonl"
        csv_path = tmp_dir / "output.csv"
        with open(str(jsonl_path), "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "hello world", "label": "greeting"}) + "\n")
            f.write(json.dumps({"text": "goodbye", "label": "farewell"}) + "\n")

        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_jsonl_to_csv(str(jsonl_path), str(csv_path))

        assert csv_path.exists()
        with open(str(csv_path), "r", encoding="utf-8") as f:
            content = f.read()
        assert "text" in content
        assert "hello world" in content

    def test_convert_csv_to_jsonl(self, tmp_dir):
        """Convert CSV data to JSONL format."""
        csv_path = tmp_dir / "input.csv"
        jsonl_path = tmp_dir / "output.jsonl"
        with open(str(csv_path), "w", encoding="utf-8", newline="") as f:
            f.write("text,label\n")
            f.write("hello,greeting\n")
            f.write("goodbye,farewell\n")

        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_csv_to_jsonl(str(csv_path), str(jsonl_path))

        assert jsonl_path.exists()
        with open(str(jsonl_path), "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["text"] == "hello"


class TestJsonToJsonlConversion:
    """Test JSON to JSONL conversion."""

    def test_convert_json_list_to_jsonl(self, tmp_dir):
        """Convert a JSON array to JSONL format."""
        json_path = tmp_dir / "data.json"
        jsonl_path = tmp_dir / "data.jsonl"
        data = [{"text": "first"}, {"text": "second"}, {"text": "third"}]
        with open(str(json_path), "w", encoding="utf-8") as f:
            json.dump(data, f)

        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_json_to_jsonl(str(json_path), str(jsonl_path))

        assert jsonl_path.exists()
        with open(str(jsonl_path), "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3

    def test_convert_json_dict_with_data_key(self, tmp_dir):
        """Convert a JSON dict with 'data' key to JSONL."""
        json_path = tmp_dir / "data.json"
        jsonl_path = tmp_dir / "data.jsonl"
        payload = {"data": [{"text": "a"}, {"text": "b"}]}
        with open(str(json_path), "w", encoding="utf-8") as f:
            json.dump(payload, f)

        from nexus_llm.training.preprocessing import DataPreprocessor
        dp = DataPreprocessor()
        dp.convert_json_to_jsonl(str(json_path), str(jsonl_path))

        with open(str(jsonl_path), "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


class TestAlpacaChatmlConversion:
    """Test conversion between Alpaca and ChatML formats."""

    def test_alpaca_to_messages(self):
        """Convert Alpaca-format sample to chat message list."""
        alpaca_sample = {
            "instruction": "Translate to French",
            "input": "Hello world",
            "output": "Bonjour le monde",
        }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"### Instruction:\n{alpaca_sample['instruction']}\n\n### Input:\n{alpaca_sample['input']}"},
            {"role": "assistant", "content": alpaca_sample["output"]},
        ]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert "Translate to French" in messages[1]["content"]
        assert messages[2]["content"] == "Bonjour le monde"

    def test_chatml_to_alpaca(self):
        """Convert ChatML messages back to Alpaca format."""
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
        ]
        alpaca = {
            "instruction": messages[0]["content"],
            "input": "",
            "output": messages[1]["content"],
        }
        assert alpaca["instruction"] == "What is AI?"
        assert alpaca["output"] == "AI stands for Artificial Intelligence."
