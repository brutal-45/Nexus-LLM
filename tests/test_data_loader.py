"""Test data loading for Nexus-LLM."""
import json
import csv
import io
import os
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path


@dataclass
class DataSample:
    id: str
    text: str
    label: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLoader:
    def __init__(self, batch_size: int = 32, shuffle: bool = False, seed: int = 42):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

    @property
    def batch_size(self):
        return self._batch_size

    def load_json(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return [data]

    def load_jsonl(self, path: str) -> List[Dict]:
        results = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def load_csv(self, path: str) -> List[Dict]:
        results = []
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(dict(row))
        return results

    def load_text(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def batch(self, data: List[Any]) -> Iterator[List[Any]]:
        import random
        items = list(data)
        if self._shuffle:
            random.seed(self._seed)
            random.shuffle(items)
        for i in range(0, len(items), self._batch_size):
            yield items[i:i + self._batch_size]

    def load_from_dict(self, data: Dict[str, List]) -> List[DataSample]:
        lengths = {k: len(v) for k, v in data.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError("All columns must have same length")
        n = max(lengths.values()) if lengths else 0
        samples = []
        for i in range(n):
            sample_data = {k: v[i] for k, v in data.items()}
            samples.append(DataSample(
                id=str(i),
                text=sample_data.get("text", ""),
                label=sample_data.get("label", None),
                metadata=sample_data,
            ))
        return samples


class TestDataLoader:
    def test_load_json(self, tmp_dir):
        data = [{"text": "hello", "label": 1}, {"text": "world", "label": 0}]
        f = tmp_dir / "data.json"
        f.write_text(json.dumps(data))
        loader = DataLoader()
        loaded = loader.load_json(str(f))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_load_json_single(self, tmp_dir):
        data = {"text": "single", "label": 1}
        f = tmp_dir / "single.json"
        f.write_text(json.dumps(data))
        loader = DataLoader()
        loaded = loader.load_json(str(f))
        assert len(loaded) == 1

    def test_load_jsonl(self, tmp_dir):
        lines = ['{"text": "a"}\n', '{"text": "b"}\n']
        f = tmp_dir / "data.jsonl"
        f.write_text("".join(lines))
        loader = DataLoader()
        loaded = loader.load_jsonl(str(f))
        assert len(loaded) == 2

    def test_load_csv(self, tmp_dir):
        f = tmp_dir / "data.csv"
        f.write_text("text,label\nhello,1\nworld,0\n")
        loader = DataLoader()
        loaded = loader.load_csv(str(f))
        assert len(loaded) == 2
        assert loaded[0]["text"] == "hello"

    def test_load_text(self, tmp_dir):
        f = tmp_dir / "data.txt"
        f.write_text("hello world")
        loader = DataLoader()
        text = loader.load_text(str(f))
        assert text == "hello world"

    def test_batch(self):
        loader = DataLoader(batch_size=3)
        data = list(range(10))
        batches = list(loader.batch(data))
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_batch_shuffle(self):
        loader = DataLoader(batch_size=5, shuffle=True, seed=42)
        data = list(range(20))
        batches = list(loader.batch(data))
        assert len(batches) == 4

    def test_load_from_dict(self):
        loader = DataLoader()
        data = {"text": ["a", "b", "c"], "label": [1, 0, 1]}
        samples = loader.load_from_dict(data)
        assert len(samples) == 3
        assert samples[0].text == "a"
        assert samples[1].label == 0

    def test_load_from_dict_mismatched_lengths(self):
        loader = DataLoader()
        data = {"text": ["a", "b"], "label": [1]}
        with pytest.raises(ValueError, match="same length"):
            loader.load_from_dict(data)

    def test_batch_size_property(self):
        loader = DataLoader(batch_size=64)
        assert loader.batch_size == 64
