"""Test data processing for Nexus-LLM."""
import pytest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional


@dataclass
class ProcessingStep:
    name: str
    fn: Callable
    description: str = ""


class DataProcessor:
    def __init__(self):
        self._steps: List[ProcessingStep] = []

    def add_step(self, name: str, fn: Callable, description: str = ""):
        self._steps.append(ProcessingStep(name=name, fn=fn, description=description))

    def process(self, data: List[Dict]) -> List[Dict]:
        for step in self._steps:
            data = step.fn(data)
        return data

    def process_single(self, item: Dict) -> Dict:
        result = dict(item)
        for step in self._steps:
            result = step.fn([result])[0]
        return result

    def list_steps(self) -> List[str]:
        return [s.name for s in self._steps]

    def remove_step(self, name: str):
        self._steps = [s for s in self._steps if s.name != name]

    def clear(self):
        self._steps.clear()


# Common processing functions
def filter_empty(data: List[Dict], text_key: str = "text") -> List[Dict]:
    return [d for d in data if d.get(text_key, "").strip()]


def filter_by_length(data: List[Dict], min_length: int = 1, max_length: int = 10000, text_key: str = "text") -> List[Dict]:
    return [d for d in data if min_length <= len(d.get(text_key, "")) <= max_length]


def normalize_text(data: List[Dict], text_key: str = "text") -> List[Dict]:
    import re
    result = []
    for d in data:
        new_d = dict(d)
        text = new_d.get(text_key, "")
        text = re.sub(r'\s+', ' ', text).strip()
        new_d[text_key] = text
        result.append(new_d)
    return result


def lowercase_text(data: List[Dict], text_key: str = "text") -> List[Dict]:
    result = []
    for d in data:
        new_d = dict(d)
        new_d[text_key] = new_d.get(text_key, "").lower()
        result.append(new_d)
    return result


def deduplicate(data: List[Dict], key: str = "text") -> List[Dict]:
    seen = set()
    result = []
    for d in data:
        val = d.get(key, "")
        if val not in seen:
            seen.add(val)
            result.append(d)
    return result


def sample_data(data: List[Dict], n: int = 100, seed: int = 42) -> List[Dict]:
    import random
    random.seed(seed)
    return random.sample(data, min(n, len(data)))


def rename_key(data: List[Dict], old_key: str, new_key: str) -> List[Dict]:
    result = []
    for d in data:
        new_d = dict(d)
        if old_key in new_d:
            new_d[new_key] = new_d.pop(old_key)
        result.append(new_d)
    return result


class TestDataProcessor:
    def test_add_step(self):
        processor = DataProcessor()
        processor.add_step("filter", filter_empty)
        assert len(processor.list_steps()) == 1

    def test_process(self):
        processor = DataProcessor()
        processor.add_step("filter_empty", filter_empty)
        data = [{"text": "hello"}, {"text": ""}, {"text": "world"}]
        result = processor.process(data)
        assert len(result) == 2

    def test_multiple_steps(self):
        processor = DataProcessor()
        processor.add_step("filter", filter_empty)
        processor.add_step("lowercase", lowercase_text)
        data = [{"text": "HELLO"}, {"text": "WORLD"}]
        result = processor.process(data)
        assert result[0]["text"] == "hello"

    def test_process_single(self):
        processor = DataProcessor()
        processor.add_step("lowercase", lowercase_text)
        item = {"text": "HELLO"}
        result = processor.process_single(item)
        assert result["text"] == "hello"

    def test_remove_step(self):
        processor = DataProcessor()
        processor.add_step("step1", filter_empty)
        processor.add_step("step2", lowercase_text)
        processor.remove_step("step1")
        assert processor.list_steps() == ["step2"]

    def test_clear(self):
        processor = DataProcessor()
        processor.add_step("step1", filter_empty)
        processor.clear()
        assert processor.list_steps() == []


class TestFilterEmpty:
    def test_filters_empty(self):
        data = [{"text": "hello"}, {"text": ""}, {"text": "world"}]
        result = filter_empty(data)
        assert len(result) == 2

    def test_filters_whitespace(self):
        data = [{"text": "hello"}, {"text": "   "}]
        result = filter_empty(data)
        assert len(result) == 1


class TestFilterByLength:
    def test_min_length(self):
        data = [{"text": "hi"}, {"text": "hello world"}]
        result = filter_by_length(data, min_length=5)
        assert len(result) == 1

    def test_max_length(self):
        data = [{"text": "short"}, {"text": "a" * 100}]
        result = filter_by_length(data, max_length=50)
        assert len(result) == 1


class TestNormalizeText:
    def test_normalizes_whitespace(self):
        data = [{"text": "hello   world"}]
        result = normalize_text(data)
        assert result[0]["text"] == "hello world"


class TestLowercaseText:
    def test_lowercase(self):
        data = [{"text": "HELLO World"}]
        result = lowercase_text(data)
        assert result[0]["text"] == "hello world"


class TestDeduplicate:
    def test_removes_duplicates(self):
        data = [{"text": "hello"}, {"text": "hello"}, {"text": "world"}]
        result = deduplicate(data)
        assert len(result) == 2


class TestSampleData:
    def test_sample(self):
        data = [{"text": f"item_{i}"} for i in range(100)]
        result = sample_data(data, n=10)
        assert len(result) == 10

    def test_sample_more_than_available(self):
        data = [{"text": "a"}, {"text": "b"}]
        result = sample_data(data, n=10)
        assert len(result) == 2


class TestRenameKey:
    def test_rename(self):
        data = [{"old": "value"}]
        result = rename_key(data, "old", "new")
        assert "new" in result[0]
        assert "old" not in result[0]
