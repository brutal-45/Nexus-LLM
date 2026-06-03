"""Test eval reports for Nexus-LLM."""
import json
import time
import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path


@dataclass
class EvalReportEntry:
    metric_name: str
    value: float
    description: str = ""
    higher_is_better: bool = True


@dataclass
class EvalReport:
    model_name: str
    timestamp: float = 0.0
    entries: List[EvalReportEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def add_entry(self, entry: EvalReportEntry):
        self.entries.append(entry)

    def get_entry(self, metric_name: str) -> Optional[EvalReportEntry]:
        for entry in self.entries:
            if entry.metric_name == metric_name:
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "metrics": [
                {"name": e.metric_name, "value": e.value, "description": e.description}
                for e in self.entries
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        lines = [f"Evaluation Report: {self.model_name}", "=" * 40]
        for entry in self.entries:
            lines.append(f"  {entry.metric_name}: {entry.value:.4f}")
        return "\n".join(lines)

    @property
    def num_metrics(self):
        return len(self.entries)


class ReportGenerator:
    def __init__(self):
        self._reports: List[EvalReport] = []

    def create_report(self, model_name: str, metrics: Dict[str, float], metadata: Dict = None) -> EvalReport:
        report = EvalReport(model_name=model_name, metadata=metadata or {})
        for name, value in metrics.items():
            report.add_entry(EvalReportEntry(metric_name=name, value=value))
        self._reports.append(report)
        return report

    def compare_reports(self, reports: List[EvalReport]) -> Dict[str, Any]:
        if not reports:
            return {}
        comparison = {}
        all_metrics = set()
        for report in reports:
            for entry in report.entries:
                all_metrics.add(entry.metric_name)

        for metric in all_metrics:
            values = {}
            for report in reports:
                entry = report.get_entry(metric)
                if entry:
                    values[report.model_name] = entry.value
            comparison[metric] = values
        return comparison

    def save_report(self, report: EvalReport, path: str):
        with open(path, "w") as f:
            f.write(report.to_json())

    def load_report(self, path: str) -> EvalReport:
        with open(path, "r") as f:
            data = json.load(f)
        report = EvalReport(model_name=data["model_name"], metadata=data.get("metadata", {}))
        for m in data.get("metrics", []):
            report.add_entry(EvalReportEntry(metric_name=m["name"], value=m["value"], description=m.get("description", "")))
        return report

    def get_reports(self) -> List[EvalReport]:
        return list(self._reports)


class TestEvalReportEntry:
    def test_creation(self):
        entry = EvalReportEntry(metric_name="accuracy", value=0.85)
        assert entry.metric_name == "accuracy"
        assert entry.higher_is_better is True

    def test_custom(self):
        entry = EvalReportEntry(metric_name="loss", value=2.5, higher_is_better=False)
        assert entry.higher_is_better is False


class TestEvalReport:
    def test_creation(self):
        report = EvalReport(model_name="test-model")
        assert report.model_name == "test-model"
        assert report.num_metrics == 0

    def test_add_entry(self):
        report = EvalReport(model_name="test")
        report.add_entry(EvalReportEntry(metric_name="accuracy", value=0.9))
        assert report.num_metrics == 1

    def test_get_entry(self):
        report = EvalReport(model_name="test")
        report.add_entry(EvalReportEntry(metric_name="accuracy", value=0.9))
        entry = report.get_entry("accuracy")
        assert entry is not None
        assert entry.value == 0.9

    def test_get_nonexistent(self):
        report = EvalReport(model_name="test")
        assert report.get_entry("nonexistent") is None

    def test_to_dict(self):
        report = EvalReport(model_name="test")
        report.add_entry(EvalReportEntry(metric_name="f1", value=0.8))
        d = report.to_dict()
        assert d["model_name"] == "test"
        assert len(d["metrics"]) == 1

    def test_to_json(self):
        report = EvalReport(model_name="test")
        report.add_entry(EvalReportEntry(metric_name="f1", value=0.8))
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["model_name"] == "test"

    def test_summary(self):
        report = EvalReport(model_name="test")
        report.add_entry(EvalReportEntry(metric_name="accuracy", value=0.85))
        s = report.summary()
        assert "test" in s
        assert "accuracy" in s


class TestReportGenerator:
    def test_create_report(self):
        gen = ReportGenerator()
        report = gen.create_report("model-a", {"accuracy": 0.9, "f1": 0.85})
        assert report.num_metrics == 2

    def test_compare_reports(self):
        gen = ReportGenerator()
        r1 = gen.create_report("model-a", {"accuracy": 0.9})
        r2 = gen.create_report("model-b", {"accuracy": 0.8})
        comparison = gen.compare_reports([r1, r2])
        assert "accuracy" in comparison
        assert comparison["accuracy"]["model-a"] == 0.9

    def test_save_and_load(self, tmp_dir):
        gen = ReportGenerator()
        report = gen.create_report("test-model", {"accuracy": 0.95}, metadata={"seed": 42})
        path = str(tmp_dir / "report.json")
        gen.save_report(report, path)
        loaded = gen.load_report(path)
        assert loaded.model_name == "test-model"
        assert loaded.get_entry("accuracy").value == 0.95

    def test_get_reports(self):
        gen = ReportGenerator()
        gen.create_report("a", {"x": 1.0})
        gen.create_report("b", {"y": 2.0})
        assert len(gen.get_reports()) == 2
