"""Evaluation module for Nexus-LLM.

Provides model evaluation, benchmarking, metrics computation,
reporting, and comparison utilities.
"""

from nexus_llm.evaluation.evaluator import Evaluator
from nexus_llm.evaluation.benchmark import BenchmarkRunner
from nexus_llm.evaluation.metrics import MetricsCalculator
from nexus_llm.evaluation.report import EvaluationReport
from nexus_llm.evaluation.comparison import ComparisonEngine, ComparisonResult

__all__ = [
    "Evaluator",
    "BenchmarkRunner",
    "MetricsCalculator",
    "EvaluationReport",
    "ComparisonEngine",
    "ComparisonResult",
]
