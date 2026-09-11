"""Evaluation module for Nexus-LLM.

Provides model evaluation, benchmarking, metrics computation,
reporting, and comparison utilities.
"""

from nexus_llm.evaluation.benchmark import BenchmarkConfig, BenchmarkResult, BenchmarkRunner
from nexus_llm.evaluation.comparison import ComparisonEngine, ComparisonResult
from nexus_llm.evaluation.evaluator import Evaluator
from nexus_llm.evaluation.generation_eval import GenerationEvaluator, GenerationQualityResult
from nexus_llm.evaluation.metrics import MetricsCalculator, tokenize
from nexus_llm.evaluation.perplexity import PerplexityCalculator, PerplexityResult
from nexus_llm.evaluation.report import EvaluationReport

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "GenerationEvaluator",
    "GenerationQualityResult",
    "PerplexityCalculator",
    "PerplexityResult",
    "tokenize",
    "ComparisonEngine",
    "ComparisonResult",
    "EvaluationReport",
    "Evaluator",
    "MetricsCalculator",
]
