"""
Nexus-LLM Evaluation Module

Provides comprehensive evaluation capabilities for language models including:
- Standard NLP benchmarks (MMLU, HellaSwag, ARC, WinoGrande)
- Evaluation metrics (BLEU, ROUGE, accuracy, F1, exact match, BERTScore)
- Perplexity calculation (sequence-level, token-level, sliding window)
- Generation quality assessment (diversity, coherence, relevance, fluency)
- Report generation (HTML, JSON, text) with visualization and comparison
"""

from nexus_llm.evaluation.evaluator import Evaluator, EvaluationResult, ModelComparison
from nexus_llm.evaluation.benchmarks import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
from nexus_llm.evaluation.metrics import (
    BLEUScore,
    ROUGEScore,
    Accuracy,
    F1Score,
    ExactMatch,
    BERTScore,
    MetricRegistry,
)
from nexus_llm.evaluation.perplexity import PerplexityCalculator, PerplexityResult
from nexus_llm.evaluation.generation_eval import GenerationEvaluator, GenerationQualityResult
from nexus_llm.evaluation.report import ReportGenerator, ReportFormat

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "ModelComparison",
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BLEUScore",
    "ROUGEScore",
    "Accuracy",
    "F1Score",
    "ExactMatch",
    "BERTScore",
    "MetricRegistry",
    "PerplexityCalculator",
    "PerplexityResult",
    "GenerationEvaluator",
    "GenerationQualityResult",
    "ReportGenerator",
    "ReportFormat",
]
