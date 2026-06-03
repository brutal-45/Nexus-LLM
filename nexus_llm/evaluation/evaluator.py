"""Evaluator for Nexus-LLM models.

High-level API that orchestrates evaluation across generation and chat
tasks, producing :class:`EvaluationReport` instances.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.evaluation.metrics import MetricsCalculator
from nexus_llm.evaluation.report import EvaluationReport

logger = logging.getLogger(__name__)

# A model-like object is expected to have .generate() and/or .chat()
ModelLike = Any


class Evaluator:
    """Evaluate a model on generation and chat tasks.

    Example::

        ev = Evaluator()
        report = ev.evaluate(model, dataset=my_prompts)
        print(report.summary())
    """

    def __init__(self, calculator: Optional[MetricsCalculator] = None) -> None:
        self._calculator = calculator or MetricsCalculator()

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: ModelLike,
        dataset: Optional[List[Dict[str, Any]]] = None,
        model_name: str = "",
    ) -> EvaluationReport:
        """Run a full evaluation of *model* against *dataset*.

        The dataset is a list of dicts with at least a ``"prompt"`` key.
        If a ``"reference"`` key is present, reference-based metrics
        (BLEU, ROUGE) are also computed.

        Args:
            model: A model object with a ``.generate(prompt)`` method.
            dataset: Optional list of evaluation examples.
            model_name: Label for the resulting report.

        Returns:
            An :class:`EvaluationReport` with all computed scores.
        """
        if dataset is None:
            dataset = self._default_dataset()

        model_name = model_name or getattr(model, "name", "unknown")
        report = EvaluationReport(model_name=model_name, dataset_name="custom")

        hypotheses: List[str] = []
        references: List[str] = []

        for item in dataset:
            prompt = item["prompt"]
            try:
                output = model.generate(prompt) if hasattr(model, "generate") else ""
            except Exception:
                logger.warning("Generation failed for prompt: %s", prompt[:80])
                output = ""

            hypotheses.append(output)
            if "reference" in item:
                references.append(item["reference"])

        # Diversity metrics
        report.add_metric("distinct_2", self._calculator.distinct_n(hypotheses, n=2))
        report.add_metric("avg_length", self._calculator.average_length(hypotheses))

        # Reference-based metrics (if references available)
        if references and len(references) == len(hypotheses):
            bleu_scores: List[float] = []
            rouge1_scores: List[float] = []
            for ref, hyp in zip(references, hypotheses):
                bleu_scores.append(self._calculator.bleu_score(ref, hyp))
                rouge1_scores.append(
                    self._calculator.rouge_score(ref, hyp)["rouge1"]
                )
            report.add_metric(
                "bleu",
                sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            )
            report.add_metric(
                "rouge1",
                sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            )

        logger.info(
            "Evaluation of %r complete — %d metric(s)",
            model_name,
            len(report.scores),
        )
        return report

    # ------------------------------------------------------------------
    # Focused evaluation methods
    # ------------------------------------------------------------------

    def evaluate_generation(
        self,
        model: ModelLike,
        prompts: List[str],
        references: Optional[List[str]] = None,
        model_name: str = "",
    ) -> Dict[str, float]:
        """Evaluate text generation quality.

        Args:
            model: Model with a ``.generate(prompt)`` method.
            prompts: List of prompt strings.
            references: Optional parallel list of reference strings.
            model_name: Label for logging.

        Returns:
            Dict of metric name → score.
        """
        hypotheses: List[str] = []
        for prompt in prompts:
            try:
                output = model.generate(prompt) if hasattr(model, "generate") else ""
            except Exception:
                output = ""
            hypotheses.append(output)

        scores: Dict[str, float] = {
            "distinct_2": self._calculator.distinct_n(hypotheses, n=2),
            "avg_length": self._calculator.average_length(hypotheses),
        }

        if references and len(references) == len(hypotheses):
            bleu_list = [
                self._calculator.bleu_score(r, h)
                for r, h in zip(references, hypotheses)
            ]
            rouge_list = [
                self._calculator.rouge_score(r, h)["rouge1"]
                for r, h in zip(references, hypotheses)
            ]
            scores["bleu"] = sum(bleu_list) / len(bleu_list)
            scores["rouge1"] = sum(rouge_list) / len(rouge_list)

        return scores

    def evaluate_chat(
        self,
        model: ModelLike,
        conversations: List[List[Dict[str, str]]],
        references: Optional[List[str]] = None,
        model_name: str = "",
    ) -> Dict[str, float]:
        """Evaluate multi-turn chat quality.

        Args:
            model: Model with a ``.chat(messages)`` method.
            conversations: Each conversation is a list of message dicts
                           (``{"role": ..., "content": ...}``).
            references: Optional list of expected final-response strings.
            model_name: Label for logging.

        Returns:
            Dict of metric name → score.
        """
        hypotheses: List[str] = []
        for conv in conversations:
            try:
                output = model.chat(conv) if hasattr(model, "chat") else ""
            except Exception:
                output = ""
            hypotheses.append(output)

        scores: Dict[str, float] = {
            "distinct_2": self._calculator.distinct_n(hypotheses, n=2),
            "avg_length": self._calculator.average_length(hypotheses),
        }

        if references and len(references) == len(hypotheses):
            bleu_list = [
                self._calculator.bleu_score(r, h)
                for r, h in zip(references, hypotheses)
            ]
            scores["bleu"] = sum(bleu_list) / len(bleu_list)

        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_dataset() -> List[Dict[str, Any]]:
        """Return a minimal placeholder dataset for quick smoke tests."""
        return [
            {"prompt": "What is the capital of France?", "reference": "Paris"},
            {"prompt": "Explain quantum computing.", "reference": "A type of computing that uses quantum mechanics."},
            {"prompt": "Write a haiku about rain.", "reference": "Gentle drops fall down"},
        ]
