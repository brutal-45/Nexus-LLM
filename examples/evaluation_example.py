#!/usr/bin/env python3
"""
Model Evaluation Example - Nexus-LLM
======================================
Demonstrates how to evaluate model performance on standard benchmarks
and custom evaluation criteria.
"""

from nexus_llm import InferenceEngine
from nexus_llm.evaluation import (
    Evaluator,
    EvalConfig,
    BenchmarkDataset,
    CustomMetric,
)


# --- Define a custom evaluation metric ---

class ConcisenessMetric(CustomMetric):
    """Measures how concise the model's responses are."""

    name = "conciseness"
    description = "Ratio of essential information to total response length"

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        """Return a score between 0 and 1, where 1 is perfectly concise."""
        if not prediction:
            return 0.0
        # Simple heuristic: compare length ratio with information overlap
        essential_length = min(len(reference), len(prediction))
        return essential_length / max(len(prediction), 1)


class FactualAccuracyMetric(CustomMetric):
    """Checks factual consistency between prediction and reference."""

    name = "factual_accuracy"
    description = "Factual consistency score using entailment checking"

    def compute(self, prediction: str, reference: str, **kwargs) -> float:
        # In practice, this would use an NLI model
        # Here we return a placeholder
        return 0.85


def main():
    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

    # --- Configure evaluation ---
    eval_config = EvalConfig(
        batch_size=8,
        max_samples=None,            # Evaluate on full dataset
        temperature=0.0,             # Deterministic for evaluation
        max_tokens=256,
        seed=42,
        output_dir="./eval_results",
    )

    evaluator = Evaluator(
        engine=engine,
        config=eval_config,
    )

    # --- Evaluate on standard benchmarks ---
    print("=" * 60)
    print("Standard Benchmark Evaluation")
    print("=" * 60)

    # MMLU (Massive Multitask Language Understanding)
    mmlu_results = evaluator.evaluate_benchmark(
        benchmark=BenchmarkDataset.MMLU,
        subsets=["stem", "humanities", "social_sciences"],
    )
    print(f"\nMMLU Results:")
    print(f"  Overall accuracy: {mmlu_results.overall_score:.3f}")
    for subset, score in mmlu_results.subset_scores.items():
        print(f"  {subset}: {score:.3f}")

    # HumanEval (Code generation)
    humaneval_results = evaluator.evaluate_benchmark(
        benchmark=BenchmarkDataset.HUMAN_EVAL,
        metric="pass@k",
        k_values=[1, 10, 100],
    )
    print(f"\nHumanEval Results:")
    for k, score in humaneval_results.scores.items():
        print(f"  pass@{k}: {score:.3f}")

    # --- Evaluate with custom metrics ---
    print("\n" + "=" * 60)
    print("Custom Metric Evaluation")
    print("=" * 60)

    evaluator.register_metric(ConcisenessMetric())
    evaluator.register_metric(FactualAccuracyMetric())

    custom_results = evaluator.evaluate_custom(
        dataset_path="./data/eval_qa_pairs.jsonl",
        input_field="question",
        reference_field="answer",
        metrics=["rouge_l", "bleu", "bertscore", "conciseness", "factual_accuracy"],
    )

    print(f"\nCustom Evaluation Results:")
    for metric_name, score in custom_results.metric_scores.items():
        print(f"  {metric_name}: {score:.3f}")

    # --- Compare models ---
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    models = ["nexus-3b-chat", "nexus-7b-chat", "nexus-13b-chat"]
    comparison = {}

    for model_name in models:
        engine = InferenceEngine(model_name=model_name, device="auto")
        evaluator = Evaluator(engine=engine, config=eval_config)
        results = evaluator.evaluate_benchmark(BenchmarkDataset.MMLU)
        comparison[model_name] = results.overall_score

    print(f"\n{'Model':<20} {'MMLU Accuracy':<15}")
    print("-" * 35)
    for model, score in comparison.items():
        print(f"{model:<20} {score:<15.3f}")

    # --- Generate evaluation report ---
    eval_config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = f"{eval_config.output_dir}/eval_report.html"
    custom_results.generate_report(
        output_path=report_path,
        include_examples=True,
        num_examples=5,
    )
    print(f"\nEvaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
