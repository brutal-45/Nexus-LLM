"""Training evaluation: validation loss, perplexity, generation quality, downstream tasks."""

import math
import logging
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates model performance with various metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None,
        fp16: bool = False,
    ):
        self.model = model
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.fp16 = fp16
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_loss(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate model loss on a dataset.

        Args:
            dataloader: DataLoader for evaluation data.
            max_batches: Maximum number of batches to evaluate.

        Returns:
            Dictionary with loss and perplexity metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        batch_count = 0

        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            batch_size = batch.get("input_ids", batch.get("labels", None)).shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            batch_count += 1

        avg_loss = total_loss / max(total_samples, 1)
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_samples": total_samples,
        }

    @torch.no_grad()
    def evaluate_accuracy(
        self,
        dataloader: DataLoader,
        ignore_index: int = -100,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate token-level accuracy.

        Args:
            dataloader: DataLoader for evaluation data.
            ignore_index: Label value to ignore in accuracy computation.
            max_batches: Maximum number of batches.

        Returns:
            Dictionary with accuracy metrics.
        """
        self.model.eval()
        correct = 0
        total = 0
        batch_count = 0

        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
            labels = batch["labels"]

            predictions = logits.argmax(dim=-1)
            mask = labels != ignore_index
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
            batch_count += 1

        accuracy = correct / max(total, 1)
        return {
            "eval_accuracy": accuracy,
            "eval_correct": correct,
            "eval_total_tokens": total,
        }

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: List[str],
        tokenizer: Any,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        num_beams: int = 1,
        do_sample: bool = True,
    ) -> List[Dict[str, str]]:
        """Generate text samples for quality evaluation.

        Args:
            prompts: List of prompt strings.
            tokenizer: Tokenizer for encoding/decoding.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            num_beams: Number of beams for beam search.
            do_sample: Whether to sample.

        Returns:
            List of dictionaries with prompt and generated text.
        """
        self.model.eval()
        results = []

        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "num_beams": num_beams,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }

            if hasattr(self.model, "generate"):
                output_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_text = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
            else:
                generated_text = ""

            results.append({
                "prompt": prompt,
                "generated": generated_text,
            })

        return results

    def evaluate_generation_quality(
        self,
        prompts: List[str],
        references: Optional[List[str]] = None,
        tokenizer: Any = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate generation quality with various metrics.

        Args:
            prompts: List of prompts.
            references: Optional reference texts for comparison.
            tokenizer: Tokenizer.
            max_new_tokens: Max tokens to generate.

        Returns:
            Dictionary with generation quality metrics.
        """
        if tokenizer is None:
            return {"error": "Tokenizer required for generation evaluation"}

        samples = self.generate_samples(
            prompts, tokenizer, max_new_tokens=max_new_tokens
        )

        metrics: Dict[str, Any] = {
            "num_samples": len(samples),
            "avg_length": 0.0,
            "min_length": 0.0,
            "max_length": 0.0,
        }

        lengths = [len(s["generated"].split()) for s in samples if s["generated"]]
        if lengths:
            metrics["avg_length"] = sum(lengths) / len(lengths)
            metrics["min_length"] = min(lengths)
            metrics["max_length"] = max(lengths)

        if references and len(references) == len(samples):
            rouge_scores = self._compute_rouge_approx(
                [s["generated"] for s in samples], references
            )
            metrics["rouge"] = rouge_scores

        return metrics

    @staticmethod
    def _compute_rouge_approx(
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute approximate ROUGE scores (unigram overlap)."""
        def get_unigrams(text: str) -> set:
            return set(text.lower().split())

        rouge_1_scores = []
        rouge_2_scores = []

        for pred, ref in zip(predictions, references):
            pred_unigrams = get_unigrams(pred)
            ref_unigrams = get_unigrams(ref)

            if not ref_unigrams:
                continue

            overlap = pred_unigrams & ref_unigrams
            precision = len(overlap) / max(len(pred_unigrams), 1)
            recall = len(overlap) / max(len(ref_unigrams), 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)
            rouge_1_scores.append(f1)

            pred_bigrams = set(
                zip(pred.lower().split(), pred.lower().split()[1:])
            )
            ref_bigrams = set(
                zip(ref.lower().split(), ref.lower().split()[1:])
            )
            if ref_bigrams:
                overlap_2 = pred_bigrams & ref_bigrams
                precision_2 = len(overlap_2) / max(len(pred_bigrams), 1)
                recall_2 = len(overlap_2) / max(len(ref_bigrams), 1)
                f1_2 = 2 * precision_2 * recall_2 / max(precision_2 + recall_2, 1e-10)
                rouge_2_scores.append(f1_2)

        return {
            "rouge-1": sum(rouge_1_scores) / max(len(rouge_1_scores), 1),
            "rouge-2": sum(rouge_2_scores) / max(len(rouge_2_scores), 1),
        }

    def evaluate_downstream(
        self,
        task_name: str,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on a downstream task.

        Args:
            task_name: Name of the downstream task.
            dataloader: DataLoader for the task.
            max_batches: Maximum batches to evaluate.

        Returns:
            Dictionary with task-specific metrics.
        """
        if task_name in ("classification", "cls"):
            return self._evaluate_classification(dataloader, max_batches)
        elif task_name in ("generation", "gen"):
            return self._evaluate_generation(dataloader, max_batches)
        else:
            return self.evaluate_loss(dataloader, max_batches)

    def _evaluate_classification(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on a classification task."""
        self.model.eval()
        correct = 0
        total = 0
        batch_count = 0

        for batch in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs.get("logits")
            labels = batch.get("labels")

            if labels is not None and logits is not None:
                predictions = logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.shape[0]

            batch_count += 1

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "total_samples": total}

    def _evaluate_generation(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate on a generation task using loss."""
        return self.evaluate_loss(dataloader, max_batches)

    def full_evaluation(
        self,
        eval_dataloader: DataLoader,
        tokenizer: Optional[Any] = None,
        sample_prompts: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a comprehensive evaluation.

        Args:
            eval_dataloader: DataLoader for evaluation data.
            tokenizer: Optional tokenizer for generation evaluation.
            sample_prompts: Optional prompts for generation quality.
            max_batches: Maximum number of batches.

        Returns:
            Complete evaluation results dictionary.
        """
        results = {}

        loss_metrics = self.evaluate_loss(eval_dataloader, max_batches)
        results.update(loss_metrics)

        accuracy_metrics = self.evaluate_accuracy(eval_dataloader, max_batches=max_batches)
        results.update(accuracy_metrics)

        if tokenizer and sample_prompts:
            gen_metrics = self.evaluate_generation_quality(
                sample_prompts, tokenizer=tokenizer
            )
            results["generation"] = gen_metrics

        return results
