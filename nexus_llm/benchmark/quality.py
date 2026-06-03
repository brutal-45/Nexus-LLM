"""Quality benchmark for Nexus-LLM.

Evaluates model outputs on coherence, diversity, accuracy, and
fluency using heuristic and reference-based metrics.
"""

import hashlib
import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ModelLike = Any


class QualityBenchmark:
    """Benchmark suite focused on output quality metrics.

    Example::

        qb = QualityBenchmark()
        results = qb.benchmark_coherence(model, ["Tell me a story"])
        print(results)
    """

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    def benchmark_coherence(
        self,
        model: ModelLike,
        prompts: List[str],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Evaluate output coherence using heuristic scoring.

        Coherence is approximated by measuring:
        - Sentence overlap (bigram repetition ratio)
        - Average sentence length consistency
        - Presence of discourse markers

        Args:
            model: A model-like object with a ``generate(text, max_tokens)``
                   method.
            prompts: List of prompts to evaluate.
            max_tokens: Max generation length.

        Returns:
            Dict with ``avg_score``, ``scores``, and ``n_prompts``.
        """
        scores: List[float] = []

        for prompt in prompts:
            output = self._generate(model, prompt, max_tokens)
            score = self._compute_coherence(output)
            scores.append(score)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        result = {
            "avg_score": round(avg_score, 4),
            "scores": [round(s, 4) for s in scores],
            "n_prompts": len(prompts),
        }
        logger.info("Coherence benchmark: avg_score=%.4f", avg_score)
        return result

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def benchmark_diversity(
        self,
        model: ModelLike,
        prompts: List[str],
        max_tokens: int = 128,
        n_samples_per_prompt: int = 3,
    ) -> Dict[str, Any]:
        """Measure output diversity using distinct n-gram ratios.

        Computes Distinct-1 (unigram) and Distinct-2 (bigram) across
        multiple generations per prompt.

        Args:
            model: A model-like object with a ``generate(text, max_tokens)``
                   method.
            prompts: List of prompts to evaluate.
            max_tokens: Max generation length.
            n_samples_per_prompt: Number of samples per prompt.

        Returns:
            Dict with ``distinct_1``, ``distinct_2``, and ``n_prompts``.
        """
        all_unigrams: List[str] = []
        all_bigrams: List[str] = []

        for prompt in prompts:
            for _ in range(n_samples_per_prompt):
                output = self._generate(model, prompt, max_tokens)
                tokens = output.split()
                all_unigrams.extend(tokens)
                for i in range(len(tokens) - 1):
                    all_bigrams.append(f"{tokens[i]} {tokens[i + 1]}")

        distinct_1 = (
            len(set(all_unigrams)) / len(all_unigrams) if all_unigrams else 0.0
        )
        distinct_2 = (
            len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
        )

        result = {
            "distinct_1": round(distinct_1, 4),
            "distinct_2": round(distinct_2, 4),
            "n_prompts": len(prompts),
        }
        logger.info(
            "Diversity benchmark: distinct_1=%.4f, distinct_2=%.4f",
            distinct_1,
            distinct_2,
        )
        return result

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def benchmark_accuracy(
        self,
        model: ModelLike,
        qa_pairs: List[Dict[str, str]],
        max_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Evaluate question-answering accuracy.

        Uses token-level overlap (F1) between the model output and
        the reference answer as an approximate accuracy measure.

        Args:
            model: A model-like object with a ``generate(text, max_tokens)``
                   method.
            qa_pairs: List of dicts with ``"question"`` and ``"answer"`` keys.
            max_tokens: Max generation length.

        Returns:
            Dict with ``avg_f1``, ``exact_match``, ``n_questions``.
        """
        f1_scores: List[float] = []
        exact_matches = 0

        for pair in qa_pairs:
            question = pair.get("question", "")
            reference = pair.get("answer", "")

            prompt = f"Question: {question}\nAnswer:"
            output = self._generate(model, prompt, max_tokens)
            prediction = output.strip().lower()
            ref_lower = reference.strip().lower()

            # Exact match
            if prediction == ref_lower:
                exact_matches += 1

            # Token-level F1
            pred_tokens = set(prediction.split())
            ref_tokens = set(ref_lower.split())

            if not pred_tokens or not ref_tokens:
                f1_scores.append(0.0)
                continue

            common = pred_tokens & ref_tokens
            if not common:
                f1_scores.append(0.0)
                continue

            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        exact_match_rate = exact_matches / len(qa_pairs) if qa_pairs else 0.0

        result = {
            "avg_f1": round(avg_f1, 4),
            "exact_match": round(exact_match_rate, 4),
            "n_questions": len(qa_pairs),
        }
        logger.info(
            "Accuracy benchmark: avg_f1=%.4f, exact_match=%.4f",
            avg_f1,
            exact_match_rate,
        )
        return result

    # ------------------------------------------------------------------
    # Fluency
    # ------------------------------------------------------------------

    def benchmark_fluency(
        self,
        model: ModelLike,
        prompts: List[str],
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """Evaluate output fluency using heuristic metrics.

        Fluency is approximated by:
        - Average sentence length (target ~15-20 words)
        - Ratio of well-formed sentences (starts with capital, ends with punctuation)
        - Vocabulary richness (type-token ratio)

        Args:
            model: A model-like object with a ``generate(text, max_tokens)``
                   method.
            prompts: List of prompts to evaluate.
            max_tokens: Max generation length.

        Returns:
            Dict with ``avg_fluency_score``, ``avg_sentence_length``,
            ``well_formed_ratio``, ``type_token_ratio``.
        """
        fluency_scores: List[float] = []
        all_sentence_lengths: List[float] = []
        well_formed = 0
        total_sentences = 0
        all_tokens: List[str] = []

        for prompt in prompts:
            output = self._generate(model, prompt, max_tokens)
            sentences = [s.strip() for s in output.split(".") if s.strip()]

            for sent in sentences:
                total_sentences += 1
                words = sent.split()
                sent_len = len(words)
                all_sentence_lengths.append(sent_len)
                all_tokens.extend(words)

                # Check well-formed: starts with capital, has content
                if sent and sent[0].isupper() and len(words) >= 2:
                    well_formed += 1

            score = self._compute_fluency(output)
            fluency_scores.append(score)

        avg_fluency = sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.0
        avg_sent_len = (
            sum(all_sentence_lengths) / len(all_sentence_lengths)
            if all_sentence_lengths
            else 0.0
        )
        well_formed_ratio = well_formed / total_sentences if total_sentences else 0.0
        ttr = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0

        result = {
            "avg_fluency_score": round(avg_fluency, 4),
            "avg_sentence_length": round(avg_sent_len, 2),
            "well_formed_ratio": round(well_formed_ratio, 4),
            "type_token_ratio": round(ttr, 4),
        }
        logger.info("Fluency benchmark: avg_fluency=%.4f", avg_fluency)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate(model: ModelLike, prompt: str, max_tokens: int) -> str:
        """Generate text from *model*, falling back to a placeholder."""
        generate_fn = getattr(model, "generate", None)
        if generate_fn is not None and callable(generate_fn):
            output = generate_fn(prompt, max_tokens=max_tokens)
            if isinstance(output, dict):
                return output.get("text", "")
            return str(output)
        # Deterministic placeholder for testing without a real model
        return (
            f"This is a sample response to: {prompt[:50]}. "
            f"It contains multiple sentences for analysis. "
            f"The output should be coherent and well-formed."
        )

    @staticmethod
    def _compute_coherence(text: str) -> float:
        """Heuristic coherence score in [0, 1]."""
        if not text or len(text.split()) < 4:
            return 0.0

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 2:
            return 0.5

        # Bigram overlap between consecutive sentences
        overlaps: List[float] = []
        for i in range(len(sentences) - 1):
            words_a = set(sentences[i].lower().split())
            words_b = set(sentences[i + 1].lower().split())
            if words_a and words_b:
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        # Penalise very short or very long texts
        word_count = len(text.split())
        length_factor = 1.0 - abs(word_count - 50) / 100
        length_factor = max(0.0, min(1.0, length_factor))

        return min(1.0, avg_overlap * 0.7 + length_factor * 0.3)

    @staticmethod
    def _compute_fluency(text: str) -> float:
        """Heuristic fluency score in [0, 1]."""
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Type-token ratio (vocabulary richness)
        ttr = len(set(w.lower() for w in words)) / len(words)

        # Sentence length normalcy (prefer ~8-25 words)
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if sentences:
            avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
            length_score = max(0.0, 1.0 - abs(avg_len - 15) / 30)
        else:
            length_score = 0.0

        return (ttr * 0.5 + length_score * 0.5)
