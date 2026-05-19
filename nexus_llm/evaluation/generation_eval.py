"""
Generation Quality Evaluation Module

Assesses the quality of generated text across four dimensions:
- Diversity: lexical variety and n-gram diversity
- Coherence: intra-sentence and inter-sentence coherence
- Relevance: semantic overlap with reference/prompt
- Fluency: language model-style scoring of naturalness

Each dimension produces a score in [0, 1] and an overall quality score.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nexus_llm.evaluation.metrics import _tokenize, _normalize_text


@dataclass
class GenerationQualityResult:
    """Container for generation quality scores."""
    diversity: float
    coherence: float
    relevance: float
    fluency: float
    overall_quality: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diversity": self.diversity,
            "coherence": self.coherence,
            "relevance": self.relevance,
            "fluency": self.fluency,
            "overall_quality": self.overall_quality,
            "details": self.details,
        }

    def summary(self) -> str:
        return (
            f"Generation Quality Report\n"
            f"  Diversity:  {self.diversity:.4f}\n"
            f"  Coherence:  {self.coherence:.4f}\n"
            f"  Relevance:  {self.relevance:.4f}\n"
            f"  Fluency:    {self.fluency:.4f}\n"
            f"  Overall:    {self.overall_quality:.4f}"
        )


class GenerationEvaluator:
    """
    Evaluate generation quality across multiple dimensions.

    Each scoring function operates on one or more generated texts and
    optional references.  When references are available, relevance is
    measured against them; otherwise, it falls back to self-assessment
    heuristics.
    """

    def __init__(
        self,
        diversity_weight: float = 0.25,
        coherence_weight: float = 0.25,
        relevance_weight: float = 0.25,
        fluency_weight: float = 0.25,
    ):
        total = diversity_weight + coherence_weight + relevance_weight + fluency_weight
        self.diversity_weight = diversity_weight / total
        self.coherence_weight = coherence_weight / total
        self.relevance_weight = relevance_weight / total
        self.fluency_weight = fluency_weight / total

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    @staticmethod
    def _distinct_ngrams(tokens: List[str], n: int) -> float:
        """Ratio of unique n-grams to total n-grams."""
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def _type_token_ratio(tokens: List[str]) -> float:
        """Type-Token Ratio: unique tokens / total tokens."""
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    def _score_diversity(self, text: str) -> float:
        """
        Compute diversity score for a single generated text.

        Combines:
        - Type-Token Ratio (unigram diversity)
        - Distinct-2 (bigram diversity)
        - Distinct-3 (trigram diversity)
        """
        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        ttr = self._type_token_ratio(tokens)
        distinct2 = self._distinct_ngrams(tokens, 2)
        distinct3 = self._distinct_ngrams(tokens, 3)

        # Weighted combination — TTR is most important
        return 0.4 * ttr + 0.35 * distinct2 + 0.25 * distinct3

    def _score_corpus_diversity(self, texts: Sequence[str]) -> float:
        """
        Compute corpus-level diversity.

        Measures inter-text variety: how different are the generated texts
        from each other.
        """
        if len(texts) < 2:
            return self._score_diversity(texts[0]) if texts else 0.0

        # Average pairwise Jaccard distance (1 - Jaccard similarity)
        token_sets = [set(_tokenize(t)) for t in texts]
        total_dist = 0.0
        pairs = 0
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                si, sj = token_sets[i], token_sets[j]
                if not si and not sj:
                    continue
                union = len(si | sj)
                if union == 0:
                    continue
                jaccard = len(si & sj) / union
                total_dist += (1 - jaccard)
                pairs += 1

        inter_text_diversity = total_dist / pairs if pairs > 0 else 0.0
        intra_text_diversity = sum(self._score_diversity(t) for t in texts) / len(texts)

        return 0.5 * intra_text_diversity + 0.5 * inter_text_diversity

    # ------------------------------------------------------------------
    # Coherence
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _sentence_overlap(s1: str, s2: str) -> float:
        """Compute token overlap between two sentences."""
        t1 = set(_tokenize(s1))
        t2 = set(_tokenize(s2))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / min(len(t1), len(t2))

    def _score_coherence(self, text: str) -> float:
        """
        Compute coherence score based on adjacent sentence overlap.

        Higher overlap between consecutive sentences suggests better
        discourse coherence.
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return 1.0 if sentences else 0.0

        overlaps = []
        for i in range(len(sentences) - 1):
            overlap = self._sentence_overlap(sentences[i], sentences[i + 1])
            overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps)

        # Penalize too much repetition (overlap > 0.9 means likely duplicate sentences)
        repetition_penalty = sum(1 for o in overlaps if o > 0.9) / len(overlaps)
        adjusted = avg_overlap - 0.3 * repetition_penalty

        return max(0.0, min(1.0, adjusted))

    # ------------------------------------------------------------------
    # Relevance
    # ------------------------------------------------------------------

    def _score_relevance(self, prediction: str, reference: str) -> float:
        """
        Compute relevance of prediction to reference.

        Uses token-level F1 as a proxy for semantic relevance.
        """
        pred_tokens = set(_tokenize(prediction))
        ref_tokens = set(_tokenize(reference))

        if not ref_tokens:
            return 0.0
        if not pred_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _score_relevance_self(self, text: str) -> float:
        """
        Fallback relevance score when no reference is available.

        Estimates relevance by checking how focused the text is —
        lower entropy over tokens suggests a more focused text.
        """
        tokens = _tokenize(text)
        if not tokens:
            return 0.0

        counter = Counter(tokens)
        total = len(tokens)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize: max entropy for uniform distribution over all tokens
        max_entropy = math.log2(len(counter)) if len(counter) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Lower entropy → more focused → higher relevance
        # Map [0, 1] entropy to relevance: high when low entropy
        relevance = 1.0 - 0.5 * normalized_entropy
        return max(0.0, min(1.0, relevance))

    # ------------------------------------------------------------------
    # Fluency
    # ------------------------------------------------------------------

    @staticmethod
    def _score_fluency(text: str) -> float:
        """
        Estimate fluency using simple heuristics:

        - Average sentence length (too short or too long is penalized)
        - Ratio of complete sentences (ending with .!? )
        - Punctuation balance
        - Absence of repeated characters/words
        """
        if not text.strip():
            return 0.0

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        # 1. Average sentence length — ideal around 10-20 words
        sent_lengths = [len(_tokenize(s)) for s in sentences]
        avg_len = sum(sent_lengths) / len(sent_lengths)
        len_score = 1.0 - abs(avg_len - 15) / 30  # peak at 15 words
        len_score = max(0.0, min(1.0, len_score))

        # 2. Sentence completeness — ratio ending with punctuation
        complete = sum(1 for s in sentences if s[-1] in ".!?") / len(sentences)

        # 3. Punctuation balance (not too many commas/periods)
        total_chars = len(text)
        if total_chars > 0:
            punct_ratio = sum(1 for c in text if c in ".,;:!?'\"") / total_chars
            punct_score = 1.0 - abs(punct_ratio - 0.05) / 0.15
            punct_score = max(0.0, min(1.0, punct_score))
        else:
            punct_score = 0.0

        # 4. Repetition penalty
        tokens = _tokenize(text)
        if len(tokens) > 1:
            bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            bigram_counts = Counter(bigrams)
            max_repeat = max(bigram_counts.values())
            repeat_penalty = min(1.0, max_repeat / 3)  # penalize if a bigram repeats >3 times
            repetition_score = 1.0 - 0.5 * repeat_penalty
        else:
            repetition_score = 1.0

        # Weighted combination
        fluency = (
            0.25 * len_score
            + 0.30 * complete
            + 0.20 * punct_score
            + 0.25 * repetition_score
        )
        return max(0.0, min(1.0, fluency))

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predictions: Sequence[str],
        references: Optional[Sequence[str]] = None,
    ) -> GenerationQualityResult:
        """
        Evaluate generation quality across all dimensions.

        Args:
            predictions: Generated texts.
            references: Optional ground-truth texts for relevance scoring.

        Returns:
            GenerationQualityResult with per-dimension and overall scores.
        """
        if not predictions:
            return GenerationQualityResult(
                diversity=0.0,
                coherence=0.0,
                relevance=0.0,
                fluency=0.0,
                overall_quality=0.0,
            )

        # Diversity
        diversity = self._score_corpus_diversity(predictions)

        # Coherence (average across predictions)
        coherence_scores = [self._score_coherence(p) for p in predictions]
        coherence = sum(coherence_scores) / len(coherence_scores)

        # Relevance
        if references is not None:
            rel_scores = [
                self._score_relevance(p, r)
                for p, r in zip(predictions, references)
            ]
            relevance = sum(rel_scores) / len(rel_scores)
        else:
            rel_scores = [self._score_relevance_self(p) for p in predictions]
            relevance = sum(rel_scores) / len(rel_scores)

        # Fluency
        fluency_scores = [self._score_fluency(p) for p in predictions]
        fluency = sum(fluency_scores) / len(fluency_scores)

        # Overall weighted score
        overall = (
            self.diversity_weight * diversity
            + self.coherence_weight * coherence
            + self.relevance_weight * relevance
            + self.fluency_weight * fluency
        )

        details = {
            "per_prediction": [
                {
                    "index": i,
                    "diversity": self._score_diversity(predictions[i]),
                    "coherence": coherence_scores[i],
                    "relevance": rel_scores[i],
                    "fluency": fluency_scores[i],
                }
                for i in range(len(predictions))
            ],
            "num_predictions": len(predictions),
            "has_references": references is not None,
        }

        return GenerationQualityResult(
            diversity=diversity,
            coherence=coherence,
            relevance=relevance,
            fluency=fluency,
            overall_quality=overall,
            details=details,
        )

    def evaluate_single(
        self,
        prediction: str,
        reference: Optional[str] = None,
    ) -> GenerationQualityResult:
        """
        Evaluate a single generated text.

        Convenience wrapper around ``evaluate``.
        """
        refs = [reference] if reference is not None else None
        return self.evaluate([prediction], refs)
