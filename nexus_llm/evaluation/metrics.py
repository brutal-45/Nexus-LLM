"""Metrics calculator for Nexus-LLM evaluation.

Provides standard NLP metrics: perplexity, BLEU, ROUGE, distinct-n,
and simple statistics.
"""

import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Stateless utility that computes common NLP metrics.

    All methods are pure functions (no internal state) so a single
    instance can be shared across threads.

    Example::

        calc = MetricsCalculator()
        ppl = calc.perplexity(logits_tensor, labels_tensor)
        bleu = calc.bleu_score("the cat sat", "the dog sat")
    """

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def perplexity(self, logits: List[List[float]], labels: List[int]) -> float:
        """Compute perplexity from logit vectors and ground-truth labels.

        Args:
            logits: 2-D array of shape ``(seq_len, vocab_size)`` — raw
                    logit scores for each position.
            labels: 1-D list of length ``seq_len`` with integer label ids.

        Returns:
            The perplexity score (float).  Returns ``float("inf")`` if
            the total log-probability is zero.
        """
        if len(logits) != len(labels):
            raise ValueError(
                f"Logits length ({len(logits)}) != labels length ({len(labels)})"
            )
        if not logits:
            return float("inf")

        total_log_prob = 0.0
        count = 0

        for step_logits, label in zip(logits, labels):
            if label < 0:
                # Convention: negative labels are ignored (padding / mask)
                continue
            # Numerically stable log-softmax
            max_logit = max(step_logits)
            shifted = [l - max_logit for l in step_logits]
            log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
            log_prob = shifted[label] - log_sum_exp
            total_log_prob += log_prob
            count += 1

        if count == 0:
            return float("inf")

        avg_neg_log_prob = -total_log_prob / count
        return math.exp(avg_neg_log_prob)

    # ------------------------------------------------------------------
    # BLEU
    # ------------------------------------------------------------------

    def bleu_score(
        self,
        reference: str,
        hypothesis: str,
        max_n: int = 4,
    ) -> float:
        """Compute corpus-level BLEU score between *reference* and *hypothesis*.

        A simple brevity penalty is applied.  This is a self-contained
        implementation that does not require ``nltk`` or ``sacrebleu``.

        Args:
            reference: Reference sentence.
            hypothesis: Hypothesis (model output) sentence.
            max_n: Maximum n-gram order (default 4).

        Returns:
            BLEU score in [0, 1].
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()

        if not hyp_tokens:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(hyp_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))

        # Modified precision for each n-gram order
        log_avg = 0.0
        for n in range(1, max_n + 1):
            ref_ngrams = self._ngrams(ref_tokens, n)
            hyp_ngrams = self._ngrams(hyp_tokens, n)
            matches = 0
            clipped = Counter()
            for ng in hyp_ngrams:
                clipped[ng] = min(hyp_ngrams[ng], ref_ngrams.get(ng, 0))
            matches = sum(clipped.values())
            total = max(sum(hyp_ngrams.values()), 1)
            precision = matches / total if total > 0 else 0.0
            if precision == 0:
                return 0.0
            log_avg += math.log(precision)

        return bp * math.exp(log_avg / max_n)

    # ------------------------------------------------------------------
    # ROUGE
    # ------------------------------------------------------------------

    def rouge_score(
        self,
        reference: str,
        hypothesis: str,
    ) -> Dict[str, float]:
        """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

        Args:
            reference: Reference sentence.
            hypothesis: Hypothesis sentence.

        Returns:
            Dict with keys ``rouge1``, ``rouge2``, ``rougeL``.
        """
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()

        rouge1 = self._rouge_n(ref_tokens, hyp_tokens, n=1)
        rouge2 = self._rouge_n(ref_tokens, hyp_tokens, n=2)
        rougeL = self._rouge_l(ref_tokens, hyp_tokens)

        return {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
        }

    # ------------------------------------------------------------------
    # Distinct-n
    # ------------------------------------------------------------------

    def distinct_n(self, texts: List[str], n: int = 2) -> float:
        """Compute Distinct-n diversity metric over a list of texts.

        Distinct-n is the ratio of unique n-grams to total n-grams across
        all texts.  Higher is more diverse.

        Args:
            texts: List of generated texts.
            n: N-gram order (default 2).

        Returns:
            Distinct-n score in [0, 1].
        """
        all_ngrams: List[Tuple[str, ...]] = []
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                all_ngrams.append(tuple(tokens[i : i + n]))

        if not all_ngrams:
            return 0.0

        return len(set(all_ngrams)) / len(all_ngrams)

    # ------------------------------------------------------------------
    # Average length
    # ------------------------------------------------------------------

    def average_length(self, texts: List[str]) -> float:
        """Compute the average token count across *texts*.

        Args:
            texts: List of generated texts.

        Returns:
            Average number of whitespace-split tokens.
        """
        if not texts:
            return 0.0
        return sum(len(t.split()) for t in texts) / len(texts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Counter:
        """Return a Counter of n-grams."""
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    @staticmethod
    def _rouge_n(ref: List[str], hyp: List[str], n: int) -> float:
        """ROUGE-N F1 score."""
        ref_ngrams = MetricsCalculator._ngrams(ref, n)
        hyp_ngrams = MetricsCalculator._ngrams(hyp, n)

        overlap = sum((ref_ngrams & hyp_ngrams).values())
        ref_total = sum(ref_ngrams.values())
        hyp_total = sum(hyp_ngrams.values())

        if ref_total == 0 or hyp_total == 0:
            return 0.0

        precision = overlap / hyp_total
        recall = overlap / ref_total

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _rouge_l(ref: List[str], hyp: List[str]) -> float:
        """ROUGE-L F1 score based on longest common subsequence."""
        m, n = len(ref), len(hyp)
        if m == 0 or n == 0:
            return 0.0

        # DP table for LCS length
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        precision = lcs_len / n
        recall = lcs_len / m

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
