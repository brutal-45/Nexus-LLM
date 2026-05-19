"""
Evaluation Metrics Module

Implements standard NLP evaluation metrics:
- BLEU (Bilingual Evaluation Understudy) with up to 4-gram precision
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation) — ROUGE-1, ROUGE-2, ROUGE-L
- Accuracy for classification tasks
- F1 Score (micro, macro, per-class)
- Exact Match
- BERTScore (simulated via token overlap when transformers unavailable)
"""

import math
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union


# ========================================================================
# Utility functions
# ========================================================================

def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer after normalization."""
    return _normalize_text(text).split()


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return list of n-gram tuples."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ========================================================================
# BLEU Score
# ========================================================================

class BLEUScore:
    """
    Compute BLEU score (Papineni et al., 2002).

    Supports configurable max n-gram order and brevity penalty.
    """

    def __init__(self, max_order: int = 4, smooth: bool = True):
        self.max_order = max_order
        self.smooth = smooth

    def _modified_precision(
        self,
        prediction_tokens: List[str],
        reference_tokens: List[str],
        n: int,
    ) -> float:
        """Compute modified n-gram precision with clipping."""
        pred_ngrams = Counter(_ngrams(prediction_tokens, n))
        ref_ngrams = Counter(_ngrams(reference_tokens, n))
        clipped = 0
        total = 0
        for ngram, count in pred_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count
        if total == 0:
            return 0.0
        if self.smooth and clipped == 0:
            return 1.0 / (total + 1)
        return clipped / total

    @staticmethod
    def _brevity_penalty(prediction_tokens: List[str], reference_tokens: List[str]) -> float:
        """Compute brevity penalty."""
        pred_len = len(prediction_tokens)
        ref_len = len(reference_tokens)
        if pred_len > ref_len:
            return 1.0
        if pred_len == 0:
            return 0.0
        return math.exp(1 - ref_len / pred_len)

    def compute(self, prediction: str, reference: str) -> float:
        """
        Compute BLEU score for a single prediction-reference pair.

        Args:
            prediction: Generated text.
            reference: Ground-truth text.

        Returns:
            BLEU score in [0, 1].
        """
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)
        if not pred_tokens:
            return 0.0

        precisions: List[float] = []
        for n in range(1, self.max_order + 1):
            p = self._modified_precision(pred_tokens, ref_tokens, n)
            precisions.append(p)

        # Geometric mean of precisions
        log_avg = 0.0
        for p in precisions:
            if p <= 0:
                return 0.0
            log_avg += math.log(p)
        log_avg /= self.max_order

        bp = self._brevity_penalty(pred_tokens, ref_tokens)
        return bp * math.exp(log_avg)

    def compute_corpus(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """Compute corpus-level BLEU (average of sentence-level BLEU)."""
        if not predictions:
            return 0.0
        scores = [self.compute(p, r) for p, r in zip(predictions, references)]
        return sum(scores) / len(scores)


# ========================================================================
# ROUGE Score
# ========================================================================

class ROUGEScore:
    """
    Compute ROUGE scores: ROUGE-1, ROUGE-2, ROUGE-L.

    Uses F-measure with configurable beta for recall weighting.
    """

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    @staticmethod
    def _f_measure(precision: float, recall: float, beta: float = 1.0) -> float:
        """Compute F-measure."""
        if precision + recall == 0:
            return 0.0
        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    def rouge_n(self, prediction: str, reference: str, n: int) -> Dict[str, float]:
        """Compute ROUGE-N precision, recall, and F-measure."""
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)

        pred_ngrams = Counter(_ngrams(pred_tokens, n))
        ref_ngrams = Counter(_ngrams(ref_tokens, n))

        overlap = 0
        for ngram, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))

        pred_total = sum(pred_ngrams.values())
        ref_total = sum(ref_ngrams.values())

        precision = overlap / pred_total if pred_total > 0 else 0.0
        recall = overlap / ref_total if ref_total > 0 else 0.0
        f_measure = self._f_measure(precision, recall, self.beta)

        return {"precision": precision, "recall": recall, "fmeasure": f_measure}

    def rouge_l(self, prediction: str, reference: str) -> Dict[str, float]:
        """Compute ROUGE-L based on Longest Common Subsequence."""
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)

        lcs_len = self._lcs_length(pred_tokens, ref_tokens)

        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        f_measure = self._f_measure(precision, recall, self.beta)

        return {"precision": precision, "recall": recall, "fmeasure": f_measure}

    @staticmethod
    def _lcs_length(x: List[str], y: List[str]) -> int:
        """Compute length of Longest Common Subsequence via DP."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        # Optimize to two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    def compute(self, prediction: str, reference: str) -> Dict[str, Dict[str, float]]:
        """
        Compute all ROUGE variants.

        Returns:
            Dict with keys 'rouge1', 'rouge2', 'rougeL', each mapping
            to {'precision', 'recall', 'fmeasure'}.
        """
        return {
            "rouge1": self.rouge_n(prediction, reference, 1),
            "rouge2": self.rouge_n(prediction, reference, 2),
            "rougeL": self.rouge_l(prediction, reference),
        }

    def compute_corpus(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute corpus-level ROUGE (average of sentence-level)."""
        agg: Dict[str, Dict[str, List[float]]] = {
            "rouge1": {"precision": [], "recall": [], "fmeasure": []},
            "rouge2": {"precision": [], "recall": [], "fmeasure": []},
            "rougeL": {"precision": [], "recall": [], "fmeasure": []},
        }
        for pred, ref in zip(predictions, references):
            scores = self.compute(pred, ref)
            for key in agg:
                for sub in agg[key]:
                    agg[key][sub].append(scores[key][sub])

        result: Dict[str, Dict[str, float]] = {}
        for key in agg:
            result[key] = {}
            for sub in agg[key]:
                vals = agg[key][sub]
                result[key][sub] = sum(vals) / len(vals) if vals else 0.0
        return result


# ========================================================================
# Accuracy
# ========================================================================

class Accuracy:
    """Compute classification accuracy."""

    def compute(self, prediction: str, reference: str) -> float:
        """
        Compute accuracy for a single example.

        Treats prediction and reference as class labels after normalization.
        Returns 1.0 if they match, 0.0 otherwise.
        """
        return float(_normalize_text(prediction) == _normalize_text(reference))

    def compute_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """Compute accuracy over a batch."""
        if not predictions:
            return 0.0
        correct = sum(self.compute(p, r) for p, r in zip(predictions, references))
        return correct / len(predictions)


# ========================================================================
# F1 Score
# ========================================================================

class F1Score:
    """
    Compute token-level F1 score between prediction and reference.

    Useful for span extraction and QA tasks where partial credit matters.
    """

    def __init__(self, average: str = "micro"):
        """
        Args:
            average: 'micro' (global counts) or 'macro' (per-example average).
        """
        if average not in ("micro", "macro"):
            raise ValueError(f"average must be 'micro' or 'macro', got '{average}'")
        self.average = average

    def compute(self, prediction: str, reference: str) -> float:
        """
        Compute token-level F1 for a single prediction-reference pair.

        Tokens are compared as sets after normalization.
        """
        pred_tokens = set(_tokenize(prediction))
        ref_tokens = set(_tokenize(reference))

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def compute_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """Compute F1 over a batch."""
        if not predictions:
            return 0.0

        if self.average == "macro":
            scores = [self.compute(p, r) for p, r in zip(predictions, references)]
            return sum(scores) / len(scores)

        # Micro: aggregate TP, FP, FN across all examples
        total_common = 0
        total_pred = 0
        total_ref = 0
        for p, r in zip(predictions, references):
            pred_tokens = set(_tokenize(p))
            ref_tokens = set(_tokenize(r))
            common = pred_tokens & ref_tokens
            total_common += len(common)
            total_pred += len(pred_tokens)
            total_ref += len(ref_tokens)

        precision = total_common / total_pred if total_pred > 0 else 0.0
        recall = total_common / total_ref if total_ref > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


# ========================================================================
# Exact Match
# ========================================================================

class ExactMatch:
    """Compute exact match after normalization."""

    def compute(self, prediction: str, reference: str) -> float:
        """
        Return 1.0 if normalized prediction equals normalized reference.
        """
        return float(_normalize_text(prediction) == _normalize_text(reference))

    def compute_batch(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> float:
        """Compute exact match over a batch."""
        if not predictions:
            return 0.0
        matches = sum(self.compute(p, r) for p, r in zip(predictions, references))
        return matches / len(predictions)


# ========================================================================
# BERTScore (token-overlap approximation)
# ========================================================================

class BERTScore:
    """
    Compute BERTScore-like metric using token overlap.

    When the ``transformers`` library is available, this uses cosine similarity
    of contextual embeddings.  Otherwise, it falls back to a token-overlap
    approximation that correlates with semantic similarity.
    """

    def __init__(self, use_transformers: bool = True, model_name: str = "bert-base-uncased"):
        self.use_transformers = use_transformers
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._fallback = False

        if use_transformers:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name)
                self._model.eval()
            except Exception:
                self._fallback = True

    def _compute_with_embeddings(
        self,
        prediction: str,
        reference: str,
    ) -> Dict[str, float]:
        """Compute BERTScore using actual contextual embeddings."""
        import torch
        import torch.nn.functional as F

        def _get_embeddings(text: str) -> torch.Tensor:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Use last hidden state, exclude [CLS] and [SEP]
            return outputs.last_hidden_state[0, 1:-1, :]

        pred_emb = _get_embeddings(prediction)
        ref_emb = _get_embeddings(reference)

        # Cosine similarity matrix
        sim_matrix = F.cosine_similarity(
            pred_emb.unsqueeze(1), ref_emb.unsqueeze(0), dim=-1
        )

        # Precision: for each pred token, max similarity to any ref token
        precision = sim_matrix.max(dim=1).values.mean().item()
        # Recall: for each ref token, max similarity to any pred token
        recall = sim_matrix.max(dim=0).values.mean().item()
        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def _compute_fallback(
        self,
        prediction: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Token-overlap approximation of BERTScore.

        Uses weighted token matching where rare tokens contribute more.
        """
        pred_tokens = _tokenize(prediction)
        ref_tokens = _tokenize(reference)

        if not pred_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)

        # IDF-like weighting: tokens that appear in fewer documents are weighted higher
        all_tokens = set(pred_tokens) | set(ref_tokens)
        doc_freq: Dict[str, int] = {}
        for t in all_tokens:
            doc_freq[t] = 0
            if t in pred_counter:
                doc_freq[t] += 1
            if t in ref_counter:
                doc_freq[t] += 1

        idf: Dict[str, float] = {}
        for t in all_tokens:
            idf[t] = math.log(2.0 / (doc_freq[t])) + 1.0  # smooth IDF

        def _weighted_match(counter_a: Counter, counter_b: Counter) -> float:
            score = 0.0
            total_weight = 0.0
            for token, count_a in counter_a.items():
                weight = idf.get(token, 1.0)
                total_weight += weight * count_a
                count_b = counter_b.get(token, 0)
                matched = min(count_a, count_b)
                score += weight * matched
            return score / total_weight if total_weight > 0 else 0.0

        precision = _weighted_match(pred_counter, ref_counter)
        recall = _weighted_match(ref_counter, pred_counter)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {"precision": precision, "recall": recall, "f1": f1}

    def compute(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute BERTScore for a single prediction-reference pair.

        Returns dict with 'precision', 'recall', and 'f1'.
        """
        if self._fallback or not self.use_transformers or self._model is None:
            return self._compute_fallback(prediction, reference)
        return self._compute_with_embeddings(prediction, reference)

    def compute_corpus(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
    ) -> Dict[str, float]:
        """Compute corpus-level BERTScore."""
        agg = {"precision": [], "recall": [], "f1": []}
        for p, r in zip(predictions, references):
            scores = self.compute(p, r)
            for k in agg:
                agg[k].append(scores[k])
        return {k: sum(v) / len(v) if v else 0.0 for k, v in agg.items()}


# ========================================================================
# Metric Registry
# ========================================================================

class MetricRegistry:
    """Registry for instantiating metrics by name."""

    _registry: Dict[str, type] = {
        "bleu": BLEUScore,
        "rouge": ROUGEScore,
        "accuracy": Accuracy,
        "f1": F1Score,
        "exact_match": ExactMatch,
        "bertscore": BERTScore,
    }

    def get(self, name: str) -> Any:
        """
        Instantiate a metric by name.

        Args:
            name: One of 'bleu', 'rouge', 'accuracy', 'f1', 'exact_match', 'bertscore'.

        Returns:
            Metric object with a ``compute`` method.
        """
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        if name_lower not in self._registry:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[name_lower]()

    @classmethod
    def available_metrics(cls) -> List[str]:
        """Return list of available metric names."""
        return list(cls._registry.keys())

    @classmethod
    def register(cls, name: str, metric_class: type) -> None:
        """Register a custom metric class."""
        cls._registry[name.lower()] = metric_class
