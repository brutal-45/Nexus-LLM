"""
Perplexity Calculation Module

Supports three modes of perplexity computation:
- Sequence-level: single perplexity for an entire sequence
- Token-level: per-token perplexity breakdown
- Sliding window: perplexity computed over a moving context window

Also provides batch computation and statistical summaries.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class PerplexityResult:
    """Container for perplexity computation results."""
    perplexity: float
    total_log_prob: float
    num_tokens: int
    per_token_perplexity: List[float] = field(default_factory=list)
    per_token_log_prob: List[float] = field(default_factory=list)
    window_perplexities: List[float] = field(default_factory=list)
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    @property
    def bits_per_token(self) -> float:
        """Average bits per token (log base 2)."""
        if self.num_tokens == 0:
            return 0.0
        return -self.total_log_prob / (self.num_tokens * math.log(2))

    @property
    def entropy(self) -> float:
        """Estimated entropy in nats."""
        if self.num_tokens == 0:
            return 0.0
        return -self.total_log_prob / self.num_tokens

    def to_dict(self) -> Dict[str, Union[float, int, List[float], Dict]]:
        return {
            "perplexity": self.perplexity,
            "total_log_prob": self.total_log_prob,
            "num_tokens": self.num_tokens,
            "per_token_perplexity": self.per_token_perplexity,
            "per_token_log_prob": self.per_token_log_prob,
            "window_perplexities": self.window_perplexities,
            "bits_per_token": self.bits_per_token,
            "entropy": self.entropy,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Perplexity: {self.perplexity:.4f}",
            f"Total log prob: {self.total_log_prob:.4f}",
            f"Num tokens: {self.num_tokens}",
            f"Bits/token: {self.bits_per_token:.4f}",
            f"Entropy: {self.entropy:.4f}",
        ]
        if self.per_token_perplexity:
            min_ppl = min(self.per_token_perplexity)
            max_ppl = max(self.per_token_perplexity)
            avg_ppl = sum(self.per_token_perplexity) / len(self.per_token_perplexity)
            lines.extend([
                f"Token-level PPL — min: {min_ppl:.4f}, max: {max_ppl:.4f}, avg: {avg_ppl:.4f}",
            ])
        if self.window_perplexities:
            lines.append(f"Window PPL — count: {len(self.window_perplexities)}, "
                         f"avg: {sum(self.window_perplexities)/len(self.window_perplexities):.4f}")
        return "\n".join(lines)


class PerplexityCalculator:
    """
    Computes perplexity from log probabilities.

    Perplexity is defined as:

        PPL = exp(-1/N * sum(log P(x_i | x_{<i})))

    where N is the number of tokens and P(x_i | x_{<i}) is the model's
    predicted probability for token x_i given its context.
    """

    def __init__(
        self,
        base: float = math.e,
        ignore_padding: bool = True,
        pad_token_id: Optional[int] = None,
    ):
        """
        Args:
            base: Logarithm base used for input log-probs (default: natural log).
            ignore_padding: Whether to skip padding tokens in computation.
            pad_token_id: Token ID used for padding (required if ignore_padding=True).
        """
        self.base = base
        self.ignore_padding = ignore_padding
        self.pad_token_id = pad_token_id

    def compute(
        self,
        log_probs: Sequence[float],
        token_ids: Optional[Sequence[int]] = None,
    ) -> PerplexityResult:
        """
        Compute sequence-level perplexity.

        Args:
            log_probs: Sequence of log probabilities for each token.
            token_ids: Optional token IDs (used for padding filtering).

        Returns:
            PerplexityResult with aggregated and per-token details.
        """
        filtered_log_probs: List[float] = []
        filtered_token_ids: List[int] = []

        for i, lp in enumerate(log_probs):
            if self.ignore_padding and token_ids is not None and self.pad_token_id is not None:
                if token_ids[i] == self.pad_token_id:
                    continue
            filtered_log_probs.append(float(lp))
            if token_ids is not None:
                filtered_token_ids.append(token_ids[i])

        if not filtered_log_probs:
            return PerplexityResult(
                perplexity=float("inf"),
                total_log_prob=0.0,
                num_tokens=0,
            )

        # Convert to natural log if base differs
        if self.base != math.e:
            nat_log_probs = [lp * math.log(self.base) for lp in filtered_log_probs]
        else:
            nat_log_probs = filtered_log_probs

        total_log_prob = sum(nat_log_probs)
        num_tokens = len(nat_log_probs)

        avg_log_prob = total_log_prob / num_tokens
        perplexity = math.exp(-avg_log_prob)

        # Per-token perplexity
        per_token_perplexity = [math.exp(-lp) for lp in nat_log_probs]

        return PerplexityResult(
            perplexity=perplexity,
            total_log_prob=total_log_prob,
            num_tokens=num_tokens,
            per_token_perplexity=per_token_perplexity,
            per_token_log_prob=nat_log_probs,
        )

    def compute_token_level(
        self,
        log_probs: Sequence[float],
        token_ids: Optional[Sequence[int]] = None,
    ) -> PerplexityResult:
        """
        Compute token-level perplexity breakdown.

        Returns the same as ``compute`` but emphasizes per-token details.
        """
        return self.compute(log_probs, token_ids)

    def compute_sliding_window(
        self,
        log_probs: Sequence[float],
        window_size: int = 128,
        stride: int = 64,
        token_ids: Optional[Sequence[int]] = None,
    ) -> PerplexityResult:
        """
        Compute perplexity using a sliding window approach.

        Useful for long sequences where local context is more relevant.

        Args:
            log_probs: Sequence of log probabilities.
            window_size: Size of the sliding window.
            stride: Step size for the window.
            token_ids: Optional token IDs for padding filtering.

        Returns:
            PerplexityResult with window-level and overall perplexities.
        """
        if not log_probs:
            return PerplexityResult(
                perplexity=float("inf"),
                total_log_prob=0.0,
                num_tokens=0,
            )

        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")

        window_perplexities: List[float] = []
        all_window_log_probs: List[float] = []

        start = 0
        while start < len(log_probs):
            end = min(start + window_size, len(log_probs))
            window_lps = list(log_probs[start:end])

            # Filter padding if needed
            if self.ignore_padding and token_ids is not None and self.pad_token_id is not None:
                window_lps = [
                    lp for i, lp in enumerate(window_lps)
                    if token_ids[start + i] != self.pad_token_id
                ]

            if not window_lps:
                start += stride
                continue

            # Convert to natural log
            if self.base != math.e:
                nat_lps = [lp * math.log(self.base) for lp in window_lps]
            else:
                nat_lps = window_lps

            total_lp = sum(nat_lps)
            n = len(nat_lps)
            window_ppl = math.exp(-total_lp / n) if n > 0 else float("inf")

            window_perplexities.append(window_ppl)
            all_window_log_probs.extend(nat_lps)
            start += stride

        # Overall perplexity from all window contributions
        if all_window_log_probs:
            overall_total = sum(all_window_log_probs)
            overall_n = len(all_window_log_probs)
            overall_ppl = math.exp(-overall_total / overall_n)
        else:
            overall_ppl = float("inf")

        return PerplexityResult(
            perplexity=overall_ppl,
            total_log_prob=sum(all_window_log_probs) if all_window_log_probs else 0.0,
            num_tokens=len(all_window_log_probs),
            window_perplexities=window_perplexities,
            metadata={"window_size": window_size, "stride": stride},
        )

    def compute_batch(
        self,
        batch_log_probs: Sequence[Sequence[float]],
        batch_token_ids: Optional[Sequence[Sequence[int]]] = None,
    ) -> List[PerplexityResult]:
        """
        Compute perplexity for a batch of sequences.

        Args:
            batch_log_probs: List of log-prob sequences.
            batch_token_ids: Optional list of token ID sequences.

        Returns:
            List of PerplexityResult, one per sequence.
        """
        results: List[PerplexityResult] = []
        for i, lps in enumerate(batch_log_probs):
            tids = batch_token_ids[i] if batch_token_ids is not None else None
            results.append(self.compute(lps, tids))
        return results

    def compute_batch_summary(
        self,
        batch_log_probs: Sequence[Sequence[float]],
        batch_token_ids: Optional[Sequence[Sequence[int]]] = None,
    ) -> Dict[str, float]:
        """
        Compute aggregate statistics over a batch.

        Returns mean, median, min, max perplexity and total log prob.
        """
        results = self.compute_batch(batch_log_probs, batch_token_ids)
        ppls = [r.perplexity for r in results if r.num_tokens > 0]

        if not ppls:
            return {
                "mean_ppl": float("inf"),
                "median_ppl": float("inf"),
                "min_ppl": float("inf"),
                "max_ppl": float("inf"),
                "total_sequences": 0,
                "total_tokens": 0,
                "total_log_prob": 0.0,
            }

        sorted_ppls = sorted(ppls)
        n = len(sorted_ppls)
        median = sorted_ppls[n // 2] if n % 2 == 1 else (sorted_ppls[n // 2 - 1] + sorted_ppls[n // 2]) / 2

        return {
            "mean_ppl": sum(ppls) / len(ppls),
            "median_ppl": median,
            "min_ppl": min(ppls),
            "max_ppl": max(ppls),
            "total_sequences": len(ppls),
            "total_tokens": sum(r.num_tokens for r in results),
            "total_log_prob": sum(r.total_log_prob for r in results),
        }
