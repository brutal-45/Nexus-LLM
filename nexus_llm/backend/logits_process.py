"""Logits processors for Nexus-LLM backend.

Implements processors that modify logits before sampling: repetition penalty,
temperature scaling, top-k filtering, top-p filtering, min-length suppression,
no-repeat n-gram suppression, and custom processors.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Callable, Dict, Set, Any
from collections import defaultdict
from abc import ABC, abstractmethod


class LogitsProcessor(ABC):
    """Abstract base class for logits processors."""

    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Process logits and return modified scores."""
        pass

    def reset(self) -> None:
        """Reset internal state for a new generation."""
        pass


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Penalize tokens that have already appeared in the sequence."""

    def __init__(self, penalty: float = 1.2):
        if penalty <= 0:
            raise ValueError(f"Repetition penalty must be positive, got {penalty}")
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            unique_ids = torch.unique(input_ids[batch_idx])
            for token_id in unique_ids:
                if scores[batch_idx, token_id] < 0:
                    scores[batch_idx, token_id] *= self.penalty
                else:
                    scores[batch_idx, token_id] /= self.penalty
        return scores


class FrequencyPenaltyLogitsProcessor(LogitsProcessor):
    """Penalize tokens based on their frequency in the input."""

    def __init__(self, penalty: float = 0.5):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            token_counts = torch.bincount(input_ids[batch_idx], minlength=scores.shape[-1]).float()
            scores[batch_idx] -= self.penalty * token_counts
        return scores


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """Penalize tokens that have appeared at least once, regardless of count."""

    def __init__(self, penalty: float = 0.5):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            token_counts = torch.bincount(input_ids[batch_idx], minlength=scores.shape[-1]).float()
            presence = (token_counts > 0).float()
            scores[batch_idx] -= self.penalty * presence
        return scores


class TemperatureLogitsProcessor(LogitsProcessor):
    """Scale logits by temperature. Higher temperature = more random."""

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores / self.temperature


class TopKLogitsProcessor(LogitsProcessor):
    """Keep only the top-k logits, setting the rest to -inf."""

    def __init__(self, top_k: int, min_tokens_to_keep: int = 1):
        self.top_k = max(min_tokens_to_keep, top_k)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        return scores.masked_fill(indices_to_remove, float("-inf"))


class TopPLogitsProcessor(LogitsProcessor):
    """Keep the smallest set of tokens with cumulative probability >= top_p."""

    def __init__(self, top_p: float = 1.0, min_tokens_to_keep: int = 1):
        if not 0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        self.top_p = top_p
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= self.top_p
        sorted_indices_to_remove[..., :self.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        return scores.masked_fill(indices_to_remove, float("-inf"))


class MinLengthLogitsProcessor(LogitsProcessor):
    """Suppress EOS token until minimum length is reached."""

    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.min_length:
            scores[:, self.eos_token_id] = float("-inf")
        return scores


class MinNewTokensLogitsProcessor(LogitsProcessor):
    """Suppress EOS token until minimum number of new tokens are generated."""

    def __init__(self, prompt_length: int, min_new_tokens: int, eos_token_id: int):
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length
        if new_tokens_length < self.min_new_tokens:
            scores[:, self.eos_token_id] = float("-inf")
        return scores


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Prevent generation of n-grams that have already appeared."""

    def __init__(self, ngram_size: int):
        if ngram_size <= 0:
            raise ValueError(f"ngram_size must be positive, got {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_idx in range(input_ids.shape[0]):
            if input_ids.shape[-1] < self.ngram_size:
                continue

            ngrams: Dict[tuple, Set[int]] = defaultdict(set)
            seq = input_ids[batch_idx].tolist()
            for i in range(len(seq) - self.ngram_size + 1):
                ngram = tuple(seq[i:i + self.ngram_size - 1])
                next_token = seq[i + self.ngram_size - 1]
                ngrams[ngram].add(next_token)

            recent_ngram = tuple(seq[-(self.ngram_size - 1):])
            if recent_ngram in ngrams:
                for token_id in ngrams[recent_ngram]:
                    scores[batch_idx, token_id] = float("-inf")

        return scores


class NoBadWordsLogitsProcessor(LogitsProcessor):
    """Suppress specified token sequences (bad words)."""

    def __init__(self, bad_words_ids: List[List[int]], eos_token_id: int):
        self.bad_words_ids = bad_words_ids
        self.eos_token_id = eos_token_id
        self._static_bad_words: Set[int] = set()
        for word_ids in bad_words_ids:
            if len(word_ids) == 1:
                self._static_bad_words.add(word_ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for token_id in self._static_bad_words:
            scores[:, token_id] = float("-inf")

        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx].tolist()
            for word_ids in self.bad_words_ids:
                if len(word_ids) <= 1:
                    continue
                word_len = len(word_ids)
                if len(seq) >= word_len - 1:
                    tail = seq[-(word_len - 1):]
                    if tail == word_ids[:-1]:
                        scores[batch_idx, word_ids[-1]] = float("-inf")
        return scores


class EpsilonLogitsProcessor(LogitsProcessor):
    """Remove tokens with probability below epsilon."""

    def __init__(self, epsilon: float = 0.002, min_tokens_to_keep: int = 1):
        self.epsilon = epsilon
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = F.softmax(scores, dim=-1)
        indices_to_remove = probs < self.epsilon
        indices_to_remove[..., :self.min_tokens_to_keep] = False
        return scores.masked_fill(indices_to_remove, float("-inf"))


class EtaLogitsProcessor(LogitsProcessor):
    """Entropy-based token filtering."""

    def __init__(self, eta: float = 0.002, min_tokens_to_keep: int = 1):
        self.eta = eta
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = F.softmax(scores, dim=-1)
        log_probs = F.log_softmax(scores, dim=-1)
        entropy = -torch.nansum(probs * log_probs, dim=-1, keepdim=True)
        eta_threshold = torch.min(
            self.eta * torch.exp(-entropy),
            torch.tensor(1.0, device=scores.device)
        )
        indices_to_remove = probs < eta_threshold
        indices_to_remove[..., :self.min_tokens_to_keep] = False
        return scores.masked_fill(indices_to_remove, float("-inf"))


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """Penalize/reward tokens based on presence in the encoder input."""

    def __init__(self, penalty: float = 1.2, encoder_input_ids: Optional[torch.LongTensor] = None):
        self.penalty = penalty
        self.encoder_input_ids = encoder_input_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.encoder_input_ids is None:
            return scores

        for batch_idx in range(self.encoder_input_ids.shape[0]):
            unique_ids = torch.unique(self.encoder_input_ids[batch_idx])
            for token_id in unique_ids:
                if scores[batch_idx, token_id] < 0:
                    scores[batch_idx, token_id] *= self.penalty
                else:
                    scores[batch_idx, token_id] /= self.penalty
        return scores


class CustomLogitsProcessor(LogitsProcessor):
    """Wrap a custom function as a logits processor."""

    def __init__(self, processor_fn: Callable[[torch.LongTensor, torch.FloatTensor], torch.FloatTensor]):
        self.processor_fn = processor_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return self.processor_fn(input_ids, scores)


class LogitsProcessorList:
    """Compose multiple logits processors, applied in order."""

    def __init__(self, processors: Optional[List[LogitsProcessor]] = None):
        self.processors: List[LogitsProcessor] = processors or []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for processor in self.processors:
            scores = processor(input_ids, scores)
        return scores

    def append(self, processor: LogitsProcessor) -> None:
        self.processors.append(processor)

    def extend(self, processors: List[LogitsProcessor]) -> None:
        self.processors.extend(processors)

    def reset(self) -> None:
        for p in self.processors:
            p.reset()

    def __len__(self) -> int:
        return len(self.processors)

    def __getitem__(self, idx: int) -> LogitsProcessor:
        return self.processors[idx]

    def __iter__(self):
        return iter(self.processors)
