"""Stopping criteria for Nexus-LLM backend.

Implements various stopping conditions: max length, EOS token, string match,
length penalty, and custom criteria composition.
"""

import torch
from typing import List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class StoppingCriterion(ABC):
    """Abstract base class for stopping criteria."""

    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Return True if generation should stop."""
        pass

    def reset(self) -> None:
        """Reset any internal state for a new generation."""
        pass


class MaxLengthCriteria(StoppingCriterion):
    """Stop when the total sequence length reaches max_length."""

    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MaxNewTokensCriteria(StoppingCriterion):
    """Stop when the number of newly generated tokens reaches max_new_tokens."""

    def __init__(self, prompt_length: int, max_new_tokens: int):
        self.prompt_length = prompt_length
        self.max_new_tokens = max_new_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_new_tokens = input_ids.shape[-1] - self.prompt_length
        return current_new_tokens >= self.max_new_tokens


class EosTokenCriteria(StoppingCriterion):
    """Stop when an end-of-sequence token is generated."""

    def __init__(self, eos_token_id: Optional[int] = None, eos_token_ids: Optional[List[int]] = None):
        if eos_token_id is not None and eos_token_ids is not None:
            self.eos_token_ids = set([eos_token_id] + eos_token_ids)
        elif eos_token_id is not None:
            self.eos_token_ids = {eos_token_id}
        elif eos_token_ids is not None:
            self.eos_token_ids = set(eos_token_ids)
        else:
            self.eos_token_ids = set()
        self._finished: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.eos_token_ids:
            return False

        batch_size = input_ids.shape[0]
        newly_finished = set()
        for i in range(batch_size):
            if i not in self._finished:
                last_token = input_ids[i, -1].item()
                if last_token in self.eos_token_ids:
                    newly_finished.add(i)

        self._finished.update(newly_finished)
        return len(self._finished) == batch_size

    def reset(self) -> None:
        self._finished = set()


class StringMatchCriteria(StoppingCriterion):
    """Stop when a specific string appears in the decoded output."""

    def __init__(self, stop_strings: List[str], tokenizer: Any):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer
        self._finished: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            if i not in self._finished:
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                if any(s in text for s in self.stop_strings):
                    self._finished.add(i)
        return len(self._finished) == batch_size

    def reset(self) -> None:
        self._finished = set()


class StopTokenIdsCriteria(StoppingCriterion):
    """Stop when any of the specified token IDs are generated."""

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = set(stop_token_ids)
        self._finished: Set[int] = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            if i not in self._finished:
                last_token = input_ids[i, -1].item()
                if last_token in self.stop_token_ids:
                    self._finished.add(i)
        return len(self._finished) == batch_size

    def reset(self) -> None:
        self._finished = set()


class MinLengthCriteria(StoppingCriterion):
    """Prevent stopping before minimum length is reached (suppresses EOS)."""

    def __init__(self, min_length: int, eos_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] < self.min_length:
            scores[:, self.eos_token_id] = float("-inf")
        return False


class MinNewTokensCriteria(StoppingCriterion):
    """Prevent stopping before minimum number of new tokens are generated."""

    def __init__(self, prompt_length: int, min_new_tokens: int, eos_token_id: int):
        self.prompt_length = prompt_length
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_new_tokens = input_ids.shape[-1] - self.prompt_length
        if current_new_tokens < self.min_new_tokens:
            scores[:, self.eos_token_id] = float("-inf")
        return False


class TimeLimitCriteria(StoppingCriterion):
    """Stop generation after a time limit (in seconds)."""

    def __init__(self, max_time_seconds: float):
        self.max_time_seconds = max_time_seconds
        self._start_time: Optional[float] = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        import time
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        return elapsed >= self.max_time_seconds

    def reset(self) -> None:
        self._start_time = None


class CustomFunctionCriteria(StoppingCriterion):
    """Stop based on a custom callable function.

    The function receives (input_ids, scores, kwargs) and returns bool.
    """

    def __init__(self, stop_fn: Callable[[torch.LongTensor, torch.FloatTensor], bool]):
        self.stop_fn = stop_fn

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.stop_fn(input_ids, scores, **kwargs)


class CompositeStoppingCriteria(StoppingCriterion):
    """Combine multiple stopping criteria. Stops when ANY criterion returns True."""

    def __init__(self, criteria: List[StoppingCriterion], mode: str = "any"):
        self.criteria = criteria
        self.mode = mode
        if mode not in ("any", "all"):
            raise ValueError(f"mode must be 'any' or 'all', got '{mode}'")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        results = [c(input_ids, scores, **kwargs) for c in self.criteria]
        if self.mode == "any":
            return any(results)
        return all(results)

    def reset(self) -> None:
        for c in self.criteria:
            c.reset()


@dataclass
class StoppingCriteriaBuilder:
    """Builder for composing stopping criteria fluently."""

    criteria: List[StoppingCriterion] = field(default_factory=list)

    def max_length(self, max_length: int) -> "StoppingCriteriaBuilder":
        self.criteria.append(MaxLengthCriteria(max_length))
        return self

    def max_new_tokens(self, prompt_length: int, max_new_tokens: int) -> "StoppingCriteriaBuilder":
        self.criteria.append(MaxNewTokensCriteria(prompt_length, max_new_tokens))
        return self

    def eos_token(self, eos_token_id: Optional[int] = None, eos_token_ids: Optional[List[int]] = None) -> "StoppingCriteriaBuilder":
        self.criteria.append(EosTokenCriteria(eos_token_id, eos_token_ids))
        return self

    def string_match(self, stop_strings: List[str], tokenizer: Any) -> "StoppingCriteriaBuilder":
        self.criteria.append(StringMatchCriteria(stop_strings, tokenizer))
        return self

    def stop_token_ids(self, stop_token_ids: List[int]) -> "StoppingCriteriaBuilder":
        self.criteria.append(StopTokenIdsCriteria(stop_token_ids))
        return self

    def min_length(self, min_length: int, eos_token_id: int) -> "StoppingCriteriaBuilder":
        self.criteria.append(MinLengthCriteria(min_length, eos_token_id))
        return self

    def min_new_tokens(self, prompt_length: int, min_new_tokens: int, eos_token_id: int) -> "StoppingCriteriaBuilder":
        self.criteria.append(MinNewTokensCriteria(prompt_length, min_new_tokens, eos_token_id))
        return self

    def time_limit(self, max_time_seconds: float) -> "StoppingCriteriaBuilder":
        self.criteria.append(TimeLimitCriteria(max_time_seconds))
        return self

    def custom(self, stop_fn: Callable) -> "StoppingCriteriaBuilder":
        self.criteria.append(CustomFunctionCriteria(stop_fn))
        return self

    def build(self, mode: str = "any") -> CompositeStoppingCriteria:
        return CompositeStoppingCriteria(self.criteria, mode=mode)
