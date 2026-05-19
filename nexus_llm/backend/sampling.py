"""Sampling strategies for Nexus-LLM backend.

Implements various sampling methods: greedy, multinomial, top-k, top-p (nucleus),
typical, eta, epsilon sampling, and temperature-based scaling.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    """Configuration for sampling parameters."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    typical_p: float = 1.0
    eta_cutoff: float = 0.0
    epsilon_cutoff: float = 0.0
    min_tokens_to_keep: int = 1
    seed: Optional[int] = None


class SamplingStrategy:
    """Base class for sampling strategies."""

    def __init__(self, config: Optional[SamplingConfig] = None):
        self.config = config or SamplingConfig()

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from logits. Returns selected token indices."""
        raise NotImplementedError("Subclasses must implement sample()")


class GreedySampling(SamplingStrategy):
    """Greedy decoding: always select the highest probability token."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


class MultinomialSampling(SamplingStrategy):
    """Multinomial sampling with temperature scaling."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.config.temperature
        if temperature <= 0:
            return GreedySampling(self.config).sample(logits)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopKSampling(SamplingStrategy):
    """Top-k sampling: only consider the k most likely tokens."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.config.temperature
        top_k = self.config.top_k
        min_tokens = max(self.config.min_tokens_to_keep, 1)

        if top_k <= 0:
            return MultinomialSampling(self.config).sample(logits)

        top_k = max(min_tokens, min(top_k, logits.size(-1)))

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if temperature > 0:
            probs = F.softmax(filtered_logits / temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class TopPSampling(SamplingStrategy):
    """Top-p (nucleus) sampling: consider smallest set of tokens with cumulative probability >= p."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = self.config.temperature
        top_p = self.config.top_p

        if top_p >= 1.0:
            return MultinomialSampling(self.config).sample(logits)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_indices_to_remove[..., :self.config.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))

        if temperature > 0:
            probs = F.softmax(filtered_logits / temperature, dim=-1)
        else:
            probs = F.softmax(filtered_logits, dim=-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)


class NucleusSampling(TopPSampling):
    """Alias for Top-P sampling (nucleus sampling is the same as top-p)."""
    pass


class TypicalSampling(SamplingStrategy):
    """Typical sampling: sample based on information content closeness to expected entropy."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        typical_p = self.config.typical_p

        if typical_p >= 1.0:
            return MultinomialSampling(self.config).sample(logits)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        temperature = self.config.temperature
        scaled_logits = logits / temperature if temperature > 0 else logits
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)

        neg_log_probs = -log_probs
        entropy = torch.nansum(probs * neg_log_probs, dim=-1, keepdim=True)
        deviation = torch.abs(neg_log_probs - entropy)

        sorted_deviation, sorted_indices = torch.sort(deviation)
        sorted_probs = probs.gather(dim=-1, index=sorted_indices)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs >= typical_p
        sorted_indices_to_remove[..., :self.config.min_tokens_to_keep] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
        filtered_probs = F.softmax(filtered_logits / (temperature if temperature > 0 else 1.0), dim=-1)

        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)


class EtaSampling(SamplingStrategy):
    """Eta sampling: filters tokens based on entropy-relative probability threshold."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        eta_cutoff = self.config.eta_cutoff

        if eta_cutoff <= 0.0:
            return MultinomialSampling(self.config).sample(logits)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        temperature = self.config.temperature
        scaled_logits = logits / temperature if temperature > 0 else logits
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.nansum(probs * log_probs, dim=-1, keepdim=True)

        eta_threshold = torch.min(
            eta_cutoff * torch.exp(-entropy),
            torch.tensor(1.0, device=logits.device)
        )
        indices_to_remove = probs < eta_threshold
        indices_to_remove[..., :self.config.min_tokens_to_keep] = False

        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
        filtered_probs = F.softmax(filtered_logits / (temperature if temperature > 0 else 1.0), dim=-1)

        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)


class EpsilonSampling(SamplingStrategy):
    """Epsilon sampling: filter out tokens with probability below epsilon."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        epsilon_cutoff = self.config.epsilon_cutoff

        if epsilon_cutoff <= 0.0:
            return MultinomialSampling(self.config).sample(logits)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        temperature = self.config.temperature
        scaled_logits = logits / temperature if temperature > 0 else logits
        probs = F.softmax(scaled_logits, dim=-1)

        indices_to_remove = probs < epsilon_cutoff
        indices_to_remove[..., :self.config.min_tokens_to_keep] = False

        filtered_logits = logits.masked_fill(indices_to_remove, float("-inf"))
        filtered_probs = F.softmax(filtered_logits / (temperature if temperature > 0 else 1.0), dim=-1)

        return torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)


class CombinedSampling(SamplingStrategy):
    """Combined sampling: applies multiple filters in sequence (top-k -> top-p -> typical -> epsilon -> eta)."""

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        config = self.config
        filtered_logits = logits.clone()

        if config.top_k > 0:
            top_k = min(config.top_k, filtered_logits.size(-1))
            indices_to_remove = filtered_logits < torch.topk(filtered_logits, top_k)[0][..., -1, None]
            filtered_logits = filtered_logits.masked_fill(indices_to_remove, float("-inf"))

        if config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= config.top_p
            sorted_indices_to_remove[..., :config.min_tokens_to_keep] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            filtered_logits = filtered_logits.masked_fill(indices_to_remove, float("-inf"))

        if config.typical_p < 1.0:
            probs = F.softmax(filtered_logits, dim=-1)
            log_probs = F.log_softmax(filtered_logits, dim=-1)
            entropy = -torch.nansum(probs * log_probs, dim=-1, keepdim=True)
            deviation = torch.abs(-log_probs - entropy)
            sorted_deviation, sorted_indices = torch.sort(deviation)
            sorted_probs = probs.gather(dim=-1, index=sorted_indices)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs >= config.typical_p
            sorted_indices_to_remove[..., :config.min_tokens_to_keep] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            filtered_logits = filtered_logits.masked_fill(indices_to_remove, float("-inf"))

        if config.epsilon_cutoff > 0.0:
            probs = F.softmax(filtered_logits, dim=-1)
            indices_to_remove = probs < config.epsilon_cutoff
            indices_to_remove[..., :config.min_tokens_to_keep] = False
            filtered_logits = filtered_logits.masked_fill(indices_to_remove, float("-inf"))

        if config.eta_cutoff > 0.0:
            probs = F.softmax(filtered_logits, dim=-1)
            log_probs = F.log_softmax(filtered_logits, dim=-1)
            entropy = -torch.nansum(probs * log_probs, dim=-1, keepdim=True)
            eta_threshold = torch.min(
                config.eta_cutoff * torch.exp(-entropy),
                torch.tensor(1.0, device=logits.device)
            )
            indices_to_remove = probs < eta_threshold
            indices_to_remove[..., :config.min_tokens_to_keep] = False
            filtered_logits = filtered_logits.masked_fill(indices_to_remove, float("-inf"))

        temperature = config.temperature if config.temperature > 0 else 1.0
        probs = F.softmax(filtered_logits / temperature, dim=-1)

        if config.seed is not None:
            torch.manual_seed(config.seed)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def create_sampler(config: SamplingConfig) -> SamplingStrategy:
    """Factory function to create the appropriate sampling strategy.

    Uses CombinedSampling when multiple filters are active, or specialized
    strategies for single-filter cases.
    """
    c = config
    has_top_k = c.top_k > 0
    has_top_p = c.top_p < 1.0
    has_typical = c.typical_p < 1.0
    has_eta = c.eta_cutoff > 0.0
    has_epsilon = c.epsilon_cutoff > 0.0

    active_filters = sum([has_top_k, has_top_p, has_typical, has_eta, has_epsilon])

    if active_filters == 0:
        if c.temperature <= 0:
            return GreedySampling(config)
        return MultinomialSampling(config)
    elif active_filters == 1:
        if has_top_k:
            return TopKSampling(config)
        elif has_top_p:
            return TopPSampling(config)
        elif has_typical:
            return TypicalSampling(config)
        elif has_eta:
            return EtaSampling(config)
        elif has_epsilon:
            return EpsilonSampling(config)

    return CombinedSampling(config)
