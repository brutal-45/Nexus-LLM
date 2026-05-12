"""
Nexus Sampling Methods
==========================
All sampling strategies used in LLM generation, implemented from scratch.

Sampling Strategies:
    - Ancestral sampling: Sample from full distribution
    - Temperature sampling: Scale logits before softmax
    - Top-k sampling: Only consider K most probable tokens
    - Top-p (nucleus) sampling: Smallest set with cumulative prob >= p
    - Gumbel-Softmax: Differentiable approximation of categorical sampling
    - Beam search: Deterministic search with multiple hypotheses
    - Repetition penalty: Penalize recently seen tokens
    - Min-p sampling: Filter tokens below min_prob * p_max

Each sampler transforms a logit distribution into a sampled token,
following the pipeline:
    logits -> temperature -> top_k -> top_p -> sample -> token_id

Reference:
    - Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020) - Nucleus
    - Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

Array = np.ndarray


class Sampler(ABC):
    """Abstract base class for samplers."""

    @abstractmethod
    def sample(self, logits: Array, temperature: float = 1.0) -> int:
        """Sample a single token from logits."""
        pass

    @abstractmethod
    def sample_batch(self, logits: Array, temperature: float = 1.0) -> Array:
        """Sample tokens for a batch."""
        pass


class Random:
    """Random number generator wrapper for reproducibility."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self.rng.uniform(low, high, size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self.rng.normal(loc, scale, size=size)

    def categorical(self, probs: Array) -> int:
        return int(self.rng.choice(len(probs), p=probs))

    def gumbel(self, size=None) -> Array:
        """Sample from Gumbel(0, 1) distribution."""
        u = self.rng.uniform(1e-10, 1.0, size=size)
        return -np.log(-np.log(u))


# ================================================================
# CORE SAMPLING METHODS
# ================================================================

def ancestral_sample(logits: Array, rng: Optional[Random] = None) -> int:
    """
    Ancestral sampling: sample directly from the softmax distribution.
    
    The most basic sampling strategy. At each step, compute:
        p(x_t | x_<t) = softmax(logits)
        x_t ~ Categorical(p)
    
    Pros: Maximally diverse, can generate any token
    Cons: Can produce low-quality or incoherent text
    
    Args:
        logits: Unnormalized log-probabilities (vocab_size,).
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64)
    probs = _stable_softmax(logits)
    
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


def temperature_sample(logits: Array, temperature: float, rng: Optional[Random] = None) -> int:
    """
    Temperature-based sampling.
    
    Scales logits before softmax to control randomness:
        p_i = softmax(logit_i / T)
    
    T > 1: Flatter distribution → more random/diverse
    T < 1: Sharper distribution → more deterministic/focused
    T → 0: Greedy (always picks the most probable token)
    T = 1: Standard sampling (no change)
    
    Effect on entropy:
        H_T = -sum p_i log p_i
        dH/dT > 0 for T > 0 (higher temperature → higher entropy)
    
    Args:
        logits: Logits (vocab_size,).
        temperature: Sampling temperature (default: 0.7).
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64)
    
    if temperature < 1e-8:
        # Near-zero temperature → greedy
        return int(np.argmax(logits))
    
    scaled_logits = logits / temperature
    probs = _stable_softmax(scaled_logits)
    
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(logits: Array, k: int, rng: Optional[Random] = None) -> int:
    """
    Top-k sampling (also called "filtered sampling").
    
    Only considers the K most probable tokens:
        1. Find the top-K logits
        2. Set all other logits to -infinity
        3. Sample from the remaining K tokens
    
    This prevents the model from sampling very unlikely tokens
    while still allowing diversity among the top candidates.
    
    Effect: Removes the "long tail" of the distribution.
    
    Args:
        logits: Logits (vocab_size,).
        k: Number of top tokens to consider.
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    
    k = min(k, len(logits))
    
    # Find the k-th largest value
    top_k_indices = np.argpartition(logits, -k)[-k:]
    
    # Zero out all tokens not in top-k
    mask = np.full(len(logits), -np.inf)
    mask[top_k_indices] = logits[top_k_indices]
    
    # Sample from the filtered distribution
    probs = _stable_softmax(mask)
    
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


def top_p_sample(logits: Array, p: float, rng: Optional[Random] = None) -> int:
    """
    Top-p (nucleus) sampling.
    
    Selects the smallest set of tokens whose cumulative probability
    exceeds p, then samples from that set.
    
    Algorithm:
        1. Sort tokens by probability (descending)
        2. Compute cumulative sum: C_i = sum_{j<=i} p_j
        3. Find the smallest set S where sum_{j in S} p_j >= p
        4. Set all tokens outside S to probability 0
        5. Re-normalize and sample
    
    Advantage over top-k: dynamically adjusts the candidate set size.
    When the model is confident (one token dominates), fewer tokens are
    considered. When uncertain, more tokens are included.
    
    Reference:
        Holtzman et al., "The Curious Case of Neural Text Degeneration" (2020)
    
    Args:
        logits: Logits (vocab_size,).
        p: Cumulative probability threshold (0, 1]. Default: 0.9.
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    
    # Compute probabilities
    probs = _stable_softmax(logits)
    
    # Sort in descending order
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    # Compute cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)
    
    # Find cutoff: first index where cumsum >= p
    # Remove tokens with cumsum > p (keep one more for the boundary)
    cutoff_idx = np.searchsorted(cumulative_probs, p)
    if cutoff_idx < len(cumulative_probs):
        cutoff_idx += 1  # Include the token that pushes us over p
    
    # Zero out tokens beyond cutoff
    sorted_probs[cutoff_idx:] = 0.0
    
    # Re-normalize
    sorted_probs = sorted_probs / sorted_probs.sum()
    
    # Map back to original indices
    final_probs = np.zeros_like(probs)
    final_probs[sorted_indices] = sorted_probs
    
    # Sample
    if rng is not None:
        return rng.categorical(final_probs)
    return int(np.random.choice(len(final_probs), p=final_probs))


def min_p_sample(logits: Array, min_p: float, rng: Optional[Random] = None) -> int:
    """
    Min-p sampling.
    
    Filters tokens whose probability is below min_p * max_probability.
    
    Unlike top-k and top-p which operate on fixed counts/thresholds,
    min-p dynamically scales with the model's confidence:
        threshold = min_p * max(probabilities)
    
    This is more adaptive than top-k and simpler than top-p.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    probs = _stable_softmax(logits)
    
    # Threshold
    max_prob = np.max(probs)
    threshold = min_p * max_prob
    
    # Filter
    probs[probs < threshold] = 0.0
    
    # Re-normalize
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = _stable_softmax(logits)  # Fallback
    
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


def repetition_penalty_sample(
    logits: Array,
    generated_tokens: List[int],
    penalty: float = 1.1,
    rng: Optional[Random] = None,
) -> int:
    """
    Apply repetition penalty and then sample.
    
    For each previously generated token t:
        If logit[t] > 0: logit[t] /= penalty
        If logit[t] < 0: logit[t] *= penalty
    
    This discourages the model from repeating itself without
    completely banning previously seen tokens.
    
    Args:
        logits: Logits (vocab_size,).
        generated_tokens: List of previously generated token IDs.
        penalty: Penalty factor (> 1.0 penalizes, < 1.0 encourages).
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    
    # Apply repetition penalty
    for token_id in set(generated_tokens):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    # Sample from modified distribution
    probs = _stable_softmax(logits)
    
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


def gumbel_softmax_sample(
    logits: Array,
    temperature: float = 1.0,
    hard: bool = False,
    rng: Optional[Random] = None,
) -> Array:
    """
    Gumbel-Softmax (concrete distribution) sampling.
    
    Differentiable approximation of categorical sampling.
    
    Algorithm:
        1. Draw g_i ~ Gumbel(0, 1): g_i = -log(-log(u_i)), u_i ~ Uniform(0,1)
        2. Compute: y_i = softmax((logit_i + g_i) / temperature)
    
    When temperature → 0, this approaches a one-hot (argmax) sample.
    When temperature → ∞, this approaches the uniform distribution.
    
    If hard=True, uses the straight-through estimator:
        y_hard = one_hot(argmax(logits + gumbel_noise))
        y = y_hard - y_soft.detach() + y_soft  (forward = hard, backward = soft)
    
    This allows gradients to flow through discrete sampling, enabling
    end-to-end training of models with categorical latent variables.
    
    Reference:
        Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (2017)
        Maddison et al., "The Concrete Distribution: A Continuous Relaxation of
                         Discrete Random Variables" (2017)
    
    Args:
        logits: Logits (vocab_size,).
        temperature: Relaxation temperature.
        hard: If True, use straight-through estimator for one-hot.
    
    Returns:
        Sampled probabilities (vocab_size,).
    """
    logits = np.asarray(logits, dtype=np.float64)
    
    # Draw Gumbel noise
    if rng is not None:
        gumbel_noise = rng.gumbel(size=logits.shape)
    else:
        u = np.random.uniform(1e-10, 1.0, size=logits.shape)
        gumbel_noise = -np.log(-np.log(u))
    
    # Gumbel-Softmax
    y = _stable_softmax((logits + gumbel_noise) / temperature)
    
    if hard:
        # Straight-through estimator
        index = np.argmax(y)
        y_hard = np.zeros_like(y)
        y_hard[index] = 1.0
        y = y_hard  # In numpy we just return the hard sample
        # In a framework with autograd, we'd do: y = y_hard - y.detach() + y
    
    return y


def beam_search_sample(
    logits_fn: callable,
    initial_input: Array,
    beam_width: int = 5,
    max_steps: int = 100,
    length_penalty: float = 1.0,
    eos_token_id: int = 2,
) -> Tuple[Array, float]:
    """
    Beam search decoding.
    
    Maintains beam_width hypotheses and expands them at each step.
    Uses log probabilities and length normalization.
    
    Algorithm:
        1. Initialize beams with initial input
        2. For each step:
           a. For each active beam, get next token logits
           b. Compute log-probs for all possible extensions
           c. Score: cumulative_log_prob + length_penalty * log(current_length)
           d. Select top beam_width hypotheses
        3. Return highest-scoring completed sequence
    
    Args:
        logits_fn: Function that takes input_ids and returns logits.
        initial_input: Starting token IDs.
        beam_width: Number of beams to maintain.
        max_steps: Maximum generation steps.
        length_penalty: Exponential penalty on length (1.0 = no penalty).
        eos_token_id: End-of-sequence token ID.
    
    Returns:
        Tuple of (best_sequence, score).
    """
    # Initialize
    beams = [(initial_input, 0.0)]  # (sequence, log_prob)
    completed = []
    
    for step in range(max_steps):
        if not beams:
            break
        
        all_candidates = []
        
        for seq, score in beams:
            # Get logits for the current sequence
            next_logits = logits_fn(seq)
            
            # Get top beam_width expansions
            next_logits = np.asarray(next_logits[-1])  # Last position
            top_k = min(beam_width, len(next_logits))
            top_indices = np.argpartition(next_logits, -top_k)[-top_k:]
            
            for token_id in top_indices:
                new_seq = np.append(seq, token_id)
                # Score with length normalization
                new_length = len(new_seq)
                new_score = score + np.log(next_logits[token_id] + 1e-10)
                # Length penalty: normalized_score = score / length^penalty
                normalized = new_score / (new_length ** length_penalty)
                
                all_candidates.append((new_seq, normalized, new_score))
                
                # Check for EOS
                if token_id == eos_token_id:
                    completed.append((new_seq, normalized))
        
        # Select top beam_width candidates
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = [(seq, raw) for seq, _, raw in all_candidates[:beam_width]
                 if seq[-1] != eos_token_id]
    
    # Add remaining beams to completed
    for seq, score in beams:
        normalized = score / (len(seq) ** length_penalty)
        completed.append((seq, normalized))
    
    # Return best
    if completed:
        completed.sort(key=lambda x: x[1], reverse=True)
        return completed[0][0], completed[0][1]
    
    return initial_input, 0.0


# ================================================================
# COMBINED SAMPLING PIPELINE
# ================================================================

def combined_sample(
    logits: Array,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    generated_tokens: Optional[List[int]] = None,
    rng: Optional[Random] = None,
) -> int:
    """
    Combined sampling pipeline used in most LLMs.
    
    Applies transformations in order:
        1. Repetition penalty
        2. Temperature scaling
        3. Top-k filtering
        4. Top-p (nucleus) filtering
        5. Sample from final distribution
    
    This is the standard pipeline used by LLaMA, Mistral, GPT-NeoX, etc.
    
    Args:
        logits: Raw logits (vocab_size,).
        temperature: Temperature for scaling.
        top_k: Number of top tokens to keep (0 = disabled).
        top_p: Nucleus threshold (1.0 = disabled).
        repetition_penalty: Token repetition penalty (1.0 = disabled).
        generated_tokens: Previously generated tokens for repetition penalty.
    
    Returns:
        Sampled token index.
    """
    logits = np.asarray(logits, dtype=np.float64).copy()
    
    # Step 1: Repetition penalty
    if repetition_penalty != 1.0 and generated_tokens:
        for token_id in set(generated_tokens):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty
    
    # Step 2: Temperature
    if temperature != 1.0:
        if temperature < 1e-8:
            return int(np.argmax(logits))
        logits = logits / temperature
    
    # Step 3: Top-k
    if top_k > 0 and top_k < len(logits):
        top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full(len(logits), -np.inf)
        mask[top_k_indices] = logits[top_k_indices]
        logits = mask
    
    # Step 4: Top-p
    if top_p < 1.0:
        probs = _stable_softmax(logits)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        
        # Remove tokens with cumulative probability above p
        cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
        sorted_probs[cutoff_idx:] = 0.0
        
        # Re-normalize
        sorted_probs = sorted_probs / sorted_probs.sum()
        final_probs = np.zeros_like(probs)
        final_probs[sorted_indices[:cutoff_idx]] = sorted_probs[:cutoff_idx]
        probs = final_probs
    else:
        probs = _stable_softmax(logits)
    
    # Step 5: Sample
    if rng is not None:
        return rng.categorical(probs)
    return int(np.random.choice(len(probs), p=probs))


# ================================================================
# UTILITIES
# ================================================================

def _stable_softmax(logits: Array) -> Array:
    """
    Numerically stable softmax.
    
    softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    
    Subtracting max(x) prevents overflow in exp().
    """
    logits = np.asarray(logits, dtype=np.float64)
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def _log_softmax(logits: Array) -> Array:
    """Numerically stable log-softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    x = logits - np.max(logits)
    return x - np.log(np.sum(np.exp(x)))
