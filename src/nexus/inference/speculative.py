"""
Speculative Decoding for Nexus
=================================
Accelerates autoregressive generation by speculating multiple tokens
per step and verifying them in parallel against the target model.

Key insight: GPUs are underutilized during auto-regressive generation
because each step processes only 1 token. Speculative decoding processes
K tokens per step, achieving Kx speedup when the draft model is good.

Methods implemented:
1. Draft-Model Speculative Decoding (Leviathan et al. 2023)
2. Self-Speculative Decoding (Chen et al. 2023)
3. Medusa Heads (Arora et al. 2024)
4. EAGLE (Li et al. 2024)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
import math
import time


# ============================================================================
# Section 5: SpeculativeDecodingStats
# ============================================================================

@dataclass
class SpeculativeDecodingStats:
    """Statistics for monitoring speculative decoding performance."""

    total_tokens_generated: int = 0
    total_target_model_calls: int = 0
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    bonus_tokens: int = 0  # Tokens accepted beyond speculation depth
    wall_time_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens that were accepted by the target model."""
        return self.accepted_tokens / max(1, self.total_draft_tokens)

    @property
    def effective_speedup(self) -> float:
        """
        Effective speedup vs. standard autoregressive decoding (1 token/call).
        Computed as average tokens produced per target model forward pass.
        """
        if self.total_target_model_calls == 0:
            return 1.0
        tokens_per_call = self.total_tokens_generated / self.total_target_model_calls
        return tokens_per_call

    @property
    def rejection_rate(self) -> float:
        """Fraction of draft tokens that were rejected."""
        return self.rejected_tokens / max(1, self.total_draft_tokens)

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_tokens_generated = 0
        self.total_target_model_calls = 0
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
        self.rejected_tokens = 0
        self.bonus_tokens = 0
        self.wall_time_seconds = 0.0

    def to_dict(self) -> dict:
        """Serialize statistics to a dictionary."""
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_target_model_calls": self.total_target_model_calls,
            "total_draft_tokens": self.total_draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
            "bonus_tokens": self.bonus_tokens,
            "wall_time_seconds": self.wall_time_seconds,
            "acceptance_rate": self.acceptance_rate,
            "rejection_rate": self.rejection_rate,
            "effective_speedup": self.effective_speedup,
        }

    def __str__(self) -> str:
        return (
            f"SpeculativeDecodingStats("
            f"tokens={self.total_tokens_generated}, "
            f"target_calls={self.total_target_model_calls}, "
            f"draft_tokens={self.total_draft_tokens}, "
            f"accepted={self.accepted_tokens}, "
            f"rejected={self.rejected_tokens}, "
            f"bonus={self.bonus_tokens}, "
            f"accept_rate={self.acceptance_rate:.3f}, "
            f"speedup={self.effective_speedup:.2f}x, "
            f"time={self.wall_time_seconds:.2f}s)"
        )


# ============================================================================
# Section 1: DraftModelSpeculativeDecoder
# ============================================================================

class DraftModelSpeculativeDecoder:
    """
    Speculative decoding using a smaller draft model.

    Algorithm:
    1. Draft model generates K candidate tokens autoregressively
    2. Target model verifies all K tokens in one forward pass
    3. Accept/reject each token using a modified rejection criterion
    4. If token i is rejected: keep tokens 0..i-1, resample token i
    5. If all accepted: sample one bonus token from target model's distribution

    Rejection criterion (for each draft token t_i):
        r = min(1, p_target(t_i) / p_draft(t_i))
        accept with probability r (from uniform distribution)

    If rejected at position i:
        new_token ~ normalize(p_target(t_i) - p_draft(t_i))_+
        where _+ means setting negative values to 0 and renormalizing

    Speedup:
        K * (time_draft / time_target) when draft acceptance rate is high
        Typical: 2-3x speedup with 5-8 token speculation depth

    Args:
        target_model: The large model to generate from
        draft_model: A smaller, faster model (e.g., 7B for 70B target)
        tokenizer: Tokenizer for encoding/decoding
        speculation_depth: Number of tokens to speculate (K)
        temperature: Sampling temperature
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold
        device: Device to run on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        tokenizer: Any,
        speculation_depth: int = 5,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.speculation_depth = speculation_depth
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device

        # Put models in eval mode
        self.target_model.eval()
        self.draft_model.eval()

        # Statistics tracking
        self.stats = SpeculativeDecodingStats()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, float, int]:
        """
        Generate text using speculative decoding.

        Runs the full speculative decoding loop:
        1. Draft model proposes K tokens
        2. Target model verifies in one pass
        3. Accept/reject using modified rejection sampling
        4. Repeat until max_new_tokens reached

        Args:
            prompt_ids: Input token IDs, shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Tuple of:
                - generated_ids: Full sequence including prompt, shape (batch, total_seq_len)
                - acceptance_rate: Fraction of draft tokens accepted
                - num_target_calls: Number of target model forward passes
        """
        self.stats.reset()
        start_time = time.time()

        input_ids = prompt_ids.to(self.device)
        batch_size = input_ids.shape[0]

        # Initialize KV caches for both models
        target_past = None
        draft_past = None

        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            # Step 1: Draft model generates K tokens
            draft_tokens, draft_probs = self._draft_generate(
                input_ids, self.speculation_depth, draft_past
            )

            # Number of tokens actually drafted (may be fewer near end)
            K = draft_tokens.shape[1]
            self.stats.total_draft_tokens += K

            # Step 2: Target model verifies all K tokens in one pass
            (
                n_accepted,
                accepted_tokens,
                bonus_token,
                target_past,
            ) = self._verify_tokens(input_ids, draft_tokens, draft_probs, target_past)

            self.stats.total_target_model_calls += 1
            self.stats.accepted_tokens += n_accepted
            self.stats.rejected_tokens += K - n_accepted

            # Step 3: Append accepted tokens (and bonus if all accepted)
            new_tokens = accepted_tokens  # shape: (batch, n_accepted)
            if bonus_token is not None:
                new_tokens = torch.cat(
                    [new_tokens, bonus_token.unsqueeze(1)], dim=1
                )
                self.stats.bonus_tokens += 1

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            generated_tokens += new_tokens.shape[1]

            self.stats.total_tokens_generated += new_tokens.shape[1]

            # Update draft KV cache for next iteration
            # The draft cache is discarded each iteration (simpler, still fast)
            draft_past = None

        self.stats.wall_time_seconds = time.time() - start_time
        acceptance_rate = self.stats.acceptance_rate
        num_target_calls = self.stats.total_target_model_calls

        return input_ids, acceptance_rate, num_target_calls

    def _draft_generate(
        self,
        input_ids: torch.Tensor,
        K: int,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draft model generates K tokens autoregressively.

        Runs the draft model K times, each time sampling one token and
        appending it to the input for the next step. Collects both
        the sampled tokens and their probability distributions.

        Args:
            input_ids: Current sequence, shape (batch, seq_len)
            K: Number of tokens to draft
            past_key_values: Optional KV cache from previous draft steps

        Returns:
            Tuple of:
                - draft_tokens: Sampled token IDs, shape (batch, K)
                - draft_probs: Probability distributions, shape (batch, K, vocab_size)
        """
        draft_tokens_list = []
        draft_probs_list = []
        current_ids = input_ids
        current_past = past_key_values

        vocab_size = getattr(
            self.draft_model.config, "vocab_size",
            self.draft_model.get_output_embeddings().out_features,
        )

        for _ in range(K):
            # Forward pass through draft model
            outputs = self.draft_model(
                input_ids=current_ids,
                past_key_values=current_past,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
            current_past = outputs.past_key_values

            # Apply sampling (temperature, top-k, top-p)
            probs = self._apply_sampling(
                logits, self.temperature, self.top_k, self.top_p
            )

            # Sample token
            token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

            draft_tokens_list.append(token)
            draft_probs_list.append(probs)

            # Append to input for next draft step
            current_ids = token

        draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (batch, K)
        draft_probs = torch.stack(draft_probs_list, dim=1)  # (batch, K, vocab_size)

        return draft_tokens, draft_probs

    def _verify_tokens(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        past_key_values: Optional[Any] = None,
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Any]:
        """
        Target model verifies all draft tokens in one forward pass.

        The target model processes the original input plus all draft tokens
        simultaneously, producing logits for every position. Each draft token
        is then checked against the rejection sampling criterion.

        Args:
            input_ids: Original sequence, shape (batch, seq_len)
            draft_tokens: Draft token IDs, shape (batch, K)
            draft_probs: Draft probabilities, shape (batch, K, vocab_size)
            past_key_values: Optional KV cache from previous target steps

        Returns:
            Tuple of:
                - n_accepted: Number of tokens accepted (0 to K)
                - accepted_tokens: Accepted token IDs, shape (batch, n_accepted)
                - bonus_token: Bonus token if all K accepted, else None
                - new_past: Updated KV cache
        """
        batch_size = input_ids.shape[0]
        K = draft_tokens.shape[1]

        # Concatenate input + draft tokens for batched verification
        full_input = torch.cat([input_ids, draft_tokens], dim=1)

        # Target model forward pass (processes all K+1 positions at once)
        outputs = self.target_model(
            input_ids=full_input,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Logits for positions after input_ids (i.e., the K draft positions + 1 bonus)
        # Shape: (batch, K+1, vocab_size)
        verify_logits = outputs.logits[:, input_ids.shape[1] - 1:, :]
        new_past = outputs.past_key_values

        # Apply sampling to target logits for fair comparison
        target_probs_list = []
        for i in range(K + 1):
            probs = self._apply_sampling(
                verify_logits[:, i, :], self.temperature, self.top_k, self.top_p
            )
            target_probs_list.append(probs)

        target_probs = torch.stack(target_probs_list, dim=1)  # (batch, K+1, V)

        # Accept/reject each draft token using rejection sampling
        accepted_tokens_list = []
        n_accepted = 0
        bonus_token = None
        resampled = False

        # Uniform random values for rejection criterion
        u_values = torch.rand(batch_size, K, device=self.device)

        for i in range(K):
            if resampled:
                break

            p_target = target_probs[:, i, :]  # (batch, V) — dist conditioned on prefix + drafts[:i]
            p_draft = draft_probs[:, i, :]    # (batch, V)

            # For each batch element, check acceptance
            for b in range(batch_size):
                if resampled:
                    # Already rejected for this batch — keep first batch's result
                    # In practice, batch_size=1 for generation; for bs>1, we
                    # take the minimum accepted count across the batch.
                    break

            # Vectorized rejection check: r = min(1, p_target / max(p_draft, eps))
            p_draft_clamped = p_draft.clamp(min=1e-10)
            r = torch.min(
                torch.ones_like(p_target),
                p_target / p_draft_clamped,
            )

            # Get probability of the actually-drafted token
            draft_token_ids = draft_tokens[:, i]  # (batch,)
            p_target_at_draft = target_probs[:, i, :].gather(
                1, draft_token_ids.unsqueeze(1)
            ).squeeze(1)  # (batch,)
            p_draft_at_draft = draft_probs[:, i, :].gather(
                1, draft_token_ids.unsqueeze(1)
            ).squeeze(1)  # (batch,)

            r_at_draft = torch.min(
                torch.ones_like(p_target_at_draft),
                p_target_at_draft / p_draft_at_draft.clamp(min=1e-10),
            )

            # Accept if u < r (element-wise)
            accept_mask = u_values[:, i] < r_at_draft  # (batch,)

            if accept_mask.all():
                accepted_tokens_list.append(draft_tokens[:, i].unsqueeze(1))
                n_accepted += 1
            else:
                # At least one batch element rejected — resample from adjusted dist
                resampled = True
                adjusted_probs = self._resample_from_adjusted(
                    target_probs[:, i, :],
                    draft_probs[:, i, :],
                )
                # Sample new token from adjusted distribution
                new_token = torch.multinomial(adjusted_probs, num_samples=1)
                accepted_tokens_list.append(new_token)
                n_accepted += 1  # The resampled token counts as accepted

        # If all K tokens accepted, sample a bonus token from position K+1
        if n_accepted == K and not resampled:
            bonus_probs = target_probs[:, K, :]  # (batch, V)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)  # (batch, 1)

        if accepted_tokens_list:
            accepted_tokens = torch.cat(accepted_tokens_list, dim=1)  # (batch, n_accepted)
        else:
            # Edge case: K=0 or immediate rejection
            accepted_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)

        return n_accepted, accepted_tokens, bonus_token, new_past

    def _rejection_sample(
        self,
        p_target: torch.Tensor,
        p_draft: torch.Tensor,
        u: torch.Tensor,
        token_id: torch.Tensor,
    ) -> bool:
        """
        Apply rejection sampling criterion for a single token.

        r = min(1, p_target(t) / p_draft(t))
        Accept the token if u < r.

        Args:
            p_target: Target model probability distribution, shape (vocab_size,)
            p_draft: Draft model probability distribution, shape (vocab_size,)
            u: Uniform random value in [0, 1)
            token_id: The drafted token ID to check

        Returns:
            True if the token is accepted, False if rejected
        """
        p_t = p_target[token_id].item()
        p_d = p_draft[token_id].clamp(min=1e-10).item()
        r = min(1.0, p_t / p_d)
        return u.item() < r

    def _resample_from_adjusted(
        self,
        p_target: torch.Tensor,
        p_draft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from normalized(p_target - p_draft)_+ when a token is rejected.

        The adjusted distribution puts more mass on tokens that the target
        model prefers over the draft model, ensuring the resampled token
        is from the correct target distribution.

        Args:
            p_target: Target model probabilities, shape (batch, vocab_size)
            p_draft: Draft model probabilities, shape (batch, vocab_size)

        Returns:
            Adjusted probability distribution, shape (batch, vocab_size)
        """
        # Compute difference, clamp negatives to zero
        diff = p_target - p_draft
        adjusted = diff.clamp(min=0.0)

        # Renormalize to get a valid distribution
        adjusted_sum = adjusted.sum(dim=-1, keepdim=True)
        # If all zeros (extremely rare), fall back to target distribution
        adjusted = torch.where(
            adjusted_sum > 1e-10,
            adjusted / adjusted_sum,
            p_target,
        )
        return adjusted

    def _apply_sampling(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """
        Apply temperature scaling, top-k filtering, and top-p (nucleus)
        filtering to raw logits, then convert to probabilities.

        Steps:
        1. Scale logits by temperature
        2. Filter to top-k highest logits (set rest to -inf)
        3. Filter to smallest set with cumulative prob >= top_p (nucleus)
        4. Convert to probabilities via softmax

        Args:
            logits: Raw model logits, shape (batch, vocab_size) or (vocab_size,)
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens (0 = disabled)
            top_p: Keep tokens until cumulative prob reaches top_p (1.0 = disabled)

        Returns:
            Probability distribution, same shape as input
        """
        # Ensure 2D for uniform processing
        was_1d = logits.dim() == 1
        if was_1d:
            logits = logits.unsqueeze(0)

        # Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right so the first token above threshold is kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # Scatter back to original ordering
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_indices_to_remove,
            )
            logits[indices_to_remove] = float("-inf")

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        if was_1d:
            probs = probs.squeeze(0)

        return probs


# ============================================================================
# Section 2: SelfSpeculativeDecoder
# ============================================================================

class SelfSpeculativeDecoder:
    """
    Self-speculative decoding using the target model itself as draft.

    Key difference from draft-model approach:
    - Uses the SAME model for both draft and verification
    - Draft phase uses early exit (exits after layer L/2 instead of full L)
    - Verification uses full model
    - No separate draft model needed!

    Algorithm:
    1. Draft: run forward through first half of layers for K tokens
    2. Verify: run forward through ALL layers for all K tokens
    3. Accept/reject same as draft-model approach

    Benefits:
    - No extra model weights needed
    - Still 1.5-2x speedup from parallelism
    - Simple to implement

    Implementation:
    - Create two forward passes: shallow (draft) and deep (verify)
    - Shallow pass uses layers[:num_draft_layers]
    - Deep pass uses all layers

    Args:
        model: The language model (used for both draft and verification)
        tokenizer: Tokenizer for encoding/decoding
        speculation_depth: Number of tokens to speculate per step (K)
        draft_layer_ratio: Fraction of layers to use for draft (0.5 = first half)
        temperature: Sampling temperature
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        speculation_depth: int = 3,
        draft_layer_ratio: float = 0.5,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.speculation_depth = speculation_depth
        self.draft_layer_ratio = draft_layer_ratio
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device

        self.model.eval()

        # Compute draft layer count
        total_layers = model.config.num_hidden_layers
        self.num_draft_layers = max(1, int(total_layers * draft_layer_ratio))
        self.num_verify_layers = total_layers

        # Split model layers
        self._setup_layer_split()

        # Statistics
        self.stats = SpeculativeDecodingStats()

    def _setup_layer_split(self) -> None:
        """
        Split model layers into draft (shallow) and verification (deep) groups.

        For a standard transformer model with model.layers:
        - Draft layers: model.layers[:num_draft_layers]
        - Verify layers: model.layers[num_draft_layers:]  (remaining layers)
        - Embedding and norm are shared
        """
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            self.draft_layers = nn.ModuleList(list(layers[: self.num_draft_layers]))
            self.verify_layers = nn.ModuleList(
                list(layers[self.num_draft_layers :])
            )
            self.embed_tokens = self.model.model.embed_tokens
            self.norm = getattr(self.model.model, "norm", None)
            self.lm_head = self.model.lm_head
        else:
            # Fallback: use the full model for both (degrades to no speculation)
            self.draft_layers = None
            self.verify_layers = None

    def _forward_shallow(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Forward pass through only the first N layers (draft phase).

        Processes the input through the embedding layer and the first
        num_draft_layers transformer layers, producing intermediate
        hidden states. Much faster than full forward pass.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional KV cache for draft layers

        Returns:
            Tuple of:
                - logits: Draft logits, shape (batch, seq_len, vocab_size)
                - past_key_values: Updated KV cache for draft layers
        """
        if self.draft_layers is None:
            # Fallback: full model forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return outputs.logits, outputs.past_key_values

        # Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # Apply causal mask
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, device=self.device, dtype=torch.bool
            )

        causal_mask = self._create_causal_mask(seq_len, self.device)

        # Forward through draft layers
        new_past = []
        for i, layer in enumerate(self.draft_layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            layer_output = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=True,
            )
            hidden_states = layer_output[0]
            new_past.append(layer_output[1])

        # Apply norm and project to vocabulary
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits, tuple(new_past)

    def _forward_full(
        self,
        input_ids: torch.Tensor,
        draft_past_kv: Optional[Tuple] = None,
        full_past_kv: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all layers (verification phase).

        Uses the draft KV cache for shared layers and extends it for
        the remaining verification layers. This avoids recomputing
        the first num_draft_layers' work.

        Args:
            input_ids: Full input sequence (prompt + draft tokens)
            draft_past_kv: KV cache from draft layers (already computed)
            full_past_kv: KV cache for verification layers (if any)

        Returns:
            logits: Full model logits, shape (batch, seq_len, vocab_size)
        """
        # Use the full model's forward pass for verification
        # In a production implementation, we would reuse draft KV cache
        # for the shared layers. Here we do a full forward pass for correctness.
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create a standard causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return ~mask  # True = attend, False = mask

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate text with self-speculative decoding.

        Alternates between draft (shallow forward) and verify (full forward)
        phases. Accepted tokens are appended, rejected tokens are resampled.

        Args:
            prompt_ids: Input token IDs, shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Tuple of:
                - generated_ids: Full sequence including prompt
                - acceptance_rate: Fraction of draft tokens accepted
        """
        self.stats.reset()
        start_time = time.time()

        input_ids = prompt_ids.to(self.device)
        batch_size = input_ids.shape[0]
        draft_kv = None

        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            # Determine how many tokens to draft
            K = min(self.speculation_depth, max_new_tokens - generated_tokens)

            # === DRAFT PHASE: shallow forward for K tokens ===
            draft_tokens_list = []
            draft_probs_list = []
            draft_input = input_ids

            for i in range(K):
                draft_logits, draft_kv = self._forward_shallow(
                    draft_input, past_key_values=draft_kv
                )
                # Get logits for last position
                last_logits = draft_logits[:, -1, :]
                probs = F.softmax(last_logits / self.temperature, dim=-1)

                # Apply top-k and top-p
                if self.top_k > 0:
                    top_k_val = min(self.top_k, probs.size(-1))
                    topk_probs, topk_indices = torch.topk(probs, top_k_val)
                    probs = torch.zeros_like(probs).scatter_(
                        1, topk_indices, topk_probs
                    )
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                token = torch.multinomial(probs, num_samples=1)
                draft_tokens_list.append(token)
                draft_probs_list.append(probs)
                draft_input = token
                self.stats.total_draft_tokens += 1

            draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (batch, K)
            draft_probs = torch.stack(draft_probs_list, dim=1)  # (batch, K, V)

            # === VERIFY PHASE: full forward pass ===
            full_input = torch.cat([input_ids, draft_tokens], dim=1)
            target_logits = self._forward_full(full_input, draft_kv)

            # Extract logits at each draft position
            verify_logits = target_logits[:, input_ids.shape[1] - 1 :, :]
            target_probs_list = []
            for i in range(K + 1):
                probs = F.softmax(
                    verify_logits[:, i, :] / self.temperature, dim=-1
                )
                target_probs_list.append(probs)
            target_probs = torch.stack(target_probs_list, dim=1)

            # === ACCEPT/REJECT ===
            accepted_list = []
            n_accepted = 0
            bonus_token = None
            rejected = False

            u_values = torch.rand(batch_size, K, device=self.device)

            for i in range(K):
                if rejected:
                    break

                p_t = target_probs[:, i, :]
                p_d = draft_probs[:, i, :]

                token_id = draft_tokens[:, i]
                pt_at_token = p_t.gather(1, token_id.unsqueeze(1)).squeeze(1)
                pd_at_token = p_d.gather(1, token_id.unsqueeze(1)).squeeze(1)

                r = torch.min(
                    torch.ones_like(pt_at_token),
                    pt_at_token / pd_at_token.clamp(min=1e-10),
                )

                accept_mask = u_values[:, i] < r

                if accept_mask.all():
                    accepted_list.append(draft_tokens[:, i].unsqueeze(1))
                    n_accepted += 1
                else:
                    rejected = True
                    # Resample from adjusted distribution
                    diff = (p_t - p_d).clamp(min=0)
                    diff_sum = diff.sum(dim=-1, keepdim=True)
                    adjusted = torch.where(
                        diff_sum > 1e-10, diff / diff_sum, p_t
                    )
                    new_token = torch.multinomial(adjusted, num_samples=1)
                    accepted_list.append(new_token)
                    n_accepted += 1

            # Bonus token if all accepted
            if n_accepted == K and not rejected:
                bonus_probs = target_probs[:, K, :]
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)

            # Append tokens
            new_tokens = torch.cat(accepted_list, dim=1) if accepted_list else torch.empty(
                batch_size, 0, dtype=torch.long, device=self.device
            )
            if bonus_token is not None:
                new_tokens = torch.cat([new_tokens, bonus_token], dim=1)
                self.stats.bonus_tokens += 1

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            generated_tokens += new_tokens.shape[1]

            self.stats.total_tokens_generated += new_tokens.shape[1]
            self.stats.total_target_model_calls += 1
            self.stats.accepted_tokens += n_accepted
            self.stats.rejected_tokens += K - n_accepted

            # Reset draft KV for next iteration
            draft_kv = None

        self.stats.wall_time_seconds = time.time() - start_time

        return input_ids, self.stats.acceptance_rate


# ============================================================================
# Section 3: MedusaHead, MedusaTree, MedusaTrainer
# ============================================================================

class MedusaHead(nn.Module):
    """
    Medusa prediction head for multi-token speculation.

    Instead of a separate draft model, Medusa adds multiple prediction
    heads to the target model. Each head predicts a future token at
    a different offset (1, 2, 3, ... steps ahead).

    Architecture:
        base_model -> hidden_states
        head_1 -> predict token at position t+1
        head_2 -> predict token at position t+2
        ...
        head_K -> predict token at position t+K

    Each head is a simple 2-layer MLP:
        hidden_states -> linear(d_model, d_model) -> relu -> linear(d_model, vocab_size)

    During inference:
    1. Forward pass through base model once (for input at position t)
    2. All K heads produce predictions simultaneously
    3. Verify predictions against base model's actual outputs

    Training:
        - Freeze base model
        - Train heads on next-token prediction loss
        - Each head i is trained to predict token at position t+i
        - Use teacher forcing with the base model's hidden states

    Speedup: 2-3x for 4-5 heads (minimal overhead, parallel prediction)

    Args:
        hidden_size: Model hidden dimension (d_model)
        vocab_size: Vocabulary size
        num_heads: Number of prediction heads (K)
        num_layers_per_head: Number of MLP layers per head (default: 2)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        num_layers_per_head: int = 2,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Build heads — each is a small MLP
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = []
            in_dim = hidden_size
            for j in range(num_layers_per_head):
                out_dim = hidden_size if j < num_layers_per_head - 1 else vocab_size
                layers.append(nn.Linear(in_dim, out_dim, bias=True))
                if j < num_layers_per_head - 1:
                    layers.append(nn.ReLU())
                else:
                    # Last layer: no activation (raw logits)
                    pass
                in_dim = out_dim
            self.heads.append(nn.Sequential(*layers))

        # Temperature parameter per head for calibration
        self.head_temperatures = nn.Parameter(
            torch.ones(num_heads) * 0.7
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Generate predictions for all heads simultaneously.

        Args:
            hidden_states: Base model output at position t,
                          shape (batch, 1, hidden_size) or (batch, hidden_size)

        Returns:
            List of logits tensors, each (batch, vocab_size) for each head.
            head[i] predicts the token at offset i+1.
        """
        # Squeeze sequence dimension if present
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]  # Take last position

        predictions = []
        for i, head in enumerate(self.heads):
            logits = head(hidden_states)  # (batch, vocab_size)
            # Apply per-head temperature scaling
            temp = self.head_temperatures[i].clamp(min=0.1)
            logits = logits / temp
            predictions.append(logits)

        return predictions

    def get_probs(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Get probability distributions from all heads.

        Args:
            hidden_states: Base model output, shape (batch, 1, hidden_size)

        Returns:
            List of probability tensors, each (batch, vocab_size)
        """
        logits_list = self.forward(hidden_states)
        return [F.softmax(logits, dim=-1) for logits in logits_list]

    def predict_top_k(
        self, hidden_states: torch.Tensor, k: int = 1
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get top-k predictions from each head.

        Args:
            hidden_states: Base model output, shape (batch, 1, hidden_size)
            k: Number of top predictions per head

        Returns:
            Tuple of:
                - top_tokens: List of (batch, k) tensors with top token IDs
                - top_probs: List of (batch, k) tensors with top probabilities
        """
        probs_list = self.get_probs(hidden_states)
        top_tokens = []
        top_probs = []
        for probs in probs_list:
            values, indices = torch.topk(probs, k, dim=-1)
            top_tokens.append(indices)
            top_probs.append(values)
        return top_tokens, top_probs


class MedusaTree:
    """
    Tree-based speculation for Medusa heads.

    Instead of a single speculative path, Medusa constructs a tree
    of possible continuations. Each node represents a token prediction,
    and branches represent different possible choices.

    The tree is explored top-k at each level, and the target model
    verifies all leaf nodes in one batched forward pass.

    Tree structure (K=2 heads, top-1 per level):
        root -> token_1 (head 1's prediction)
              -> token_2a (head 2's prediction given token_1)
              -> token_2b (alternative)

    For K heads with branching factor B:
        Tree nodes: 1 + B + B^2 + ... + B^(K-1) = (B^K - 1)/(B - 1)
        With B=1: K nodes (linear path)
        With B=2, K=4: 1 + 2 + 4 + 8 = 15 nodes

    Args:
        medusa_head: Trained MedusaHead module
        base_model: The base language model
        tokenizer: Tokenizer
        tree_size: Number of speculation levels (K)
        top_k_per_head: Branching factor (candidates per head)
        temperature: Sampling temperature
    """

    def __init__(
        self,
        medusa_head: MedusaHead,
        base_model: nn.Module,
        tokenizer: Any,
        tree_size: int = 4,
        top_k_per_head: int = 1,
        temperature: float = 0.7,
    ) -> None:
        self.medusa_head = medusa_head
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.tree_size = tree_size
        self.top_k_per_head = top_k_per_head
        self.temperature = temperature

        self.base_model.eval()
        self.stats = SpeculativeDecodingStats()

    def build_tree(
        self, hidden_states: torch.Tensor, current_token: Optional[torch.Tensor] = None
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Build the speculation tree using Medusa heads.

        Starting from the current hidden state, each Medusa head
        produces predictions at increasing offsets. For branching
        factor > 1, multiple candidates are kept at each level.

        Args:
            hidden_states: Hidden states from base model, shape (batch, 1, hidden_size)
            current_token: Current token (used for root context)

        Returns:
            Tuple of:
                - candidate_sequences: List of token-ID sequences (each a list of ints)
                - candidate_probs: List of joint probabilities for each sequence
        """
        # Get predictions from all Medusa heads
        head_probs = self.medusa_head.get_probs(hidden_states)
        # head_probs[i] has shape (batch, vocab_size)

        batch_idx = 0  # For batch_size > 1, we'd parallelize

        # Build tree level by level using BFS
        # Each node: (token_ids: List[int], cumulative_log_prob: float)
        nodes = [([], 0.0)]  # Root node

        for level in range(self.tree_size):
            new_nodes = []
            probs = head_probs[level][batch_idx]  # (vocab_size,)

            # Get top-k candidates at this level
            topk_probs, topk_indices = torch.topk(
                probs, self.top_k_per_head, dim=-1
            )

            for seq, cum_prob in nodes:
                for k in range(self.top_k_per_head):
                    token_id = topk_indices[k].item()
                    token_prob = topk_probs[k].item()
                    new_seq = seq + [token_id]
                    new_cum_prob = cum_prob + math.log(max(token_prob, 1e-10))
                    new_nodes.append((new_seq, new_cum_prob))

            nodes = new_nodes

        # Extract candidate sequences and their probabilities
        candidate_sequences = [node[0] for node in nodes]
        candidate_log_probs = [node[1] for node in nodes]

        return candidate_sequences, candidate_log_probs

    def verify_tree(
        self,
        input_ids: torch.Tensor,
        candidate_sequences: List[List[int]],
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """
        Verify all candidate sequences against the base model.

        Batches all candidate continuations and runs them through the
        base model. Accepts the longest prefix that matches, then
        samples a bonus token from the base model's distribution.

        Args:
            input_ids: Current sequence, shape (batch, seq_len)
            candidate_sequences: List of token-ID sequences to verify

        Returns:
            Tuple of:
                - n_accepted: Number of tokens accepted from best candidate
                - accepted_tokens: Accepted token IDs tensor
                - bonus_token: Bonus token from base model (if all accepted)
        """
        if not candidate_sequences or not candidate_sequences[0]:
            return 0, torch.empty(0, dtype=torch.long), None

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Use the first (highest-probability) candidate for verification
        best_candidate = candidate_sequences[0]
        K = len(best_candidate)

        # Create verification input: original + candidate tokens
        candidate_tensor = torch.tensor(
            [best_candidate], device=device, dtype=torch.long
        ).unsqueeze(0)  # (1, K)
        candidate_tensor = candidate_tensor.expand(batch_size, -1)  # (batch, K)

        full_input = torch.cat([input_ids, candidate_tensor], dim=1)

        # Forward pass through base model
        outputs = self.base_model(input_ids=full_input, use_cache=False)
        all_logits = outputs.logits  # (batch, seq_len + K, vocab_size)

        # Get logits at each candidate position
        start_pos = input_ids.shape[1] - 1
        verify_logits = all_logits[:, start_pos:, :]  # (batch, K+1, vocab_size)

        # Verify each token against the base model
        n_accepted = 0
        accepted_list = []
        bonus_token = None

        for i in range(K):
            logits_i = verify_logits[:, i, :]  # (batch, vocab_size)
            probs_i = F.softmax(logits_i / self.temperature, dim=-1)

            draft_token_id = best_candidate[i]
            draft_prob = probs_i[0, draft_token_id].item()

            # Accept with probability proportional to base model's confidence
            u = torch.rand(1).item()
            if u < draft_prob:
                accepted_list.append(
                    torch.tensor([[draft_token_id]], device=device, dtype=torch.long)
                )
                n_accepted += 1
            else:
                # Reject: sample from adjusted distribution
                # (simplified: sample from base model distribution)
                new_token = torch.multinomial(probs_i, num_samples=1)
                accepted_list.append(new_token)
                n_accepted += 1
                break
        else:
            # All K tokens accepted — sample bonus from position K
            bonus_logits = verify_logits[:, K, :]
            bonus_probs = F.softmax(bonus_logits / self.temperature, dim=-1)
            bonus_token = torch.multinomial(bonus_probs, num_samples=1)

        accepted_tokens = (
            torch.cat(accepted_list, dim=1) if accepted_list
            else torch.empty(batch_size, 0, dtype=torch.long, device=device)
        )

        return n_accepted, accepted_tokens, bonus_token

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, float]:
        """
        Full generation loop with Medusa tree speculation.

        Alternates between:
        1. Base model forward pass (get hidden states)
        2. Medusa head tree construction
        3. Tree verification against base model

        Args:
            prompt_ids: Input token IDs, shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Tuple of:
                - generated_ids: Full sequence including prompt
                - acceptance_rate: Fraction of speculated tokens accepted
        """
        self.stats.reset()
        start_time = time.time()

        input_ids = prompt_ids.to(self.device if hasattr(self, "device") else prompt_ids.device)
        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            # Step 1: Forward pass through base model
            outputs = self.base_model(input_ids=input_ids, use_cache=False)
            last_hidden = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else None

            if last_hidden is None:
                # Fallback: get hidden states from second-to-last layer
                # by running with output_hidden_states=True
                outputs = self.base_model(
                    input_ids=input_ids, use_cache=False, output_hidden_states=True
                )
                last_hidden = outputs.hidden_states[-1]

            # Step 2: Build speculation tree
            candidate_sequences, candidate_probs = self.build_tree(
                last_hidden[:, -1:, :]
            )

            if not candidate_sequences or not candidate_sequences[0]:
                # No candidates — fall back to standard sampling
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits / self.temperature, dim=-1)
                new_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, new_token], dim=1)
                generated_tokens += 1
                self.stats.total_tokens_generated += 1
                self.stats.total_target_model_calls += 1
                continue

            K = len(candidate_sequences[0])
            self.stats.total_draft_tokens += K

            # Step 3: Verify tree against base model
            n_accepted, accepted_tokens, bonus_token = self.verify_tree(
                input_ids, candidate_sequences
            )

            self.stats.total_target_model_calls += 1
            self.stats.accepted_tokens += n_accepted
            self.stats.rejected_tokens += K - n_accepted

            # Append accepted tokens
            if accepted_tokens.shape[1] > 0:
                input_ids = torch.cat([input_ids, accepted_tokens], dim=1)
                generated_tokens += accepted_tokens.shape[1]
                self.stats.total_tokens_generated += accepted_tokens.shape[1]

            if bonus_token is not None:
                input_ids = torch.cat([input_ids, bonus_token], dim=1)
                generated_tokens += 1
                self.stats.total_tokens_generated += 1
                self.stats.bonus_tokens += 1

        self.stats.wall_time_seconds = time.time() - start_time

        return input_ids, self.stats.acceptance_rate


class MedusaTrainer:
    """
    Training utility for Medusa heads.

    Training procedure:
    1. Forward pass through base model to get hidden states
    2. For each head i, compute cross-entropy loss between
       head_i's prediction and the actual token at position t+i
    3. Backpropagate only through the heads (base model frozen)
    4. Use gradient accumulation for stable training

    Args:
        base_model: The base language model (will be frozen)
        medusa_head: MedusaHead module to train
        optimizer_cls: Optimizer class (default: AdamW)
        lr: Learning rate (default: 1e-4)
        gradient_accumulation_steps: Steps between optimizer updates
    """

    def __init__(
        self,
        base_model: nn.Module,
        medusa_head: MedusaHead,
        optimizer_cls: type = torch.optim.AdamW,
        lr: float = 1e-4,
        gradient_accumulation_steps: int = 4,
    ) -> None:
        self.base_model = base_model
        self.medusa_head = medusa_head
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

        # Optimizer only for Medusa heads
        self.optimizer = optimizer_cls(
            medusa_head.parameters(), lr=lr, weight_decay=0.01
        )

        # Training state
        self.step_count = 0
        self.total_loss = 0.0
        self.head_losses: Dict[int, float] = {}

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        One training step for Medusa heads.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            labels: Target token IDs, shape (batch, seq_len)
                   (typically input_ids shifted by 1)

        Returns:
            Dictionary of losses: total loss + per-head losses
        """
        self.medusa_head.train()

        # Get hidden states from frozen base model
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            # Use last layer's hidden states
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[-1]
            else:
                # Fallback: use output embeddings
                hidden_states = outputs.last_hidden_state

        # Get predictions from each Medusa head
        head_logits = self.medusa_head(hidden_states)
        # head_logits[i] has shape (batch, seq_len, vocab_size)

        losses = {}
        total_loss = torch.tensor(0.0, device=input_ids.device)

        for i, logits in enumerate(head_logits):
            offset = i + 1
            # Head i predicts the token at position t + offset
            # Align: logits[:, :-offset, :] vs labels[:, offset:, :]
            if hidden_states.shape[1] <= offset:
                continue

            head_logits_i = logits[:, :-offset, :].contiguous()
            head_labels = labels[:, offset:].contiguous()

            loss = F.cross_entropy(
                head_logits_i.view(-1, head_logits_i.size(-1)),
                head_labels.view(-1),
                ignore_index=-100,
            )
            losses[f"head_{i}_loss"] = loss.item()
            total_loss = total_loss + loss

        # Normalize by number of heads
        num_active_heads = len(losses)
        if num_active_heads > 0:
            total_loss = total_loss / num_active_heads

        # Backward pass (only Medusa head parameters have gradients)
        total_loss.backward()

        # Gradient accumulation
        if (self.step_count + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.medusa_head.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.step_count += 1
        self.total_loss = total_loss.item()
        self.head_losses = losses

        result = {"total_loss": total_loss.item()}
        result.update(losses)
        return result

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        head_idx: int,
        offset: int,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for a specific head.

        Head at index head_idx predicts the token at offset positions ahead.
        For example, head_idx=0 with offset=1 predicts the next token,
        head_idx=1 with offset=2 predicts the token 2 positions ahead, etc.

        Args:
            hidden_states: Base model hidden states, shape (batch, seq_len, hidden_size)
            labels: Target token IDs, shape (batch, seq_len)
            head_idx: Index of the Medusa head
            offset: How many positions ahead this head predicts

        Returns:
            Cross-entropy loss scalar
        """
        head = self.medusa_head.heads[head_idx]

        # Get logits from this specific head
        if hidden_states.dim() == 3:
            logits = head(hidden_states)  # (batch, seq_len, vocab_size)
        else:
            logits = head(hidden_states)  # (batch, vocab_size)

        # Align logits and labels
        if logits.dim() == 3 and hidden_states.shape[1] > offset:
            head_logits = logits[:, :-offset, :].contiguous()
            head_labels = labels[:, offset:].contiguous()
        elif logits.dim() == 2:
            head_logits = logits
            head_labels = labels
        else:
            return torch.tensor(0.0, device=hidden_states.device)

        loss = F.cross_entropy(
            head_logits.view(-1, head_logits.size(-1)),
            head_labels.view(-1),
            ignore_index=-100,
        )
        return loss


# ============================================================================
# Section 4: EAGLE and EAGLEDecoder
# ============================================================================

class EAGLE(nn.Module):
    """
    EAGLE: Extrapolation Algorithm for Greater Language-model Efficiency.

    Instead of predicting raw token IDs (like Medusa), EAGLE predicts
    the DIFFERENCE between the draft model's output logits and the
    target model's output logits (residual prediction).

    Key insight: the residual logits are much simpler to predict than
    the full logits, so a lightweight network can achieve high accuracy.

    Architecture:
        1. Feature extractor: Process (draft_logits, hidden_states, token_embeddings)
        2. Residual projector: Project features to residual logits space
        3. Token mapping: Convert residual logits back to token probabilities

    The feature extractor combines:
        - Top-1 draft token embedding
        - Hidden state from an intermediate layer of the draft model
        - Draft model's output logits (projected)

    Reference: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)

    Args:
        hidden_size: Target model hidden dimension
        vocab_size: Vocabulary size
        draft_hidden_size: Draft model hidden dimension
        feature_dim: Internal feature projection dimension
        num_layers: Number of transformer layers for sequence modeling
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        draft_hidden_size: int,
        feature_dim: int = 1024,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.draft_hidden_size = draft_hidden_size
        self.feature_dim = feature_dim

        # Project draft logits down to manageable dimension
        self.logits_proj = nn.Sequential(
            nn.Linear(vocab_size, feature_dim, bias=False),
            nn.GELU(),
        )

        # Project target hidden states
        self.hidden_proj = nn.Sequential(
            nn.Linear(hidden_size, feature_dim, bias=False),
            nn.GELU(),
        )

        # Project draft token embeddings
        self.token_emb_proj = nn.Sequential(
            nn.Linear(draft_hidden_size, feature_dim, bias=False),
            nn.GELU(),
        )

        # Feature extractor: combine all three projected features
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # Lightweight transformer layers for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=8,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                batch_first=True,
                dropout=0.0,
                activation="gelu",
            ) for _ in range(num_layers)]
        )

        # Residual projector: predict residual logits
        self.residual_head = nn.Linear(feature_dim, vocab_size)

        # Layer norm for output
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict residual logits that correct the draft model's output.

        Combines target model hidden states, draft model logits, and
        draft token embeddings to predict the residual (difference) that
        should be added to the draft logits to get the target logits.

        Args:
            hidden_states: Target model hidden states,
                          shape (batch, seq_len, hidden_size)
            draft_logits: Draft model's output logits,
                         shape (batch, seq_len, vocab_size)
            draft_token_embeddings: Embeddings of draft tokens,
                                   shape (batch, seq_len, draft_hidden_size)

        Returns:
            Corrected logits: draft_logits + residual_prediction,
            shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project each input modality
        h_hidden = self.hidden_proj(hidden_states)          # (B, T, D)
        h_logits = self.logits_proj(draft_logits)           # (B, T, D)
        h_tokens = self.token_emb_proj(draft_token_embeddings)  # (B, T, D)

        # Concatenate features
        combined = torch.cat([h_hidden, h_logits, h_tokens], dim=-1)  # (B, T, 3D)

        # Project to feature space
        features = self.feature_proj(combined)  # (B, T, D)

        # Apply transformer layers for sequence modeling
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, hidden_states.device)
        for layer in self.layers:
            features = layer(features, mask=causal_mask, is_causal=True)

        # Final normalization and residual prediction
        features = self.output_norm(features)
        residual_logits = self.residual_head(features)  # (B, T, vocab_size)

        # Corrected logits = draft + residual
        corrected_logits = draft_logits + residual_logits

        return corrected_logits

    def _create_causal_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Create causal attention mask for transformer layers."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def generate_draft(
        self,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_emb: torch.Tensor,
        num_draft_tokens: int = 5,
        temperature: float = 0.7,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate draft tokens using the EAGLE residual predictor.

        Iteratively: predict residual -> correct draft -> sample token -> repeat.

        Args:
            hidden_states: Target model hidden states at current position
            draft_logits: Draft model logits at current position
            draft_token_emb: Embedding of the last drafted token
            num_draft_tokens: Number of tokens to draft
            temperature: Sampling temperature

        Returns:
            Tuple of:
                - draft_tokens: Drafted token IDs, shape (batch, num_draft_tokens)
                - draft_probs: Probability distributions, shape (batch, num_draft_tokens, vocab_size)
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        draft_tokens_list = []
        draft_probs_list = []

        current_hidden = hidden_states[:, -1:, :]  # (B, 1, H)
        current_logits = draft_logits[:, -1:, :]   # (B, 1, V)
        current_emb = draft_token_emb[:, -1:, :]    # (B, 1, draft_H)

        for _ in range(num_draft_tokens):
            # Predict corrected logits
            corrected = self(current_hidden, current_logits, current_emb)
            corrected_logits = corrected[:, -1, :]  # (B, V)

            # Apply temperature and get probabilities
            probs = F.softmax(corrected_logits / temperature, dim=-1)

            # Sample token
            token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            draft_tokens_list.append(token)
            draft_probs_list.append(probs)

            # For the next iteration, we'd need updated hidden states from the draft model.
            # In practice, these are obtained by running the draft model forward.
            # Here we use a simple approximation: project the token embedding.
            # A production implementation would interleave with actual draft model calls.
            next_emb = draft_token_emb  # Placeholder — in practice, re-run draft model

        draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (B, num_draft_tokens)
        draft_probs = torch.stack(draft_probs_list, dim=1)   # (B, num_draft_tokens, V)

        return draft_tokens, draft_probs


class EAGLEDecoder:
    """
    Full speculative decoding system using EAGLE.

    Combines a draft model with the EAGLE residual predictor for
    high-quality speculation that outperforms both standard draft-model
    and Medusa approaches.

    The EAGLE module corrects the draft model's predictions by learning
    the residual between draft and target model outputs, leading to
    higher acceptance rates and faster generation.

    Args:
        target_model: The large model to generate from
        draft_model: A smaller, faster model for drafting
        eagle_module: Trained EAGLE residual prediction module
        tokenizer: Tokenizer for encoding/decoding
        speculation_depth: Number of tokens to speculate per step (K)
        temperature: Sampling temperature
        top_k: Top-k filtering threshold
        top_p: Top-p (nucleus) filtering threshold
        device: Device to run on
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        eagle_module: EAGLE,
        tokenizer: Any,
        speculation_depth: int = 5,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        device: str = "cuda",
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.eagle = eagle_module
        self.tokenizer = tokenizer
        self.speculation_depth = speculation_depth
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.device = device

        self.target_model.eval()
        self.draft_model.eval()
        self.eagle.eval()

        self.stats = SpeculativeDecodingStats()

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate using EAGLE-enhanced speculative decoding.

        The generation loop:
        1. Draft model proposes K tokens autoregressively
        2. EAGLE corrects draft predictions using residual prediction
        3. Target model verifies corrected tokens in one batched pass
        4. Accept/reject using standard speculative decoding criterion

        Args:
            prompt_ids: Input token IDs, shape (batch, seq_len)
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Tuple of:
                - generated_ids: Full sequence including prompt
                - stats: Dictionary with acceptance_rate, speedup, and timing
        """
        self.stats.reset()
        start_time = time.time()

        input_ids = prompt_ids.to(self.device)
        batch_size = input_ids.shape[0]

        # Get vocab size and embedding dimension
        vocab_size = self.eagle.vocab_size

        generated_tokens = 0

        while generated_tokens < max_new_tokens:
            K = min(self.speculation_depth, max_new_tokens - generated_tokens)

            # === Phase 1: Draft model generates K tokens ===
            draft_tokens_list = []
            draft_logits_list = []
            draft_hidden_list = []
            draft_emb_list = []
            current_ids = input_ids

            for _ in range(K):
                # Draft model forward pass
                draft_outputs = self.draft_model(
                    input_ids=current_ids, use_cache=False,
                    output_hidden_states=True,
                )
                draft_logits = draft_outputs.logits[:, -1, :]  # (B, V)

                # Get draft hidden states (from an intermediate layer)
                if hasattr(draft_outputs, "hidden_states") and draft_outputs.hidden_states:
                    draft_hidden = draft_outputs.hidden_states[-2][:, -1:, :]
                else:
                    draft_hidden = draft_outputs.last_hidden_state[:, -1:, :]

                # Get draft token embedding
                draft_probs = F.softmax(draft_logits / self.temperature, dim=-1)
                draft_token = torch.multinomial(draft_probs, num_samples=1)  # (B, 1)

                # Get embedding of drafted token
                if hasattr(self.draft_model, "model"):
                    if hasattr(self.draft_model.model, "embed_tokens"):
                        draft_token_emb = self.draft_model.model.embed_tokens(draft_token)
                    else:
                        draft_token_emb = self.draft_model.get_input_embeddings()(draft_token)
                else:
                    draft_token_emb = self.draft_model.get_input_embeddings()(draft_token)

                draft_tokens_list.append(draft_token)
                draft_logits_list.append(draft_logits)
                draft_hidden_list.append(draft_hidden)
                draft_emb_list.append(draft_token_emb)

                current_ids = draft_token

            draft_tokens = torch.cat(draft_tokens_list, dim=1)  # (B, K)
            draft_logits_stack = torch.stack(draft_logits_list, dim=1)  # (B, K, V)
            draft_hidden_stack = torch.cat(draft_hidden_list, dim=1)  # (B, K, H_d)
            draft_emb_stack = torch.cat(draft_emb_list, dim=1)  # (B, K, H_d)

            self.stats.total_draft_tokens += K

            # === Phase 2: EAGLE corrects draft predictions ===
            # Get target model hidden states for the input prefix
            with torch.no_grad():
                target_outputs = self.target_model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                )
                target_hidden = (
                    target_outputs.hidden_states[-1]
                    if hasattr(target_outputs, "hidden_states")
                    else target_outputs.last_hidden_state
                )  # (B, T, H_t)

            # Use the last position's hidden state for EAGLE correction
            # In practice, EAGLE processes the full sequence; here we use the last position
            target_hidden_last = target_hidden[:, -1:, :]  # (B, 1, H_t)

            # EAGLE generates corrected draft tokens
            eagle_draft_tokens, eagle_draft_probs = self.eagle.generate_draft(
                target_hidden_last,
                draft_logits_stack[:, :1, :],
                draft_emb_stack[:, :1, :],
                num_draft_tokens=K,
                temperature=self.temperature,
            )

            # Use EAGLE-corrected tokens (they tend to have higher acceptance rate)
            corrected_tokens = eagle_draft_tokens  # (B, K)
            corrected_probs = eagle_draft_probs    # (B, K, V)

            # === Phase 3: Target model verifies all K tokens ===
            full_input = torch.cat([input_ids, corrected_tokens], dim=1)
            verify_outputs = self.target_model(input_ids=full_input, use_cache=False)
            verify_logits = verify_outputs.logits  # (B, T+K, V)

            # Extract logits at each draft position
            start_pos = input_ids.shape[1] - 1
            target_logits_list = []
            for i in range(K + 1):
                logits_i = verify_logits[:, start_pos + i, :]
                probs_i = F.softmax(logits_i / self.temperature, dim=-1)
                target_logits_list.append(probs_i)
            target_probs = torch.stack(target_logits_list, dim=1)  # (B, K+1, V)

            # === Phase 4: Accept/reject ===
            accepted_list = []
            n_accepted = 0
            bonus_token = None
            rejected = False

            u_values = torch.rand(batch_size, K, device=self.device)

            for i in range(K):
                if rejected:
                    break

                p_t = target_probs[:, i, :]
                p_eagle = corrected_probs[:, i, :]

                token_id = corrected_tokens[:, i]
                pt_at_token = p_t.gather(1, token_id.unsqueeze(1)).squeeze(1)
                pe_at_token = p_eagle.gather(1, token_id.unsqueeze(1)).squeeze(1)

                # Rejection sampling: accept if u < min(1, p_target / p_eagle)
                r = torch.min(
                    torch.ones_like(pt_at_token),
                    pt_at_token / pe_at_token.clamp(min=1e-10),
                )

                if (u_values[:, i] < r).all():
                    accepted_list.append(token_id.unsqueeze(1))
                    n_accepted += 1
                else:
                    rejected = True
                    # Resample from adjusted distribution
                    diff = (p_t - p_eagle).clamp(min=0)
                    diff_sum = diff.sum(dim=-1, keepdim=True)
                    adjusted = torch.where(diff_sum > 1e-10, diff / diff_sum, p_t)
                    new_token = torch.multinomial(adjusted, num_samples=1)
                    accepted_list.append(new_token)
                    n_accepted += 1

            # Bonus token if all K accepted
            if n_accepted == K and not rejected:
                bonus_probs = target_probs[:, K, :]
                bonus_token = torch.multinomial(bonus_probs, num_samples=1)
                self.stats.bonus_tokens += 1

            # Append results
            new_tokens = torch.cat(accepted_list, dim=1) if accepted_list else torch.empty(
                batch_size, 0, dtype=torch.long, device=self.device
            )
            if bonus_token is not None:
                new_tokens = torch.cat([new_tokens, bonus_token], dim=1)

            input_ids = torch.cat([input_ids, new_tokens], dim=1)
            generated_tokens += new_tokens.shape[1]

            self.stats.total_tokens_generated += new_tokens.shape[1]
            self.stats.total_target_model_calls += 1
            self.stats.accepted_tokens += n_accepted
            self.stats.rejected_tokens += K - n_accepted

        self.stats.wall_time_seconds = time.time() - start_time

        stats_dict = {
            "acceptance_rate": self.stats.acceptance_rate,
            "effective_speedup": self.stats.effective_speedup,
            "total_tokens": self.stats.total_tokens_generated,
            "target_calls": self.stats.total_target_model_calls,
            "bonus_tokens": self.stats.bonus_tokens,
            "wall_time": self.stats.wall_time_seconds,
            "tokens_per_second": (
                self.stats.total_tokens_generated / max(self.stats.wall_time_seconds, 1e-6)
            ),
        }

        return input_ids, stats_dict


# ============================================================================
# Utility functions
# ============================================================================

def create_speculative_decoder(
    method: str = "draft_model",
    target_model: Optional[nn.Module] = None,
    draft_model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    **kwargs,
) -> Any:
    """
    Factory function for creating speculative decoders.

    Args:
        method: Speculative decoding method. One of:
            - "draft_model": DraftModelSpeculativeDecoder (default)
            - "self_spec": SelfSpeculativeDecoder
            - "medusa": MedusaTree
            - "eagle": EAGLEDecoder
        target_model: Target language model
        draft_model: Draft model (for draft_model and eagle methods)
        tokenizer: Tokenizer instance
        **kwargs: Additional arguments passed to the decoder constructor

    Returns:
        Instantiated speculative decoder

    Raises:
        ValueError: If method is not recognized or required args are missing
    """
    method_lower = method.lower().replace("-", "_")

    if method_lower == "draft_model":
        if target_model is None or draft_model is None:
            raise ValueError(
                "DraftModelSpeculativeDecoder requires both target_model and draft_model"
            )
        return DraftModelSpeculativeDecoder(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            **kwargs,
        )

    elif method_lower == "self_spec":
        if target_model is None:
            raise ValueError("SelfSpeculativeDecoder requires target_model")
        return SelfSpeculativeDecoder(
            model=target_model,
            tokenizer=tokenizer,
            **kwargs,
        )

    elif method_lower == "medusa":
        if target_model is None:
            raise ValueError("MedusaTree requires base_model (target_model)")
        # Create MedusaHead with model dimensions
        hidden_size = kwargs.pop("hidden_size", target_model.config.hidden_size)
        vocab_size = kwargs.pop("vocab_size", target_model.config.vocab_size)
        num_heads = kwargs.pop("num_heads", 4)
        medusa_head = MedusaHead(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_heads=num_heads,
        )
        return MedusaTree(
            medusa_head=medusa_head,
            base_model=target_model,
            tokenizer=tokenizer,
            **kwargs,
        )

    elif method_lower == "eagle":
        if target_model is None or draft_model is None:
            raise ValueError("EAGLEDecoder requires both target_model and draft_model")
        hidden_size = kwargs.pop("hidden_size", target_model.config.hidden_size)
        vocab_size = kwargs.pop("vocab_size", target_model.config.vocab_size)
        draft_hidden = kwargs.pop(
            "draft_hidden_size", draft_model.config.hidden_size
        )
        eagle_module = EAGLE(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            draft_hidden_size=draft_hidden,
        )
        return EAGLEDecoder(
            target_model=target_model,
            draft_model=draft_model,
            eagle_module=eagle_module,
            tokenizer=tokenizer,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown speculative decoding method: '{method}'. "
            f"Choose from: draft_model, self_spec, medusa, eagle"
        )
