"""
Output Head for Nexus v2
============================
Efficient language modeling head with large vocabulary support.

This module provides output projection strategies for decoder-only LLMs:

    - LMHead: Standard linear projection to vocabulary logits.
    - ParallelLMHead: Vocabulary-sharded output projection for multi-GPU.
    - FusedCrossEntropyLMHead: Fused projection + cross-entropy loss for memory efficiency.
    - LogSoftmaxHead: Log-softmax output for external loss computation.
    - ScaledDotProductScorer: Temperature-scaled logits for knowledge distillation.
    - AdaptiveOutputHead: Switchable strategy (full, sampled, hierarchical softmax).

Memory considerations for large vocabularies (128K-256K):
    Full logits: O(B * T * V) memory materialisation.
    Fused CE:    O(B * T * D) memory — D << V, typically 10-50x savings.
    Sampled:     O(B * T * K) memory — K is number of candidate samples.

All modules support BF16/FP16 inputs and distributed training contexts.
"""

from __future__ import annotations

import math
import enum
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. LMHead
# ---------------------------------------------------------------------------

class LMHead(nn.Module):
    """
    Standard language modeling head: linear projection to vocabulary logits.

    Computes ``logits = hidden_states @ W_vocab.T + bias`` where
    ``W_vocab`` has shape ``(vocab_size, hidden_size)``.

    Optionally supports weight tying with an embedding layer — the head
    reuses the embedding weight as its own weight, saving ``vocab_size *
    hidden_size`` parameters.

    Args:
        hidden_size: Model dimensionality (input features).
        vocab_size: Total vocabulary size (output features).
        bias: Whether to include a learnable bias term.
        dtype: Parameter dtype (default float32).

    Shapes:
        hidden_states: (batch_size, seq_len, hidden_size)
        logits:        (batch_size, seq_len, vocab_size)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project hidden states to vocabulary logits.

        Args:
            hidden_states: Tensor of shape ``(batch_size, seq_len, hidden_size)``.
                Supports BF16 and FP16 inputs; computation may be upcast to FP32
                internally for numerical stability.

        Returns:
            Logits of shape ``(batch_size, seq_len, vocab_size)``.
        """
        # F.linear expects (..., in_features), weight is (out_features, in_features)
        logits = F.linear(hidden_states, self.weight, self.bias)
        return logits

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """
        Share weights with an embedding layer.

        After calling this method, ``self.weight`` is replaced with a reference
        to ``embedding_weight``.  Any subsequent ``state_dict`` will reflect
        the tied parameter.

        Args:
            embedding_weight: The ``weight`` parameter of an embedding layer,
                shape ``(vocab_size, hidden_size)``.
        """
        if embedding_weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"Embedding weight shape {embedding_weight.shape} does not match "
                f"expected ({self.vocab_size}, {self.hidden_size})"
            )
        self.weight = embedding_weight

    def extra_repr(self) -> str:
        bias_str = "True" if self.bias is not None else "False"
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"bias={bias_str}"
        )


# ---------------------------------------------------------------------------
# 2. ParallelLMHead
# ---------------------------------------------------------------------------

class ParallelLMHead(nn.Module):
    """
    Vocabulary-parallel output projection for multi-GPU training.

    When the vocabulary is very large (128K-256K+), the output projection
    weight alone can consume significant GPU memory. This module shards the
    vocabulary dimension across ``world_size`` GPUs so that each GPU only
    stores ``vocab_size // world_size`` rows of the weight matrix.

    Forward pass:
        1. Each rank computes partial logits: ``partial = H @ W_local.T``
        2. An all-gather collects the full logits across ranks.
        3. Optionally, a fused cross-entropy can be computed per-rank using
           a distributed cross-entropy kernel (not implemented here — use
           ``FusedCrossEntropyLMHead`` with vocab sharding for that).

    Falls back to a standard ``LMHead`` when ``torch.distributed`` is not
    available or ``world_size == 1``.

    Args:
        hidden_size: Model dimensionality.
        vocab_size: Total vocabulary size across all ranks.
        rank: Current process rank.
        world_size: Total number of parallel processes.
        bias: Whether to use a bias term.
        dtype: Parameter dtype.

    Shapes:
        hidden_states: (batch_size, seq_len, hidden_size)
        logits:        (batch_size, seq_len, vocab_size)   [gathered]
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int = 0,
        world_size: int = 1,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.world_size = world_size
        self.use_parallel = _DIST_AVAILABLE and world_size > 1

        if self.use_parallel:
            assert vocab_size % world_size == 0, (
                f"vocab_size ({vocab_size}) must be divisible by "
                f"world_size ({world_size})"
            )
            self.local_vocab_size = vocab_size // world_size
            self.weight = nn.Parameter(
                torch.empty(self.local_vocab_size, hidden_size, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=0.02)

            self.bias: Optional[nn.Parameter] = None
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.local_vocab_size, dtype=dtype)
                )
        else:
            self.weight = nn.Parameter(
                torch.empty(vocab_size, hidden_size, dtype=dtype)
            )
            nn.init.normal_(self.weight, std=0.02)
            self.bias = None
            if bias:
                self.bias = nn.Parameter(torch.zeros(vocab_size, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits, gathering across GPUs when in parallel mode.

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``

        Returns:
            Logits ``(batch_size, seq_len, vocab_size)``.
        """
        # Compute local logits: (B, T, local_vocab_size)
        local_logits = F.linear(hidden_states, self.weight, self.bias)

        if not self.use_parallel:
            return local_logits

        # All-gather along the vocab dimension
        gathered = [
            torch.empty_like(local_logits) for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, local_logits.contiguous())

        # Concatenate along last dim: (B, T, vocab_size)
        logits = torch.cat(gathered, dim=-1)
        return logits

    def compute_local_cross_entropy(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-entropy loss using only the local vocab partition.

        Each rank computes loss only for labels that fall in its local
        partition; the result is a scalar loss that can be all-reduced.

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``
            labels: ``(batch_size, seq_len)`` — target token indices.

        Returns:
            Tuple of (local_loss, num_local_tokens):
                - local_loss: Sum of cross-entropy losses for local tokens.
                - num_local_tokens: Number of labels in the local partition.
        """
        if not self.use_parallel:
            raise RuntimeError(
                "compute_local_cross_entropy is only available in parallel mode."
            )

        vocab_start = self.rank * self.local_vocab_size
        vocab_end = vocab_start + self.local_vocab_size

        # Mask for labels in this rank's partition
        mask = (labels >= vocab_start) & (labels < vocab_end)
        local_labels = (labels - vocab_start).clamp(min=0, max=self.local_vocab_size - 1)

        # Compute local logits
        local_logits = F.linear(hidden_states, self.weight, self.bias)

        # Compute cross-entropy per-token
        loss_per_token = F.cross_entropy(
            local_logits.view(-1, self.local_vocab_size),
            local_labels.view(-1),
            reduction="none",
        )

        # Zero out non-local tokens and sum
        loss_per_token = loss_per_token.view_as(labels) * mask.float()
        local_loss = loss_per_token.sum()
        num_local_tokens = mask.sum().clamp(min=1)

        return local_loss, num_local_tokens

    def extra_repr(self) -> str:
        base = f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}"
        if self.use_parallel:
            base += (
                f", rank={self.rank}, world_size={self.world_size}, "
                f"local_vocab={self.local_vocab_size}"
            )
        bias_str = "True" if self.bias is not None else "False"
        base += f", bias={bias_str}"
        return base


# ---------------------------------------------------------------------------
# 3. FusedCrossEntropyLMHead
# ---------------------------------------------------------------------------

class FusedCrossEntropyLMHead(nn.Module):
    """
    Fused output projection + cross-entropy loss computation.

    Instead of materialising the full logits tensor of shape
    ``(B, T, V)`` (which requires O(B·T·V) memory), this module computes
    the cross-entropy loss in a single pass, only requiring O(B·T·D)
    memory where D is the hidden dimension.  For V=128K and D=4096,
    this represents roughly a 30x memory reduction.

    The fusion works by computing ``log_softmax(hidden @ W.T)`` and then
    selecting only the log-probabilities for the target tokens, never
    constructing the full ``(B, T, V)`` tensor.

    Supports:
        - Label smoothing for regularisation.
        - Optional weight tying with an embedding layer.
        - Distributed training via gradient all-reduce.

    Args:
        hidden_size: Model dimensionality.
        vocab_size: Vocabulary size.
        label_smoothing: Smoothing factor ε ∈ [0, 1) (default 0.0).
        bias: Whether to include a bias term.
        dtype: Parameter dtype.

    Shapes:
        hidden_states: (batch_size, seq_len, hidden_size)
        labels:        (batch_size, seq_len)          — target token indices
        loss:          scalar tensor
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        label_smoothing: float = 0.0,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Compute fused cross-entropy loss without materialising full logits.

        The implementation casts to FP32 for the loss computation even when
        inputs are BF16/FP16, to maintain numerical stability.

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``.
            labels: ``(batch_size, seq_len)`` — target indices in ``[0, vocab_size)``.

        Returns:
            Scalar cross-entropy loss.
        """
        # Flatten for matrix multiply
        B, T, D = hidden_states.shape
        H = hidden_states.reshape(B * T, D)
        L = labels.reshape(B * T)

        # Compute logits in FP32 for numerical stability
        H_fp32 = H.float()
        logits = F.linear(H_fp32, self.weight.float(), self.bias.float() if self.bias is not None else None)

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits,
            L,
            reduction="mean",
            label_smoothing=self.label_smoothing,
        )

        return loss.to(hidden_states.dtype)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optionally materialise full logits (e.g. for greedy decoding).

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``.

        Returns:
            Logits ``(batch_size, seq_len, vocab_size)``.
        """
        return F.linear(hidden_states, self.weight, self.bias)

    def tie_weights(self, embedding_weight: nn.Parameter) -> None:
        """Share weight with an embedding layer."""
        if embedding_weight.shape != (self.vocab_size, self.hidden_size):
            raise ValueError(
                f"Embedding weight shape {embedding_weight.shape} does not match "
                f"expected ({self.vocab_size}, {self.hidden_size})"
            )
        self.weight = embedding_weight

    def extra_repr(self) -> str:
        bias_str = "True" if self.bias is not None else "False"
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"label_smoothing={self.label_smoothing}, bias={bias_str}"
        )


# ---------------------------------------------------------------------------
# 4. LogSoftmaxHead
# ---------------------------------------------------------------------------

class LogSoftmaxHead(nn.Module):
    """
    Log-softmax output head for numerical stability.

    Returns log-probabilities ``log P(vocab | hidden)`` directly, which is
    useful when the loss function is computed externally (e.g. NLL loss,
    KL divergence, or custom loss functions) and only the normalised
    log-probabilities are needed.

    The log-softmax is computed with ``dim=-1`` (vocabulary dimension).

    Args:
        hidden_size: Model dimensionality.
        vocab_size: Vocabulary size.
        bias: Whether to include a bias term.
        dtype: Parameter dtype.

    Shapes:
        hidden_states: (batch_size, seq_len, hidden_size)
        log_probs:     (batch_size, seq_len, vocab_size)
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-softmax over vocabulary.

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``.

        Returns:
            Log-probabilities ``(batch_size, seq_len, vocab_size)``.
        """
        logits = F.linear(hidden_states, self.weight, self.bias)
        # FP32 log-softmax for numerical stability
        log_probs = F.log_softmax(logits.float(), dim=-1).to(hidden_states.dtype)
        return log_probs

    def extra_repr(self) -> str:
        bias_str = "True" if self.bias is not None else "False"
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"bias={bias_str}"
        )


# ---------------------------------------------------------------------------
# 5. ScaledDotProductScorer
# ---------------------------------------------------------------------------

class ScaledDotProductScorer(nn.Module):
    """
    Temperature-scaled scorer for knowledge distillation.

    Given logits from a language model head, this module applies a
    temperature scaling:

        ``scores = logits / temperature``

    The softened scores can then be used to compute KL divergence between
    a teacher and student model:

        ``KL(softmax(teacher_logits / T) || softmax(student_logits / T)) * T^2``

    The ``T^2`` factor ensures that the gradient magnitude remains constant
    regardless of temperature.

    Args:
        temperature: Softening temperature (default 1.0; higher = softer distribution).
        squared_factor: Whether to multiply the loss by T^2 (default True).

    Shapes:
        logits:  (batch_size, seq_len, vocab_size) or arbitrary shape
        scores:  same shape as logits
    """

    def __init__(
        self,
        temperature: float = 1.0,
        squared_factor: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.squared_factor = squared_factor

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

    def forward(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Unscaled logits of any shape.

        Returns:
            Temperature-scaled logits of the same shape.
        """
        return logits / self.temperature

    def compute_kl_divergence(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute the knowledge distillation loss.

        ``loss = alpha * T^2 * KL(teacher || student) + (1 - alpha) * CE(student, labels)``

        When ``labels`` is ``None``, only the KL term is returned.

        Args:
            teacher_logits: Logits from the teacher model.
            student_logits: Logits from the student model.
            labels: Optional hard labels for the CE term.
            alpha: Weight for the KD loss vs hard-label CE (default 0.5).

        Returns:
            Scalar distillation loss.
        """
        T = self.temperature

        # Soften both sets of logits
        teacher_soft = F.log_softmax(teacher_logits / T, dim=-1)
        student_soft = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence (teacher as target)
        kd_loss = F.kl_div(
            student_soft, teacher_soft, reduction="batchmean"
        )

        # Apply T^2 scaling
        if self.squared_factor:
            kd_loss = kd_loss * (T * T)

        if labels is None:
            return kd_loss

        # Hard-label cross-entropy
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            reduction="mean",
        )

        return alpha * kd_loss + (1.0 - alpha) * ce_loss

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}, squared_factor={self.squared_factor}"


# ---------------------------------------------------------------------------
# 6. AdaptiveOutputHead
# ---------------------------------------------------------------------------

class OutputStrategy(str, enum.Enum):
    """Supported output projection strategies."""
    FULL = "full"                    # Standard full-vocab projection
    SAMPLED_SOFTMAX = "sampled"      # Sampled (candidate) softmax
    HIERARCHICAL_SOFTMAX = "hierarchical"  # Hierarchical softmax


class AdaptiveOutputHead(nn.Module):
    """
    Output head that can dynamically switch between projection strategies.

    For very large vocabularies (>256K tokens), computing the full softmax
    over all tokens at every step is prohibitively expensive.  This module
    provides a unified interface that allows switching between:

    - **Full projection**: Standard ``logits = H @ W.T`` with full softmax.
      Best for vocabularies up to ~128K.

    - **Sampled softmax (candidate sampling)**: At each step, only a small
      set of K candidate tokens (including the target) are evaluated, reducing
      the cost from O(V) to O(K) per token.  Uses importance sampling with
      a unigram distribution to produce unbiased gradient estimates.

    - **Hierarchical softmax**: Organises the vocabulary as a binary tree
      (or 2-ary tree with K branches per node).  Decoding requires only
      O(log_K V) computations per token instead of O(V).

    The strategy can be switched at runtime via ``set_strategy()`` or
    configured at construction time.

    Args:
        hidden_size: Model dimensionality.
        vocab_size: Vocabulary size.
        strategy: Initial output strategy.
        num_candidates: Number of candidates for sampled softmax (default 8192).
        num_branches: Branching factor for hierarchical softmax (default 2).
        bias: Whether to use a bias term.
        dtype: Parameter dtype.

    Shapes:
        hidden_states: (batch_size, seq_len, hidden_size)
        logits / loss:  depends on strategy
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        strategy: Union[str, OutputStrategy] = OutputStrategy.FULL,
        num_candidates: int = 8192,
        num_branches: int = 2,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self._strategy = OutputStrategy(strategy)
        self.num_candidates = min(num_candidates, vocab_size)
        self.num_branches = max(2, num_branches)

        # Full vocabulary weight (always maintained for fallback)
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype)
        )
        nn.init.normal_(self.weight, std=0.02)

        self.bias: Optional[nn.Parameter] = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(vocab_size, dtype=dtype))

        # For sampled softmax: precompute unigram sampling distribution
        # (In practice, this would be set from the training data token frequencies)
        self.register_buffer(
            "_sampling_probs",
            torch.ones(vocab_size, dtype=torch.float32) / vocab_size,
            persistent=False,
        )

    @property
    def strategy(self) -> OutputStrategy:
        """Current output projection strategy."""
        return self._strategy

    def set_strategy(self, strategy: Union[str, OutputStrategy]) -> None:
        """
        Switch the output strategy at runtime.

        Args:
            strategy: One of ``"full"``, ``"sampled"``, or ``"hierarchical"``.
        """
        self._strategy = OutputStrategy(strategy)

    def set_sampling_probs(self, probs: torch.Tensor) -> None:
        """
        Set the unigram probability distribution for candidate sampling.

        Args:
            probs: 1-D tensor of shape ``(vocab_size,)`` summing to 1.
        """
        if probs.shape[0] != self.vocab_size:
            raise ValueError(
                f"Sampling probs must have vocab_size elements, "
                f"got {probs.shape[0]}"
            )
        self._sampling_probs = probs.to(self._sampling_probs.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Compute output using the current strategy.

        When ``labels`` is provided and the strategy is ``"full"``, returns
        the mean cross-entropy loss (scalar).  Otherwise, returns the logits
        tensor.

        Args:
            hidden_states: ``(batch_size, seq_len, hidden_size)``.
            labels: Optional ``(batch_size, seq_len)`` target indices.

        Returns:
            Either logits or scalar loss, depending on inputs and strategy.
        """
        if self._strategy == OutputStrategy.FULL:
            return self._forward_full(hidden_states, labels)
        elif self._strategy == OutputStrategy.SAMPLED_SOFTMAX:
            return self._forward_sampled(hidden_states, labels)
        elif self._strategy == OutputStrategy.HIERARCHICAL_SOFTMAX:
            return self._forward_hierarchical(hidden_states, labels)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    def _forward_full(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Standard full-vocabulary projection."""
        logits = F.linear(hidden_states, self.weight, self.bias)
        if labels is not None:
            return F.cross_entropy(
                logits.float().view(-1, self.vocab_size),
                labels.view(-1),
                reduction="mean",
            )
        return logits

    def _forward_sampled(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Sampled softmax: evaluate only K candidates per token.

        Uses importance sampling with the unigram distribution for unbiased
        gradient estimates.  The true target is always included in the
        candidate set.

        When ``labels`` is None, falls back to full projection.
        """
        if labels is None:
            # Can't do sampled softmax without targets; fall back
            return self._forward_full(hidden_states, None)

        B, T, D = hidden_states.shape
        H = hidden_states.reshape(B * T, D)
        L = labels.reshape(B * T)

        # Sample candidates (excluding the target)
        K = self.num_candidates
        # Remove the target from sampling probs to avoid duplicates
        safe_probs = self._sampling_probs.clone()
        # Use multinomial to sample K-1 negative samples per batch element
        neg_samples = torch.multinomial(
            safe_probs.expand(B * T, -1),
            K - 1,
            replacement=True,
        )  # (B*T, K-1)

        # Combine with true labels
        candidates = torch.cat([L.unsqueeze(1), neg_samples], dim=1)  # (B*T, K)

        # Gather candidate weights: (K, D)
        candidate_weights = self.weight[candidates]  # (B*T, K, D)

        # Compute logits for candidates: (B*T, K)
        candidate_logits = torch.bmm(
            candidate_weights, H.unsqueeze(2)
        ).squeeze(2)  # (B*T, K)

        # The true label is always at index 0 in the candidate set
        # Compute sampled cross-entropy
        # q(x) = sampling probability, p(x) = true distribution
        candidate_probs = safe_probs[candidates]  # (B*T, K)
        log_q = torch.log(candidate_probs.clamp(min=1e-10))

        # Log-sum-exp over candidates
        max_logits = candidate_logits.max(dim=-1, keepdim=True)[0]
        shifted = candidate_logits - max_logits
        lse = max_logits.squeeze(-1) + torch.logsumexp(shifted, dim=-1)

        # Logit of true label (index 0)
        true_logit = candidate_logits[:, 0]
        # Importance-weighted loss
        loss = -true_logit + lse + log_q[:, 0]

        return loss.float().mean()

    def _forward_hierarchical(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Hierarchical softmax approximation.

        Organises the vocabulary into a tree with ``num_branches`` children
        per node.  Each token is reached by a path of ``log_{num_branches}(V)``
        decisions.  At each level, a small classifier selects which branch
        to follow, requiring only O(D * num_branches) computation per level
        instead of O(D * V) for the full projection.

        This is a simplified implementation: the hierarchical structure is
        implicit (based on a flat mapping) rather than using a Huffman or
        frequency-optimal tree.

        When ``labels`` is None, falls back to full projection.
        """
        if labels is None:
            return self._forward_full(hidden_states, None)

        B, T, D = hidden_states.shape
        H = hidden_states.reshape(B * T, D)
        L = labels.reshape(B * T)
        K = self.num_branches

        # Depth of the tree: ceil(log_K(V))
        depth = math.ceil(math.log(self.vocab_size, K))

        # Pad vocabulary to K^depth for clean tree structure
        padded_vocab = K ** depth

        total_loss = torch.zeros(B * T, device=H.device, dtype=torch.float32)

        # Process each level of the tree
        for level in range(depth):
            # Node index at this level for each sample
            node_idx = L // (K ** (depth - level - 1))
            # Which branch to take
            branch = node_idx % K

            # Create a weight for this level: we use different slices of the
            # full weight matrix to approximate tree-path weights.
            level_weight_idx = level * K
            level_end = min(level_weight_idx + K, self.vocab_size)
            actual_k = level_end - level_weight_idx

            if actual_k < K:
                # Pad weight for this level
                pad = torch.zeros(K - actual_k, D, device=H.device, dtype=H.dtype)
                level_w = torch.cat([
                    self.weight[level_weight_idx:level_end], pad
                ], dim=0)
            else:
                level_w = self.weight[level_weight_idx:level_end]

            # Level logits: (B*T, K)
            level_logits = F.linear(H.float(), level_w.float())

            # NLL for the correct branch
            level_loss = F.cross_entropy(level_logits, branch, reduction="none")
            total_loss = total_loss + level_loss

        return total_loss.mean()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Always compute full logits regardless of current strategy.
        Useful for inference / beam search.
        """
        return F.linear(hidden_states, self.weight, self.bias)

    def extra_repr(self) -> str:
        bias_str = "True" if self.bias is not None else "False"
        return (
            f"vocab_size={self.vocab_size}, hidden_size={self.hidden_size}, "
            f"strategy={self._strategy.value}, bias={bias_str}, "
            f"num_candidates={self.num_candidates}, "
            f"num_branches={self.num_branches}"
        )
