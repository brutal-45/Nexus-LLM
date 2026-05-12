"""
Feed-Forward Network Variants for Nexus v2
==============================================
Complete FFN implementations including standard, gated, and Mixture of Experts.

For a 100B model with SwiGLU FFN (8/3 ratio):
    d_model = 12288, intermediate = 32768
    W_gate: 12288 -> 32768 (400M params)
    W_up:   12288 -> 32768 (400M params)
    W_down: 32768 -> 12288 (400M params)
    Total per layer: ~1.2B params
    80 layers: ~96B FFN params

For MoE with 128 experts, top-2 routing:
    Total params: 80 * 128 * 1.2B = ~12.3T total, but only 2/128 = 1.56% active
    Effective active params per forward: ~80 * 2 * 1.2B = ~192B
    With shared expert: adds ~1.2B per layer

Reference:
    - Shazeer, "GLU Variants Improve Transformer" (2020)
    - Fedus et al., "Switch Transformers" (2022)
    - Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (2022)
    - Du et al., "GLaM: Efficient Scaling of Language Models" (2022)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "StandardFFN",
    "SwiGLUFFNv2",
    "GeGLUFFN",
    "ReGLUFFN",
    "Expert",
    "ExpertRouter",
    "MixtureOfExperts",
    "FineGrainedMoE",
    "create_ffn",
]


# ---------------------------------------------------------------------------
# Activation registry
# ---------------------------------------------------------------------------

_ACTIVATION_REGISTRY: Dict[str, nn.Module] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


def _get_activation(act_name: str) -> nn.Module:
    """Return an activation module by name. Falls back to GELU on unknown."""
    act_name = act_name.lower().strip()
    if act_name in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_name]()
    raise ValueError(
        f"Unknown activation '{act_name}'. "
        f"Choose from: {list(_ACTIVATION_REGISTRY.keys())}"
    )


# ===================================================================
# 1. Standard FFN
# ===================================================================

class StandardFFN(nn.Module):
    """
    Standard two-layer feed-forward network.

    FFN(x) = W2 * activation(W1 * x + b1) + b2

    expansion_ratio = 4: d_model -> 4*d_model -> d_model

    This is the classic position-wise FFN used in the original Transformer
    (Vaswani et al., 2017). The intermediate dimension is typically 4x the
    model hidden size.

    Args:
        hidden_size:      Model hidden dimension (d_model).
        intermediate_size: FFN intermediate dimension (d_ff).
        activation:       Name of the activation function (default "gelu").
        bias:             Whether to use bias in linear layers (default True).
        dropout:          Dropout probability (default 0.0).
        pre_norm:         If True, apply RMSNorm before the first projection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        bias: bool = True,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Optional pre-norm inside the FFN
        self.pre_norm: Optional[nn.RMSNorm] = None
        if pre_norm:
            self.pre_norm = nn.RMSNorm(hidden_size)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.act = _get_activation(activation)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using scaled normal distribution."""
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        residual = x

        if self.pre_norm is not None:
            x = self.pre_norm(x)

        x = self.up_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)

        return x + residual

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"pre_norm={self.pre_norm is not None}"
        )


# ===================================================================
# 2. SwiGLU FFN v2
# ===================================================================

class SwiGLUFFNv2(nn.Module):
    """
    SwiGLU FFN with enhanced features.

    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Default ratio: 8/3 * d_model (from LLaMA).

    The SwiGLU variant uses three weight matrices and applies the SiLU
    activation to the gate branch only, while the up branch passes through
    linearly. Element-wise multiplication creates a dynamic gating signal.

    Compared to a standard FFN with 4x expansion, SwiGLU uses (8/3)x
    expansion to keep the parameter count similar because of the extra
    projection matrix.

    Args:
        hidden_size:       Model hidden dimension.
        intermediate_size: FFN intermediate dimension. If None, set to
                           int(hidden_size * 8 / 3) rounded to nearest
                           multiple of 256 for tensor core efficiency.
        bias:              Whether to add bias to projections (default False).
        dropout:           Dropout probability after down projection.
        pre_norm:          Apply RMSNorm before gate/up projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        if intermediate_size is None:
            # LLaMA-style: round to nearest multiple of 256
            intermediate_size = int((hidden_size * 8) / 3)
            intermediate_size = ((intermediate_size + 255) // 256) * 256
        self.intermediate_size = intermediate_size

        # Optional pre-norm inside the FFN
        self.pre_norm: Optional[nn.RMSNorm] = None
        if pre_norm:
            self.pre_norm = nn.RMSNorm(hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with scaled normal for stability."""
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02 / math.sqrt(2 * self.hidden_size / self.intermediate_size))
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        residual = x

        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # Gate: SiLU(x @ W_gate)
        gate = F.silu(self.gate_proj(x))
        # Up: x @ W_up (no activation)
        up = self.up_proj(x)
        # Gated combination
        hidden = gate * up
        # Down projection
        output = self.down_proj(hidden)

        return self.dropout(output) + residual

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"pre_norm={self.pre_norm is not None}"
        )


# ===================================================================
# 3. GeGLU FFN
# ===================================================================

class GeGLUFFN(nn.Module):
    """
    Gated Linear Unit with GELU activation.

    GeGLU(x) = GELU(x @ W_gate) * (x @ W_up) @ W_down

    GeGLU replaces the SiLU activation in SwiGLU with GELU, offering
    slightly different gradient dynamics. GELU smoothly interpolates
    between ReLU behavior and a linear function near zero:

        GELU(x) = x * Phi(x)

    where Phi is the standard Gaussian CDF.

    Used in PaLM and some Flan-T5 variants.

    Args:
        hidden_size:       Model hidden dimension.
        intermediate_size: FFN intermediate dimension (default None → 4*hidden).
        bias:              Whether to add bias (default False).
        dropout:           Dropout probability (default 0.0).
        pre_norm:          Apply RMSNorm before projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.intermediate_size = intermediate_size

        self.pre_norm: Optional[nn.RMSNorm] = None
        if pre_norm:
            self.pre_norm = nn.RMSNorm(hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        residual = x

        if self.pre_norm is not None:
            x = self.pre_norm(x)

        gate = F.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)

        return self.dropout(output) + residual

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"pre_norm={self.pre_norm is not None}"
        )


# ===================================================================
# 4. ReGLU FFN
# ===================================================================

class ReGLUFFN(nn.Module):
    """
    Gated Linear Unit with ReLU activation.

    ReGLU(x) = ReLU(x @ W_gate) * (x @ W_up) @ W_down

    ReGLU is the simplest gated variant, using ReLU for gating. Despite its
    simplicity, ReGLU has been shown to outperform standard ReLU FFN by a
    significant margin on language modelling benchmarks (Shazeer, 2020).

    The ReLU gating creates hard 0/1 gates that can be more sparse than
    SiLU/GELU gating, which may be beneficial for certain architectures.

    Args:
        hidden_size:       Model hidden dimension.
        intermediate_size: FFN intermediate dimension (default None → 4*hidden).
        bias:              Whether to add bias (default False).
        dropout:           Dropout probability (default 0.0).
        pre_norm:          Apply RMSNorm before projections.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.intermediate_size = intermediate_size

        self.pre_norm: Optional[nn.RMSNorm] = None
        if pre_norm:
            self.pre_norm = nn.RMSNorm(hidden_size)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)
        if self.gate_proj.bias is not None:
            nn.init.zeros_(self.gate_proj.bias)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            (batch, seq_len, hidden_size)
        """
        residual = x

        if self.pre_norm is not None:
            x = self.pre_norm(x)

        gate = F.relu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)

        return self.dropout(output) + residual

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}, "
            f"pre_norm={self.pre_norm is not None}"
        )


# ===================================================================
# 5. Expert Router
# ===================================================================

class ExpertRouter(nn.Module):
    """
    Router network for Mixture of Experts.

    Computes expert assignment scores and dispatches tokens to selected
    experts. Supports multiple routing strategies:

    - **top_k** (default): Standard softmax routing with additive noise,
      selects top-k experts per token.
    - **expert_choice**: Experts choose their preferred tokens instead of
      tokens choosing experts. Better load balancing with heterogeneous
      experts (Zhou et al., 2022).
    - **hash**: Deterministic hash-based routing. No learned parameters,
      perfectly balanced. Uses token position hashing.

    Load balancing auxiliary loss:
        L_aux = alpha * N * sum(f_i * P_i)

    where:
        f_i = fraction of tokens dispatched to expert i
        P_i = average routing probability for expert i
        N   = number of experts
        alpha = load_balance_coeff

    When f_i ≈ P_i for all experts (uniform distribution), the loss is
    minimized, encouraging balanced utilization.

    Args:
        hidden_size:             Model hidden dimension.
        num_experts:             Total number of experts.
        top_k:                   Number of experts per token (default 2).
        routing_method:          One of "top_k", "expert_choice", "hash".
        load_balance_coeff:      Weight for auxiliary loss (default 0.01).
        jitter_noise:            Std dev of noise added to routing logits
                                 during training for exploration (default 0.1).
        expert_capacity_factor:  Multiplier for per-expert token capacity.
                                 capacity = (tokens / experts) * factor * top_k
    """

    SUPPORTED_METHODS = {"top_k", "expert_choice", "hash"}

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        routing_method: str = "top_k",
        load_balance_coeff: float = 0.01,
        jitter_noise: float = 0.1,
        expert_capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        if routing_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown routing_method '{routing_method}'. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_method = routing_method
        self.load_balance_coeff = load_balance_coeff
        self.jitter_noise = jitter_noise
        self.capacity_factor = expert_capacity_factor

        # Learned router projection (not used for hash routing)
        if routing_method != "hash":
            self.gate = nn.Linear(hidden_size, num_experts, bias=False)
            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize gate projection with small weights for stable routing."""
        nn.init.normal_(self.gate.weight, std=0.02)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            routing_weights:  (batch, seq_len, top_k) normalized weights.
            selected_experts: (batch, seq_len, top_k) expert indices (int64).
            aux_loss:         scalar load balancing auxiliary loss.
        """
        if self.routing_method == "top_k":
            # Compute logits: (batch, seq_len, num_experts)
            logits = self.gate(hidden_states)
            return self._top_k_routing(logits)
        elif self.routing_method == "expert_choice":
            logits = self.gate(hidden_states)
            return self._expert_choice_routing(logits, hidden_states)
        else:
            return self._hash_routing(hidden_states)

    # ------------------------------------------------------------------
    # Top-k routing
    # ------------------------------------------------------------------

    def _top_k_routing(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard top-k routing with optional noise.

        Steps:
        1. Add jitter noise during training for exploration.
        2. Compute softmax over experts.
        3. Select top-k experts per token.
        4. Re-normalize weights over selected experts.
        5. Compute auxiliary load balancing loss.
        """
        batch, seq_len, num_experts = logits.shape

        # Add noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise

        # Full routing probabilities for aux loss computation
        routing_probs = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            routing_probs, self.top_k, dim=-1
        )

        # Re-normalize over selected experts
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Auxiliary loss
        aux_loss = self.compute_aux_loss(routing_probs, top_k_indices)

        return top_k_weights, top_k_indices, aux_loss

    # ------------------------------------------------------------------
    # Expert choice routing
    # ------------------------------------------------------------------

    def _expert_choice_routing(
        self, logits: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert choice routing: experts select their preferred tokens.

        Instead of each token choosing its top-k experts, each expert
        independently selects its top-k tokens. This naturally balances
        load across experts because each expert receives exactly the same
        number of tokens.

        For expert i with capacity c:
            expert_i selects tokens with the highest router logits[i, :]

        The output is restructured to (batch, seq_len, top_k) to match
        the standard top-k interface.

        Args:
            logits:        (batch, seq_len, num_experts) raw routing logits.
            hidden_states: (batch, seq_len, hidden_size) for shape reference.

        Returns:
            routing_weights, selected_experts, aux_loss
        """
        batch, seq_len, num_experts = logits.shape

        # Each expert picks (seq_len * top_k / num_experts) tokens
        tokens_per_expert = max(1, (seq_len * self.top_k) // num_experts)

        # Compute routing probabilities
        routing_probs = F.softmax(logits, dim=-1)

        # Transpose for expert-centric view: (num_experts, batch, seq_len)
        expert_probs = routing_probs.permute(2, 0, 1)

        # Each expert selects its top tokens_per_expert tokens
        # flat: (num_experts, batch * seq_len)
        expert_probs_flat = expert_probs.reshape(num_experts, batch * seq_len)

        k = min(tokens_per_expert, seq_len)
        expert_weights, expert_token_indices = torch.topk(
            expert_probs_flat, k, dim=-1
        )

        # Re-normalize per expert
        expert_weights = expert_weights / (
            expert_weights.sum(dim=-1, keepdim=True) + 1e-9
        )

        # Build output in (batch, seq_len, top_k) format
        # For simplicity, map back: each (expert, token) -> token's perspective
        selected_experts = torch.zeros(
            batch, seq_len, self.top_k, dtype=torch.long, device=logits.device
        )
        routing_weights = torch.zeros(
            batch, seq_len, self.top_k, device=logits.device
        )

        # Scatter: for each expert's selected tokens, fill into token arrays
        slot_idx = torch.zeros(batch, seq_len, dtype=torch.long, device=logits.device)
        for e in range(num_experts):
            for i in range(k):
                flat_idx = expert_token_indices[e, i]  # flat token index
                b = flat_idx // seq_len
                t = flat_idx % seq_len
                s = slot_idx[b, t]
                if s < self.top_k:
                    selected_experts[b, t, s] = e
                    routing_weights[b, t, s] = expert_weights[e, i]
                    slot_idx[b, t] = s + 1

        # Compute aux loss (reuse standard formula with approximate counts)
        aux_loss = self.compute_aux_loss(routing_probs, selected_experts)

        return routing_weights, selected_experts, aux_loss

    # ------------------------------------------------------------------
    # Hash routing
    # ------------------------------------------------------------------

    def _hash_routing(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deterministic hash-based routing.

        No learned parameters. Uses a simple hash of token positions
        to deterministically assign experts, ensuring perfect balance.

        hash(token_position, expert_index) % num_experts

        We generate top_k assignments by offsetting the hash:

        Returns:
            routing_weights: uniform weights (1/top_k) for selected experts.
            selected_experts: deterministic expert indices.
            aux_loss: zero (already perfectly balanced).
        """
        batch, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        selected_experts = torch.zeros(
            batch, seq_len, self.top_k, dtype=torch.long, device=device
        )
        routing_weights = torch.full(
            (batch, seq_len, self.top_k), 1.0 / self.top_k, device=device
        )

        for b in range(batch):
            for s in range(seq_len):
                for k in range(self.top_k):
                    # Simple deterministic hash using prime multipliers
                    h = (b * 2654435761 + s * 40503 + k * 48271) % self.num_experts
                    selected_experts[b, s, k] = h

        # Hash routing has zero aux loss (perfectly balanced by construction)
        aux_loss = torch.zeros(1, device=device)

        return routing_weights, selected_experts, aux_loss

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def compute_aux_loss(
        self, routing_probs: torch.Tensor, selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        L_aux = alpha * N * sum_i(f_i * P_i)

        f_i = fraction of tokens dispatched to expert i
        P_i = mean routing probability for expert i

        When f_i ≈ P_i ≈ 1/N for all experts: loss is minimized (balanced).

        Args:
            routing_probs:    (batch, seq_len, num_experts) full softmax probs.
            selected_experts: (batch, seq_len, top_k) selected expert indices.

        Returns:
            Scalar auxiliary loss.
        """
        # P_i: mean routing probability per expert
        # Average over batch and sequence dimensions
        probs_mean = routing_probs.mean(dim=(0, 1))  # (num_experts,)

        # f_i: fraction of tokens dispatched to each expert
        batch, seq_len, top_k = selected_experts.shape
        total_tokens = batch * seq_len * top_k

        # Flatten selected experts and count occurrences
        flat_experts = selected_experts.reshape(-1)  # (batch*seq*top_k,)
        # Count tokens per expert using bincount
        token_counts = torch.bincount(
            flat_experts, minlength=self.num_experts
        ).float()
        token_fractions = token_counts / (total_tokens + 1e-9)  # (num_experts,)

        # Auxiliary loss
        aux_loss = (
            self.load_balance_coeff
            * self.num_experts
            * (token_fractions * probs_mean).sum()
        )

        return aux_loss

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"routing_method='{self.routing_method}', "
            f"load_balance_coeff={self.load_balance_coeff}"
        )


# ===================================================================
# 6. Expert
# ===================================================================

class Expert(nn.Module):
    """
    Single expert in a Mixture of Experts layer.

    Each expert is a standard SwiGLU feed-forward network. The expert
    is only activated when selected by the router, making MoE models
    highly parameter-efficient at inference time.

    For a 100B MoE model with 128 experts:
        Each expert: ~1.5B params (same as a single FFN in dense model)
        Total: 128 * 1.5B = 192B params per MoE layer
        But only top-2 of 128 are active: ~1.56% parameter utilization

    Args:
        hidden_size:       Model hidden dimension.
        intermediate_size: FFN intermediate dimension.
        expert_id:         Unique identifier for this expert (default 0).
        bias:              Whether to use bias in projections (default False).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_id: int = 0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Full SwiGLU FFN: 3 matrices
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize with small weights for stable expert outputs."""
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_tokens, hidden_size)
        Returns:
            (num_tokens, hidden_size)
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def extra_repr(self) -> str:
        return (
            f"expert_id={self.expert_id}, "
            f"hidden_size={self.hidden_size}, "
            f"intermediate_size={self.intermediate_size}"
        )


# ===================================================================
# 7. Mixture of Experts
# ===================================================================

class MixtureOfExperts(nn.Module):
    """
    Full Mixture of Experts layer.

    Architecture:
        x -> Router -> (routing_weights, expert_indices) per token
        x -> Expert_0, Expert_1, ..., Expert_N -> selected outputs
        output = weighted_sum(selected_expert_outputs, routing_weights)

    Features:
    - Top-k expert selection (default k=2)
    - Optional shared expert (always active for common knowledge)
    - Expert capacity factor (limit tokens per expert)
    - Load balancing auxiliary loss
    - Expert parallelism support (annotations for distributed)
    - Dropout on expert outputs

    For MoE with shared expert:
        output = shared_expert(x) + sum_k(w_k * expert_k(x))

    Expert parallelism:
        Each GPU holds a subset of experts. All-to-all communication
        distributes tokens to expert GPUs. After computation, all-to-all
        returns results to token GPUs. This module provides the interface;
        the actual all-to-all is handled by the distributed runtime.

    Args:
        hidden_size:              Model hidden dimension.
        intermediate_size:        FFN intermediate dimension per expert.
        num_experts:              Total number of routed experts (default 128).
        top_k:                    Number of experts per token (default 2).
        shared_expert:            If True, add a shared expert always active.
        expert_capacity_factor:   Capacity multiplier per expert (default 1.25).
        load_balance_coeff:       Auxiliary loss coefficient (default 0.01).
        routing_method:           "top_k", "expert_choice", or "hash".
        dropout:                  Dropout on combined expert output.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 128,
        top_k: int = 2,
        shared_expert: bool = True,
        expert_capacity_factor: float = 1.25,
        load_balance_coeff: float = 0.01,
        routing_method: str = "top_k",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert_flag = shared_expert
        self.expert_capacity_factor = expert_capacity_factor

        # Router network
        self.router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            routing_method=routing_method,
            load_balance_coeff=load_balance_coeff,
            expert_capacity_factor=expert_capacity_factor,
        )

        # Expert modules
        self.experts = nn.ModuleList(
            [
                Expert(hidden_size, intermediate_size, expert_id=i)
                for i in range(num_experts)
            ]
        )

        # Optional shared expert (always active for common knowledge)
        self.shared_expert: Optional[Expert] = None
        if shared_expert:
            self.shared_expert = Expert(
                hidden_size, intermediate_size, expert_id=-1
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer.

        Steps:
        1. Route: compute expert assignments via router.
        2. Apply capacity constraint (drop tokens beyond capacity).
        3. Dispatch: send tokens to their assigned experts.
        4. Compute: each expert processes its assigned tokens.
        5. Combine: weighted sum of expert outputs.
        6. Optional: add shared expert output.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output:   (batch, seq_len, hidden_size)
            aux_loss: scalar auxiliary loss for load balancing
        """
        batch, seq_len, hidden = hidden_states.shape
        original_shape = hidden_states.shape

        # Step 1: Route tokens to experts
        routing_weights, selected_experts, aux_loss = self.router(hidden_states)

        # Step 2: Apply capacity constraints
        if self.training and self.expert_capacity_factor > 0:
            routing_weights, selected_experts = self._apply_capacity_factor(
                routing_weights, selected_experts, seq_len
            )

        # Step 3-4: Dispatch tokens to experts and compute
        expert_output = self._dispatch_and_compute(
            hidden_states, routing_weights, selected_experts
        )

        # Step 5: Apply dropout
        expert_output = self.dropout(expert_output)

        # Step 6: Add shared expert output
        if self.shared_expert is not None:
            shared_output = self.shared_exp(hidden_states.reshape(-1, hidden))
            shared_output = shared_output.reshape(original_shape)
            expert_output = expert_output + shared_output

        return expert_output, aux_loss

    def _dispatch_and_compute(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently dispatch tokens to experts and compute outputs.

        Uses grouped computation:
        - Group tokens by expert assignment.
        - Compute each expert's batch in parallel.
        - Scatter results back to original positions.

        For GPU efficiency:
        - Groups tokens by (expert_index, k_slot).
        - Uses torch.index_select for gathering inputs.
        - Uses scatter_add for distributing results.

        Args:
            hidden_states:    (batch, seq_len, hidden_size)
            routing_weights:  (batch, seq_len, top_k)
            selected_experts: (batch, seq_len, top_k) expert indices

        Returns:
            output: (batch, seq_len, hidden_size)
        """
        batch, seq_len, hidden = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten batch and sequence dimensions
        flat_hidden = hidden_states.reshape(-1, hidden)  # (B*S, H)
        flat_weights = routing_weights.reshape(-1, self.top_k)  # (B*S, K)
        flat_experts = selected_experts.reshape(-1, self.top_k)  # (B*S, K)

        num_tokens = batch * seq_len
        output = torch.zeros(num_tokens, hidden, device=device, dtype=dtype)

        # Group tokens by expert for efficient batched computation
        for expert_idx in range(self.num_experts):
            # Find all (token, k_slot) pairs where this expert is selected
            # flat_experts: (num_tokens, top_k)
            expert_mask = (flat_experts == expert_idx)

            if not expert_mask.any():
                continue

            # Get token indices and corresponding k-slots
            token_indices, k_slots = torch.where(expert_mask)

            # Gather the input tokens for this expert
            expert_input = flat_hidden[token_indices]  # (n_selected, H)

            # Compute expert output
            expert_result = self.experts[expert_idx](expert_input)  # (n_selected, H)

            # Get routing weights for these (token, k_slot) pairs
            weights = flat_weights[token_indices, k_slots]  # (n_selected,)

            # Weighted scatter-add to output
            output.index_add_(
                0, token_indices, expert_result * weights.unsqueeze(-1)
            )

        return output.reshape(batch, seq_len, hidden)

    def _apply_capacity_factor(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply expert capacity constraint.

        Each expert can process at most:
            capacity = (seq_len / num_experts) * capacity_factor * top_k

        Tokens beyond capacity get their routing weight zeroed (dropped).
        This prevents any single expert from becoming a bottleneck.

        Args:
            routing_weights:  (batch, seq_len, top_k)
            selected_experts: (batch, seq_len, top_k)
            seq_len:          sequence length for capacity calculation

        Returns:
            Updated routing_weights and selected_experts (in-place modified).
        """
        capacity = int(
            (seq_len / self.num_experts) * self.capacity_factor * self.top_k
        )
        capacity = max(capacity, 1)

        batch_size = routing_weights.shape[0]
        device = routing_weights.device

        for b in range(batch_size):
            for k in range(self.top_k):
                experts_k = selected_experts[b, :, k]  # (seq_len,)

                for e in range(self.num_experts):
                    expert_token_mask = experts_k == e
                    count = expert_token_mask.sum().item()

                    if count > capacity:
                        # Keep only the first `capacity` tokens (highest weights
                        # are already at the top due to top_k selection)
                        # Zero out the excess
                        indices = torch.where(expert_token_mask)[0]
                        drop_indices = indices[capacity:]
                        routing_weights[b, drop_indices, k] = 0.0

        return routing_weights, selected_experts

    def get_num_active_params(self) -> int:
        """
        Calculate number of parameters active during a forward pass.

        With top-k routing:
            active = (num_experts_per_token + shared) * params_per_expert
        """
        params_per_expert = (
            self.hidden_size * self.intermediate_size * 3  # gate + up + down
        )
        active_experts = self.top_k
        if self.shared_expert_flag:
            active_experts += 1
        return active_experts * params_per_expert

    def get_total_params(self) -> int:
        """Total parameters in the MoE layer (all experts)."""
        params_per_expert = self.hidden_size * self.intermediate_size * 3
        total = self.num_experts * params_per_expert
        if self.shared_expert_flag:
            total += params_per_expert
        # Add router parameters
        if hasattr(self.router, "gate"):
            total += self.hidden_size * self.num_experts
        return total

    def extra_repr(self) -> str:
        active_params = self.get_num_active_params()
        total_params = self.get_total_params()
        utilization = (active_params / total_params * 100) if total_params > 0 else 0
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"shared_expert={self.shared_expert_flag}, "
            f"active_params={active_params:,}, "
            f"total_params={total_params:,}, "
            f"utilization={utilization:.2f}%"
        )


# ===================================================================
# 8. Fine-Grained Mixture of Experts
# ===================================================================

class FineGrainedMoE(nn.Module):
    """
    Fine-grained Mixture of Experts with many small experts.

    Instead of 128 experts with 1.5B params each, use:
    - 256 experts with 750M params each (more specialized)
    - 512 experts with 375M params each (highly specialized)
    - 1024 experts with 187M params each (very specialized)

    Benefits:
    - Better expert specialization (smaller, more focused experts)
    - Improved routing (more choices per token)
    - Better load balancing (more experts to distribute across)

    Trade-off:
    - Higher routing overhead
    - More all-to-all communication in distributed setting
    - Smaller experts may have less capacity for complex patterns

    Implementation wraps MixtureOfExperts with fine-grained defaults.
    The underlying computation is identical; this class provides a
    convenient interface with sensible defaults for fine-grained setups.

    Args:
        hidden_size:              Model hidden dimension.
        intermediate_size:        Total intermediate dimension distributed
                                 across all experts. Each expert gets
                                 intermediate_size // num_experts.
        num_experts:              Number of fine-grained experts (default 256).
        top_k:                    Experts per token (default 8).
        shared_expert:            Include a shared expert (default True).
        expert_capacity_factor:   Capacity multiplier (default 1.0).
        load_balance_coeff:       Aux loss coefficient (default 0.01).
        routing_method:           Routing strategy (default "top_k").
        dropout:                  Output dropout (default 0.0).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 256,
        top_k: int = 8,
        shared_expert: bool = True,
        expert_capacity_factor: float = 1.0,
        load_balance_coeff: float = 0.01,
        routing_method: str = "top_k",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Distribute the intermediate dimension across experts
        # Each expert gets a fraction of the total intermediate size
        expert_intermediate = max(
            intermediate_size // num_experts,
            hidden_size,  # minimum: same as hidden size
        )
        # Round up to multiple of 64 for hardware efficiency
        expert_intermediate = ((expert_intermediate + 63) // 64) * 64

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_intermediate = expert_intermediate

        # Router with potentially higher top-k
        self.router = ExpertRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            routing_method=routing_method,
            load_balance_coeff=load_balance_coeff,
            expert_capacity_factor=expert_capacity_factor,
        )

        # Fine-grained experts
        self.experts = nn.ModuleList(
            [
                Expert(hidden_size, expert_intermediate, expert_id=i)
                for i in range(num_experts)
            ]
        )

        # Optional shared expert
        self.shared_expert: Optional[Expert] = None
        if shared_expert:
            self.shared_expert = Expert(
                hidden_size, expert_intermediate, expert_id=-1
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through fine-grained MoE.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            output:   (batch, seq_len, hidden_size)
            aux_loss: scalar auxiliary loss
        """
        batch, seq_len, hidden = hidden_states.shape

        # Route
        routing_weights, selected_experts, aux_loss = self.router(hidden_states)

        # Dispatch and compute
        output = self._dispatch_and_compute(
            hidden_states, routing_weights, selected_experts
        )

        output = self.dropout(output)

        # Shared expert
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states.reshape(-1, hidden))
            shared_output = shared_output.reshape(batch, seq_len, hidden)
            output = output + shared_output

        return output, aux_loss

    def _dispatch_and_compute(
        self,
        hidden_states: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch and compute with fine-grained expert grouping.

        Same algorithm as MixtureOfExperts but optimized for many small
        experts. Groups multiple experts together for batch efficiency
        when expert assignment is sparse relative to total expert count.
        """
        batch, seq_len, hidden = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        flat_hidden = hidden_states.reshape(-1, hidden)
        flat_weights = routing_weights.reshape(-1, self.top_k)
        flat_experts = selected_experts.reshape(-1, self.top_k)

        num_tokens = batch * seq_len
        output = torch.zeros(num_tokens, hidden, device=device, dtype=dtype)

        # Build index map for all experts at once
        for expert_idx in range(self.num_experts):
            expert_mask = flat_experts == expert_idx
            if not expert_mask.any():
                continue

            token_indices, k_slots = torch.where(expert_mask)
            expert_input = flat_hidden[token_indices]
            expert_result = self.experts[expert_idx](expert_input)
            weights = flat_weights[token_indices, k_slots]
            output.index_add_(
                0, token_indices, expert_result * weights.unsqueeze(-1)
            )

        return output.reshape(batch, seq_len, hidden)

    def get_active_fraction(self) -> float:
        """
        Fraction of total experts active per token.

        active_fraction = top_k / num_experts
        """
        return self.top_k / self.num_experts

    def get_total_params(self) -> int:
        """Total parameter count across all experts and router."""
        params_per_expert = self.hidden_size * self.expert_intermediate * 3
        total = self.num_experts * params_per_expert
        if self.shared_expert is not None:
            total += params_per_expert
        if hasattr(self.router, "gate"):
            total += self.hidden_size * self.num_experts
        return total

    def extra_repr(self) -> str:
        total_params = self.get_total_params()
        active_frac = self.get_active_fraction()
        return (
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"expert_intermediate={self.expert_intermediate}, "
            f"active_fraction={active_frac:.4f}, "
            f"total_params={total_params:,}"
        )


# ===================================================================
# 9. FFN Factory
# ===================================================================

def create_ffn(config) -> nn.Module:
    """
    Factory function to create FFN from config.

    Supports:
    - "standard": StandardFFN (two-layer with configurable activation)
    - "swiglu":   SwiGLUFFNv2 (gated with SiLU, LLaMA-style)
    - "geglu":    GeGLUFFN (gated with GELU)
    - "reglu":    ReGLUFFN (gated with ReLU)
    - "moe":      MixtureOfExperts (full MoE layer with router + experts)
    - "fine_grained_moe": FineGrainedMoE (many small experts)

    The config object must expose the following attributes (depending on type):
    - All types:  hidden_size
    - Non-MoE:    intermediate_size, hidden_act, bias, hidden_dropout
    - MoE:        num_experts, num_experts_per_token, moe_aux_loss_coeff

    Args:
        config: ModelConfig or dict-like with FFN configuration.

    Returns:
        nn.Module: The instantiated FFN layer.

    Raises:
        ValueError: If ffn_type is not recognized.

    Example:
        >>> ffn = create_ffn(config)  # config.ffn_type == "swiglu"
        >>> output = ffn(x)
    """
    # Handle dict-like configs
    if isinstance(config, dict):
        cfg = type("Cfg", (), config)()
    else:
        cfg = config

    ffn_type = getattr(cfg, "ffn_type", "swiglu")
    hidden_size = getattr(cfg, "hidden_size", 4096)

    if ffn_type == "standard":
        return StandardFFN(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", hidden_size * 4),
            activation=getattr(cfg, "hidden_act", "gelu"),
            bias=getattr(cfg, "bias", True),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
            pre_norm=getattr(cfg, "ffn_pre_norm", False),
        )

    elif ffn_type == "swiglu":
        return SwiGLUFFNv2(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", None),
            bias=getattr(cfg, "bias", False),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
            pre_norm=getattr(cfg, "ffn_pre_norm", False),
        )

    elif ffn_type == "geglu":
        return GeGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", None),
            bias=getattr(cfg, "bias", False),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
            pre_norm=getattr(cfg, "ffn_pre_norm", False),
        )

    elif ffn_type == "reglu":
        return ReGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", None),
            bias=getattr(cfg, "bias", False),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
            pre_norm=getattr(cfg, "ffn_pre_norm", False),
        )

    elif ffn_type == "moe":
        return MixtureOfExperts(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", hidden_size * 4),
            num_experts=getattr(cfg, "num_experts", 128),
            top_k=getattr(cfg, "num_experts_per_token", 2),
            shared_expert=getattr(cfg, "moe_shared_expert", True),
            expert_capacity_factor=getattr(cfg, "moe_capacity_factor", 1.25),
            load_balance_coeff=getattr(cfg, "moe_aux_loss_coeff", 0.01),
            routing_method=getattr(cfg, "moe_routing_method", "top_k"),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
        )

    elif ffn_type == "fine_grained_moe":
        return FineGrainedMoE(
            hidden_size=hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size", hidden_size * 4),
            num_experts=getattr(cfg, "num_experts", 256),
            top_k=getattr(cfg, "num_experts_per_token", 8),
            shared_expert=getattr(cfg, "moe_shared_expert", True),
            expert_capacity_factor=getattr(cfg, "moe_capacity_factor", 1.0),
            load_balance_coeff=getattr(cfg, "moe_aux_loss_coeff", 0.01),
            routing_method=getattr(cfg, "moe_routing_method", "top_k"),
            dropout=getattr(cfg, "hidden_dropout", 0.0),
        )

    else:
        raise ValueError(
            f"Unknown ffn_type '{ffn_type}'. "
            f"Choose from: standard, swiglu, geglu, reglu, moe, fine_grained_moe"
        )
