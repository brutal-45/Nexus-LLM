"""
Nexus Mixture of Experts (MoE)
================================
Complete Mixture of Experts implementation for scaling Nexus LLM.

Architecture overview:
    - Expert: single feed-forward expert (Linear -> GELU -> Linear)
    - TopKGating: router network with load balancing
    - MoELayer: dispatch tokens to top-k experts, capacity factor, token dropping
    - SparseMoETransformerLayer: MoE replacing FFN in transformer
    - MixtureOfExpertsTransformer: full transformer with MoE layers

Key features:
    - Expert parallelism pattern for distributed training
    - Load balancing loss to prevent expert collapse
    - Capacity factor to limit expert workload
    - Token dropping when expert is over capacity
    - Configurable MoE frequency (every N layers)
    - Expert dropout for training robustness
    - Residual MoE connections
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple, List, Dict, Any, Union, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts layers.

    Attributes:
        num_experts: Total number of experts in the MoE layer.
        num_selected: Number of experts selected per token (top-k).
        expert_dim: Hidden dimension of each expert's intermediate layer.
                     If 0, uses hidden_size * 4 (standard expansion).
        capacity_factor: Multiplicative factor for expert capacity.
            Each expert processes at most capacity_factor * (tokens / num_experts) tokens.
        load_balancing_loss_coef: Coefficient for the load balancing auxiliary loss.
        router_jitter_noise: Noise added to router logits during training
            to encourage exploration and improve load balancing.
        aux_loss_coef: Coefficient for the combined auxiliary loss.
        router_bias: Whether to add bias to the router linear layer.
        router_hidden_dim: If > 0, use a 2-layer router with this hidden dim.
        expert_dropout: Dropout applied inside each expert.
        gate_softmax_temperature: Temperature for softmax over router logits.
            Lower values sharpen the distribution.
        drop_tokens: Whether to drop tokens when experts are over capacity.
            If False, route overflow tokens to next-best expert.
        expert_group_size: If > 0, group experts and route within groups.
            Useful for very large expert counts.
        use_residual: Whether to use residual connection from MoE input to output.
        residual_coef: Coefficient for the residual connection.
        parallel_loss: Whether to compute load balancing loss in parallel
            using the scatter/gather pattern.
        normalize_router_input: Whether to apply LayerNorm before the router.
        use_bfloat16_router: Whether to compute router in bfloat16.
    """

    num_experts: int = 8
    num_selected: int = 2
    expert_dim: int = 0
    capacity_factor: float = 1.0
    load_balancing_loss_coef: float = 0.01
    router_jitter_noise: float = 0.1
    aux_loss_coef: float = 0.01
    router_bias: bool = True
    router_hidden_dim: int = 0
    expert_dropout: float = 0.0
    gate_softmax_temperature: float = 1.0
    drop_tokens: bool = True
    expert_group_size: int = 0
    use_residual: bool = False
    residual_coef: float = 0.1
    parallel_loss: bool = True
    normalize_router_input: bool = False
    use_bfloat16_router: bool = False

    def __post_init__(self):
        if self.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.num_experts}")
        if self.num_selected < 1:
            raise ValueError(f"num_selected must be >= 1, got {self.num_selected}")
        if self.num_selected > self.num_experts:
            raise ValueError(
                f"num_selected ({self.num_selected}) must be <= "
                f"num_experts ({self.num_experts})"
            )
        if self.capacity_factor <= 0:
            raise ValueError(
                f"capacity_factor must be > 0, got {self.capacity_factor}"
            )
        if self.gate_softmax_temperature <= 0:
            raise ValueError(
                f"gate_softmax_temperature must be > 0, "
                f"got {self.gate_softmax_temperature}"
            )
        if self.expert_group_size > 0:
            if self.num_experts % self.expert_group_size != 0:
                raise ValueError(
                    f"num_experts ({self.num_experts}) must be divisible by "
                    f"expert_group_size ({self.expert_group_size})"
                )


# =============================================================================
# Utility Functions
# =============================================================================


def _get_expert_dim(expert_dim: int, hidden_size: int) -> int:
    """Resolve expert intermediate dimension.

    If expert_dim is 0, use standard 4x expansion of hidden_size.
    """
    if expert_dim > 0:
        return expert_dim
    return hidden_size * 4


def _stable_top_k(
    logits: torch.Tensor,
    k: int,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Perform top-k with deterministic tie-breaking.

    When logits have equal values, break ties by index to ensure
    reproducibility across devices.

    Args:
        logits: Input tensor of shape (..., num_experts).
        k: Number of top elements to select.
        dim: Dimension along which to select.

    Returns:
        Tuple of (values, indices) from top-k selection.
    """
    if logits.shape[dim] <= k:
        return torch.sort(logits, descending=True, dim=dim)

    values, indices = torch.topk(logits, k=k, dim=dim, largest=True, sorted=True)
    return values, indices


def _compute_capacity(
    num_tokens: int,
    num_experts: int,
    capacity_factor: float,
) -> int:
    """Compute the capacity of each expert.

    Capacity determines the maximum number of tokens each expert processes.
    Tokens beyond capacity are either dropped or routed to other experts.

    Args:
        num_tokens: Total number of tokens to route.
        num_experts: Number of experts.
        capacity_factor: Multiplicative capacity factor.

    Returns:
        Capacity per expert (integer).
    """
    return int(math.ceil(num_tokens * capacity_factor / num_experts))


def _scatter_to_expert(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_capacity: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter tokens to their assigned experts with capacity tracking.

    Tokens are sorted by expert index and padded to capacity for efficient
    batched expert computation.

    Args:
        tokens: Input tokens of shape (num_tokens, hidden_dim).
        expert_indices: Expert assignment of shape (num_tokens,).
        expert_capacity: Maximum tokens per expert.
        num_experts: Total number of experts.

    Returns:
        Tuple of:
            - dispatched: Tokens arranged for expert processing
              (num_experts, expert_capacity, hidden_dim).
            - expert_mask: Boolean mask indicating valid tokens
              (num_experts, expert_capacity).
            - sorted_indices: Original indices for gathering results.
    """
    num_tokens = tokens.shape[0]
    hidden_dim = tokens.shape[1]

    # Create expert assignment with capacity tracking
    # Track how many tokens each expert has received
    expert_counts = torch.zeros(
        num_experts, dtype=torch.long, device=tokens.device
    )

    # Sort tokens by expert index for grouped processing
    sorted_indices = torch.argsort(expert_indices, dim=0, stable=True)
    sorted_tokens = tokens[sorted_indices]
    sorted_expert_ids = expert_indices[sorted_indices]

    # Build dispatched tensor: (num_experts, expert_capacity, hidden_dim)
    dispatched = torch.zeros(
        num_experts, expert_capacity, hidden_dim,
        device=tokens.device, dtype=tokens.dtype,
    )
    expert_mask = torch.zeros(
        num_experts, expert_capacity,
        device=tokens.device, dtype=torch.bool,
    )

    # Create position within each expert for each sorted token
    cumulative = torch.zeros(num_tokens, dtype=torch.long, device=tokens.device)
    for i in range(1, num_tokens):
        if sorted_expert_ids[i] == sorted_expert_ids[i - 1]:
            cumulative[i] = cumulative[i - 1] + 1
        else:
            cumulative[i] = 0

    # Determine which tokens fit within capacity
    within_capacity = cumulative < expert_capacity

    # Compute flat indices into dispatched tensor
    expert_flat = sorted_expert_ids[within_capacity]
    pos_flat = cumulative[within_capacity]

    # Scatter tokens into dispatched tensor
    dispatched[expert_flat, pos_flat] = sorted_tokens[within_capacity]
    expert_mask[expert_flat, pos_flat] = True

    return dispatched, expert_mask, sorted_indices


def _gather_from_expert(
    dispatched_output: torch.Tensor,
    expert_mask: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    gates: torch.Tensor,
    expert_indices: torch.Tensor,
    num_experts: int,
    expert_capacity: int,
) -> torch.Tensor:
    """Gather expert outputs and combine with gating weights.

    Args:
        dispatched_output: Expert outputs of shape
            (num_experts, expert_capacity, hidden_dim).
        expert_mask: Valid token mask of shape (num_experts, expert_capacity).
        sorted_indices: Original token indices for reordering.
        num_tokens: Total number of input tokens.
        gates: Gating weights of shape (num_tokens, num_selected).
        expert_indices: Expert assignments of shape (num_tokens, num_selected).
        num_experts: Total number of experts.
        expert_capacity: Maximum tokens per expert.

    Returns:
        Combined output of shape (num_tokens, hidden_dim).
    """
    hidden_dim = dispatched_output.shape[-1]
    output = torch.zeros(
        num_tokens, hidden_dim,
        device=dispatched_output.device, dtype=dispatched_output.dtype,
    )

    # For each expert selection (k selections per token)
    for selection_idx in range(expert_indices.shape[1]):
        # Get the expert indices for this selection
        sel_expert_ids = expert_indices[:, selection_idx]
        sel_gates = gates[:, selection_idx]

        # Sort by expert id for this selection
        sort_idx = torch.argsort(sel_expert_ids, dim=0, stable=True)
        sorted_sel_expert_ids = sel_expert_ids[sort_idx]
        sorted_sel_gates = sel_gates[sort_idx]
        sorted_original_idx = sort_idx

        # Compute position within expert
        sel_num = sel_expert_ids.shape[0]
        cumulative = torch.zeros(
            sel_num, dtype=torch.long, device=dispatched_output.device
        )
        for i in range(1, sel_num):
            if sorted_sel_expert_ids[i] == sorted_sel_expert_ids[i - 1]:
                cumulative[i] = cumulative[i - 1] + 1
            else:
                cumulative[i] = 0

        # Only gather tokens that fit within capacity
        within_capacity = cumulative < expert_capacity

        expert_flat = sorted_sel_expert_ids[within_capacity]
        pos_flat = cumulative[within_capacity]
        orig_idx = sorted_original_idx[within_capacity]
        gate_vals = sorted_sel_gates[within_capacity]

        # Gather from dispatched output
        gathered = dispatched_output[expert_flat, pos_flat]

        # Apply gating weights and accumulate
        output[orig_idx] += gathered * gate_vals.unsqueeze(-1)

    return output


def _efficient_scatter(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_capacity: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Efficient scatter implementation using bincount and index arithmetic.

    This version avoids explicit Python loops for better GPU utilization.

    Args:
        tokens: Input tokens of shape (num_tokens, hidden_dim).
        expert_indices: Expert assignment of shape (num_tokens,).
        expert_capacity: Maximum tokens per expert.
        num_experts: Total number of experts.

    Returns:
        Tuple of (dispatched, expert_mask, sorted_indices, position_ids).
    """
    num_tokens = tokens.shape[0]
    hidden_dim = tokens.shape[1]

    # Count tokens per expert
    expert_counts = torch.bincount(
        expert_indices, minlength=num_experts
    )

    # Compute positions within each expert using cumulative sum
    # First, sort by expert index
    sorted_indices = torch.argsort(expert_indices, dim=0, stable=True)
    sorted_tokens = tokens[sorted_indices]
    sorted_expert_ids = expert_indices[sorted_indices]

    # Compute position within each expert group
    # Use cumsum trick: shift when expert id changes
    is_same_expert = sorted_expert_ids[1:] == sorted_expert_ids[:-1]
    is_same_expert = torch.cat([
        torch.tensor([False], device=tokens.device),
        is_same_expert,
    ])
    position_ids = torch.cumsum(is_same_expert.long(), dim=0)

    # Determine which tokens fit within capacity
    within_capacity = position_ids < expert_capacity

    # Initialize dispatched tensor
    dispatched = torch.zeros(
        num_experts, expert_capacity, hidden_dim,
        device=tokens.device, dtype=tokens.dtype,
    )
    expert_mask = torch.zeros(
        num_experts, expert_capacity,
        device=tokens.device, dtype=torch.bool,
    )

    # Only scatter tokens within capacity
    valid_expert_ids = sorted_expert_ids[within_capacity]
    valid_positions = position_ids[within_capacity]
    valid_tokens = sorted_tokens[within_capacity]

    dispatched[valid_expert_ids, valid_positions] = valid_tokens
    expert_mask[valid_expert_ids, valid_positions] = True

    # Track how many tokens were dropped
    dropped = (~within_capacity).sum()

    return dispatched, expert_mask, sorted_indices, position_ids


def _efficient_gather(
    dispatched_output: torch.Tensor,
    expert_mask: torch.Tensor,
    sorted_indices: torch.Tensor,
    position_ids: torch.Tensor,
    gates: torch.Tensor,
    expert_indices: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    expert_capacity: int,
) -> torch.Tensor:
    """Efficient gather implementation.

    Args:
        dispatched_output: Expert outputs (num_experts, capacity, hidden_dim).
        expert_mask: Valid mask (num_experts, capacity).
        sorted_indices: Original token indices for reordering.
        position_ids: Position within each expert.
        gates: Gating weights (num_tokens, num_selected).
        expert_indices: Expert assignments (num_tokens, num_selected).
        num_tokens: Total input tokens.
        num_experts: Total number of experts.
        expert_capacity: Capacity per expert.

    Returns:
        Combined output (num_tokens, hidden_dim).
    """
    hidden_dim = dispatched_output.shape[-1]
    output = torch.zeros(
        num_tokens, hidden_dim,
        device=dispatched_output.device, dtype=dispatched_output.dtype,
    )

    for k in range(expert_indices.shape[1]):
        sel_expert_ids = expert_indices[:, k]
        sel_gates = gates[:, k]

        # Sort by expert for this selection
        sort_idx = torch.argsort(sel_expert_ids, dim=0, stable=True)
        sorted_eids = sel_expert_ids[sort_idx]
        sorted_gates = sel_gates[sort_idx]

        # Positions within expert
        sel_n = sorted_eids.shape[0]
        same = torch.zeros(sel_n, dtype=torch.bool, device=dispatched_output.device)
        same[1:] = sorted_eids[1:] == sorted_eids[:-1]
        positions = torch.cumsum(same.long(), dim=0)

        # Capacity filter
        valid = positions < expert_capacity
        eids_valid = sorted_eids[valid]
        pos_valid = positions[valid]
        gates_valid = sorted_gates[valid]
        orig_idx = sort_idx[valid]

        # Gather and accumulate
        gathered = dispatched_output[eids_valid, pos_valid]
        output[orig_idx] += gathered * gates_valid.unsqueeze(-1)

    return output


# =============================================================================
# Expert Module
# =============================================================================


class Expert(nn.Module):
    """Single feed-forward expert network.

    Each expert is a standard two-layer feed-forward network with GELU
    activation. The architecture follows the standard transformer FFN
    pattern: Linear -> GELU -> Linear.

    Architecture:
        input (hidden_size) -> Linear(hidden_size, expert_dim) -> GELU
        -> Linear(expert_dim, hidden_size) -> output (hidden_size)

    Weight initialization follows Xavier/Glorot uniform for good gradient
    flow, with bias initialized to zero.

    Args:
        hidden_size: Input and output dimension.
        expert_dim: Intermediate (hidden) dimension of the expert.
        dropout: Dropout probability applied after GELU activation.
        bias: Whether to include bias in linear layers.
        activation: Activation function name ('gelu', 'relu', 'silu', 'tanh').
        init_std: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        self.dropout_rate = dropout

        # First linear projection: hidden_size -> expert_dim
        self.up_proj = nn.Linear(hidden_size, expert_dim, bias=bias)

        # Activation function
        if activation == "gelu":
            self.act_fn = nn.GELU()
        elif activation == "relu":
            self.act_fn = nn.ReLU(inplace=True)
        elif activation == "silu" or activation == "swish":
            self.act_fn = nn.SiLU(inplace=True)
        elif activation == "tanh":
            self.act_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Dropout after activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Second linear projection: expert_dim -> hidden_size
        self.down_proj = nn.Linear(expert_dim, hidden_size, bias=bias)

        # Initialize weights
        self._init_weights(init_std)

    def _init_weights(self, std: float):
        """Initialize weights using truncated normal approximation.

        Uses Xavier uniform initialization scaled by std. This provides
        good initial signal propagation through the network.
        """
        nn.init.trunc_normal_(
            self.up_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

        nn.init.trunc_normal_(
            self.down_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).
        """
        # Project up to expert dimension
        x = self.up_proj(x)
        # Apply activation
        x = self.act_fn(x)
        # Apply dropout
        x = self.dropout(x)
        # Project back down to hidden dimension
        x = self.down_proj(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"expert_dim={self.expert_dim}, "
            f"dropout={self.dropout_rate}"
        )


class GatedExpert(nn.Module):
    """Gated expert with a SwiGLU-style gating mechanism.

    This variant uses a gate projection alongside the standard up
    projection, following the SwiGLU pattern used in LLaMA:

        gate = sigmoid(Linear_gate(x))
        up = Linear_up(x)
        output = Linear_down(gate * silu(up))

    This typically outperforms the standard Expert with GELU activation.

    Args:
        hidden_size: Input and output dimension.
        expert_dim: Intermediate dimension (typically hidden_size * 4).
        dropout: Dropout probability.
        bias: Whether to include bias in linear layers.
        init_std: Standard deviation for weight initialization.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim

        self.gate_proj = nn.Linear(hidden_size, expert_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_size, expert_dim, bias=bias)
        self.down_proj = nn.Linear(expert_dim, hidden_size, bias=bias)
        self.act_fn = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(init_std)

    def _init_weights(self, std: float):
        """Initialize weights with truncated normal distribution."""
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.trunc_normal_(
                proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std
            )
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with SwiGLU gating.

        Args:
            x: Input tensor of shape (..., hidden_size).

        Returns:
            Output tensor of shape (..., hidden_size).
        """
        gate = self.gate_proj(x)
        gate = torch.sigmoid(gate)
        up = self.up_proj(x)
        up = self.act_fn(up)
        x = self.dropout(gate * up)
        x = self.down_proj(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"expert_dim={self.expert_dim}"
        )


class ExpertGroup(nn.Module):
    """Group of experts that can be processed together efficiently.

    This module wraps a subset of experts and provides batched forward
    pass through all experts in the group. Useful for expert parallelism
    where experts are sharded across devices.

    Args:
        hidden_size: Input and output dimension.
        expert_dim: Intermediate dimension of each expert.
        num_experts_in_group: Number of experts in this group.
        dropout: Dropout probability for each expert.
        bias: Whether to use bias in expert linear layers.
        activation: Activation function name.
        use_gating: Whether to use gated (SwiGLU) experts.
        init_std: Weight initialization standard deviation.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_dim: int,
        num_experts_in_group: int,
        dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        use_gating: bool = False,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts_in_group

        if use_gating:
            ExpertClass = GatedExpert
        else:
            ExpertClass = Expert

        self.experts = nn.ModuleList([
            ExpertClass(
                hidden_size=hidden_size,
                expert_dim=expert_dim,
                dropout=dropout,
                bias=bias,
                activation=activation,
                init_std=init_std,
            )
            for _ in range(num_experts_in_group)
        ])

    def forward(
        self,
        x: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through all experts in the group.

        Args:
            x: Input tokens arranged as (num_experts, capacity, hidden_dim).
            expert_mask: Boolean mask of shape (num_experts, capacity)
                indicating valid token positions.

        Returns:
            Expert outputs of shape (num_experts, capacity, hidden_dim).
            Invalid positions contain zeros.
        """
        num_experts, capacity, hidden_dim = x.shape
        output = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            expert_input = x[i]
            # Only process valid tokens
            valid_mask = expert_mask[i]
            if valid_mask.any():
                # Get valid tokens
                valid_tokens = expert_input[valid_mask]
                # Process through expert
                expert_output = expert(valid_tokens)
                # Place back in output
                output[i, valid_mask] = expert_output
            # Invalid positions remain zero

        return output

    def forward_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass processing all experts as a single batch.

        This reshapes the input to batch all experts together and runs
        a single forward pass, which is more efficient when the expert
        implementations are identical.

        Args:
            x: Input of shape (num_experts, capacity, hidden_dim).

        Returns:
            Output of shape (num_experts, capacity, hidden_dim).
        """
        num_experts, capacity, hidden_dim = x.shape
        # Flatten experts and capacity into batch dimension
        batched = x.view(num_experts * capacity, hidden_dim)
        # Process each expert's tokens through their respective expert
        outputs = []
        offset = 0
        for expert in self.experts:
            expert_batch = batched[offset:offset + capacity]
            outputs.append(expert(expert_batch))
            offset += capacity
        output = torch.cat(outputs, dim=0)
        return output.view(num_experts, capacity, hidden_dim)


# =============================================================================
# Top-K Gating (Router)
# =============================================================================


class TopKGating(nn.Module):
    """Top-K gating network (router) for Mixture of Experts.

    The router maps each input token to a probability distribution over
    experts, then selects the top-k experts for each token. It also
    computes auxiliary losses for load balancing.

    Architecture:
        input (hidden_size) -> Linear(hidden_size, num_experts)
        -> add noise (training) -> softmax -> top-k -> gates, indices

    The load balancing loss encourages uniform expert utilization:
        L_bal = num_experts * sum(f_i * P_i)
    where f_i is the fraction of tokens dispatched to expert i and
    P_i is the average routing probability for expert i.

    Args:
        hidden_size: Input hidden dimension.
        config: MoEConfig with gating parameters.
    """

    def __init__(
        self,
        hidden_size: int,
        config: MoEConfig,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = config.num_experts
        self.num_selected = config.num_selected
        self.jitter_noise = config.router_jitter_noise
        self.temperature = config.gate_softmax_temperature
        self.load_balancing_loss_coef = config.load_balancing_loss_coef

        # Router network
        if config.router_hidden_dim > 0:
            # Two-layer router with hidden projection
            self.router = nn.Sequential(
                nn.Linear(hidden_size, config.router_hidden_dim, bias=config.router_bias),
                nn.ReLU(inplace=True),
                nn.Linear(config.router_hidden_dim, self.num_experts, bias=config.router_bias),
            )
        else:
            # Single-layer router
            self.router = nn.Linear(
                hidden_size, self.num_experts, bias=config.router_bias
            )

        # Optional input normalization
        self.input_norm = None
        if config.normalize_router_input:
            self.input_norm = nn.LayerNorm(hidden_size)

        # Initialize router weights
        self._init_router_weights()

    def _init_router_weights(self):
        """Initialize router weights with small values for stable gating.

        The router should start with near-uniform routing and gradually
        learn to specialize. We use small normal initialization.
        """
        if isinstance(self.router, nn.Linear):
            nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
            if self.router.bias is not None:
                nn.init.zeros_(self.router.bias)
        elif isinstance(self.router, nn.Sequential):
            for module in self.router.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gating weights and expert assignments.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
                or (num_tokens, hidden_size).

        Returns:
            Tuple of:
                - gates: Top-k gating weights, shape
                    (num_tokens, num_selected).
                - selected_experts: Selected expert indices, shape
                    (num_tokens, num_selected).
                - aux_loss: Load balancing auxiliary loss (scalar).
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.hidden_size)

        # Optional normalization
        if self.input_norm is not None:
            x = self.input_norm(x)

        # Compute router logits
        router_logits = self.router(x)

        # Apply temperature scaling
        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature

        # Add noise during training for exploration
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Compute probabilities with softmax
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        gates, selected_experts = _stable_top_k(
            router_probs, self.num_selected, dim=-1
        )

        # Normalize gates to sum to 1
        gates_sum = gates.sum(dim=-1, keepdim=True)
        gates = gates / (gates_sum + 1e-9)

        # Compute load balancing auxiliary loss
        aux_loss = self._compute_load_balancing_loss(
            router_probs, selected_experts
        )

        return gates, selected_experts, aux_loss

    def _compute_load_balancing_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        The loss encourages uniform distribution of tokens across experts:

        L = num_experts * sum_i(f_i * P_i)

        where:
            f_i = fraction of tokens assigned to expert i
            P_i = mean routing probability for expert i

        This loss is minimized when f_i = P_i = 1/num_experts for all i.

        Args:
            router_probs: Router probabilities (num_tokens, num_experts).
            selected_experts: Selected expert indices (num_tokens, num_selected).

        Returns:
            Load balancing loss (scalar tensor).
        """
        num_tokens = router_probs.shape[0]

        # Compute fraction of tokens dispatched to each expert
        # selected_experts: (num_tokens, num_selected)
        one_hot = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()  # (num_tokens, num_selected, num_experts)
        tokens_per_expert = one_hot.sum(dim=(0, 1))  # (num_experts,)
        fraction = tokens_per_expert / (num_tokens * self.num_selected + 1e-9)

        # Mean routing probability per expert
        mean_prob = router_probs.mean(dim=0)  # (num_experts,)

        # Load balancing loss
        aux_loss = self.num_experts * (fraction * mean_prob).sum()

        return aux_loss * self.load_balancing_loss_coef

    def compute_importance_loss(
        self,
        router_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute importance-based auxiliary loss.

        This variant uses the squared coefficients of variation of the
        importance (mean probability) across experts. This penalizes
        both high and low utilization experts.

        Args:
            router_probs: Router probabilities (num_tokens, num_experts).

        Returns:
            Importance loss (scalar).
        """
        importance = router_probs.mean(dim=0)
        mean_importance = importance.mean()
        variance = ((importance - mean_importance) ** 2).mean()
        return variance

    def compute_z_loss(
        self,
        router_logits: torch.Tensor,
        z_loss_coef: float = 0.001,
    ) -> torch.Tensor:
        """Compute z-loss for numerical stability.

        This encourages the router logits to remain small, preventing
        numerical overflow in softmax. Used in Switch Transformer.

        L_z = (1/N) * sum_i(log(sum_j(exp(logit_ij)))^2)

        Args:
            router_logits: Raw router logits (num_tokens, num_experts).
            z_loss_coef: Coefficient for z-loss term.

        Returns:
            Z-loss (scalar).
        """
        log_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = (log_z ** 2).mean()
        return z_loss * z_loss_coef


class GroupedTopKGating(nn.Module):
    """Grouped top-k gating for large numbers of experts.

    When the number of experts is very large (e.g., 128+), routing to
    all experts simultaneously can be expensive. This module groups experts
    and routes within groups, reducing computation.

    First selects the top group, then routes to top-k experts within
    that group.

    Args:
        hidden_size: Input hidden dimension.
        num_experts: Total number of experts.
        num_selected: Number of experts per token.
        num_groups: Number of expert groups.
        experts_per_group: Experts in each group.
        group_capacity_factor: Capacity factor for group selection.
        jitter_noise: Router noise for exploration.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_selected: int,
        num_groups: int = 4,
        experts_per_group: int = 0,
        group_capacity_factor: float = 1.0,
        jitter_noise: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.num_groups = num_groups
        self.jitter_noise = jitter_noise

        if experts_per_group == 0:
            self.experts_per_group = num_experts // num_groups
        else:
            self.experts_per_group = experts_per_group

        assert self.num_groups * self.experts_per_group == num_experts, (
            f"num_groups ({num_groups}) * experts_per_group "
            f"({self.experts_per_group}) != num_experts ({num_experts})"
        )

        # Group-level router
        self.group_router = nn.Linear(hidden_size, num_groups, bias=True)

        # Expert-level router within each group
        self.expert_routers = nn.ModuleList([
            nn.Linear(hidden_size, self.experts_per_group, bias=True)
            for _ in range(num_groups)
        ])

        self._init_weights()

    def _init_weights(self):
        """Initialize router weights with small values."""
        nn.init.normal_(self.group_router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.group_router.bias)
        for router in self.expert_routers:
            nn.init.normal_(router.weight, mean=0.0, std=0.01)
            nn.init.zeros_(router.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute grouped gating weights and expert assignments.

        Args:
            x: Input tensor of shape (num_tokens, hidden_size).

        Returns:
            Tuple of (gates, selected_experts, aux_loss).
        """
        num_tokens = x.shape[0]

        # Add noise during training
        if self.training and self.jitter_noise > 0:
            noise_x = x + torch.randn_like(x) * self.jitter_noise
        else:
            noise_x = x

        # Group selection
        group_logits = self.group_router(noise_x)
        group_probs = F.softmax(group_logits, dim=-1)

        # Select top groups for each token
        num_groups_to_select = max(1, self.num_selected // self.experts_per_group + 1)
        group_gates, selected_groups = torch.topk(
            group_probs, min(num_groups_to_select, self.num_groups), dim=-1
        )

        # Within each selected group, select top experts
        all_gates = torch.zeros(
            num_tokens, self.num_selected,
            device=x.device, dtype=x.dtype,
        )
        all_experts = torch.zeros(
            num_tokens, self.num_selected,
            device=x.device, dtype=torch.long,
        )

        aux_loss = torch.tensor(0.0, device=x.device)

        expert_idx = 0
        for g_sel in range(selected_groups.shape[1]):
            group_ids = selected_groups[:, g_sel]
            group_gate_vals = group_gates[:, g_sel]

            for group_id in range(self.num_groups):
                mask = group_ids == group_id
                if not mask.any():
                    continue

                token_indices = mask.nonzero(as_tuple=True)[0]
                token_inputs = noise_x[token_indices]

                # Expert routing within this group
                expert_logits = self.expert_routers[group_id](token_inputs)
                expert_probs = F.softmax(expert_logits, dim=-1)

                # Select top experts within group
                remaining = self.num_selected - expert_idx
                k = min(remaining, self.experts_per_group)
                if k <= 0:
                    break

                exp_gates, exp_selected = torch.topk(expert_probs, k, dim=-1)

                # Map local expert indices to global
                global_expert_ids = exp_selected + group_id * self.experts_per_group
                combined_gates = group_gate_vals[token_indices].unsqueeze(-1) * exp_gates

                # Assign to output
                for j in range(k):
                    remaining_slots = self.num_selected - expert_idx
                    if remaining_slots <= 0:
                        break
                    all_experts[token_indices, expert_idx] = global_expert_ids[:, j]
                    all_gates[token_indices, expert_idx] = combined_gates[:, j]
                    expert_idx += 1

                # Accumulate load balancing loss
                f_i = torch.bincount(
                    global_expert_ids.view(-1),
                    minlength=self.num_experts
                ).float()
                f_i = f_i / (num_tokens * self.num_selected + 1e-9)
                p_i = expert_probs.mean(dim=0)
                aux_loss = aux_loss + self.num_experts * (f_i * p_i).sum()

        return all_gates, all_experts, aux_loss


# =============================================================================
# Expert Load Balancer
# =============================================================================


class ExpertLoadBalancer:
    """Monitor and manage expert utilization across forward passes.

    Tracks statistics about expert usage and computes auxiliary losses
    for load balancing. Supports both per-step and running statistics.

    Args:
        num_experts: Number of experts to monitor.
        window_size: Number of recent steps to average over.
    """

    def __init__(self, num_experts: int, window_size: int = 100):
        self.num_experts = num_experts
        self.window_size = window_size

        self._reset_stats()

    def _reset_stats(self):
        """Reset all tracking statistics."""
        self.step_count = 0
        self.recent_fractions = []
        self.recent_probs = []
        self.total_tokens_dispatched = 0
        self.total_tokens_dropped = 0
        self.expert_dispatch_counts = torch.zeros(self.num_experts)
        self.expert_prob_sums = torch.zeros(self.num_experts)

    def update(
        self,
        selected_experts: torch.Tensor,
        router_probs: torch.Tensor,
        tokens_dropped: int = 0,
    ):
        """Update statistics with a new batch of routing decisions.

        Args:
            selected_experts: Expert indices (num_tokens, num_selected).
            router_probs: Router probabilities (num_tokens, num_experts).
            tokens_dropped: Number of tokens dropped due to capacity.
        """
        self.step_count += 1
        num_tokens = router_probs.shape[0]
        num_selected = selected_experts.shape[1]
        device = router_probs.device

        # Compute token fractions per expert
        one_hot = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()
        tokens_per_expert = one_hot.sum(dim=(0, 1))
        fraction = tokens_per_expert / (num_tokens * num_selected + 1e-9)

        # Compute mean routing probability per expert
        mean_prob = router_probs.mean(dim=0)

        # Store recent stats
        self.recent_fractions.append(fraction.detach().cpu())
        self.recent_probs.append(mean_prob.detach().cpu())

        # Trim to window size
        if len(self.recent_fractions) > self.window_size:
            self.recent_fractions = self.recent_fractions[-self.window_size:]
            self.recent_probs = self.recent_probs[-self.window_size:]

        # Accumulate totals
        self.total_tokens_dispatched += num_tokens
        self.total_tokens_dropped += tokens_dropped
        self.expert_dispatch_counts += tokens_per_expert.detach().cpu()
        self.expert_prob_sums += mean_prob.detach().cpu() * num_tokens

    def get_load_balance_loss(self) -> torch.Tensor:
        """Compute load balancing loss from recent statistics.

        Returns:
            Load balance loss averaged over the recent window.
        """
        if not self.recent_fractions:
            return torch.tensor(0.0)

        avg_fraction = torch.stack(self.recent_fractions).mean(dim=0)
        avg_prob = torch.stack(self.recent_probs).mean(dim=0)

        loss = self.num_experts * (avg_fraction * avg_prob).sum()
        return loss

    def get_auxiliary_loss(
        self,
        coef: float = 0.01,
    ) -> torch.Tensor:
        """Compute combined auxiliary loss.

        Includes load balancing loss and a variance penalty on expert
        utilization.

        Args:
            coef: Coefficient for the auxiliary loss.

        Returns:
            Combined auxiliary loss.
        """
        lb_loss = self.get_load_balance_loss()

        if len(self.recent_fractions) > 1:
            # Variance penalty on expert utilization
            fractions_stack = torch.stack(self.recent_fractions)
            util_variance = fractions_stack.var(dim=0).mean()
            var_loss = util_variance * 0.1
        else:
            var_loss = torch.tensor(0.0)

        return coef * (lb_loss + var_loss)

    def get_expert_utilization(self) -> torch.Tensor:
        """Get current expert utilization fractions.

        Returns:
            Tensor of shape (num_experts,) with utilization fractions.
        """
        total = self.expert_dispatch_counts.sum()
        if total == 0:
            return torch.zeros(self.num_experts)
        return self.expert_dispatch_counts / total

    def get_dropout_rate(self) -> float:
        """Get the rate of dropped tokens.

        Returns:
            Fraction of tokens dropped due to expert capacity.
        """
        total = self.total_tokens_dispatched + self.total_tokens_dropped
        if total == 0:
            return 0.0
        return self.total_tokens_dropped / total

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics.

        Returns:
            Dictionary with load balancing statistics.
        """
        utilization = self.get_expert_utilization()
        return {
            "step_count": self.step_count,
            "total_tokens_dispatched": self.total_tokens_dispatched,
            "total_tokens_dropped": self.total_tokens_dropped,
            "dropout_rate": self.get_dropout_rate(),
            "expert_utilization": utilization.tolist(),
            "utilization_mean": utilization.mean().item(),
            "utilization_std": utilization.std().item(),
            "utilization_min": utilization.min().item(),
            "utilization_max": utilization.max().item(),
            "load_balance_loss": self.get_load_balance_loss().item(),
            "expert_dispatch_counts": self.expert_dispatch_counts.tolist(),
        }

    def reset(self):
        """Reset all tracking statistics."""
        self._reset_stats()

    def log_statistics(self):
        """Log current statistics using the logger."""
        stats = self.get_statistics()
        logger.info(
            f"MoE Load Balancer (step {stats['step_count']}): "
            f"dispatched={stats['total_tokens_dispatched']}, "
            f"dropped={stats['total_tokens_dropped']} "
            f"(rate={stats['dropout_rate']:.4f}), "
            f"util_mean={stats['utilization_mean']:.4f}, "
            f"util_std={stats['utilization_std']:.4f}, "
            f"lb_loss={stats['load_balance_loss']:.6f}"
        )


# =============================================================================
# MoE Layer
# =============================================================================


class MoELayerOutput(NamedTuple):
    """Output of the MoE layer.

    Attributes:
        output: Combined expert outputs (batch, seq_len, hidden_size).
        aux_loss: Auxiliary load balancing loss (scalar).
        routing_details: Dictionary with routing statistics.
    """

    output: torch.Tensor
    aux_loss: torch.Tensor
    routing_details: Dict[str, Any]


class MoELayer(nn.Module):
    """Full Mixture of Experts layer.

    Implements token routing to experts with capacity factor, token
    dropping, and expert parallelism support. This is the core MoE
    computation module.

    Pipeline:
        1. Router (TopKGating) assigns each token to top-k experts
        2. Tokens are scattered to their assigned experts
        3. Each expert processes its assigned tokens (up to capacity)
        4. Expert outputs are gathered and combined using gating weights
        5. Auxiliary loss is computed for load balancing

    Args:
        hidden_size: Input and output dimension.
        config: MoEConfig with all MoE parameters.
        use_gated_experts: Whether to use SwiGLU-gated experts.
        activation: Activation function for standard experts.
        init_std: Weight initialization standard deviation.
    """

    def __init__(
        self,
        hidden_size: int,
        config: MoEConfig,
        use_gated_experts: bool = False,
        activation: str = "gelu",
        init_std: float = 0.02,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.num_experts = config.num_experts
        self.num_selected = config.num_selected
        self.capacity_factor = config.capacity_factor
        self.drop_tokens = config.drop_tokens
        self.use_residual = config.use_residual
        self.residual_coef = config.residual_coef

        # Resolve expert dimension
        expert_dim = _get_expert_dim(config.expert_dim, hidden_size)

        # Router (gating network)
        self.gating = TopKGating(hidden_size, config)

        # Expert networks
        if use_gated_experts:
            ExpertClass = GatedExpert
        else:
            ExpertClass = Expert

        self.experts = nn.ModuleList([
            ExpertClass(
                hidden_size=hidden_size,
                expert_dim=expert_dim,
                dropout=config.expert_dropout,
                bias=config.router_bias,
                activation=activation,
                init_std=init_std,
            )
            for _ in range(config.num_experts)
        ])

        # Load balancer for monitoring
        self.load_balancer = ExpertLoadBalancer(
            num_experts=config.num_experts
        )

        # Residual projection (optional)
        if self.use_residual:
            self.residual_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.weight)

    def _compute_expert_outputs(
        self,
        dispatched: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens through all experts.

        Args:
            dispatched: Tokens arranged by expert
                (num_experts, capacity, hidden_dim).
            expert_mask: Valid token mask (num_experts, capacity).

        Returns:
            Expert outputs (num_experts, capacity, hidden_dim).
        """
        num_experts, capacity, hidden_dim = dispatched.shape
        output = torch.zeros_like(dispatched)

        for i in range(self.num_experts):
            if not expert_mask[i].any():
                continue

            expert_input = dispatched[i][expert_mask[i]]
            expert_output = self.experts[i](expert_input)
            output[i][expert_mask[i]] = expert_output

        return output

    def _compute_expert_outputs_batched(
        self,
        dispatched: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Process tokens through experts using a more batched approach.

        For experts with the same number of valid tokens, we can batch
        their processing together for better GPU utilization.

        Args:
            dispatched: Tokens arranged by expert.
            expert_mask: Valid token mask.

        Returns:
            Expert outputs.
        """
        num_experts = dispatched.shape[0]
        output = torch.zeros_like(dispatched)

        # Group experts by their valid token count for batching
        count_to_experts: Dict[int, List[int]] = defaultdict(list)
        for i in range(num_experts):
            count = int(expert_mask[i].sum().item())
            count_to_experts[count].append(i)

        for count, expert_ids in count_to_experts.items():
            if count == 0:
                continue

            # Stack all expert inputs with this count
            expert_inputs = torch.stack([
                dispatched[i][expert_mask[i]] for i in expert_ids
            ], dim=0)  # (len(expert_ids), count, hidden_dim)

            # Process each expert
            for j, eid in enumerate(expert_ids):
                expert_output = self.experts[eid](expert_inputs[j])
                output[eid][expert_mask[eid]] = expert_output

        return output

    def forward(
        self,
        x: torch.Tensor,
    ) -> MoELayerOutput:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
                or (num_tokens, hidden_size).

        Returns:
            MoELayerOutput with combined output and auxiliary loss.
        """
        original_shape = x.shape
        was_3d = x.dim() == 3
        if was_3d:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)
        else:
            batch_size = 1
            seq_len = x.shape[0]
            hidden_dim = x.shape[1]

        num_tokens = x.shape[0]

        # === Step 1: Route tokens to experts ===
        gates, selected_experts, aux_loss = self.gating(x)

        # === Step 2: Compute capacity ===
        capacity = _compute_capacity(
            num_tokens, self.num_experts, self.capacity_factor
        )

        # === Step 3: Dispatch tokens ===
        dispatched, expert_mask, sorted_indices, position_ids = \
            _efficient_scatter(
                x, selected_experts[:, 0],
                capacity, self.num_experts,
            )

        # === Step 4: Process through experts ===
        dispatched_output = self._compute_expert_outputs(
            dispatched, expert_mask
        )

        # === Step 5: Gather and combine ===
        # Combine all expert selections
        output = torch.zeros(
            num_tokens, hidden_dim,
            device=x.device, dtype=x.dtype,
        )

        for k in range(self.num_selected):
            sel_eids = selected_experts[:, k]
            sel_gates = gates[:, k]

            # Dispatch for this selection
            disp_k, mask_k, sort_k, pos_k = _efficient_scatter(
                x, sel_eids, capacity, self.num_experts
            )
            out_k = self._compute_expert_outputs(disp_k, mask_k)

            # Gather
            num_valid = mask_k.sum(dim=1)  # (num_experts,)
            valid_experts = (num_valid > 0).nonzero(as_tuple=True)[0]

            for eid in valid_experts:
                eid_int = eid.item()
                valid_mask = mask_k[eid_int]
                # Find original token indices
                sort_indices_for_expert = sort_k[valid_mask]
                output[sort_indices_for_expert] += (
                    out_k[eid_int][valid_mask] *
                    sel_gates[sort_indices_for_expert].unsqueeze(-1)
                )

        # === Step 6: Residual connection (optional) ===
        if self.use_residual:
            residual_output = self.residual_proj(x)
            output = output + self.residual_coef * residual_output

        # === Step 7: Track statistics ===
        tokens_dropped = int((~expert_mask).sum().item())
        self.load_balancer.update(
            selected_experts=selected_experts,
            router_probs=self.gating(
                x
            )[0].mean(dim=0, keepdim=True).expand(
                num_tokens, -1
            ) if self.training else torch.zeros(
                num_tokens, self.num_experts, device=x.device
            ),
            tokens_dropped=tokens_dropped,
        )

        # Reshape output
        if was_3d:
            output = output.view(batch_size, seq_len, hidden_dim)

        routing_details = {
            "num_tokens": num_tokens,
            "capacity_per_expert": capacity,
            "tokens_dropped": tokens_dropped,
            "dropout_rate": tokens_dropped / max(num_tokens, 1),
        }

        return MoELayerOutput(
            output=output,
            aux_loss=aux_loss,
            routing_details=routing_details,
        )

    def forward_single_dispatch(
        self,
        x: torch.Tensor,
    ) -> MoELayerOutput:
        """Simplified forward using only the first expert selection.

        This is useful for inference efficiency or when only one expert
        per token is desired.

        Args:
            x: Input tensor (batch, seq_len, hidden_size) or (tokens, hidden).

        Returns:
            MoELayerOutput with single-expert output.
        """
        original_shape = x.shape
        was_3d = x.dim() == 3
        if was_3d:
            x = x.view(-1, x.shape[-1])

        num_tokens = x.shape[0]

        # Route to single expert
        gates, selected_experts, aux_loss = self.gating(x)
        expert_ids = selected_experts[:, 0]
        gate_weights = gates[:, 0]

        # Compute capacity
        capacity = _compute_capacity(
            num_tokens, self.num_experts, self.capacity_factor
        )

        # Dispatch
        dispatched, expert_mask, sorted_indices, position_ids = \
            _efficient_scatter(x, expert_ids, capacity, self.num_experts)

        # Process through experts
        dispatched_output = self._compute_expert_outputs(dispatched, expert_mask)

        # Gather back
        output = torch.zeros_like(x)
        num_valid = expert_mask.sum(dim=1)
        valid_experts = (num_valid > 0).nonzero(as_tuple=True)[0]

        for eid in valid_experts:
            eid_int = eid.item()
            valid = expert_mask[eid_int]
            orig_idx = sorted_indices[valid]
            output[orig_idx] = (
                dispatched_output[eid_int][valid] *
                gate_weights[orig_idx].unsqueeze(-1)
            )

        tokens_dropped = int((~expert_mask).sum().item())

        if was_3d:
            output = output.view(original_shape)

        routing_details = {
            "num_tokens": num_tokens,
            "capacity_per_expert": capacity,
            "tokens_dropped": tokens_dropped,
            "dropout_rate": tokens_dropped / max(num_tokens, 1),
        }

        return MoELayerOutput(
            output=output,
            aux_loss=aux_loss,
            routing_details=routing_details,
        )


# =============================================================================
# Expert Parallelism
# =============================================================================


class ExpertParallelism:
    """Manage expert sharding across multiple GPUs for distributed training.

    Expert parallelism distributes different experts across different devices,
    allowing each device to hold only a subset of the expert parameters.
    This is essential for models with hundreds of experts.

    Implementation:
        - Shard experts evenly across devices
        - AllGather for router computation (all devices need full routing info)
        - Scatter inputs to appropriate devices
        - Each device processes its local experts
        - AllGather to combine expert outputs

    Args:
        num_experts: Total number of experts.
        num_devices: Number of devices to shard across.
        device_ids: Explicit device IDs (if None, uses range(num_devices)).
    """

    def __init__(
        self,
        num_experts: int,
        num_devices: int,
        device_ids: Optional[List[int]] = None,
    ):
        self.num_experts = num_experts
        self.num_devices = num_devices
        self.device_ids = device_ids or list(range(num_devices))

        assert len(self.device_ids) == num_devices
        assert num_experts >= num_devices, (
            f"num_experts ({num_experts}) must be >= num_devices ({num_devices})"
        )

        # Compute expert-to-device mapping
        self.expert_to_device: Dict[int, int] = {}
        self.device_to_experts: Dict[int, List[int]] = {d: [] for d in self.device_ids}

        for i in range(num_experts):
            device = self.device_ids[i % num_devices]
            self.expert_to_device[i] = device
            self.device_to_experts[device].append(i)

    def get_expert_device(self, expert_id: int) -> int:
        """Get the device ID for a given expert.

        Args:
            expert_id: Expert index.

        Returns:
            Device ID where this expert resides.
        """
        return self.expert_to_device[expert_id]

    def get_device_experts(self, device_id: int) -> List[int]:
        """Get expert indices for a given device.

        Args:
            device_id: Device ID.

        Returns:
            List of expert indices on this device.
        """
        return self.device_to_experts[device_id]

    def shard_experts(
        self,
        experts: nn.ModuleList,
    ) -> Dict[int, nn.ModuleList]:
        """Shard expert modules across devices.

        Args:
            experts: Complete ModuleList of all experts.

        Returns:
            Dictionary mapping device ID to local expert ModuleList.
        """
        sharded: Dict[int, nn.ModuleList] = {}
        for device_id in self.device_ids:
            local_expert_ids = self.device_to_experts[device_id]
            local_experts = nn.ModuleList([
                experts[i] for i in local_expert_ids
            ])
            sharded[device_id] = local_experts

        return sharded

    def scatter_inputs(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        gates: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Scatter tokens to their respective devices.

        Args:
            tokens: Input tokens (num_tokens, hidden_dim).
            expert_ids: Expert assignments (num_tokens,).
            gates: Gating weights (num_tokens,).

        Returns:
            Dict mapping device_id to (local_tokens, local_gates, local_expert_ids).
        """
        scattered: Dict[int, List[torch.Tensor]] = {d: [] for d in self.device_ids}
        scattered_gates: Dict[int, List[torch.Tensor]] = {d: [] for d in self.device_ids}
        scattered_eids: Dict[int, List[torch.Tensor]] = {d: [] for d in self.device_ids}

        for token_idx in range(tokens.shape[0]):
            eid = expert_ids[token_idx].item()
            device = self.expert_to_device[eid]
            local_eid = self.device_to_experts[device].index(eid)

            scattered[device].append(tokens[token_idx])
            scattered_gates[device].append(gates[token_idx])
            scattered_eids[device].append(
                torch.tensor(local_eid, device=tokens.device, dtype=torch.long)
            )

        result = {}
        for device_id in self.device_ids:
            if len(scattered[device_id]) > 0:
                result[device_id] = (
                    torch.stack(scattered[device_id]),
                    torch.stack(scattered_gates[device_id]),
                    torch.stack(scattered_eids[device_id]),
                )

        return result

    def gather_outputs(
        self,
        device_outputs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        num_tokens: int,
        hidden_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Gather expert outputs from all devices.

        Args:
            device_outputs: Dict mapping device_id to (outputs, gates, token_indices).
            num_tokens: Total number of tokens.
            hidden_dim: Hidden dimension.
            device: Target device.

        Returns:
            Combined output tensor (num_tokens, hidden_dim).
        """
        output = torch.zeros(num_tokens, hidden_dim, device=device)

        for device_id, (dev_outputs, dev_gates, token_indices) in device_outputs.items():
            for i in range(dev_outputs.shape[0]):
                token_idx = token_indices[i].item()
                output[token_idx] = dev_outputs[i] * dev_gates[i]

        return output

    def get_sharding_info(self) -> Dict[str, Any]:
        """Get information about the expert sharding.

        Returns:
            Dictionary with sharding details.
        """
        return {
            "num_experts": self.num_experts,
            "num_devices": self.num_devices,
            "device_ids": self.device_ids,
            "experts_per_device": {
                d: len(eids) for d, eids in self.device_to_experts.items()
            },
            "expert_to_device": self.expert_to_device,
        }


class DeviceAwareMoELayer(nn.Module):
    """MoE layer with expert parallelism support.

    Extends the base MoELayer to support sharding experts across multiple
    devices. During forward pass, tokens are routed to the appropriate
    device, processed by local experts, and results are gathered.

    Args:
        hidden_size: Input and output dimension.
        config: MoEConfig.
        parallelism: ExpertParallelism instance.
        use_gated_experts: Whether to use gated experts.
    """

    def __init__(
        self,
        hidden_size: int,
        config: MoEConfig,
        parallelism: ExpertParallelism,
        use_gated_experts: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.parallelism = parallelism

        # Router stays on the primary device
        self.gating = TopKGating(hidden_size, config)

        # Create experts and shard across devices
        expert_dim = _get_expert_dim(config.expert_dim, hidden_size)
        ExpertClass = GatedExpert if use_gated_experts else Expert

        self.experts = nn.ModuleList([
            ExpertClass(hidden_size, expert_dim, config.expert_dropout)
            for _ in range(config.num_experts)
        ])

        # Place experts on their assigned devices
        for i, expert in enumerate(self.experts):
            device_id = parallelism.get_expert_device(i)
            expert.to(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> MoELayerOutput:
        """Forward pass with expert parallelism.

        Args:
            x: Input tensor (batch, seq_len, hidden_size) or (tokens, hidden).

        Returns:
            MoELayerOutput with combined output and auxiliary loss.
        """
        original_shape = x.shape
        was_3d = x.dim() == 3
        if was_3d:
            x = x.view(-1, x.shape[-1])

        num_tokens = x.shape[0]

        # Route tokens
        gates, selected_experts, aux_loss = self.gating(x)
        expert_ids = selected_experts[:, 0]
        gate_weights = gates[:, 0]

        # Compute capacity
        capacity = _compute_capacity(
            num_tokens, self.config.num_experts, self.config.capacity_factor
        )

        # Process each expert (device-aware)
        output = torch.zeros_like(x)

        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            token_mask = expert_ids == expert_idx
            if not token_mask.any():
                continue

            count = token_mask.sum().item()
            if count > capacity:
                # Drop excess tokens
                valid_indices = token_mask.nonzero(as_tuple=True)[0][:capacity]
            else:
                valid_indices = token_mask.nonzero(as_tuple=True)[0]

            # Process on expert's device
            expert_input = x[valid_indices].to(self.experts[expert_idx].weight.device)
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output.to(x.device)

            # Accumulate weighted output
            output[valid_indices] += (
                expert_output * gate_weights[valid_indices].unsqueeze(-1)
            )

        if was_3d:
            output = output.view(original_shape)

        tokens_dropped = num_tokens - int((expert_ids == expert_ids).sum().item())

        return MoELayerOutput(
            output=output,
            aux_loss=aux_loss,
            routing_details={
                "num_tokens": num_tokens,
                "capacity_per_expert": capacity,
                "tokens_dropped": max(0, tokens_dropped),
            },
        )


# =============================================================================
# Expert Dropout
# =============================================================================


class ExpertDropout(nn.Module):
    """Randomly drop experts during training for robustness.

    Expert dropout prevents over-reliance on specific experts by randomly
    zeroing out expert outputs during training. This is analogous to
    standard dropout but operates at the expert level rather than
    the neuron level.

    During inference, all experts are used (no dropout).

    Args:
        num_experts: Number of experts.
        dropout_rate: Probability of dropping each expert during training.
        scale_in_eval: Whether to scale outputs by 1/(1 - dropout_rate)
            during evaluation to match expected value.
    """

    def __init__(
        self,
        num_experts: int,
        dropout_rate: float = 0.1,
        scale_in_eval: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate
        self.scale_in_eval = scale_in_eval

        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

    def forward(
        self,
        expert_outputs: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply expert dropout.

        Args:
            expert_outputs: Expert outputs (num_experts, capacity, hidden_dim).
            expert_mask: Valid token mask (num_experts, capacity).
                If None, all positions are considered valid.

        Returns:
            Tuple of (dropped_outputs, dropout_mask).
        """
        if not self.training or self.dropout_rate == 0:
            scale = 1.0 / (1.0 - self.dropout_rate) if self.scale_in_eval else 1.0
            return expert_outputs * scale, torch.ones(
                expert_outputs.shape[0], device=expert_outputs.device, dtype=torch.bool
            )

        # Generate dropout mask: True means expert is kept
        keep_mask = torch.rand(
            self.num_experts, device=expert_outputs.device
        ) > self.dropout_rate

        # Ensure at least one expert is always kept
        if not keep_mask.any():
            keep_mask[torch.randint(0, self.num_experts, (1,))] = True

        # Apply mask
        scale = 1.0 / (1.0 - self.dropout_rate)
        dropped_outputs = expert_outputs.clone()
        for i in range(self.num_experts):
            if not keep_mask[i]:
                dropped_outputs[i] = 0.0

        dropped_outputs = dropped_outputs * scale

        return dropped_outputs, keep_mask

    def get_keep_probability(self) -> float:
        """Get the probability of keeping each expert.

        Returns:
            Keep probability (1 - dropout_rate).
        """
        return 1.0 - self.dropout_rate


class AdaptiveExpertDropout(nn.Module):
    """Adaptive expert dropout based on utilization statistics.

    Instead of a fixed dropout rate, this module adjusts the dropout
    probability per expert based on its recent utilization. Experts
    with higher utilization get higher dropout rates, encouraging
    more uniform expert usage.

    Args:
        num_experts: Number of experts.
        base_dropout_rate: Base dropout probability.
        momentum: Momentum for updating utilization statistics.
        max_dropout_rate: Maximum dropout rate per expert.
    """

    def __init__(
        self,
        num_experts: int,
        base_dropout_rate: float = 0.05,
        momentum: float = 0.1,
        max_dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.base_dropout_rate = base_dropout_rate
        self.momentum = momentum
        self.max_dropout_rate = max_dropout_rate

        # Running utilization statistics (EMA)
        self.register_buffer(
            "expert_utilization",
            torch.ones(num_experts) / num_experts,
        )
        self.register_buffer(
            "total_dispatches",
            torch.zeros(1),
        )

    def update_utilization(
        self,
        selected_experts: torch.Tensor,
    ):
        """Update running utilization statistics.

        Args:
            selected_experts: Expert assignments (num_tokens, num_selected).
        """
        num_tokens = selected_experts.shape[0]
        num_selected = selected_experts.shape[1]

        # Current batch utilization
        current = torch.zeros(
            self.num_experts, device=selected_experts.device
        )
        one_hot = F.one_hot(
            selected_experts, num_classes=self.num_experts
        ).float()
        current = one_hot.sum(dim=(0, 1))
        current = current / (num_tokens * num_selected + 1e-9)

        # Update EMA
        self.expert_utilization = (
            (1 - self.momentum) * self.expert_utilization +
            self.momentum * current
        )
        self.total_dispatches += num_tokens

    def get_dropout_rates(self) -> torch.Tensor:
        """Compute per-expert dropout rates based on utilization.

        Experts with utilization above mean get higher dropout rates.

        Returns:
            Tensor of shape (num_experts,) with dropout rates.
        """
        mean_util = self.expert_utilization.mean()
        excess_util = self.expert_utilization - mean_util

        # Scale excess utilization to dropout rate
        dropout_rates = self.base_dropout_rate + excess_util * 2.0
        dropout_rates = dropout_rates.clamp(0, self.max_dropout_rate)

        return dropout_rates

    def forward(
        self,
        expert_outputs: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive expert dropout.

        Args:
            expert_outputs: Expert outputs (num_experts, capacity, hidden_dim).
            expert_mask: Valid token mask.

        Returns:
            Tuple of (dropped_outputs, keep_mask).
        """
        if not self.training:
            return expert_outputs, torch.ones(
                expert_outputs.shape[0], device=expert_outputs.device, dtype=torch.bool
            )

        dropout_rates = self.get_dropout_rates()

        keep_mask = torch.rand(
            self.num_experts, device=expert_outputs.device
        ) > dropout_rates

        if not keep_mask.any():
            keep_mask[dropout_rates.argmin()] = True

        dropped = expert_outputs.clone()
        for i in range(self.num_experts):
            if not keep_mask[i]:
                dropped[i] = 0.0

        return dropped, keep_mask


# =============================================================================
# Residual MoE
# =============================================================================


class ResidualMoE(nn.Module):
    """Mixture of Experts with residual connections.

    Adds a residual connection from the input directly to the output
    of the MoE layer. This helps with training stability and ensures
    that tokens always receive some useful signal even when experts
    produce suboptimal outputs.

    Architecture:
        output = MoE(x) + alpha * projection(x)
        where alpha is the residual coefficient and projection is an
        optional learned linear transformation.

    Args:
        hidden_size: Input and output dimension.
        config: MoEConfig.
        residual_coef: Coefficient for the residual connection.
        learn_residual: Whether to learn the residual coefficient.
        use_projection: Whether to use a learned projection in the residual.
        use_gated_experts: Whether to use gated experts.
    """

    def __init__(
        self,
        hidden_size: int,
        config: MoEConfig,
        residual_coef: float = 0.1,
        learn_residual: bool = False,
        use_projection: bool = True,
        use_gated_experts: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        self.residual_coef = residual_coef

        # Main MoE layer
        self.moe = MoELayer(
            hidden_size=hidden_size,
            config=config,
            use_gated_experts=use_gated_experts,
        )

        # Residual projection
        self.residual_proj = None
        if use_projection:
            self.residual_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.GELU(),
            )
            # Initialize with near-zero weights for gradual learning
            nn.init.normal_(
                self.residual_proj[0].weight, mean=0.0, std=0.001
            )

        # Learnable residual coefficient
        self.residual_alpha = None
        if learn_residual:
            self.residual_alpha = nn.Parameter(torch.tensor(residual_coef))

        # Pre-normalization
        self.input_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> MoELayerOutput:
        """Forward pass with residual connection.

        Args:
            x: Input tensor (batch, seq_len, hidden_size) or (tokens, hidden).

        Returns:
            MoELayerOutput with residual-augmented output.
        """
        residual = x

        # Normalize input
        x_normed = self.input_norm(x)

        # MoE forward
        moe_output = self.moe(x_normed)

        # Apply residual connection
        if self.residual_proj is not None:
            residual_transformed = self.residual_proj(residual)
        else:
            residual_transformed = residual

        if self.residual_alpha is not None:
            alpha = torch.sigmoid(self.residual_alpha)
        else:
            alpha = self.residual_coef

        combined_output = moe_output.output + alpha * residual_transformed

        return MoELayerOutput(
            output=combined_output,
            aux_loss=moe_output.aux_loss,
            routing_details=moe_output.routing_details,
        )


class StochasticDepthMoE(nn.Module):
    """MoE with stochastic depth for regularized training.

    Randomly drops entire MoE layers during training (similar to
    Stochastic Depth / DropPath in ResNets). This regularizes the
    model by preventing co-adaptation of MoE layers.

    Args:
        hidden_size: Input and output dimension.
        config: MoEConfig.
        drop_path_rate: Probability of dropping the entire MoE layer.
        use_gated_experts: Whether to use gated experts.
    """

    def __init__(
        self,
        hidden_size: int,
        config: MoEConfig,
        drop_path_rate: float = 0.1,
        use_gated_experts: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop_path_rate = drop_path_rate
        self.config = config

        self.moe = MoELayer(
            hidden_size=hidden_size,
            config=config,
            use_gated_experts=use_gated_experts,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> MoELayerOutput:
        """Forward pass with stochastic depth.

        During training, the MoE layer is dropped with probability
        drop_path_rate. When dropped, the input is passed through
        unchanged.

        Args:
            x: Input tensor.

        Returns:
            MoELayerOutput.
        """
        if self.training and torch.rand(1).item() < self.drop_path_rate:
            # Drop the MoE layer, pass input through
            aux_loss = torch.tensor(0.0, device=x.device)
            return MoELayerOutput(
                output=x,
                aux_loss=aux_loss,
                routing_details={"dropped": True},
            )

        return self.moe(x)


# =============================================================================
# Sparse MoE Transformer Layer
# =============================================================================


class SparseMoETransformerLayer(nn.Module):
    """Transformer layer with MoE replacing the standard FFN.

    This layer follows the pre-normalization transformer pattern:
        x' = x + Attention(LayerNorm(x))
        x'' = x' + MoE(LayerNorm(x'))

    The MoE layer replaces the standard dense FFN, providing sparse
    computation where each token is only processed by a subset of experts.

    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (for GQA).
        moe_config: Configuration for the MoE layer.
        head_dim: Dimension of each attention head.
        rms_norm_eps: Epsilon for RMS normalization.
        attention_dropout: Dropout for attention weights.
        max_position_embeddings: Maximum sequence length.
        rope_theta: Base for RoPE positional encoding.
        layer_idx: Index of this layer.
        use_gated_experts: Whether to use gated experts.
        use_flash_attention: Whether to use flash attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        moe_config: MoEConfig,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 8192,
        rope_theta: float = 10000.0,
        layer_idx: int = 0,
        use_gated_experts: bool = False,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        self.head_dim = head_dim
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads

        # Pre-attention normalization
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

        # Multi-head attention (GQA)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0 else nn.Identity()

        # RoPE
        self.rope = RotaryEmbedding(
            dim=head_dim,
            max_seq_len=max_position_embeddings,
            base=rope_theta,
        )

        # Pre-MoE normalization
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=rms_norm_eps)

        # MoE layer (replaces standard FFN)
        self.moe = MoELayer(
            hidden_size=hidden_size,
            config=moe_config,
            use_gated_experts=use_gated_experts,
        )

        # Expert dropout wrapper
        self.expert_dropout = ExpertDropout(
            num_experts=moe_config.num_experts,
            dropout_rate=0.05,
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        std = 0.02
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=std)

    def _apply_rope(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Position Embeddings.

        Args:
            x: Input tensor (batch, num_heads, seq_len, head_dim).
            seq_len: Sequence length.

        Returns:
            Tuple of (rope_cos, rope_sin) tensors.
        """
        cos, sin = self.rope.get_embeddings(seq_len, x.device, x.dtype)
        return cos, sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K.

        Args:
            q: Query tensor (batch, heads, seq_len, head_dim).
            k: Key tensor (batch, heads, seq_len, head_dim).
            cos: Cosine embeddings (seq_len, head_dim).
            sin: Sine embeddings (seq_len, head_dim).

        Returns:
            Rotated (q, k) tensors.
        """
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed

    def _repeat_kv(
        self,
        hidden_states: torch.Tensor,
        n_rep: int,
    ) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Args:
            hidden_states: KV states (batch, num_kv_heads, seq_len, head_dim).
            n_rep: Number of repetitions.

        Returns:
            Expanded KV states (batch, num_heads, seq_len, head_dim).
        """
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, seq_len, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            q: Query (batch, heads, seq_q, head_dim).
            k: Key (batch, heads, seq_kv, head_dim).
            v: Value (batch, heads, seq_kv, head_dim).
            attention_mask: Optional mask.

        Returns:
            Attention output (batch, heads, seq_q, head_dim).
        """
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward pass through the sparse MoE transformer layer.

        Args:
            hidden_states: Input (batch, seq_len, hidden_size).
            attention_mask: Causal attention mask.
            position_ids: Position indices for RoPE.
            past_key_value: Optional KV cache.
            use_cache: Whether to use and return KV cache.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (hidden_states, present_kv, aux_loss).
        """
        batch_size, seq_len, _ = hidden_states.shape
        residual = hidden_states

        # === Self-Attention Block ===
        normed = self.input_layernorm(hidden_states)

        # Project to Q, K, V
        q = self.q_proj(normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(normed).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(normed).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE
        cos, sin = self._apply_rope(q, seq_len)
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Repeat KV for GQA
        k = self._repeat_kv(k, self.num_kv_groups)
        v = self._repeat_kv(v, self.num_kv_groups)

        # Create causal mask
        if attention_mask is None:
            total_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, total_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=total_len - seq_len + 1,
            )
            attention_mask = torch.zeros(seq_len, total_len, device=hidden_states.device)
            attention_mask = attention_mask.masked_fill(causal_mask, float("-inf"))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Attention
        attn_output = self._scaled_dot_product_attention(q, k, v, attention_mask)

        # Project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        # Residual connection
        hidden_states = residual + attn_output

        # === MoE Block ===
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        # MoE forward
        moe_output = self.moe(normed)

        # Apply expert dropout
        expert_out_3d = moe_output.output.view(
            self.config.num_experts if hasattr(self, 'config') else self.moe.config.num_experts,
            -1,
            self.hidden_size,
        ) if moe_output.output.view(-1).shape[0] == \
            self.moe.config.num_experts * seq_len else moe_output.output

        hidden_states = residual + moe_output.output

        return hidden_states, present_kv, moe_output.aux_loss


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) module.

    Computes cosine and sine embeddings for rotary position encoding.

    Args:
        dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length.
        base: Base for the frequency computation.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute embeddings
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute cos and sin embeddings up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(torch.float32), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(torch.float32), persistent=False)

    def get_embeddings(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE embeddings for the given sequence length.

        Args:
            seq_len: Sequence length.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos, sin) tensors of shape (seq_len, dim).
        """
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
            self.max_seq_len = seq_len

        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


# =============================================================================
# Full Mixture of Experts Transformer
# =============================================================================


class MoETransformerConfig:
    """Configuration for the full MoE Transformer model.

    Args:
        vocab_size: Vocabulary size.
        hidden_size: Model hidden dimension.
        num_hidden_layers: Total number of transformer layers.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads.
        max_position_embeddings: Maximum sequence length.
        rms_norm_eps: RMS normalization epsilon.
        rope_theta: RoPE base frequency.
        attention_dropout: Attention dropout rate.
        moe_config: MoE layer configuration.
        moe_frequency: Apply MoE every N layers (1 = every layer).
        moe_layer_indices: Explicit layer indices for MoE (overrides frequency).
        initializer_range: Weight initialization range.
        tie_word_embeddings: Whether to tie embedding and LM head weights.
        use_gradient_checkpointing: Whether to use gradient checkpointing.
        use_gated_experts: Whether to use gated (SwiGLU) experts.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        moe_config: Optional[MoEConfig] = None,
        moe_frequency: int = 1,
        moe_layer_indices: Optional[List[int]] = None,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        use_gradient_checkpointing: bool = False,
        use_gated_experts: bool = False,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.moe_config = moe_config or MoEConfig()
        self.moe_frequency = moe_frequency
        self.moe_layer_indices = moe_layer_indices
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gated_experts = use_gated_experts

        # Compute head dimension
        self.head_dim = hidden_size // num_attention_heads

        # Validate
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

    def get_moe_layers(self) -> List[int]:
        """Get the indices of layers that use MoE.

        Returns:
            List of layer indices with MoE.
        """
        if self.moe_layer_indices is not None:
            return self.moe_layer_indices

        return list(range(0, self.num_hidden_layers, self.moe_frequency))

    @property
    def num_moe_layers(self) -> int:
        """Number of MoE layers."""
        return len(self.get_moe_layers())

    @property
    def num_dense_layers(self) -> int:
        """Number of dense (non-MoE) layers."""
        return self.num_hidden_layers - self.num_moe_layers


class MixtureOfExpertsTransformer(nn.Module):
    """Full transformer model with Mixture of Experts layers.

    This model replaces standard dense FFN layers with MoE layers at
    configurable positions. The architecture follows the pre-norm
    transformer pattern with GQA attention and RoPE.

    Layer structure:
        For MoE layers:
            x' = x + Attention(LayerNorm(x))
            x'' = x' + MoE(LayerNorm(x'))
        For dense layers:
            x' = x + Attention(LayerNorm(x))
            x'' = x' + FFN(LayerNorm(x'))

    Args:
        config: MoETransformerConfig with all model parameters.
    """

    def __init__(self, config: MoETransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        # RoPE (shared across all layers)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Transformer layers
        self.layers = nn.ModuleList()
        moe_layer_set = set(config.get_moe_layers())

        for i in range(config.num_hidden_layers):
            if i in moe_layer_set:
                layer = SparseMoETransformerLayer(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    moe_config=config.moe_config,
                    head_dim=config.head_dim,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_dropout=config.attention_dropout,
                    max_position_embeddings=config.max_position_embeddings,
                    rope_theta=config.rope_theta,
                    layer_idx=i,
                    use_gated_experts=config.use_gated_experts,
                )
            else:
                layer = self._create_dense_layer(config, i)
            self.layers.append(layer)

        # Final normalization
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Gradient checkpointing
        self.gradient_checkpointing = config.use_gradient_checkpointing

    def _create_dense_layer(
        self,
        config: MoETransformerConfig,
        layer_idx: int,
    ) -> nn.Module:
        """Create a dense (non-MoE) transformer layer.

        Args:
            config: Model configuration.
            layer_idx: Layer index.

        Returns:
            Dense transformer layer module.
        """
        # Create a dense FFN MoE config with single expert
        dense_moe_config = MoEConfig(
            num_experts=1,
            num_selected=1,
            expert_dim=config.hidden_size * 4,
            capacity_factor=1.0,
            load_balancing_loss_coef=0.0,
            router_jitter_noise=0.0,
        )
        return SparseMoETransformerLayer(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            moe_config=dense_moe_config,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_dropout=config.attention_dropout,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            layer_idx=layer_idx,
            use_gated_experts=True,
        )

    def _init_weights(self, module: nn.Module):
        """Initialize module weights.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=self.config.initializer_range
            )
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, Any]:
        """Forward pass through the MoE Transformer.

        Args:
            input_ids: Token indices (batch, seq_len).
            attention_mask: Optional attention mask.
            position_ids: Optional position indices.
            past_key_values: Optional KV cache.
            inputs_embeds: Optional embedded inputs.
            labels: Optional target labels for loss computation.
            use_cache: Whether to use KV caching.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return hidden states.

        Returns:
            Dictionary with logits, loss, aux_loss, and optional extras.
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        # Prepare position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        # RoPE embeddings
        cos, sin = self.rope.get_embeddings(seq_len, inputs_embeds.device, inputs_embeds.dtype)

        # Process through layers
        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None
        all_aux_losses = []
        next_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    False,
                    output_attentions,
                    use_reentrant=False,
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[idx] if past_key_values is not None else None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if use_cache and layer_outputs[1] is not None:
                next_cache.append(layer_outputs[1])

            # Collect auxiliary losses from MoE layers
            if len(layer_outputs) > 2 and layer_outputs[2] is not None:
                all_aux_losses.append(layer_outputs[2])

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        # Compute combined auxiliary loss
        total_aux_loss = torch.tensor(0.0, device=logits.device)
        if all_aux_losses:
            total_aux_loss = torch.stack(all_aux_losses).sum()

        # Compute cross-entropy loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )
            loss = loss + total_aux_loss

        result = {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
        }

        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        if use_cache:
            result["past_key_values"] = next_cache

        return result

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters.

        Args:
            trainable_only: Count only trainable parameters.

        Returns:
            Total parameter count.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_moe_statistics(self) -> Dict[str, Any]:
        """Get MoE-related statistics from all MoE layers.

        Returns:
            Dictionary with aggregate MoE statistics.
        """
        stats = {
            "num_moe_layers": 0,
            "num_experts_per_layer": [],
            "total_aux_loss": 0.0,
        }

        for layer in self.layers:
            if isinstance(layer, SparseMoETransformerLayer):
                stats["num_moe_layers"] += 1
                stats["num_experts_per_layer"].append(
                    layer.moe.num_experts
                )

        return stats


# =============================================================================
# Utility classes for MoE management
# =============================================================================


class MoELayerSelector:
    """Selects which layers should be MoE vs dense.

    Supports multiple strategies for MoE layer placement:
        - 'bottom': MoE in the bottom N% of layers
        - 'top': MoE in the top N% of layers
        - 'every_n': MoE every N layers
        - 'alternating': Alternating MoE and dense
        - 'uniform': Random uniform selection
        - 'explicit': User-provided indices

    Args:
        total_layers: Total number of transformer layers.
        strategy: Selection strategy name.
        moe_fraction: Fraction of layers to make MoE (for fraction-based).
        every_n: N for 'every_n' strategy.
        explicit_indices: Layer indices for 'explicit' strategy.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        total_layers: int,
        strategy: str = "every_n",
        moe_fraction: float = 0.5,
        every_n: int = 2,
        explicit_indices: Optional[List[int]] = None,
        seed: int = 42,
    ):
        self.total_layers = total_layers
        self.strategy = strategy
        self.moe_fraction = moe_fraction
        self.every_n = every_n
        self.explicit_indices = explicit_indices or []
        self.seed = seed

    def select(self) -> List[int]:
        """Select MoE layer indices based on the strategy.

        Returns:
            List of layer indices that should use MoE.
        """
        if self.strategy == "bottom":
            n = max(1, int(self.total_layers * self.moe_fraction))
            return list(range(n))

        elif self.strategy == "top":
            n = max(1, int(self.total_layers * self.moe_fraction))
            return list(range(self.total_layers - n, self.total_layers))

        elif self.strategy == "every_n":
            return list(range(0, self.total_layers, self.every_n))

        elif self.strategy == "alternating":
            return list(range(0, self.total_layers, 2))

        elif self.strategy == "uniform":
            rng = torch.Generator()
            rng.manual_seed(self.seed)
            n = max(1, int(self.total_layers * self.moe_fraction))
            indices = torch.randperm(self.total_layers, generator=rng)[:n]
            return sorted(indices.tolist())

        elif self.strategy == "explicit":
            return [i for i in self.explicit_indices if 0 <= i < self.total_layers]

        else:
            raise ValueError(f"Unknown MoE selection strategy: {self.strategy}")


class MoEExpertPruner:
    """Prune (remove) underutilized experts from a trained MoE model.

    After training, some experts may receive very few tokens, contributing
    little to the model's output. This module identifies and removes such
    experts, creating a more efficient model.

    Args:
        utilization_threshold: Minimum utilization fraction to keep an expert.
        min_experts: Minimum number of experts to keep.
        max_experts_to_prune: Maximum experts to prune in one step.
    """

    def __init__(
        self,
        utilization_threshold: float = 0.01,
        min_experts: int = 2,
        max_experts_to_prune: int = 4,
    ):
        self.utilization_threshold = utilization_threshold
        self.min_experts = min_experts
        self.max_experts_to_prune = max_experts_to_prune

    def identify_prunable_experts(
        self,
        utilization: torch.Tensor,
    ) -> List[int]:
        """Identify experts that can be pruned.

        Args:
            utilization: Expert utilization fractions (num_experts,).

        Returns:
            List of expert indices to prune.
        """
        total_tokens = utilization.sum()
        if total_tokens == 0:
            return []

        normalized = utilization / total_tokens

        prunable = []
        for i in range(len(normalized)):
            if normalized[i] < self.utilization_threshold:
                prunable.append(i)

        # Keep minimum number of experts
        remaining = len(utilization) - len(prunable)
        if remaining < self.min_experts:
            prunable = prunable[:-(self.min_experts - remaining)]

        # Limit pruning
        prunable = prunable[:self.max_experts_to_prune]

        return prunable

    def prune_moe_layer(
        self,
        moe_layer: MoELayer,
        experts_to_keep: List[int],
    ) -> MoELayer:
        """Create a new MoE layer with only the specified experts.

        Args:
            moe_layer: Original MoE layer.
            experts_to_keep: Indices of experts to keep.

        Returns:
            New MoE layer with pruned experts.
        """
        new_config = MoEConfig(
            num_experts=len(experts_to_keep),
            num_selected=min(moe_layer.config.num_selected, len(experts_to_keep)),
            expert_dim=moe_layer.config.expert_dim,
            capacity_factor=moe_layer.config.capacity_factor,
            load_balancing_loss_coef=moe_layer.config.load_balancing_loss_coef,
            router_jitter_noise=moe_layer.config.router_jitter_noise,
        )

        new_layer = MoELayer(
            hidden_size=moe_layer.hidden_size,
            config=new_config,
        )

        # Copy expert weights
        for new_idx, old_idx in enumerate(experts_to_keep):
            new_layer.experts[new_idx].load_state_dict(
                moe_layer.experts[old_idx].state_dict()
            )

        return new_layer


class MoEScheduler:
    """Schedule MoE hyperparameters during training.

    Adjusts MoE-related hyperparameters over training steps, such as:
        - Gradually increasing capacity factor
        - Annealing noise in the router
        - Adjusting auxiliary loss coefficient

    Args:
        total_steps: Total training steps.
        initial_capacity_factor: Starting capacity factor.
        final_capacity_factor: Ending capacity factor.
        initial_noise: Starting router noise.
        final_noise: Ending router noise.
        warmup_steps: Steps before starting schedule changes.
    """

    def __init__(
        self,
        total_steps: int,
        initial_capacity_factor: float = 0.5,
        final_capacity_factor: float = 1.5,
        initial_noise: float = 0.2,
        final_noise: float = 0.01,
        warmup_steps: int = 1000,
    ):
        self.total_steps = total_steps
        self.initial_capacity_factor = initial_capacity_factor
        self.final_capacity_factor = final_capacity_factor
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self) -> Dict[str, float]:
        """Advance one step and return current hyperparameters.

        Returns:
            Dictionary with current hyperparameter values.
        """
        self.current_step += 1

        # Linear interpolation after warmup
        if self.current_step < self.warmup_steps:
            progress = 0.0
        else:
            effective_steps = self.total_steps - self.warmup_steps
            current = self.current_step - self.warmup_steps
            progress = min(1.0, current / max(effective_steps, 1))

        capacity = (
            self.initial_capacity_factor +
            (self.final_capacity_factor - self.initial_capacity_factor) * progress
        )
        noise = (
            self.initial_noise +
            (self.final_noise - self.initial_noise) * progress
        )

        return {
            "capacity_factor": capacity,
            "router_noise": noise,
            "progress": progress,
        }

    def get_current_values(self) -> Dict[str, float]:
        """Get current hyperparameter values without advancing.

        Returns:
            Dictionary with current hyperparameter values.
        """
        if self.current_step < self.warmup_steps:
            progress = 0.0
        else:
            effective_steps = self.total_steps - self.warmup_steps
            current = self.current_step - self.warmup_steps
            progress = min(1.0, current / max(effective_steps, 1))

        return {
            "capacity_factor": self.initial_capacity_factor +
                (self.final_capacity_factor - self.initial_capacity_factor) * progress,
            "router_noise": self.initial_noise +
                (self.final_noise - self.initial_noise) * progress,
            "progress": progress,
        }


class ExpertUtilityAnalyzer:
    """Analyze expert utility and specialization patterns.

    Provides tools for understanding how experts specialize during
    training, including token distribution analysis and expert
    similarity computation.

    Args:
        num_experts: Number of experts.
        hidden_size: Model hidden dimension.
    """

    def __init__(self, num_experts: int, hidden_size: int):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.router_prob_history: List[torch.Tensor] = []
        self.expert_input_history: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(num_experts)
        }

    def record_routing(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor,
    ):
        """Record routing decisions for analysis.

        Args:
            router_probs: Router probabilities (num_tokens, num_experts).
            selected_experts: Selected expert indices (num_tokens, num_selected).
        """
        self.router_prob_history.append(router_probs.detach().cpu())

        for k in range(selected_experts.shape[1]):
            for i in range(selected_experts.shape[0]):
                eid = selected_experts[i, k].item()
                if 0 <= eid < self.num_experts:
                    self.expert_input_history[eid].append(
                        router_probs[i].detach().cpu()
                    )

    def compute_expert_entropy(self) -> torch.Tensor:
        """Compute the entropy of each expert's routing distribution.

        Higher entropy indicates more uniform routing to the expert.

        Returns:
            Tensor of shape (num_experts,) with entropy per expert.
        """
        if not self.router_prob_history:
            return torch.zeros(self.num_experts)

        all_probs = torch.cat(self.router_prob_history, dim=0)
        avg_probs = all_probs.mean(dim=0)

        # Entropy: -sum(p * log(p))
        mask = avg_probs > 1e-10
        entropy = torch.zeros(self.num_experts)
        entropy[mask] = -avg_probs[mask] * torch.log(avg_probs[mask])

        return entropy

    def compute_expert_similarity_matrix(self) -> torch.Tensor:
        """Compute similarity between experts based on routing patterns.

        Returns:
            Similarity matrix of shape (num_experts, num_experts).
        """
        if not self.router_prob_history:
            return torch.eye(self.num_experts)

        all_probs = torch.cat(self.router_prob_history, dim=0)
        avg_probs = all_probs.mean(dim=0)

        # Cosine similarity
        similarity = torch.zeros(self.num_experts, self.num_experts)
        for i in range(self.num_experts):
            for j in range(self.num_experts):
                dot = (avg_probs[i] * avg_probs[j]).sum()
                norm_i = avg_probs[i].norm()
                norm_j = avg_probs[j].norm()
                if norm_i > 0 and norm_j > 0:
                    similarity[i, j] = dot / (norm_i * norm_j)

        return similarity

    def compute_load_balance_coefficient(self) -> float:
        """Compute the coefficient of variation of expert utilization.

        Lower values indicate better load balance.

        Returns:
            Coefficient of variation (std/mean).
        """
        if not self.router_prob_history:
            return 0.0

        all_probs = torch.cat(self.router_prob_history, dim=0)
        mean_probs = all_probs.mean(dim=0)

        utilization = mean_probs / (mean_probs.sum() + 1e-9)
        cv = utilization.std().item() / (utilization.mean().item() + 1e-9)

        return cv

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive expert utility report.

        Returns:
            Dictionary with analysis results.
        """
        entropy = self.compute_expert_entropy()
        similarity = self.compute_expert_similarity_matrix()
        cv = self.compute_load_balance_coefficient()

        avg_probs = torch.zeros(self.num_experts)
        if self.router_prob_history:
            all_probs = torch.cat(self.router_prob_history, dim=0)
            avg_probs = all_probs.mean(dim=0)

        return {
            "expert_entropy": entropy.tolist(),
            "mean_entropy": entropy.mean().item(),
            "expert_avg_prob": avg_probs.tolist(),
            "load_balance_cv": cv,
            "expert_similarity_matrix": similarity.tolist(),
            "num_routing_records": len(self.router_prob_history),
            "total_tokens_routed": sum(
                p.shape[0] for p in self.router_prob_history
            ) if self.router_prob_history else 0,
        }

    def reset(self):
        """Reset all recorded history."""
        self.router_prob_history = []
        self.expert_input_history = {
            i: [] for i in range(self.num_experts)
        }


class TokenRoutingVisualizer:
    """Visualize token routing patterns for MoE analysis.

    Provides utilities for creating routing visualizations including
    expert assignment heatmaps and utilization bar charts.

    Args:
        num_experts: Number of experts.
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.routing_maps: List[torch.Tensor] = []

    def record(
        self,
        selected_experts: torch.Tensor,
        gates: torch.Tensor,
    ):
        """Record routing map for a batch.

        Args:
            selected_experts: Expert assignments (num_tokens, num_selected).
            gates: Gating weights (num_tokens, num_selected).
        """
        routing_map = torch.zeros(
            selected_experts.shape[0], self.num_experts
        )
        for i in range(selected_experts.shape[0]):
            for k in range(selected_experts.shape[1]):
                eid = selected_experts[i, k].item()
                if 0 <= eid < self.num_experts:
                    routing_map[i, eid] = gates[i, k].item()
        self.routing_maps.append(routing_map)

    def get_aggregate_routing_map(self) -> torch.Tensor:
        """Get aggregate routing map across all recorded batches.

        Returns:
            Average routing map (num_tokens, num_experts) or
            aggregated statistics.
        """
        if not self.routing_maps:
            return torch.zeros(1, self.num_experts)

        return torch.cat(self.routing_maps, dim=0).mean(dim=0)

    def get_expert_load(self) -> torch.Tensor:
        """Get total token load per expert.

        Returns:
            Token counts per expert (num_experts,).
        """
        if not self.routing_maps:
            return torch.zeros(self.num_experts)

        all_maps = torch.cat(self.routing_maps, dim=0)
        load = (all_maps > 0).float().sum(dim=0)
        return load

    def reset(self):
        """Clear recorded routing maps."""
        self.routing_maps = []


# =============================================================================
# Noise-based Top-K Gating Variants
# =============================================================================


class NoisyTopKGating(TopKGating):
    """Top-K gating with learnable noise for exploration.

    Extends TopKGating with a learned noise distribution rather than
    fixed Gaussian noise. The noise parameters are learned during
    training to optimize exploration-exploitation tradeoff.

    Args:
        hidden_size: Input hidden dimension.
        config: MoEConfig.
    """

    def __init__(self, hidden_size: int, config: MoEConfig):
        super().__init__(hidden_size, config)

        # Learnable noise parameters: mean and std per expert
        self.noise_weight = nn.Parameter(
            torch.zeros(config.num_experts)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward with learnable noise.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (gates, selected_experts, aux_loss).
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.hidden_size)

        if self.input_norm is not None:
            x = self.input_norm(x)

        router_logits = self.router(x)

        if self.temperature != 1.0:
            router_logits = router_logits / self.temperature

        # Add learnable noise during training
        if self.training:
            noise = torch.randn_like(router_logits)
            noise = noise * F.softplus(self.noise_weight)
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)
        gates, selected_experts = _stable_top_k(router_probs, self.num_selected)

        gates_sum = gates.sum(dim=-1, keepdim=True)
        gates = gates / (gates_sum + 1e-9)

        aux_loss = self._compute_load_balancing_loss(
            router_probs, selected_experts
        )

        return gates, selected_experts, aux_loss


class HashRouter(nn.Module):
    """Deterministic hash-based routing for efficient inference.

    Uses a hash function to map tokens to experts, avoiding the
    computational cost of the learned router. Useful for fast
    inference or as a baseline.

    Args:
        num_experts: Number of experts.
        num_selected: Experts per token.
        hidden_size: Input dimension (for compatibility).
    """

    def __init__(
        self,
        num_experts: int,
        num_selected: int = 2,
        hidden_size: int = 0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_selected = num_selected

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens using hash function.

        Uses the sum of token embeddings as hash input for
        deterministic routing.

        Args:
            x: Input tensor (num_tokens, hidden_dim).

        Returns:
            Tuple of (gates, selected_experts, aux_loss=0).
        """
        num_tokens = x.shape[0]

        # Use hash of token sum for routing
        token_sums = x.sum(dim=-1)

        # Simple hash to expert mapping
        hash_values = (token_sums * 2654435761.0) % self.num_experts
        hash_values = hash_values.long()

        # Select top-k experts (using hash variations)
        selected_experts = torch.zeros(
            num_tokens, self.num_selected,
            device=x.device, dtype=torch.long,
        )
        gates = torch.ones(
            num_tokens, self.num_selected,
            device=x.device, dtype=x.dtype,
        ) / self.num_selected

        selected_experts[:, 0] = hash_values % self.num_experts

        for k in range(1, self.num_selected):
            variation = (hash_values * (k + 1) * 15485863) % self.num_experts
            selected_experts[:, k] = variation

        # Normalize gates
        gates = gates / gates.sum(dim=-1, keepdim=True)

        aux_loss = torch.tensor(0.0, device=x.device)

        return gates, selected_experts, aux_loss


class SinkhornRouting(nn.Module):
    """Sinkhorn-based routing for balanced expert assignment.

    Uses the Sinkhorn-Knopp algorithm to compute a doubly-stochastic
    routing matrix, ensuring that each expert receives approximately
    the same number of tokens.

    Args:
        hidden_size: Input dimension.
        num_experts: Number of experts.
        num_selected: Experts per token.
        num_iterations: Sinkhorn iterations.
        temperature: Softmax temperature.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_selected: int = 2,
        num_iterations: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.num_iterations = num_iterations
        self.temperature = temperature

        self.router = nn.Linear(hidden_size, num_experts, bias=True)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.router.bias)

    def _sinkhorn(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Sinkhorn normalization.

        Args:
            logits: Raw routing logits (num_tokens, num_experts).

        Returns:
            Doubly-stochastic routing matrix.
        """
        Q = logits / self.temperature
        Q = torch.exp(Q - Q.max(dim=-1, keepdim=True).values)

        for _ in range(self.num_iterations):
            # Row normalization
            row_sums = Q.sum(dim=-1, keepdim=True) + 1e-9
            Q = Q / row_sums

            # Column normalization
            col_sums = Q.sum(dim=0, keepdim=True) + 1e-9
            Q = Q / col_sums

        return Q

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens using Sinkhorn-based routing.

        Args:
            x: Input tensor (num_tokens, hidden_dim).

        Returns:
            Tuple of (gates, selected_experts, aux_loss).
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.hidden_size)

        num_tokens = x.shape[0]

        router_logits = self.router(x)
        routing_matrix = self._sinkhorn(router_logits)

        # Top-k selection
        gates, selected_experts = torch.topk(
            routing_matrix, self.num_selected, dim=-1
        )

        # Normalize gates
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-9)

        # Load balancing loss (should be small with Sinkhorn)
        mean_routing = routing_matrix.mean(dim=0)
        ideal = 1.0 / self.num_experts
        aux_loss = ((mean_routing - ideal) ** 2).sum()

        return gates, selected_experts, aux_loss


# =============================================================================
# Expert Weight Sharing
# =============================================================================


class ExpertWeightSharing:
    """Manage weight sharing between experts for parameter efficiency.

    Experts can share weights either partially (some layers shared) or
    fully (complete weight tying). This reduces memory and compute while
    maintaining most of the MoE benefit.

    Args:
        num_experts: Total number of experts.
        num_shared_groups: Number of weight-sharing groups.
    """

    def __init__(
        self,
        num_experts: int,
        num_shared_groups: int = 4,
    ):
        self.num_experts = num_experts
        self.num_shared_groups = num_shared_groups

        assert num_experts >= num_shared_groups

        # Assign experts to sharing groups
        self.group_assignments: Dict[int, int] = {}
        self.group_members: Dict[int, List[int]] = defaultdict(list)

        for i in range(num_experts):
            group = i % num_shared_groups
            self.group_assignments[i] = group
            self.group_members[group].append(i)

    def share_weights(
        self,
        experts: nn.ModuleList,
        share_up_proj: bool = True,
        share_down_proj: bool = False,
    ):
        """Share weights between experts in the same group.

        Args:
            experts: ModuleList of expert modules.
            share_up_proj: Whether to share the up projection weights.
            share_down_proj: Whether to share the down projection weights.
        """
        for group_id, member_indices in self.group_members.items():
            if len(member_indices) < 2:
                continue

            # Use the first expert in the group as the shared reference
            reference_idx = member_indices[0]

            for idx in member_indices[1:]:
                if share_up_proj and hasattr(experts[idx], 'up_proj'):
                    experts[idx].up_proj.weight = experts[reference_idx].up_proj.weight
                    if experts[idx].up_proj.bias is not None:
                        experts[idx].up_proj.bias = experts[reference_idx].up_proj.bias
                if share_down_proj and hasattr(experts[idx], 'down_proj'):
                    experts[idx].down_proj.weight = experts[reference_idx].down_proj.weight
                    if experts[idx].down_proj.bias is not None:
                        experts[idx].down_proj.bias = experts[reference_idx].down_proj.bias

    def get_num_unique_weights(self) -> int:
        """Get the number of unique weight sets.

        Returns:
            Number of unique expert parameter sets.
        """
        return self.num_shared_groups

    def get_sharing_ratio(self) -> float:
        """Get the weight sharing ratio.

        Returns:
            Fraction of parameters that are shared.
        """
        unique = self.get_num_unique_weights()
        return 1.0 - (unique / self.num_experts)


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    "MoEConfig",
    "Expert",
    "GatedExpert",
    "ExpertGroup",
    "TopKGating",
    "GroupedTopKGating",
    "NoisyTopKGating",
    "HashRouter",
    "SinkhornRouting",
    "ExpertLoadBalancer",
    "MoELayer",
    "MoELayerOutput",
    "DeviceAwareMoELayer",
    "ExpertParallelism",
    "ExpertDropout",
    "AdaptiveExpertDropout",
    "ResidualMoE",
    "StochasticDepthMoE",
    "SparseMoETransformerLayer",
    "MoETransformerConfig",
    "MixtureOfExpertsTransformer",
    "RotaryEmbedding",
    "MoELayerSelector",
    "MoEExpertPruner",
    "MoEScheduler",
    "ExpertUtilityAnalyzer",
    "ExpertWeightSharing",
    "TokenRoutingVisualizer",
]
