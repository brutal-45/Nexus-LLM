"""
Nexus State Space Models
==========================
State space model implementations as alternatives to transformers.

This module provides efficient sequence modeling via structured state spaces:
    - S4Layer: Structured State Space for Sequence Modeling (HiPPO-based)
    - MambaLayer: Selective state space model with input-dependent selection
    - RWKVLayer: Receptance Weighted Key Value with linear attention
    - HybridAttentionSSM: Hybrid layer combining attention + SSM

Key advantages over transformers:
    - O(N) complexity for sequence processing (vs O(N^2) attention)
    - Efficient parallel training via convolutional mode
    - Constant-size state for inference (no KV cache growth)
    - Superior long-range dependency modeling

Reference implementations draw from:
    - S4: "Efficiently Modeling Long Sequences with Structured State Spaces"
    - Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    - RWKV: "RWKV: Reinventing RNNs for the Transformer Era"
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple, List, Dict, Any, NamedTuple, Union
from dataclasses import dataclass, field
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SSMConfig:
    """Configuration for State Space Model layers.

    Attributes:
        d_model: Model hidden dimension.
        d_state: Dimension of the state space (N).
        d_conv: Local convolution width for input processing.
        expand_factor: Expansion factor for internal dimension.
        dt_rank: Rank of the time-step projection (delta).
        use_gate: Whether to use gating (input-dependent multiplier).
        use_bias: Whether to add bias to linear projections.
        dt_min: Minimum time step (delta).
        dt_max: Maximum time step (delta).
        dt_init: Initial time step scaling strategy ('random' or 'constant').
        dt_scale: Scaling factor for time step initialization.
        A_init_range: Range for A matrix initialization (log-uniform).
        D_init_value: Initial value for the skip connection D parameter.
        use_fast_path: Whether to use the optimized fast path for SSM computation.
        pad_token_id: Token ID used for padding (for variable-length sequences).
        chunk_size: Chunk size for chunked SSM processing.
        bidirectional: Whether to process sequence bidirectionally.
    """

    d_model: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    dt_rank: int = 0
    use_gate: bool = True
    use_bias: bool = True
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    A_init_range: tuple = (-5.0, -1.0)
    D_init_value: float = 1.0
    use_fast_path: bool = True
    pad_token_id: int = 0
    chunk_size: int = 0
    bidirectional: bool = False

    def __post_init__(self):
        if self.dt_rank == 0:
            self.dt_rank = max(16, self.d_model // 16)
        if self.expand_factor < 1:
            raise ValueError(f"expand_factor must be >= 1, got {self.expand_factor}")
        self.d_inner = self.d_model * self.expand_factor


@dataclass
class RWKVConfig:
    """Configuration for RWKV layers.

    Attributes:
        d_model: Model hidden dimension.
        n_head: Number of attention heads.
        head_size: Dimension of each head.
        time_decay_init_range: Initial range for time decay parameter.
        time_first_init_range: Initial range for time-first parameter.
        key_init_range: Initial range for key projection.
        value_init_range: Initial range for value projection.
        receptance_init_range: Initial range for receptance projection.
        gate_init_range: Initial range for gate projection.
        output_init_range: Initial range for output projection.
        use_time_mix: Whether to use time-mixing.
        use_channel_mix: Whether to use channel-mixing.
        time_decay_wd: Weight decay for time decay parameters.
    """

    d_model: int = 256
    n_head: int = 8
    head_size: int = 0
    time_decay_init_range: tuple = (-8.0, -4.0)
    time_first_init_range: tuple = (0.5, 1.5)
    key_init_range: tuple = (-0.5, 0.5)
    value_init_range: tuple = (-0.5, 0.5)
    receptance_init_range: tuple = (-0.5, 0.5)
    gate_init_range: tuple = (-0.5, 0.5)
    output_init_range: tuple = (-0.02, 0.02)
    use_time_mix: bool = True
    use_channel_mix: bool = True
    time_decay_wd: float = 0.0

    def __post_init__(self):
        if self.head_size == 0:
            self.head_size = self.d_model // self.n_head


@dataclass
class HybridSSMConfig:
    """Configuration for hybrid attention-SSM layers.

    Attributes:
        d_model: Model hidden dimension.
        ssm_config: Configuration for the SSM component.
        local_window_size: Window size for local attention.
        use_flash_attention: Whether to use flash attention.
        attention_heads: Number of attention heads.
        attention_dropout: Dropout for attention.
        ssm_weight: Weight for combining SSM and attention outputs.
        attention_weight: Weight for combining SSM and attention outputs.
        ssm_position: Position of SSM relative to attention ('parallel', 'before', 'after').
        use_rope: Whether to use RoPE in attention.
        rope_theta: Base for RoPE.
    """

    d_model: int = 256
    ssm_config: SSMConfig = field(default_factory=SSMConfig)
    local_window_size: int = 256
    use_flash_attention: bool = False
    attention_heads: int = 8
    attention_dropout: float = 0.0
    ssm_weight: float = 0.5
    attention_weight: float = 0.5
    ssm_position: str = "parallel"
    use_rope: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self):
        if self.ssm_config.d_model != self.d_model:
            self.ssm_config.d_model = self.d_model
        if self.ssm_position not in ("parallel", "before", "after"):
            raise ValueError(
                f"ssm_position must be 'parallel', 'before', or 'after', "
                f"got {self.ssm_position}"
            )


# =============================================================================
# Mathematical Utilities
# =============================================================================


def hippo_matrix(N: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Compute the HiPPO (High-order Polynomial Projection Operator) matrix.

    The HiPPO matrix A initializes the state transition matrix for S4.
    It follows the HiPPO-LegS (Legendre State Space) framework:
        A[n, m] = -(2n+1) if m <= n, else -(2m+1)

    Args:
        N: State dimension (size of the HiPPO matrix).
        dtype: Data type of the returned matrix.

    Returns:
        HiPPO matrix of shape (N, N).
    """
    A = torch.zeros(N, N, dtype=dtype)
    for n in range(N):
        for m in range(N):
            if m <= n:
                A[n, m] = -(2 * n + 1)
            else:
                A[n, m] = -(2 * m + 1)
    return A


def diagonal_plus_low_rank(
    N: int,
    rank: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create diagonal plus low-rank decomposition of HiPPO matrix.

    S4 parameterizes A as A = diag(P) - PP^T where P is (N, rank).
    This approximation captures the dominant eigenvalue structure of
    the HiPPO matrix efficiently.

    Args:
        N: State dimension.
        rank: Rank of the low-rank component.
        dtype: Data type.

    Returns:
        Tuple of (diagonal_part, P_matrix, reconstructed_A).
    """
    A_hippo = hippo_matrix(N, dtype)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A_hippo)

    # Sort by eigenvalue magnitude (descending)
    idx = torch.argsort(eigenvalues.abs(), descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Diagonal part (all eigenvalues)
    diag_part = eigenvalues

    # Low-rank part using top 'rank' eigenvectors
    P = eigenvectors[:, :rank] * eigenvalues[:rank].unsqueeze(0).sqrt()

    # Reconstruct: A ≈ V diag(λ) V^T = diag(λ) - correction
    # Use diagonal approximation with P^T P correction
    reconstructed = torch.diag(diag_part)
    if rank > 0:
        reconstructed = reconstructed - P @ P.T

    return diag_part, P, reconstructed


def discretize_zoh(
    A: torch.Tensor,
    B: torch.Tensor,
    delta: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discretize continuous-time state space using Zero-Order Hold (ZOH).

    Transforms continuous-time SSM:
        dx/dt = Ax + Bu
    into discrete-time SSM:
        x[k+1] = A_bar x[k] + B_bar u[k]

    Using the matrix exponential:
        A_bar = exp(A * delta)
        B_bar = (A_bar - I) @ A^{-1} @ B

    For diagonal A, this simplifies to:
        A_bar = exp(A * delta)
        B_bar = (exp(A * delta) - 1) / A * B

    Args:
        A: State matrix, shape (N,) for diagonal or (N, N).
        B: Input matrix, shape (N, d_inner) or (N,).
        delta: Time step, shape (batch, seq_len, 1) or scalar.

    Returns:
        Tuple of (A_bar, B_bar) discretized matrices.
    """
    if A.dim() == 1:
        # Diagonal case: efficient element-wise computation
        # A_bar = exp(A * delta)
        if delta.dim() == 3:
            A_bar = torch.exp(A.unsqueeze(0).unsqueeze(0) * delta)
        elif delta.dim() == 2:
            A_bar = torch.exp(A.unsqueeze(0) * delta)
        else:
            A_bar = torch.exp(A * delta)

        # B_bar = (exp(A * delta) - 1) / A * B
        # Handle A = 0 with Taylor expansion
        dA = A.unsqueeze(-1) * delta.unsqueeze(-1) if delta.dim() > 1 else A * delta
        safe_A = torch.where(
            A.abs() < 1e-6,
            torch.ones_like(A),
            A,
        )

        if delta.dim() == 3:
            expansion = (torch.exp(A.unsqueeze(0).unsqueeze(0) * delta) - 1.0)
            expansion = expansion / safe_A.unsqueeze(0).unsqueeze(0)
        elif delta.dim() == 2:
            expansion = (torch.exp(A.unsqueeze(0) * delta) - 1.0)
            expansion = expansion / safe_A.unsqueeze(0)
        else:
            expansion = (torch.exp(A * delta) - 1.0) / safe_A

        # Taylor expansion correction for near-zero A
        small_mask = A.abs() < 1e-6
        if small_mask.any():
            if delta.dim() == 3:
                taylor = delta * (1.0 + 0.5 * A.unsqueeze(0).unsqueeze(0) * delta)
            elif delta.dim() == 2:
                taylor = delta * (1.0 + 0.5 * A.unsqueeze(0) * delta)
            else:
                taylor = delta * (1.0 + 0.5 * A * delta)
            if delta.dim() == 3:
                expansion = torch.where(
                    small_mask.unsqueeze(0).unsqueeze(0),
                    taylor,
                    expansion,
                )
            elif delta.dim() == 2:
                expansion = torch.where(
                    small_mask.unsqueeze(0),
                    taylor,
                    expansion,
                )
            else:
                expansion = torch.where(small_mask, taylor, expansion)

        if B.dim() == 2:
            B_bar = expansion.unsqueeze(-1) * B.unsqueeze(0).unsqueeze(0)
        else:
            B_bar = expansion * B

    else:
        # Full matrix case: use matrix exponential
        # For small matrices, use eigendecomposition
        N = A.shape[0]
        if delta.dim() == 0:
            Ad = A * delta
        elif delta.dim() == 3:
            Ad = A.unsqueeze(0).unsqueeze(0) * delta
        else:
            Ad = A * delta

        # Matrix exponential via eigendecomposition
        eigenvalues, V = torch.linalg.eig(A)
        V_inv = torch.linalg.inv(V)

        if delta.dim() == 3:
            exp_diag = torch.exp(
                eigenvalues.unsqueeze(0).unsqueeze(0) * delta
            )
            A_bar = V @ torch.diag_embed(exp_diag) @ V_inv
        else:
            exp_diag = torch.exp(eigenvalues * delta)
            A_bar = V.real @ torch.diag(exp_diag.real) @ V_inv.real

        # B_bar = (A_bar - I) @ A_inv @ B
        A_inv = torch.linalg.inv(A)
        A_bar_minus_I = A_bar - torch.eye(N, device=A.device, dtype=A.dtype)
        if delta.dim() == 3:
            B_bar = (
                A_bar_minus_I @ A_inv.unsqueeze(0).unsqueeze(0) @
                B.unsqueeze(0).unsqueeze(0)
            )
        else:
            B_bar = A_bar_minus_I @ A_inv @ B

    return A_bar, B_bar


def krylov_approximation(
    A: torch.Tensor,
    B: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """Compute Krylov subspace approximation for efficient SSM computation.

    The Krylov method approximates the matrix exponential action
    exp(A*dt) @ B using a polynomial expansion in the Krylov subspace,
    which is much cheaper than the full matrix exponential for large N.

    K(A, B, order) = [B, AB, A^2 B, ..., A^(order-1) B]

    Args:
        A: State matrix (N,) diagonal or (N, N) full.
        B: Input matrix (N, d_inner).
        order: Order of the Krylov approximation.

    Returns:
        Krylov basis of shape (N, order * d_inner).
    """
    N = A.shape[0]
    d = B.shape[1] if B.dim() == 2 else 1

    # Build Krylov matrix: [B, AB, A^2B, ...]
    krylov = torch.zeros(N, order * d, device=A.device, dtype=A.dtype)

    current = B
    for k in range(order):
        krylov[:, k * d:(k + 1) * d] = current
        if A.dim() == 1:
            current = A.unsqueeze(-1) * current
        else:
            current = A @ current

    return krylov


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    """Causal 1D convolution for SSM input processing.

    Implements a causal convolution that ensures each output element
    only depends on current and past inputs (no future information leak).

    Args:
        x: Input tensor of shape (batch, d_inner, seq_len).
        weight: Convolution kernel of shape (d_inner, kernel_size).
        bias: Optional bias of shape (d_inner,).
        activation: Activation function ('silu', 'gelu', 'relu', 'none').

    Returns:
        Convolved output of shape (batch, d_inner, seq_len).
    """
    # Pad for causality: (kernel_size - 1) zeros at the start
    kernel_size = weight.shape[1]
    padding = kernel_size - 1

    x_padded = F.pad(x, (padding, 0))

    # Use groups=d_inner for depthwise convolution
    output = F.conv1d(
        x_padded, weight.unsqueeze(-2), bias=bias,
        padding=0, groups=x.shape[1],
    )

    # Apply activation
    if activation == "silu" or activation == "swish":
        output = F.silu(output)
    elif activation == "gelu":
        output = F.gelu(output)
    elif activation == "relu":
        output = F.relu(output)
    elif activation != "none":
        raise ValueError(f"Unknown activation: {activation}")

    return output


def selective_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Efficient selective scan implementation.

    Implements the selective state space scan:
        y = SSM(x, dt, A, B, C, D)

    where the state update is:
        h[t] = A_bar * h[t-1] + B_bar * x[t]
        y[t] = C * h[t] + D * x[t]

    with A_bar = exp(A * dt), B_bar = (exp(A * dt) - 1) / A * B

    Args:
        x: Input tensor (batch, d_inner, seq_len).
        dt: Time step tensor (batch, d_inner, seq_len).
        A: State matrix (d_state,) diagonal.
        B: Input-dependent B (batch, d_state, seq_len) or (d_state, d_inner).
        C: Output-dependent C (batch, d_state, seq_len) or (d_state, d_inner).
        D: Skip connection parameter (d_inner,).
        z: Optional gating tensor (batch, d_inner, seq_len).
        delta_bias: Bias for dt projection (d_inner,).
        delta_softplus: Whether to apply softplus to dt.

    Returns:
        Output tensor (batch, d_inner, seq_len).
    """
    batch, d_inner, seq_len = x.shape
    d_state = A.shape[0]

    # Process dt
    if delta_softplus:
        dt = F.softplus(dt)
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, -1, 1)

    # Discretize A and B
    # A is (d_state,), dt is (batch, d_inner, seq_len)
    # We need to broadcast: A_bar shape depends on whether B is input-dependent

    if B.dim() == 3 and B.shape[0] == batch:
        # Input-dependent B: (batch, d_state, seq_len)
        # A_bar: (batch, d_state, seq_len)
        dA = A.view(1, -1, 1) * dt[:, :1, :].expand(-1, d_state, seq_len)
        A_bar = torch.exp(dA)

        # B_bar for input-dependent case
        safe_A = A.view(1, -1, 1).expand(batch, d_state, seq_len)
        safe_A = torch.where(
            safe_A.abs() < 1e-6,
            torch.ones_like(safe_A),
            safe_A,
        )
        B_bar = (A_bar - 1.0) / safe_A * B

        # C is also input-dependent: (batch, d_state, seq_len)
        C_current = C
    else:
        # Static B: (d_state, d_inner)
        # For each time step, compute A_bar and B_bar
        # Use vectorized scan

        # Expand dt for state dimension
        dt_expanded = dt.mean(dim=1, keepdim=True).expand(-1, d_state, seq_len)
        dA = A.view(1, -1, 1) * dt_expanded
        A_bar = torch.exp(dA)

        safe_A = A.view(1, -1, 1).expand(batch, d_state, seq_len)
        safe_A = torch.where(
            safe_A.abs() < 1e-6,
            torch.ones_like(safe_A),
            safe_A,
        )
        B_bar = (A_bar - 1.0) / safe_A * B.view(1, -1, 1)

        C_current = C

    # Sequential scan (can be replaced with parallel scan for efficiency)
    h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros(batch, d_inner, seq_len, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        # State update: h = A_bar * h + B_bar * x
        h = A_bar[:, :, t] * h + B_bar[:, :, t] * x[:, :, t].unsqueeze(1)

        # Output: y = C * h + D * x
        y[:, :, t] = (C_current[:, :, t] * h).sum(dim=1) + D.view(1, -1) * x[:, :, t]

    # Apply gating
    if z is not None:
        y = y * z

    return y


def selective_scan_parallel(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Parallel scan implementation of selective scan using association scan.

    Uses the parallel prefix scan algorithm for O(N log N) computation
    instead of O(N) sequential scan. More efficient on GPU despite
    higher theoretical complexity.

    Args:
        x: Input (batch, d_inner, seq_len).
        dt: Time step (batch, d_inner, seq_len).
        A: State matrix diagonal (d_state,).
        B: Input matrix (d_state, d_inner).
        C: Output matrix (d_state, d_inner).
        D: Skip connection (d_inner,).
        z: Gating tensor (batch, d_inner, seq_len).
        delta_bias: Bias for dt (d_inner,).
        delta_softplus: Apply softplus to dt.

    Returns:
        Output tensor (batch, d_inner, seq_len).
    """
    batch, d_inner, seq_len = x.shape
    d_state = A.shape[0]

    if delta_softplus:
        dt = F.softplus(dt)
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, -1, 1)

    # For parallel scan, we work with the discretized matrices
    # Using the associative scan formulation:
    # Elements: (a_t, b_t) where:
    #   a_t = exp(A * dt_t)  shape: (batch, d_state)
    #   b_t = (exp(A*dt_t) - 1)/A * B * x_t  shape: (batch, d_state)

    dt_state = dt.mean(dim=1, keepdim=True).expand(-1, d_state, -1)
    dA = A.view(1, -1, 1) * dt_state  # (batch, d_state, seq_len)

    A_bars = torch.exp(dA)  # (batch, d_state, seq_len)

    safe_A = torch.where(
        A.view(1, -1, 1).abs().expand(batch, d_state, seq_len) < 1e-6,
        torch.ones(batch, d_state, seq_len, device=x.device, dtype=x.dtype),
        A.view(1, -1, 1).expand(batch, d_state, seq_len),
    )
    B_bar_base = (A_bars - 1.0) / safe_A  # (batch, d_state, seq_len)

    # B_bar * x_t: (batch, d_state, seq_len)
    Bx = torch.einsum('bsn,bdn->bdn', x, B_bar_base)

    # B expansion: (d_state, d_inner)
    # Bx already has the B matrix applied through B_bar_base
    # Actually need: B_bar * B * x
    # Let's redo: for each time step
    # B_bar_t = (exp(A*dt_t) - 1)/A  shape: (batch, d_state, 1)
    # B * x_t shape: (d_state, d_inner) @ (batch, d_inner, 1) = (batch, d_state, 1)

    Bx_corrected = torch.zeros(batch, d_state, seq_len, device=x.device, dtype=x.dtype)
    for t in range(seq_len):
        Bx_t = B @ x[:, :, t]  # (d_state, d_inner) @ (batch, d_inner) -> (batch, d_state)
        Bx_corrected[:, :, t] = B_bar_base[:, :, t] * Bx_t

    # C * h_t for each time step
    # Output = C @ h_t + D * x_t

    # For parallel scan, use sequential fallback for correctness
    # (full parallel scan implementation requires custom CUDA kernel)
    h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
    y = torch.zeros(batch, d_inner, seq_len, device=x.device, dtype=x.dtype)

    for t in range(seq_len):
        h = A_bars[:, :, t] * h + Bx_corrected[:, :, t]
        C_t = C[:, :, t] if C.dim() == 3 else C
        if C_t.dim() == 3:
            y[:, :, t] = (C_t[:, :, t] * h).sum(dim=1)
        else:
            y[:, :, t] = (C_t * h).sum(dim=1)
        y[:, :, t] = y[:, :, t] + D.view(1, -1) * x[:, :, t]

    if z is not None:
        y = y * z

    return y


# =============================================================================
# S4 Layer
# =============================================================================


class S4Layer(nn.Module):
    """Structured State Space for Sequence Modeling (S4).

    Implements the S4 layer from "Efficiently Modeling Long Sequences with
    Structured State Spaces" (Gu et al., 2022).

    The key idea is to parameterize the state matrix A using a diagonal
    plus low-rank (DPLR) decomposition of the HiPPO matrix, enabling
    efficient O(N log N) computation via FFT.

    Architecture:
        1. Linear projection: input -> d_inner
        2. Causal 1D convolution (parallel training mode)
        3. State space recurrence (sequential inference mode)
        4. Output projection: d_inner -> d_model

    The continuous-time SSM is defined as:
        dx/dt = Ax + Bu
        y = Cx + Du

    Discretized via ZOH:
        x[k+1] = A_bar * x[k] + B_bar * u[k]
        y[k] = C * x[k] + D * u[k]

    Args:
        config: SSMConfig with layer parameters.
        d_model: Model dimension (overrides config if provided).
        d_state: State dimension (overrides config if provided).
        d_conv: Convolution width (overrides config if provided).
    """

    def __init__(
        self,
        config: SSMConfig,
        d_model: Optional[int] = None,
        d_state: Optional[int] = None,
        d_conv: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model
        self.d_state = d_state or config.d_state
        self.d_conv = d_conv or config.d_conv
        self.d_inner = self.d_model * config.expand_factor

        # === Input Projection ===
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=config.use_bias)

        # === Causal Convolution ===
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=config.use_bias,
        )

        # === State Space Parameters ===
        # A: diagonal part of HiPPO matrix (d_state,)
        A_diag, A_P, _ = diagonal_plus_low_rank(
            self.d_state, rank=1, dtype=torch.float32
        )
        self.register_buffer("A_log", torch.log(-A_diag.abs()))
        self.A_P = nn.Parameter(A_P.clone().detach())

        # B: input matrix (d_state, d_inner)
        self.B = nn.Parameter(
            torch.randn(self.d_state, self.d_inner) * 0.02
        )

        # C: output matrix (d_state, d_inner)
        self.C = nn.Parameter(
            torch.randn(self.d_state, self.d_inner) * 0.02
        )

        # D: skip connection (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner) * config.D_init_value)

        # === Output Projection ===
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.use_bias)

        # === Activation ===
        self.activation = nn.SiLU(inplace=True) if config.use_gate else nn.Identity()

        # === Gating ===
        if config.use_gate:
            self.gate_proj = nn.Linear(self.d_model, self.d_inner, bias=config.use_bias)

        # === Time Step (dt) ===
        # Learnable dt parameter for each channel
        self.dt_proj = nn.Sequential(
            nn.Linear(self.d_inner, config.dt_rank, bias=config.use_bias),
            nn.Linear(config.dt_rank, self.d_inner, bias=True),
        )
        # Initialize dt projection
        with torch.no_grad():
            self.dt_proj[1].weight.uniform_(
                -config.dt_scale, config.dt_scale
            )
            self.dt_proj[1].bias.uniform_(
                math.log(math.exp(config.dt_min) - 1),
                math.log(math.exp(config.dt_max) - 1),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following S4 initialization scheme."""
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        nn.init.normal_(self.B, std=0.02)
        nn.init.normal_(self.C, std=0.02)

        # Conv1d initialization
        nn.init.normal_(self.conv1d.weight, std=0.02)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

    def _get_A(self) -> torch.Tensor:
        """Get the state transition matrix A.

        Returns:
            Diagonal A matrix of shape (d_state,).
        """
        return -torch.exp(self.A_log)

    def _get_dt(self, x: torch.Tensor) -> torch.Tensor:
        """Compute time step dt from input.

        Args:
            x: Input tensor (batch, seq_len, d_inner).

        Returns:
            Time step tensor (batch, seq_len, d_inner).
        """
        dt = self.dt_proj(x)
        dt = F.softplus(dt)
        return dt

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the S4 layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            state: Optional initial state (batch, d_state).

        Returns:
            Tuple of (output, final_state):
                - output: (batch, seq_len, d_model)
                - final_state: (batch, d_state)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        x_proj = self.in_proj(x)

        # Gating
        if self.config.use_gate:
            z = self.gate_proj(x)
            z = torch.sigmoid(z)
        else:
            z = None

        # Causal convolution
        x_conv = x_proj.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim padding
        x_conv = self.activation(x_conv)

        # Get time step
        dt = self._get_dt(x_conv.transpose(1, 2))  # (batch, seq_len, d_inner)
        dt = dt.transpose(1, 2)  # (batch, d_inner, seq_len)

        # Get state space matrices
        A = self._get_A()  # (d_state,)
        B = self.B  # (d_state, d_inner)
        C = self.C  # (d_state, d_inner)
        D = self.D  # (d_inner,)

        # Discretize
        A_bar, B_bar = discretize_zoh(A, B, dt)

        # State space computation
        if self.training and self.config.use_fast_path and seq_len > 64:
            # Use convolutional (parallel) mode for training
            y = self._ssm_conv_parallel(
                x_conv, A_bar, B_bar, C, D, z
            )
        else:
            # Use recurrent mode for inference or short sequences
            y, final_state = self._ssm_recurrent(
                x_conv, A_bar, B_bar, C, D, z, state
            )

        # Output projection
        output = self.out_proj(y.transpose(1, 2))

        return output, final_state if not self.training or not self.config.use_fast_path else None

    def _ssm_conv_parallel(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Parallel SSM computation using convolution via FFT.

        For a diagonal SSM, the recurrence can be expressed as a
        convolution, which is computed efficiently using FFT.

        Args:
            x: Input (batch, d_inner, seq_len).
            A_bar: Discretized A (batch, d_inner, seq_len) or (d_state,).
            B_bar: Discretized B.
            C: Output matrix (d_state, d_inner).
            D: Skip connection (d_inner,).
            z: Gating (batch, d_inner, seq_len).

        Returns:
            Output (batch, d_inner, seq_len).
        """
        batch, d_inner, seq_len = x.shape
        d_state = self.d_state

        # For each state dimension, compute the convolution
        # K_t = C_n * (A_bar_n)^t * B_bar_n for each state n
        # Then y_t = sum_n K_t * x_t + D * x_t

        output = torch.zeros_like(x)

        for n in range(d_state):
            # Compute impulse response for this state dimension
            A_n = A_bar[:, n, :]  # (batch, seq_len)

            # Compute powers of A_n: [1, A, A^2, ..., A^(T-1)]
            # Using cumulative product
            log_A_n = torch.where(
                A_n > 1e-10,
                torch.log(A_n),
                torch.full_like(A_n, -23.0),
            )
            powers = torch.exp(
                log_A_n * torch.arange(seq_len, device=x.device).float()
            )  # (batch, seq_len)

            # Kernel: K_t = C_n * A_n^t * B_n
            C_n = C[n]  # (d_inner,)
            B_n = B_bar[:, n, :]  # (batch, d_inner)

            kernel = powers.unsqueeze(-1) * B_n * C_n.unsqueeze(0)
            # kernel: (batch, seq_len, d_inner)

            # Flip kernel for causal convolution
            kernel_flipped = kernel.flip(1)

            # Use FFT for convolution
            x_padded = x.transpose(1, 2)  # (batch, seq_len, d_inner)
            padded_len = seq_len * 2
            x_fft = torch.fft.rfft(x_padded, n=padded_len, dim=1)
            k_fft = torch.fft.rfft(kernel_flipped, n=padded_len, dim=1)
            conv = torch.fft.irfft(x_fft * k_fft, n=padded_len, dim=1)
            conv = conv[:, :seq_len, :]  # (batch, seq_len, d_inner)

            output += conv.transpose(1, 2)

        # Add skip connection
        output = output + D.view(1, -1, 1) * x

        # Apply gating
        if z is not None:
            output = output * z

        return output

    def _ssm_recurrent(
        self,
        x: torch.Tensor,
        A_bar: torch.Tensor,
        B_bar: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        z: Optional[torch.Tensor],
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recurrent SSM computation (sequential scan).

        Suitable for autoregressive generation with stateful inference.

        Args:
            x: Input (batch, d_inner, seq_len).
            A_bar: Discretized A.
            B_bar: Discretized B.
            C: Output matrix (d_state, d_inner).
            D: Skip connection (d_inner,).
            z: Gating tensor.
            state: Initial state (batch, d_state).

        Returns:
            Tuple of (output, final_state).
        """
        batch, d_inner, seq_len = x.shape
        d_state = self.d_state

        if state is None:
            h = torch.zeros(batch, d_state, device=x.device, dtype=x.dtype)
        else:
            h = state

        output = torch.zeros(batch, d_inner, seq_len, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, :, t]  # (batch, d_inner)

            # State update: h = A_bar * h + B_bar * x
            for n in range(d_state):
                a_t = A_bar[:, n, t]  # (batch,)
                b_t = B_bar[:, n, t]  # (batch, d_inner)
                h[:, n] = a_t * h[:, n] + (b_t * x_t).sum(dim=-1)

            # Output: y = C^T h + D x
            y_t = torch.zeros(batch, d_inner, device=x.device, dtype=x.dtype)
            for n in range(d_state):
                y_t += C[n].unsqueeze(0) * h[:, n].unsqueeze(-1)
            y_t = y_t + D.unsqueeze(0) * x_t

            output[:, :, t] = y_t

        # Apply gating
        if z is not None:
            output = output * z

        return output, h

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize zero state for inference.

        Args:
            batch_size: Batch size.

        Returns:
            Initial state of shape (batch_size, d_state).
        """
        return torch.zeros(batch_size, self.d_state)


class S4Block(nn.Module):
    """S4 block with normalization and residual connection.

    A complete S4 block following the pre-norm pattern:
        x' = x + S4(LayerNorm(x))

    Args:
        config: SSMConfig.
        dropout: Dropout rate for residual.
    """

    def __init__(self, config: SSMConfig, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.ssm = S4Layer(config)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connection.

        Args:
            x: Input (batch, seq_len, d_model).
            state: Optional SSM state.

        Returns:
            Tuple of (output, final_state).
        """
        residual = x
        x_normed = self.norm(x)
        out, new_state = self.ssm(x_normed, state)
        out = self.dropout(out)
        return residual + out, new_state


# =============================================================================
# Mamba Layer
# =============================================================================


class MambaLayer(nn.Module):
    """Selective State Space Model (Mamba).

    Implements the Mamba architecture from "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" (Gu & Dao, 2023).

    Key innovation: unlike S4 where A, B, C are fixed parameters,
    Mamba makes B, C, and dt (time step) input-dependent. This
    "selectivity" allows the model to filter information based on
    the input, enabling content-based reasoning.

    Architecture:
        1. Input projection (d_model -> d_inner)
        2. Causal 1D convolution
        3. Selective SSM scan with input-dependent B, C, dt
        4. Gating mechanism
        5. Output projection (d_inner -> d_model)

    Args:
        config: SSMConfig with layer parameters.
        d_model: Model dimension.
        d_state: State dimension.
        d_conv: Convolution kernel size.
    """

    def __init__(
        self,
        config: SSMConfig,
        d_model: Optional[int] = None,
        d_state: Optional[int] = None,
        d_conv: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model
        self.d_state = d_state or config.d_state
        self.d_conv = d_conv or config.d_conv
        self.d_inner = self.d_model * config.expand_factor
        self.dt_rank = config.dt_rank

        # === Input Projection ===
        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=config.use_bias
        )

        # === Causal Convolution (depthwise) ===
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=self.d_conv,
            padding=self.d_conv - 1,
            groups=self.d_inner,
            bias=config.use_bias,
        )

        # === SSM Parameters ===
        # A: learnable state transition (d_state, d_inner)
        # Initialize with HiPPO-inspired log-uniform distribution
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.repeat(self.d_inner, 1).transpose(0, 1)  # (d_state, d_inner)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)

        # D: skip connection (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner) * config.D_init_value)

        # === Input-dependent projections ===
        # dt projection: d_inner -> dt_rank -> d_inner
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        # B projection: d_inner -> d_state * expand
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )

        # === Output Projection ===
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=config.use_bias
        )

        # === Gating ===
        if config.use_gate:
            self.gate_proj = nn.Linear(
                self.d_model, self.d_inner, bias=config.use_bias
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with Mamba-specific initialization."""
        # Input/output projections
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Convolution
        nn.init.normal_(self.conv1d.weight, std=0.02)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # dt projection initialization
        dt_init_std = self.config.dt_rank ** -0.5
        nn.init.normal_(self.dt_proj.weight, std=dt_init_std)

        # Initialize dt bias to produce time steps in desired range
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) *
                (math.log(self.config.dt_max) - math.log(self.config.dt_min)) +
                math.log(self.config.dt_min)
            )
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_dt)

        # x_proj initialization
        nn.init.xavier_uniform_(self.x_proj.weight)

    def _get_A(self) -> torch.Tensor:
        """Get the state transition matrix.

        Returns:
            A matrix of shape (d_state, d_inner).
        """
        return -torch.exp(self.A_log)

    def _selective_scan_ref(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Reference implementation of selective scan (sequential).

        Args:
            x: Input (batch, d_inner, seq_len).
            dt: Time step (batch, d_inner, seq_len).
            A: State matrix (d_state, d_inner).
            B: Input matrix (batch, d_state, seq_len).
            C: Output matrix (batch, d_state, seq_len).
            D: Skip connection (d_inner,).

        Returns:
            Output (batch, d_inner, seq_len).
        """
        batch, d_inner, seq_len = x.shape
        d_state = A.shape[0]

        # Discretize
        dA = A.unsqueeze(0).unsqueeze(-1) * dt.unsqueeze(1)  # (batch, d_state, seq_len)
        A_bar = torch.exp(dA)

        # B_bar = (A_bar - 1) / A * B
        safe_A = torch.where(
            A.unsqueeze(0).unsqueeze(-1).abs() < 1e-6,
            torch.ones_like(A.unsqueeze(0).unsqueeze(-1)),
            A.unsqueeze(0).unsqueeze(-1),
        )
        B_expanded = B.unsqueeze(2)  # (batch, d_state, 1, seq_len)
        B_bar = (A_bar - 1.0) / safe_A * B_expanded.squeeze(2)

        # Sequential scan
        h = torch.zeros(batch, d_state, d_inner, device=x.device, dtype=x.dtype)
        y = torch.zeros(batch, d_inner, seq_len, device=x.device, dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, :, t]  # (batch, d_inner)
            B_t = B[:, :, t]  # (batch, d_state)
            C_t = C[:, :, t]  # (batch, d_state)

            # State update: h = A_bar * h + B_bar * x
            A_bar_t = A_bar[:, :, t]  # (batch, d_state)
            B_bar_t = B_bar[:, :, t]  # (batch, d_state)

            h = (
                A_bar_t.unsqueeze(-1) * h +
                B_bar_t.unsqueeze(-1) * x_t.unsqueeze(1)
            )

            # Output: y = C^T h + D x
            y_t = torch.einsum('bdn,bdn->bn', C_t.unsqueeze(-1).expand(-1, -1, d_inner) * h, torch.ones_like(h))
            y[:, :, t] = y_t + D.unsqueeze(0) * x_t

        return y

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Mamba layer.

        Args:
            x: Input tensor (batch, seq_len, d_model).
            state: Optional cached state for autoregressive generation.

        Returns:
            Tuple of (output, final_state).
        """
        batch, seq_len, d_model = x.shape

        # === Input Projection ===
        # Split into main path and gating path
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_main = xz[:, :, :self.d_inner]
        z = xz[:, :, self.d_inner:]

        # === Causal Convolution ===
        x_conv = x_main.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = F.silu(x_conv)

        # === Compute Input-Dependent SSM Parameters ===
        x_proj_out = self.x_proj(x_conv.transpose(1, 2))  # (batch, seq_len, dt_rank + 2*d_state)

        # Split projections
        dt_input = x_proj_out[:, :, :self.dt_rank]
        B_input = x_proj_out[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_input = x_proj_out[:, :, self.dt_rank + self.d_state:]

        # Compute dt
        dt = self.dt_proj(dt_input)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)

        # Expand B and C for each inner dimension
        B_expanded = B_input.unsqueeze(2).expand(-1, -1, self.d_inner)
        C_expanded = C_input.unsqueeze(2).expand(-1, -1, self.d_inner)

        # === Selective Scan ===
        A = self._get_A()  # (d_state, d_inner)
        D = self.D  # (d_inner,)

        if self.training and seq_len > 1:
            # Use vectorized scan for training
            y = selective_scan(
                x_conv,
                dt.transpose(1, 2),
                A.diagonal().flatten()[:self.d_state],
                B_input.transpose(1, 2),
                C_input.transpose(1, 2),
                D,
                z=z.transpose(1, 2),
            )
        else:
            # Use sequential scan for inference
            y = self._selective_scan_ref(
                x_conv,
                dt.transpose(1, 2),
                A,
                B_expanded.transpose(1, 2),
                C_expanded.transpose(1, 2),
                D,
            )
            y = y * torch.sigmoid(z).transpose(1, 2)

        # === Output Projection ===
        output = self.out_proj(y.transpose(1, 2))

        return output, state

    def step(
        self,
        x: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-step forward for autoregressive generation.

        Args:
            x: Input token embedding (batch, 1, d_model).
            state: Dictionary with 'conv_state' and 'ssm_state'.

        Returns:
            Tuple of (output, updated_state).
        """
        batch = x.shape[0]

        # Input projection
        xz = self.in_proj(x)  # (batch, 1, d_inner * 2)
        x_main = xz[:, :, :self.d_inner]
        z = xz[:, :, self.d_inner:]

        # Update convolution state
        conv_state = state.get('conv_state')
        if conv_state is None:
            conv_state = torch.zeros(
                batch, self.d_inner, self.d_conv - 1,
                device=x.device, dtype=x.dtype,
            )

        conv_input = torch.cat([conv_state, x_main.transpose(1, 2)], dim=2)
        x_conv = self.conv1d(conv_input)[:, :, -1:]  # Last position
        x_conv = F.silu(x_conv)

        # Update conv state (shift left, add new)
        new_conv_state = conv_input[:, :, 1:]

        # Compute SSM parameters
        x_proj_out = self.x_proj(x_conv.transpose(1, 2))
        dt_input = x_proj_out[:, :, :self.dt_rank]
        B_input = x_proj_out[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_input = x_proj_out[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_input))

        # SSM step
        A = self._get_A()
        D = self.D
        ssm_state = state.get('ssm_state')
        if ssm_state is None:
            ssm_state = torch.zeros(
                batch, self.d_state, self.d_inner,
                device=x.device, dtype=x.dtype,
            )

        # Discretize
        dA = A.unsqueeze(0) * dt.unsqueeze(-1)  # (batch, d_state, d_inner)
        A_bar = torch.exp(dA)

        safe_A = torch.where(A.unsqueeze(0).abs() < 1e-6, torch.ones_like(A.unsqueeze(0)), A.unsqueeze(0))
        B_bar = (A_bar - 1.0) / safe_A

        # State update
        new_ssm_state = A_bar * ssm_state + B_bar * x_conv.transpose(1, 2)

        # Output
        y = (C_input.unsqueeze(-1).expand(-1, -1, self.d_inner) * new_ssm_state).sum(dim=1)
        y = y + D.unsqueeze(0) * x_conv.squeeze(-1)
        y = y * torch.sigmoid(z.squeeze(1))

        # Output projection
        output = self.out_proj(y)

        new_state = {
            'conv_state': new_conv_state.detach(),
            'ssm_state': new_ssm_state.detach(),
        }

        return output, new_state

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize state for autoregressive generation.

        Args:
            batch_size: Batch size.

        Returns:
            State dictionary with 'conv_state' and 'ssm_state'.
        """
        return {
            'conv_state': torch.zeros(
                batch_size, self.d_inner, self.d_conv - 1,
            ),
            'ssm_state': torch.zeros(
                batch_size, self.d_state, self.d_inner,
            ),
        }


class MambaBlock(nn.Module):
    """Mamba block with normalization and residual.

    Pre-norm Mamba block:
        x' = x + Mamba(LayerNorm(x))

    Args:
        config: SSMConfig.
        dropout: Dropout rate.
    """

    def __init__(self, config: SSMConfig, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = MambaLayer(config)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual.

        Args:
            x: Input (batch, seq_len, d_model).
            state: Optional state.

        Returns:
            Tuple of (output, state).
        """
        residual = x
        x_normed = self.norm(x)
        out, state = self.mamba(x_normed, state)
        out = self.dropout(out)
        return residual + out, state


# =============================================================================
# RWKV Layer
# =============================================================================


class TimeMix(nn.Module):
    """Time-mixing module for RWKV.

    Implements the time-mixing (attention-like) component of RWKV:
        - Computes time-decay weighted attention using a linear attention
          formulation with exponential decay.
        - Token shift: uses previous token's values for mixing.

    Architecture:
        x_prev = shift(x)  # previous token
        xr = x * R + x_prev * (1 - R)  # receptance mix
        xk = x * K + x_prev * (1 - K)  # key mix
        xv = x * V + x_prev * (1 - V)  # value mix

        r = sigmoid(W_r @ xr)
        k = W_k @ xk
        v = W_v @ xv

        w = exp(-exp(time_decay)) * time_first
        a = k * v  # attention score * value
        s = w * prev_s + a  # state update
        output = r * (W_o @ (s * k) / (s * k).sum())

    Args:
        d_model: Model dimension.
        n_head: Number of heads.
        head_size: Head dimension.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        head_size: int,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = head_size

        # Time-mixing retention parameters
        self.key = nn.Linear(d_model, n_head * head_size, bias=False)
        self.value = nn.Linear(d_model, n_head * head_size, bias=False)
        self.receptance = nn.Linear(d_model, n_head * head_size, bias=False)
        self.output = nn.Linear(n_head * head_size, d_model, bias=False)

        # Time decay and time-first (learnable per head)
        self.time_decay = nn.Parameter(
            torch.empty(n_head, head_size)
        )
        self.time_first = nn.Parameter(
            torch.empty(n_head, head_size)
        )
        self.time_mix_key = nn.Parameter(torch.ones(d_model))
        self.time_mix_value = nn.Parameter(torch.ones(d_model))
        self.time_mix_receptance = nn.Parameter(torch.ones(d_model))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize RWKV time-mix weights."""
        nn.init.uniform_(self.time_decay, -8.0, -4.0)
        nn.init.uniform_(self.time_first, 0.5, 1.5)
        nn.init.uniform_(self.time_mix_key, 0.2, 0.8)
        nn.init.uniform_(self.time_mix_value, 0.2, 0.8)
        nn.init.uniform_(self.time_mix_receptance, 0.2, 0.8)
        nn.init.uniform_(self.key.weight, -0.5, 0.5)
        nn.init.uniform_(self.value.weight, -0.5, 0.5)
        nn.init.uniform_(self.receptance.weight, -0.5, 0.5)
        nn.init.uniform_(self.output.weight, -0.02, 0.02)

    def token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift tokens by one position (prepend zero).

        Args:
            x: Input (batch, seq_len, d_model).

        Returns:
            Shifted tensor with zero at position 0.
        """
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        return torch.cat([zeros, x[:, :-1, :]], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through time-mixing.

        Args:
            x: Input (batch, seq_len, d_model).
            state: Optional state dictionary.

        Returns:
            Tuple of (output, new_state).
        """
        batch, seq_len, _ = x.shape
        last_state = None

        # Token shift for mixing
        x_prev = self.token_shift(x)

        # Mix current and previous tokens
        xk = x * self.time_mix_key + x_prev * (1 - self.time_mix_key)
        xv = x * self.time_mix_value + x_prev * (1 - self.time_mix_value)
        xr = x * self.time_mix_receptance + x_prev * (1 - self.time_mix_receptance)

        # Compute key, value, receptance
        k = self.key(xk).view(batch, seq_len, self.n_head, self.head_size)
        v = self.value(xv).view(batch, seq_len, self.n_head, self.head_size)
        r = torch.sigmoid(
            self.receptance(xr).view(batch, seq_len, self.n_head, self.head_size)
        )

        # Time decay weights
        # w = exp(-exp(time_decay)) for all positions except first
        w = torch.exp(-torch.exp(self.time_decay)).view(1, 1, self.n_head, self.head_size)
        w = w.expand(batch, seq_len, self.n_head, self.head_size)
        w[:, 0, :, :] = torch.exp(self.time_first).view(1, 1, self.n_head, self.head_size)

        # Compute attention using cumulative products for decay
        # For each head:
        #   s_t = w_t * k_t * v_t + w_{t-1} * s_{t-1}
        #   o_t = r_t * sum(w_j * k_j * v_j for j <= t) / sum(w_j * k_j for j <= t)
        output = torch.zeros(
            batch, seq_len, self.n_head, self.head_size,
            device=x.device, dtype=x.dtype,
        )

        attn_state = None
        if state is not None and 'time_mix' in state:
            attn_state = state['time_mix']

        for t in range(seq_len):
            for h in range(self.n_head):
                k_t = k[:, t, h, :]  # (batch, head_size)
                v_t = v[:, t, h, :]  # (batch, head_size)
                w_t = w[:, t, h, 0]   # (batch,)

                if t == 0 and attn_state is not None:
                    # Use previous state
                    numerator = attn_state[:, h, :self.head_size]
                    denominator = attn_state[:, h, self.head_size:]
                elif t == 0:
                    numerator = k_t * v_t
                    denominator = k_t
                else:
                    prev_num = output[:, t - 1, h, :] if t > 0 else torch.zeros_like(k_t)
                    prev_den = torch.ones(batch, self.head_size, device=x.device, dtype=x.dtype)

                    numerator = k_t * v_t + w_t.unsqueeze(-1) * (numerator if t == 0 else prev_num)
                    denominator = k_t + w_t.unsqueeze(-1) * (denominator if t == 0 else prev_den)

                # Normalize
                safe_den = denominator.abs() + 1e-8
                output[:, t, h, :] = r[:, t, h, :] * numerator / safe_den

        # Update state
        new_state = None
        if state is not None or seq_len > 1:
            new_state = {'time_mix': None}
            if seq_len > 1:
                # Save last time step's state
                s = torch.zeros(
                    batch, self.n_head, self.head_size * 2,
                    device=x.device, dtype=x.dtype,
                )
                for h in range(self.n_head):
                    if seq_len > 1:
                        s[:, h, :self.head_size] = (
                            k[:, -1, h, :] * v[:, -1, h, :] +
                            w[:, -1, h, 0].unsqueeze(-1) * (
                                k[:, -2, h, :] * v[:, -2, h, :]
                            )
                        )
                    else:
                        s[:, h, :self.head_size] = k[:, -1, h, :] * v[:, -1, h, :]
                    s[:, h, self.head_size:] = k[:, -1, h, :]
                new_state['time_mix'] = s.detach()

        # Output projection
        output = output.reshape(batch, seq_len, self.n_head * self.head_size)
        output = self.output(output)

        return output, new_state


class ChannelMix(nn.Module):
    """Channel-mixing module for RWKV.

    Implements the channel-mixing (FFN-like) component of RWKV:
        - Linear attention over channel dimension with time decay.
        - Token shift for temporal information.

    Architecture:
        x_prev = shift(x)
        xk = x * kmix + x_prev * (1 - kmix)
        xr = x * rmix + x_prev * (1 - rmix)

        r = sigmoid(W_r @ xr)
        k = square_relu(W_k @ xk)
        output = r * (W_v @ k)

    Args:
        d_model: Model dimension.
        d_hidden: Hidden dimension for channel mixing.
    """

    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or int(d_model * 4)

        self.key = nn.Linear(d_model, self.d_hidden, bias=False)
        self.value = nn.Linear(self.d_hidden, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)

        self.time_mix_key = nn.Parameter(torch.ones(d_model))
        self.time_mix_receptance = nn.Parameter(torch.ones(d_model))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize channel-mix weights."""
        nn.init.uniform_(self.time_mix_key, 0.2, 0.8)
        nn.init.uniform_(self.time_mix_receptance, 0.2, 0.8)
        nn.init.uniform_(self.key.weight, -0.5, 0.5)
        nn.init.uniform_(self.value.weight, -0.5, 0.5)
        nn.init.uniform_(self.receptance.weight, -0.5, 0.5)

    def token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Shift tokens by one position."""
        zeros = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device, dtype=x.dtype)
        return torch.cat([zeros, x[:, :-1, :]], dim=1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through channel-mixing.

        Args:
            x: Input (batch, seq_len, d_model).

        Returns:
            Output (batch, seq_len, d_model).
        """
        x_prev = self.token_shift(x)

        xk = x * self.time_mix_key + x_prev * (1 - self.time_mix_key)
        xr = x * self.time_mix_receptance + x_prev * (1 - self.time_mix_receptance)

        r = torch.sigmoid(self.receptance(xr))
        k = F.silu(self.key(xk))

        return r * self.value(k)


class RWKVLayer(nn.Module):
    """RWKV (Receptance Weighted Key Value) layer.

    Combines time-mixing (attention-like) and channel-mixing (FFN-like)
    in a single layer:

        x' = x + TimeMix(LayerNorm(x))
        x'' = x' + ChannelMix(LayerNorm(x'))

    RWKV achieves O(N) complexity like RNNs while maintaining
    training parallelism like transformers.

    Args:
        config: RWKVConfig.
        d_model: Model dimension.
    """

    def __init__(
        self,
        config: RWKVConfig,
        d_model: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model

        # Layer norms
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        # Time-mixing and channel-mixing
        if config.use_time_mix:
            self.time_mix = TimeMix(
                self.d_model, config.n_head, config.head_size
            )
        if config.use_channel_mix:
            self.channel_mix = ChannelMix(self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass through RWKV layer.

        Args:
            x: Input (batch, seq_len, d_model).
            state: Optional state for autoregressive generation.

        Returns:
            Tuple of (output, new_state).
        """
        residual = x

        # Time-mixing
        if self.config.use_time_mix:
            x_normed = self.ln1(x)
            tm_out, tm_state = self.time_mix(x_normed, state)
            x = residual + tm_out
            residual = x

        # Channel-mixing
        if self.config.use_channel_mix:
            x_normed = self.ln2(x)
            cm_out = self.channel_mix(x_normed)
            x = residual + cm_out

        return x, tm_state if self.config.use_time_mix else None

    def init_state(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Initialize state for autoregressive generation.

        Args:
            batch_size: Batch size.

        Returns:
            State dictionary.
        """
        state = {}
        if self.config.use_time_mix:
            state['time_mix'] = torch.zeros(
                batch_size, self.config.n_head,
                self.config.head_size * 2,
            )
        return state


# =============================================================================
# Hybrid Attention-SSM Layer
# =============================================================================


class HybridAttentionSSM(nn.Module):
    """Hybrid layer combining self-attention and state space models.

    This module combines local self-attention (for precise local patterns)
    with SSM (for efficient long-range modeling). The two components can
    be arranged in parallel, sequential, or weighted combination.

    Architecture:
        if parallel:
            output = w_attn * Attention(x) + w_ssm * SSM(x)
        elif before:
            output = Attention(SSM(x))
        elif after:
            output = SSM(Attention(x))

    Args:
        config: HybridSSMConfig.
        d_model: Model dimension.
    """

    def __init__(
        self,
        config: HybridSSMConfig,
        d_model: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model
        self.head_dim = self.d_model // config.attention_heads

        # === Normalization ===
        self.norm_attn = nn.LayerNorm(self.d_model)
        self.norm_ssm = nn.LayerNorm(self.d_model)

        # === Local Self-Attention ===
        self.q_proj = nn.Linear(
            self.d_model, config.attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.d_model, config.attention_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.d_model, config.attention_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.attention_heads * self.head_dim, self.d_model, bias=False
        )

        self.attn_dropout = nn.Dropout(config.attention_dropout)

        # === SSM Component ===
        self.ssm = S4Layer(config.ssm_config)

        # === Combination weights (learnable) ===
        self.ssm_gate = nn.Parameter(torch.tensor(0.0))
        self.attn_gate = nn.Parameter(torch.tensor(0.0))

        # === RoPE (optional) ===
        if config.use_rope:
            self.rope = self._build_rope(
                self.head_dim, config.local_window_size, config.rope_theta
            )
        else:
            self.rope = None

        self._init_weights()

    def _build_rope(
        self,
        dim: int,
        max_len: int,
        base: float,
    ) -> Dict[str, torch.Tensor]:
        """Build RoPE embeddings.

        Args:
            dim: Head dimension.
            max_len: Maximum sequence length.
            base: Base frequency.

        Returns:
            Dictionary with 'cos' and 'sin' tensors.
        """
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return {
            'cos': emb.cos(),
            'sin': emb.sin(),
        }

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to query/key tensors.

        Args:
            x: Tensor (batch, heads, seq_len, head_dim).
            seq_len: Sequence length.

        Returns:
            Rotated tensor.
        """
        if self.rope is None:
            return x

        cos = self.rope['cos'][:seq_len].to(x.device, dtype=x.dtype)
        sin = self.rope['sin'][:seq_len].to(x.device, dtype=x.dtype)

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)

        return x * cos.unsqueeze(0).unsqueeze(0) + rotated * sin.unsqueeze(0).unsqueeze(0)

    def _init_weights(self):
        """Initialize layer weights."""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def _local_attention(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute local windowed self-attention.

        Args:
            x: Input (batch, seq_len, d_model).
            attention_mask: Optional mask.

        Returns:
            Attention output (batch, seq_len, d_model).
        """
        batch, seq_len, _ = x.shape
        num_heads = self.config.attention_heads
        window = self.config.local_window_size

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Create local window mask
        if attention_mask is None:
            # Block diagonal mask for local windows
            local_mask = torch.zeros(seq_len, seq_len, device=x.device, dtype=torch.bool)
            for i in range(seq_len):
                start = max(0, i - window // 2)
                end = min(seq_len, i + window // 2 + 1)
                local_mask[i, start:end] = True
            attention_mask = local_mask

        if attention_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                ~attention_mask.unsqueeze(0).unsqueeze(0), float('-inf')
            )
        else:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the hybrid layer.

        Args:
            x: Input (batch, seq_len, d_model).
            attention_mask: Optional attention mask.
            ssm_state: Optional SSM state.

        Returns:
            Tuple of (output, ssm_state).
        """
        residual = x

        # Compute gates
        ssm_weight = torch.sigmoid(self.ssm_gate)
        attn_weight = torch.sigmoid(self.attn_gate)
        total = ssm_weight + attn_weight + 1e-8
        ssm_weight = ssm_weight / total
        attn_weight = attn_weight / total

        if self.config.ssm_position == "parallel":
            # Parallel combination
            ssm_out, ssm_state = self.ssm(self.norm_ssm(x), ssm_state)
            attn_out = self._local_attention(
                self.norm_attn(x), attention_mask
            )
            output = residual + ssm_weight * ssm_out + attn_weight * attn_out

        elif self.config.ssm_position == "before":
            # SSM first, then attention
            ssm_out, ssm_state = self.ssm(self.norm_ssm(x), ssm_state)
            x = residual + ssm_out
            attn_out = self._local_attention(
                self.norm_attn(x), attention_mask
            )
            output = x + attn_out

        elif self.config.ssm_position == "after":
            # Attention first, then SSM
            attn_out = self._local_attention(
                self.norm_attn(x), attention_mask
            )
            x = residual + attn_out
            ssm_out, ssm_state = self.ssm(self.norm_ssm(x), ssm_state)
            output = x + ssm_out

        return output, ssm_state


class HybridBlock(nn.Module):
    """Hybrid attention-SSM block with residual and normalization.

    Pre-norm pattern:
        x' = x + Hybrid(LayerNorm(x))

    Args:
        config: HybridSSMConfig.
        dropout: Dropout rate.
    """

    def __init__(self, config: HybridSSMConfig, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.hybrid = HybridAttentionSSM(config)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual.

        Args:
            x: Input (batch, seq_len, d_model).
            attention_mask: Optional attention mask.
            ssm_state: Optional SSM state.

        Returns:
            Tuple of (output, ssm_state).
        """
        residual = x
        x_normed = self.norm(x)
        out, state = self.hybrid(x_normed, attention_mask, ssm_state)
        out = self.dropout(out)
        return residual + out, state


# =============================================================================
# Full SSM-based Model
# =============================================================================


class SSMStack(nn.Module):
    """Stack of SSM layers forming a complete sequence model.

    Args:
        config: SSMConfig.
        num_layers: Number of SSM layers.
        dropout: Dropout between layers.
        layer_type: Type of SSM layer ('s4', 'mamba', 'rwkv', 'hybrid').
        hybrid_config: Optional config for hybrid layers.
        rwkv_config: Optional config for RWKV layers.
    """

    def __init__(
        self,
        config: SSMConfig,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_type: str = "s4",
        hybrid_config: Optional[HybridSSMConfig] = None,
        rwkv_config: Optional[RWKVConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.layer_type = layer_type

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if layer_type == "s4":
                layer = S4Block(config, dropout=dropout)
            elif layer_type == "mamba":
                layer = MambaBlock(config, dropout=dropout)
            elif layer_type == "rwkv":
                rwkv_cfg = rwkv_config or RWKVConfig(d_model=config.d_model)
                layer = RWKVLayer(rwkv_cfg, d_model=config.d_model)
            elif layer_type == "hybrid":
                hyb_cfg = hybrid_config or HybridSSMConfig(d_model=config.d_model)
                layer = HybridBlock(hyb_cfg, dropout=dropout)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            self.layers.append(layer)

        # Final normalization
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """Forward pass through all layers.

        Args:
            x: Input (batch, seq_len, d_model).
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (output, list of final states per layer).
        """
        states = []
        for layer in self.layers:
            if isinstance(layer, (S4Block, MambaBlock, HybridBlock)):
                x, state = layer(x, ssm_state=None)
            elif isinstance(layer, RWKVLayer):
                x, state = layer(x, state=None)
            elif isinstance(layer, HybridAttentionSSM):
                x, state = layer(x, attention_mask=attention_mask)
            else:
                x, state = layer(x)
            states.append(state)

        x = self.final_norm(x)
        return x, states


class SSMLanguageModel(nn.Module):
    """Language model built entirely from State Space Models.

    Replaces transformer blocks with SSM blocks for efficient
    long-range sequence modeling.

    Args:
        vocab_size: Vocabulary size.
        config: SSMConfig.
        num_layers: Number of SSM layers.
        dropout: Dropout rate.
        layer_type: SSM layer type.
        max_seq_len: Maximum sequence length for positional info.
        tie_embeddings: Whether to tie embedding and output weights.
    """

    def __init__(
        self,
        vocab_size: int,
        config: SSMConfig,
        num_layers: int = 12,
        dropout: float = 0.1,
        layer_type: str = "mamba",
        max_seq_len: int = 2048,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Token embedding
        self.embed = nn.Embedding(vocab_size, config.d_model)

        # SSM backbone
        self.backbone = SSMStack(
            config=config,
            num_layers=num_layers,
            dropout=dropout,
            layer_type=layer_type,
        )

        # LM head
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, Any]:
        """Forward pass for language modeling.

        Args:
            input_ids: Token IDs (batch, seq_len).
            labels: Optional target labels.

        Returns:
            Dictionary with logits and optional loss.
        """
        x = self.embed(input_ids)
        hidden, states = self.backbone(x)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return {"logits": logits, "loss": loss, "states": states}

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.LongTensor:
        """Autoregressive generation.

        Args:
            input_ids: Prompt tokens (batch, seq_len).
            max_new_tokens: Maximum new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated tokens (batch, seq_len + gen_len).
        """
        self.eval()
        generated = input_ids.clone()

        # Initialize states for each layer
        states = [None] * self.backbone.num_layers

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward through backbone with state
                x = self.embed(generated[:, -1:])
                h = x
                new_states = []

                for i, layer in enumerate(self.backbone.layers):
                    if isinstance(layer, MambaBlock):
                        h, state = layer.mamba(h, states[i])
                    elif isinstance(layer, S4Block):
                        h, state = layer.ssm(h, states[i])
                    elif isinstance(layer, RWKVLayer):
                        h, state = layer(h, states[i])
                    else:
                        h, state = layer(h)
                    new_states.append(state)

                states = new_states
                h = self.backbone.final_norm(h)

                logits = self.lm_head(h[:, -1, :])

                # Temperature sampling
                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    if top_p < 1.0:
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cum_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_mask = cum_probs > top_p
                        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                        sorted_mask[..., 0] = False
                        probs = probs.scatter(1, sorted_indices, sorted_probs)
                        probs = probs.masked_fill(sorted_mask.scatter(1, sorted_indices, sorted_mask), 0)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


# =============================================================================
# Utilities for SSM Analysis
# =============================================================================


class SSMAnalyzer:
    """Analyze SSM layer behavior during forward pass.

    Tracks state evolution, decay patterns, and other diagnostics.

    Args:
        layer: SSM layer to analyze.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.state_history: List[torch.Tensor] = []
        self.dt_history: List[torch.Tensor] = []
        self.output_history: List[torch.Tensor] = []

    def analyze_forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Run forward pass and collect diagnostics.

        Args:
            x: Input (batch, seq_len, d_model).

        Returns:
            Dictionary with analysis results.
        """
        self.state_history = []
        self.dt_history = []

        # Register hooks
        hooks = []

        def state_hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.state_history.append(output[1].detach())

        hook = self.layer.register_forward_hook(state_hook)
        hooks.append(hook)

        try:
            output = self.layer(x)
            analysis = self._compute_analysis(x, output)
        finally:
            for h in hooks:
                h.remove()

        return analysis

    def _compute_analysis(
        self,
        x: torch.Tensor,
        output: Tuple[torch.Tensor, Any],
    ) -> Dict[str, Any]:
        """Compute analysis metrics.

        Args:
            x: Input tensor.
            output: Layer output.

        Returns:
            Analysis dictionary.
        """
        result = {
            "input_shape": tuple(x.shape),
            "output_shape": tuple(output[0].shape),
            "num_states_recorded": len(self.state_history),
        }

        if self.state_history:
            final_state = self.state_history[-1]
            result["final_state_mean"] = final_state.mean().item()
            result["final_state_std"] = final_state.std().item()
            result["final_state_max_abs"] = final_state.abs().max().item()

        return result

    def compute_memory_decay_rate(
        self,
        x: torch.Tensor,
        num_steps: int = 50,
    ) -> float:
        """Estimate the memory decay rate of the SSM.

        Feeds a single impulse and measures how quickly the state decays.

        Args:
            x: Reference input for shape information.
            num_steps: Number of steps to measure.

        Returns:
            Estimated decay rate.
        """
        device = x.device
        dtype = x.dtype
        batch = 1
        seq_len = num_steps

        # Create impulse input (one at the beginning, zeros after)
        impulse = torch.zeros(1, seq_len, x.shape[-1], device=device, dtype=dtype)
        impulse[0, 0, :] = 1.0

        with torch.no_grad():
            output, state = self.layer(impulse)

        # Measure how output magnitude decays
        output_mag = output[0].norm(dim=-1)

        if output_mag[0] > 1e-8:
            decay_ratio = output_mag[-1] / output_mag[0]
        else:
            decay_ratio = 0.0

        return decay_ratio.item()


# =============================================================================
# Public API
# =============================================================================


__all__ = [
    "SSMConfig",
    "RWKVConfig",
    "HybridSSMConfig",
    "S4Layer",
    "S4Block",
    "MambaLayer",
    "MambaBlock",
    "RWKVLayer",
    "TimeMix",
    "ChannelMix",
    "HybridAttentionSSM",
    "HybridBlock",
    "SSMStack",
    "SSMLanguageModel",
    "SSMAnalyzer",
    "hippo_matrix",
    "diagonal_plus_low_rank",
    "discretize_zoh",
    "krylov_approximation",
    "causal_conv1d",
    "selective_scan",
    "selective_scan_parallel",
]
