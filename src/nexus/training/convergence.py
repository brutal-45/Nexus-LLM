"""
Convergence Analysis Tools for Large Language Model Training.

This module provides a comprehensive suite of tools for analyzing the training
dynamics of large language models, including:

- Loss landscape geometry (Hessian spectrum, sharpness, curvature)
- Saddle point detection and escape strategies
- Gradient noise scale tracking and phase transition detection
- Critical batch size estimation for optimal throughput
- Unified training dynamics monitoring

All functions operate on PyTorch tensors and are designed to be memory-efficient,
working with both CPU and CUDA devices. Support for distributed training is included
where applicable.

Key References:
    - "Don't Decay the Learning Rate, Increase the Batch Size" (Smith et al., 2018)
    - "Measuring the Effects of Data Parallelism" (Shallue et al., 2018)
    - "Sharpness-Aware Minimization" (Foret et al., 2021)
    - "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
    - "Escaping Saddle Points with Efficient Second-Order Methods" (Agarwal et al., 2017)
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _flatten_parameters(
    model: nn.Module,
    detach: bool = True,
) -> Tuple[torch.Tensor, List[Tuple[str, torch.Size]]]:
    """Return a flattened vector of all model parameters and shape metadata."""
    shapes: List[Tuple[str, torch.Size]] = []
    flat_parts: List[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        p = param.detach() if detach else param
        shapes.append((name, param.shape))
        flat_parts.append(p.reshape(-1))
    return torch.cat(flat_parts, dim=0), shapes


def _unflatten_vector(
    vector: torch.Tensor,
    shapes: List[Tuple[str, torch.Size]],
) -> Dict[str, torch.Tensor]:
    """Reshape a flat vector back into per-parameter tensors."""
    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in shapes:
        numel = shape.numel()
        result[name] = vector[offset : offset + numel].reshape(shape)
        offset += numel
    return result


def _apply_direction(
    model: nn.Module,
    direction: torch.Tensor,
    shapes: List[Tuple[str, torch.Size]],
    epsilon: float,
) -> None:
    """Perturb model parameters in-place: θ ← θ + ε * d."""
    with torch.no_grad():
        offset = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            param.add_(direction[offset : offset + numel].reshape(param.shape) * epsilon)
            offset += numel


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Division that returns 0 when denominator is near zero."""
    return torch.where(
        denominator.abs() > eps,
        numerator / denominator,
        torch.zeros_like(numerator),
    )


def _is_distributed() -> bool:
    """Check if torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _get_device(model: nn.Module) -> torch.device:
    """Return the device of the first parameter in the model."""
    for p in model.parameters():
        return p.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Loss Landscape Analysis
# ---------------------------------------------------------------------------

class LossLandscapeAnalyzer:
    """Analyzes the geometry of the loss landscape around the current solution.

    Provides tools for computing the Hessian spectrum, evaluating sharpness,
    measuring curvature, and probing the loss surface along arbitrary directions
    in parameter space.

    Mathematical background:
        The Hessian H = ∇²L(θ) encodes local curvature.  Its eigenvalues λ_i
        determine the landscape shape near the current parameters:
        - All λ_i > 0  → local minimum
        - All λ_i < 0  → local maximum
        - Mixed signs   → saddle point
        - |λ_i| large   → steep / sharp region
        - |λ_i| small   → flat region

    Args:
        model: The neural network model to analyze.
        loss_fn: A callable ``loss_fn(model_output, target) → Tensor``.
        device: Device for computation (``'cuda'`` or ``'cpu'``).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    # ------------------------------------------------------------------
    # Hessian-vector product
    # ------------------------------------------------------------------

    def compute_hessian_vector_product(
        self,
        vector: torch.Tensor,
        dataloader: DataLoader,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """Compute H · v using Pearlmutter's method (double back-propagation).

        Pearlmutter's trick computes H·v in O(n) time (same as a backward pass)
        by noting that H·v = ∇_θ (v · ∇_θ L).

        Steps:
            1. Compute gradient g = ∇_θ L.
            2. Compute the inner product s = v · g.
            3. Back-propagate through s to get ∇_θ s = H · v.

        Args:
            vector: A flat direction vector with the same number of elements
                    as the model's trainable parameters.
            dataloader: DataLoader providing (input, target) batches.
            num_samples: Number of mini-batches to average over.

        Returns:
            Flat tensor H · v with the same shape as *vector*.
        """
        self.model.train()
        hv = torch.zeros_like(vector, device=self.device)
        shapes: List[Tuple[str, torch.Size]] = []
        offset = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            shapes.append((name, param.shape))
            offset += param.numel()

        num_done = 0
        data_iter = iter(dataloader)
        while num_done < num_samples:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward(create_graph=True)

            # Compute v · g
            dot = torch.tensor(0.0, device=self.device, requires_grad=True)
            offset = 0
            for param in self.model.parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                numel = param.numel()
                v_slice = vector[offset : offset + numel].reshape(param.shape)
                dot = dot + torch.sum(v_slice * param.grad)
                offset += numel

            # ∇_θ (v · g) = H · v
            self.model.zero_grad()
            dot.backward()

            offset = 0
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                numel = param.numel()
                if param.grad is not None:
                    hv[offset : offset + numel] += param.grad.reshape(-1).to(self.device)
                offset += numel

            num_done += 1

        hv /= num_done
        return hv

    # ------------------------------------------------------------------
    # Hessian spectrum
    # ------------------------------------------------------------------

    def compute_hessian_spectrum(
        self,
        dataloader: DataLoader,
        num_samples: int = 100,
        max_power_iter: int = 50,
    ) -> Dict[str, Any]:
        """Compute top-K and bottom-K eigenvalues of the Hessian.

        Uses **power iteration** for the top eigenvalues and **inverse power
        iteration** for the bottom eigenvalues.  Also includes Hutchinson's
        trace estimator for the diagonal approximation.

        Power iteration recurrence:
            v_{k+1} = H · v_k / ||H · v_k||

        Inverse power iteration:
            v_{k+1} = H^{-1} · v_k / ||H^{-1} · v_k||
        (implemented implicitly via gradient descent on the Rayleigh quotient.)

        Args:
            dataloader: DataLoader for computing Hessian-vector products.
            num_samples: Number of batches for Hessian-vector product averaging.
            max_power_iter: Maximum power iteration steps.

        Returns:
            Dictionary with keys:
            - ``top_eigenvalues``   – Tensor of top-5 eigenvalues (descending).
            - ``bottom_eigenvalues`` – Tensor of bottom-5 eigenvalues (ascending).
            - ``condition_number``   – max(|λ|) / min(|λ|) (excluding near-zero).
            - ``spectrum``          – Full list of eigenvalues found.
        """
        num_eigs = 5

        # ---- Top eigenvalues via power iteration with deflation ----
        top_eigs: List[float] = []
        residual_directions: List[torch.Tensor] = []

        # Get flat parameter count
        flat_params, _ = _flatten_parameters(self.model, detach=True)
        n_params = flat_params.numel()

        for k in range(num_eigs):
            torch.manual_seed(42 + k)
            v = torch.randn(n_params, device=self.device)
            v = v / v.norm()

            for _ in range(max_power_iter):
                # Orthogonalise against previous eigenvectors
                for prev_v in residual_directions:
                    v = v - torch.dot(v, prev_v) * prev_v
                v_norm = v.norm()
                if v_norm < 1e-12:
                    v = torch.randn(n_params, device=self.device)
                    for prev_v in residual_directions:
                        v = v - torch.dot(v, prev_v) * prev_v
                    v_norm = v.norm()
                v = v / v_norm

                Hv = self.compute_hessian_vector_product(v, dataloader, num_samples=num_samples)
                eigenvalue = torch.dot(v, Hv).item()
                v = Hv / (Hv.norm() + 1e-30)

            Hv = self.compute_hessian_vector_product(v, dataloader, num_samples=num_samples)
            eigenvalue = torch.dot(v, Hv).item()
            top_eigs.append(eigenvalue)
            residual_directions.append(v.clone())

        # ---- Bottom eigenvalues via inverse power iteration ----
        bottom_eigs: List[float] = []
        inv_residuals: List[torch.Tensor] = []

        for k in range(num_eigs):
            torch.manual_seed(1000 + k)
            v = torch.randn(n_params, device=self.device)
            v = v / v.norm()

            # Inverse iteration: minimize Rayleigh quotient via gradient descent
            lr_inv = 0.1
            for it in range(max_power_iter):
                for prev_v in inv_residuals:
                    v = v - torch.dot(v, prev_v) * prev_v
                v_norm = v.norm()
                if v_norm < 1e-12:
                    v = torch.randn(n_params, device=self.device)
                    for prev_v in inv_residuals:
                        v = v - torch.dot(v, prev_v) * prev_v
                    v_norm = v.norm()
                v = v / v_norm

                Hv = self.compute_hessian_vector_product(v, dataloader, num_samples=num_samples)
                eigenvalue = torch.dot(v, Hv).item()

                # Shift to target smallest eigenvalue: (H - μI)^{-1}v ≈ step in -Hv
                # Simple approach: move away from Hv direction
                v = v - lr_inv * Hv
                v = v / (v.norm() + 1e-30)

            Hv = self.compute_hessian_vector_product(v, dataloader, num_samples=num_samples)
            eigenvalue = torch.dot(v, Hv).item()
            bottom_eigs.append(eigenvalue)
            inv_residuals.append(v.clone())

        all_eigs = sorted(bottom_eigs + top_eigs, reverse=True)

        # Condition number: max(|λ|) / min_nonzero(|λ|)
        abs_eigs = [abs(e) for e in all_eigs if abs(e) > 1e-10]
        cond_num = max(abs_eigs) / min(abs_eigs) if abs_eigs else float("inf")

        return {
            "top_eigenvalues": torch.tensor(top_eigs, device=self.device),
            "bottom_eigenvalues": torch.tensor(sorted(bottom_eigs), device=self.device),
            "condition_number": cond_num,
            "spectrum": all_eigs,
        }

    # ------------------------------------------------------------------
    # Filter normalization
    # ------------------------------------------------------------------

    def random_direction_filter_normalization(
        self,
        direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Normalize a random direction using filter normalization.

        Filter normalization (Li et al., 2018) rescales each filter (weight
        tensor) in the direction to have the same norm as the corresponding
        filter in the original model.  This produces interpretable visualisations
        where all layers contribute equally.

        Per-layer scaling:
            d_layer ← d_layer × (||θ_layer|| / ||d_layer||)

        Args:
            direction: Flat direction vector.  If ``None``, a random direction
                       is generated.

        Returns:
            Normalized flat direction vector.
        """
        if direction is None:
            flat_params, _ = _flatten_parameters(self.model, detach=True)
            direction = torch.randn_like(flat_params)

        normalized_parts: List[torch.Tensor] = []
        offset = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            d_slice = direction[offset : offset + numel].reshape(param.shape)
            param_norm = param.data.norm()
            dir_norm = d_slice.norm()
            if dir_norm > 1e-30:
                d_slice = d_slice * (param_norm / dir_norm)
            normalized_parts.append(d_slice.reshape(-1))
            offset += numel

        return torch.cat(normalized_parts, dim=0)

    # ------------------------------------------------------------------
    # 1-D loss surface
    # ------------------------------------------------------------------

    def plot_loss_surface_1d(
        self,
        dataloader: DataLoader,
        direction1: torch.Tensor,
        num_points: int = 50,
        epsilon_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> Dict[str, Any]:
        """Compute loss along a 1-D direction in parameter space.

        For each ε in *epsilon_range*, evaluates L(θ + ε·d₁).

        Args:
            dataloader: DataLoader providing evaluation data.
            direction1: Flat direction vector (same size as model parameters).
            num_points: Number of points to evaluate.
            epsilon_range: (ε_min, ε_max) range of the interpolation.

        Returns:
            ``{'coordinates': Tensor, 'losses': Tensor}`` suitable for plotting.
        """
        # Save original parameters
        original_state = {n: p.clone() for n, p in self.model.named_parameters()}

        epsilons = torch.linspace(epsilon_range[0], epsilon_range[1], num_points, device=self.device)
        losses = torch.zeros(num_points, device=self.device)
        shapes: List[Tuple[str, torch.Size]] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shapes.append((name, param.shape))

        with torch.no_grad():
            for i, eps in enumerate(epsilons):
                # Restore original params
                self._restore_params(original_state)
                _apply_direction(self.model, direction1, shapes, eps.item())

                # Evaluate loss on one batch
                data_iter = iter(dataloader)
                batch = next(data_iter)
                inputs, targets = self._parse_batch(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.model.eval()
                outputs = self.model(inputs)
                losses[i] = self.loss_fn(outputs, targets).item()

        # Restore original parameters
        self._restore_params(original_state)

        return {"coordinates": epsilons.cpu(), "losses": losses.cpu()}

    # ------------------------------------------------------------------
    # 2-D loss surface
    # ------------------------------------------------------------------

    def plot_loss_surface_2d(
        self,
        dataloader: DataLoader,
        direction1: torch.Tensor,
        direction2: torch.Tensor,
        resolution: int = 50,
        epsilon_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> Dict[str, Any]:
        """Compute loss along a 2-D plane in parameter space.

        Evaluates L(θ + α·d₁ + β·d₂) on a grid of (α, β) values.

        Args:
            dataloader: DataLoader providing evaluation data.
            direction1: First flat direction vector.
            direction2: Second flat direction vector.
            resolution: Grid resolution (resolution × resolution points).
            epsilon_range: (ε_min, ε_max) range for both axes.

        Returns:
            ``{'x': Tensor, 'y': Tensor, 'losses': Tensor(resolution, resolution)}``
            for contour plotting.
        """
        original_state = {n: p.clone() for n, p in self.model.named_parameters()}

        epsilons = torch.linspace(epsilon_range[0], epsilon_range[1], resolution, device=self.device)
        losses = torch.zeros(resolution, resolution, device=self.device)
        shapes: List[Tuple[str, torch.Size]] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shapes.append((name, param.shape))

        with torch.no_grad():
            data_iter = iter(dataloader)
            batch = next(data_iter)
            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.eval()
            for i, alpha in enumerate(epsilons):
                for j, beta in enumerate(epsilons):
                    self._restore_params(original_state)
                    _apply_direction(self.model, direction1, shapes, alpha.item())
                    _apply_direction(self.model, direction2, shapes, beta.item())

                    outputs = self.model(inputs)
                    losses[i, j] = self.loss_fn(outputs, targets).item()

        self._restore_params(original_state)

        return {
            "x": epsilons.cpu(),
            "y": epsilons.cpu(),
            "losses": losses.cpu(),
        }

    # ------------------------------------------------------------------
    # Sharpness (SAM-style)
    # ------------------------------------------------------------------

    def compute_sharpness(
        self,
        dataloader: DataLoader,
        epsilon: float = 1e-3,
    ) -> Dict[str, Any]:
        """Compute sharpness of the loss landscape around current parameters.

        Sharpness = max_{d: ||d||=1} L(θ + ε·d) − L(θ).

        In practice we approximate the max over a finite set of random directions
        (or use the top eigenvector of the Hessian).  Used in Sharpness-Aware
        Minimization (SAM, Foret et al., 2021).

        Args:
            dataloader: DataLoader for evaluation.
            epsilon: Perturbation radius ε.

        Returns:
            ``{'sharpness': float, 'worst_direction_norm': float,
               'original_loss': float, 'perturbed_losses': Tensor}``
        """
        original_state = {n: p.clone() for n, p in self.model.named_parameters()}

        # Compute original loss
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(dataloader)
            batch = next(data_iter)
            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            original_loss = self.loss_fn(outputs, targets).item()

        # Try several random directions
        num_directions = 10
        flat_params, _ = _flatten_parameters(self.model, detach=True)
        perturbed_losses = torch.zeros(num_directions, device=self.device)
        shapes: List[Tuple[str, torch.Size]] = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shapes.append((name, param.shape))

        with torch.no_grad():
            for k in range(num_directions):
                torch.manual_seed(2000 + k)
                direction = torch.randn_like(flat_params)
                direction = direction / direction.norm()

                self._restore_params(original_state)
                _apply_direction(self.model, direction, shapes, epsilon)

                outputs = self.model(inputs)
                perturbed_losses[k] = self.loss_fn(outputs, targets).item()

        self._restore_params(original_state)

        sharpness = (perturbed_losses.max() - original_loss).item()

        return {
            "sharpness": max(sharpness, 0.0),
            "worst_direction_norm": 1.0,
            "original_loss": original_loss,
            "perturbed_losses": perturbed_losses.cpu(),
        }

    # ------------------------------------------------------------------
    # Eigenvalue proportion (sharpness indicator)
    # ------------------------------------------------------------------

    def eigenvalue_proportion(
        self,
        dataloader: DataLoader,
        threshold_ratio: float = 0.99,
    ) -> Dict[str, Any]:
        """Fraction of variance explained by the top eigenvalues.

        High ratio → most curvature concentrated in a few directions
        → **sharp** minimum (potentially poor generalization).

        Low ratio → curvature spread evenly → **flat** minimum (better generalization).

        Proportion:
            R(K) = Σ_{i=1}^{K} λ_i / Σ_{i} λ_i

        We use the absolute values of eigenvalues since the Hessian may have
        negative entries near saddle points.

        Args:
            dataloader: DataLoader for Hessian computation.
            threshold_ratio: Target cumulative ratio (default 0.99).

        Returns:
            ``{'proportion_at_5': float, 'proportion_at_10': float,
               'num_eigs_for_threshold': int, 'is_sharp': bool}``
        """
        spectrum_result = self.compute_hessian_spectrum(dataloader, num_samples=50, max_power_iter=30)
        all_eigs = spectrum_result["spectrum"]
        abs_eigs = sorted([abs(e) for e in all_eigs], reverse=True)
        total = sum(abs_eigs) if abs_eigs else 1.0

        cumulative = 0.0
        num_for_threshold = 0
        proportions: Dict[int, float] = {}
        for i, e in enumerate(abs_eigs):
            cumulative += e
            if i + 1 in {5, 10, 20}:
                proportions[i + 1] = cumulative / total
            if cumulative / total >= threshold_ratio and num_for_threshold == 0:
                num_for_threshold = i + 1

        # Heuristic: sharp if 90% of variance in top 20% of eigenvalues
        n_top = max(1, len(abs_eigs) // 5)
        top_sum = sum(abs_eigs[:n_top])
        is_sharp = (top_sum / total) > 0.9

        return {
            "proportion_at_5": proportions.get(5, cumulative / total),
            "proportion_at_10": proportions.get(10, cumulative / total),
            "num_eigs_for_threshold": num_for_threshold,
            "is_sharp": is_sharp,
        }

    # ------------------------------------------------------------------
    # Loss curvature (Hutchinson's trace estimator)
    # ------------------------------------------------------------------

    def compute_loss_curvature(
        self,
        dataloader: DataLoader,
        num_samples: int = 20,
    ) -> Dict[str, Any]:
        """Average trace of the Hessian using Hutchinson's stochastic estimator.

        trace(H) = Σ_i λ_i  (sum of all eigenvalues = sum of diagonal).

        Hutchinson's estimator:
            trace(H) ≈ (1/m) Σ_{k=1}^{m} z_k^T H z_k,   z_k ~ N(0, I)

        Each z_k^T H z_k = z_k · (H z_k) can be computed via one Hessian-vector
        product, making this O(m × n) where n is the number of parameters.

        Args:
            dataloader: DataLoader for computing Hessian-vector products.
            num_samples: Number of random probe vectors (m).

        Returns:
            ``{'trace_estimate': float, 'mean_curvature': float, 'std_error': float}``
        """
        flat_params, _ = _flatten_parameters(self.model, detach=True)
        n_params = flat_params.numel()
        traces = torch.zeros(num_samples, device=self.device)

        for k in range(num_samples):
            torch.manual_seed(3000 + k)
            z = torch.randn(n_params, device=self.device)

            Hz = self.compute_hessian_vector_product(z, dataloader, num_samples=5)
            traces[k] = torch.dot(z, Hz).item()

        mean_trace = traces.mean().item()
        std_error = traces.std().item() / math.sqrt(num_samples) if num_samples > 1 else 0.0

        # Mean curvature per parameter
        mean_curvature = mean_trace / n_params if n_params > 0 else 0.0

        return {
            "trace_estimate": mean_trace,
            "mean_curvature": mean_curvature,
            "std_error": std_error,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch(batch: Any) -> Tuple[Any, Any]:
        """Parse a batch into (inputs, targets)."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        if isinstance(batch, dict):
            return batch.get("input_ids", batch.get("inputs")), batch.get("labels", batch.get("targets"))
        raise ValueError(f"Cannot parse batch of type {type(batch)}")

    def _restore_params(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore model parameters from saved state."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name])


# ---------------------------------------------------------------------------
# 2. Saddle Point Detection and Escape
# ---------------------------------------------------------------------------

class SaddlePointDetector:
    """Detect and analyze saddle points in the loss landscape.

    A saddle point is characterized by the Hessian having both positive and
    negative eigenvalues — curvature is concave in some directions and convex
    in others.  High-dimensional loss landscapes of neural networks are
    conjectured to be dominated by saddle points rather than local minima
    (Dauphin et al., 2014).

    This class provides:
    - Saddle point detection via Hessian eigenvalue analysis
    - Finding directions of negative curvature for escape
    - Perturbation strategies to escape saddle points

    Args:
        model: The neural network model.
        loss_fn: Loss function callable.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def is_saddle_point(
        self,
        dataloader: DataLoader,
        hessian_threshold: float = 1e-5,
    ) -> Dict[str, Any]:
        """Determine whether the current point is a saddle point.

        A point θ is a saddle point iff the Hessian H has at least one
        negative eigenvalue and at least one positive eigenvalue.

        Formally:
            saddle ↔  min(λ_i) < 0 AND max(λ_i) > 0

        Args:
            dataloader: DataLoader for computing Hessian-vector products.
            hessian_threshold: Eigenvalues with |λ| < threshold are treated
                               as zero (near-flat directions).

        Returns:
            Dictionary with:
            - ``is_saddle``               – bool
            - ``num_negative_eigenvalues`` – int
            - ``min_eigenvalue``          – float
            - ``eigenvalue_spectrum``     – list of floats
            - ``hessian_trace``           – float (≈ Σ λ_i)
            - ``hessian_logdet``          – float (≈ Σ log|λ_i|)
        """
        analyzer = LossLandscapeAnalyzer(self.model, self.loss_fn, self.device)
        spectrum_result = analyzer.compute_hessian_spectrum(
            dataloader, num_samples=50, max_power_iter=30,
        )
        all_eigs = spectrum_result["spectrum"]

        num_negative = sum(1 for e in all_eigs if e < -hessian_threshold)
        num_positive = sum(1 for e in all_eigs if e > hessian_threshold)
        min_eig = min(all_eigs) if all_eigs else 0.0
        max_eig = max(all_eigs) if all_eigs else 0.0
        is_saddle = (num_negative > 0) and (num_positive > 0)

        hessian_trace = sum(all_eigs) if all_eigs else 0.0

        # Log-determinant: log|det(H)| = Σ log|λ_i|
        logdet = 0.0
        for e in all_eigs:
            if abs(e) > 1e-30:
                logdet += math.log(abs(e))
            else:
                logdet += -69.0  # ≈ log(1e-30)

        return {
            "is_saddle": is_saddle,
            "num_negative_eigenvalues": num_negative,
            "min_eigenvalue": min_eig,
            "eigenvalue_spectrum": all_eigs,
            "hessian_trace": hessian_trace,
            "hessian_logdet": logdet,
        }

    def compute_negative_curvature_direction(
        self,
        dataloader: DataLoader,
        num_power_iter: int = 100,
    ) -> torch.Tensor:
        """Find the direction of most negative curvature.

        Uses inverse power iteration targeting the smallest eigenvalue.
        The direction d of most negative curvature satisfies:

            d = argmin_{||d||=1}  d^T H d

        Implementation:
            Repeatedly compute Hv and move opposite: d ← d − η·H·d
            This drives d toward the eigenspace of the smallest eigenvalue.

        Args:
            dataloader: DataLoader for Hessian-vector products.
            num_power_iter: Number of iterations.

        Returns:
            Flat direction vector (unit norm) pointing in the direction
            of most negative curvature.
        """
        analyzer = LossLandscapeAnalyzer(self.model, self.loss_fn, self.device)
        flat_params, _ = _flatten_parameters(self.model, detach=True)
        n_params = flat_params.numel()

        torch.manual_seed(42)
        v = torch.randn(n_params, device=self.device)
        v = v / v.norm()

        lr = 0.05
        for _ in range(num_power_iter):
            Hv = analyzer.compute_hessian_vector_product(v, dataloader, num_samples=3)
            # Move in direction that *decreases* Rayleigh quotient
            v = v - lr * Hv
            v_norm = v.norm()
            if v_norm < 1e-30:
                break
            v = v / v_norm

        return v

    def perturb_to_escape(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        learning_rate: float = 1e-3,
    ) -> Dict[str, Any]:
        """Apply perturbation to escape a saddle point.

        Computes the direction of most negative curvature and perturbs the
        model parameters along that direction:

            θ ← θ + α · d_neg

        where d_neg is the negative-curvature direction and α is the
        *learning_rate*.

        Args:
            model: Model to perturb (modified in-place).
            dataloader: DataLoader for evaluation.
            learning_rate: Step size for the perturbation.

        Returns:
            ``{'perturbation_norm': float, 'new_loss': float, 'improvement': float}``
        """
        # Compute current loss
        self.model.eval()
        with torch.no_grad():
            data_iter = iter(dataloader)
            batch = next(data_iter)
            inputs, targets = self._parse_batch(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            old_loss = self.loss_fn(outputs, targets).item()

        # Find negative curvature direction
        neg_dir = self.compute_negative_curvature_direction(dataloader, num_power_iter=50)

        # Apply perturbation
        shapes: List[Tuple[str, torch.Size]] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                shapes.append((name, param.shape))

        with torch.no_grad():
            _apply_direction(model, neg_dir, shapes, learning_rate)

        # Compute new loss
        with torch.no_grad():
            outputs = self.model(inputs)
            new_loss = self.loss_fn(outputs, targets).item()

        improvement = old_loss - new_loss

        return {
            "perturbation_norm": abs(learning_rate),
            "new_loss": new_loss,
            "improvement": improvement,
        }

    def hessian_spectrum_analysis(
        self,
        dataloader: DataLoader,
        num_eigenvalues: int = 20,
    ) -> Dict[str, Any]:
        """Full analysis of the Hessian eigenvalue spectrum.

        Categorises eigenvalues into positive, negative, and near-zero,
        and returns summary statistics.

        Args:
            dataloader: DataLoader for Hessian computation.
            num_eigenvalues: Number of eigenvalues to compute per end.

        Returns:
            Dictionary with breakdown and statistics.
        """
        analyzer = LossLandscapeAnalyzer(self.model, self.loss_fn, self.device)
        spectrum_result = analyzer.compute_hessian_spectrum(
            dataloader, num_samples=50, max_power_iter=30,
        )
        all_eigs = spectrum_result["spectrum"]

        threshold = 1e-5
        positive_eigs = sorted([e for e in all_eigs if e > threshold], reverse=True)
        negative_eigs = sorted([e for e in all_eigs if e < -threshold])
        zero_eigs = [e for e in all_eigs if abs(e) <= threshold]

        sorted_eigs = sorted(all_eigs, reverse=True)
        top_k = sorted_eigs[:num_eigenvalues]
        bottom_k = sorted_eigs[-num_eigenvalues:] if len(sorted_eigs) >= num_eigenvalues else sorted_eigs

        return {
            "num_positive": len(positive_eigs),
            "num_negative": len(negative_eigs),
            "num_zero": len(zero_eigs),
            "total_computed": len(all_eigs),
            "top_k_values": top_k,
            "bottom_k_values": bottom_k,
            "max_eigenvalue": max(all_eigs) if all_eigs else 0.0,
            "min_eigenvalue": min(all_eigs) if all_eigs else 0.0,
            "mean_eigenvalue": sum(all_eigs) / len(all_eigs) if all_eigs else 0.0,
            "condition_number": spectrum_result["condition_number"],
            "is_saddle": len(negative_eigs) > 0 and len(positive_eigs) > 0,
            "is_minimum": len(negative_eigs) == 0 and len(positive_eigs) > 0,
            "is_maximum": len(positive_eigs) == 0 and len(negative_eigs) > 0,
        }

    def loss_hessian_approx(
        self,
        loss: torch.Tensor,
        param: torch.nn.Parameter,
    ) -> torch.Tensor:
        """Approximate the Hessian of *loss* w.r.t. *param*.

        For small parameters (≤ 4096 elements) the full Hessian is computed
        via the double backward pass.  For larger parameters, a **block-diagonal
        approximation** is returned using finite differences of the gradient:

            H[i,i] ≈ (∂L/∂θ_i(θ + ε·e_i) − ∂L/∂θ_i(θ − ε·e_i)) / (2ε)

        Note: For very large parameters, prefer using
        :meth:`compute_hessian_vector_product` which is O(n) instead of O(n²).

        Args:
            loss: Scalar loss tensor (requires_grad=True upstream).
            param: Model parameter to compute Hessian for.

        Returns:
            Hessian matrix of shape ``[param.numel(), param.numel()]``.
            For large params, returns a diagonal approximation.
        """
        numel = param.numel()

        if numel > 4096:
            # Block-diagonal / diagonal approximation via finite differences
            eps = 1e-5
            grad_orig = torch.autograd.grad(loss, param, create_graph=False, retain_graph=True)[0]
            diag_hessian = torch.zeros(numel, device=param.device)

            for i in range(min(numel, 1024)):
                param_flat = param.data.flatten()
                param_flat_save = param_flat.clone()

                # θ + ε·e_i
                param_flat[i] += eps
                param.data.copy_(param_flat.reshape(param.shape))
                loss_p = loss.clone()  # recompute if needed
                grad_p = torch.autograd.grad(loss, param, create_graph=False, retain_graph=True)[0].flatten()

                # θ − ε·e_i
                param_flat[i] = param_flat_save[i] - eps
                param.data.copy_(param_flat.reshape(param.shape))
                grad_m = torch.autograd.grad(loss, param, create_graph=False, retain_graph=True)[0].flatten()

                diag_hessian[i] = (grad_p[i] - grad_m[i]) / (2 * eps)

                # Restore
                param.data.copy_(param_flat_save.reshape(param.shape))

            return torch.diag(diag_hessian)

        # Full Hessian via double backward (small params only)
        grads = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
        hessian_rows = []
        for i in range(numel):
            row_grad = torch.autograd.grad(grads[i], param, retain_graph=(i < numel - 1))[0].flatten()
            hessian_rows.append(row_grad)

        return torch.stack(hessian_rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch(batch: Any) -> Tuple[Any, Any]:
        """Parse a batch into (inputs, targets)."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        if isinstance(batch, dict):
            return batch.get("input_ids", batch.get("inputs")), batch.get("labels", batch.get("targets"))
        raise ValueError(f"Cannot parse batch of type {type(batch)}")


# ---------------------------------------------------------------------------
# 3. Gradient Noise Scale Tracking
# ---------------------------------------------------------------------------

class GradientNoiseTracker:
    """Tracks the gradient noise scale throughout training.

    The gradient noise scale quantifies the ratio of gradient variance to
    squared mean gradient norm:

        B_noise = E[||g − ḡ||²] / ||ḡ||²

    where ḡ = E[g] is the mean gradient across mini-batch samples.

    Interpretation:
    - **Low B_noise** → gradients are consistent across samples → can use large batch.
    - **High B_noise** → gradients are noisy → prefer small batch for sample efficiency.

    Reference: "Don't Decay the Learning Rate, Increase the Batch Size"
    (Smith et al., 2018).

    Args:
        model: The neural network model.
        window_size: Number of past steps to keep in the rolling buffer.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 100,
    ) -> None:
        self.model = model
        self.window_size = window_size
        self.grad_buffer: List[torch.Tensor] = []
        self.noise_scale_history: List[float] = []
        self.signal_to_noise_ratio_history: List[float] = []

    def accumulate_gradient(
        self,
        dataloader: DataLoader,
        num_samples: int = 10,
    ) -> None:
        """Compute gradients on multiple random mini-batches and store them.

        For each of *num_samples* mini-batches:
            1. Compute gradient g_k on batch k
            2. Flatten and store g_k in the rolling buffer

        Args:
            dataloader: DataLoader for sampling mini-batches.
            num_samples: Number of gradient snapshots to collect.
        """
        self.model.train()
        data_iter = iter(dataloader)

        for _ in range(num_samples):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            inputs, targets = self._parse_batch(batch)
            device = _get_device(self.model)
            inputs = inputs.to(device)
            targets = targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = outputs.mean() if outputs.dim() > 0 else outputs
            if hasattr(loss, "requires_grad"):
                loss.backward(retain_graph=False)

            # Flatten all gradients
            grad_parts: List[torch.Tensor] = []
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    grad_parts.append(param.grad.detach().flatten())
            if grad_parts:
                flat_grad = torch.cat(grad_parts, dim=0)
                self.grad_buffer.append(flat_grad)

        # Trim to window size
        if len(self.grad_buffer) > self.window_size:
            self.grad_buffer = self.grad_buffer[-self.window_size :]

    def compute_noise_scale(self) -> float:
        """Compute current gradient noise scale B_noise.

            B_noise = E[||g − ḡ||²] / ||ḡ||²

        Uses all gradients in the buffer.

        Returns:
            Noise scale (float).  Returns ``float('inf')`` if mean gradient
            norm is near zero.
        """
        if len(self.grad_buffer) < 2:
            return 0.0

        grads = torch.stack(self.grad_buffer, dim=0)  # (N, D)
        mean_grad = grads.mean(dim=0)  # (D,)
        mean_grad_norm_sq = mean_grad.norm().item() ** 2

        if mean_grad_norm_sq < 1e-30:
            return float("inf")

        deviations = grads - mean_grad  # (N, D)
        variance = (deviations.norm(dim=1) ** 2).mean().item()

        noise_scale = variance / mean_grad_norm_sq
        self.noise_scale_history.append(noise_scale)

        return noise_scale

    def compute_gradient_covariance(self) -> torch.Tensor:
        """Compute the covariance matrix of accumulated gradients.

            Σ = (1/N) Σ_{k=1}^{N} (g_k − ḡ)(g_k − ḡ)^T

        Warning: For large models this matrix is O(D²) in memory.
        For models with > 4096 parameters, returns the **diagonal** of the
        covariance (variances only).

        Returns:
            Covariance matrix (or diagonal) as a tensor.
        """
        if len(self.grad_buffer) < 2:
            return torch.tensor(0.0)

        grads = torch.stack(self.grad_buffer, dim=0)  # (N, D)
        mean_grad = grads.mean(dim=0)
        centered = grads - mean_grad  # (N, D)

        if centered.shape[1] > 4096:
            # Return diagonal only
            var = (centered ** 2).mean(dim=0)
            return var

        # Full covariance (D×D)
        cov = (centered.T @ centered) / centered.shape[0]
        return cov

    def compute_signal_to_noise_ratio(self) -> float:
        """Compute signal-to-noise ratio (SNR) of gradients.

            SNR = ||ḡ||² / E[||g − ḡ||²]  =  1 / B_noise

        High SNR → gradients are signal-dominated → training is stable.

        Returns:
            SNR value (float).  Returns ``float('inf')`` if variance is near zero.
        """
        if len(self.grad_buffer) < 2:
            return float("inf")

        grads = torch.stack(self.grad_buffer, dim=0)
        mean_grad = grads.mean(dim=0)
        mean_grad_norm_sq = mean_grad.norm().item() ** 2

        deviations = grads - mean_grad
        variance = (deviations.norm(dim=1) ** 2).mean().item()

        if variance < 1e-30:
            snr = float("inf")
        else:
            snr = mean_grad_norm_sq / variance

        self.signal_to_noise_ratio_history.append(snr)
        return snr

    def get_noise_scale_estimate(self) -> Dict[str, Any]:
        """Return current noise scale estimate with confidence interval.

        Uses the bootstrap method over the rolling buffer to estimate
        uncertainty.

        Returns:
            ``{'noise_scale': float, 'confidence_interval': (float, float),
               'num_samples': int}``
        """
        if len(self.grad_buffer) < 5:
            return {
                "noise_scale": self.compute_noise_scale(),
                "confidence_interval": (0.0, float("inf")),
                "num_samples": len(self.grad_buffer),
            }

        # Bootstrap
        n = len(self.grad_buffer)
        grads = torch.stack(self.grad_buffer, dim=0)
        bootstrap_estimates: List[float] = []

        for _ in range(50):
            indices = torch.randint(0, n, (n,))
            sample = grads[indices]
            mean_g = sample.mean(dim=0)
            mean_norm_sq = mean_g.norm().item() ** 2
            if mean_norm_sq < 1e-30:
                bootstrap_estimates.append(float("inf"))
                continue
            devs = sample - mean_g
            var = (devs.norm(dim=1) ** 2).mean().item()
            bootstrap_estimates.append(var / mean_norm_sq)

        point_estimate = self.compute_noise_scale()
        finite_estimates = [e for e in bootstrap_estimates if math.isfinite(e)]
        if finite_estimates:
            lo = float(torch.tensor(finite_estimates).quantile(0.05).item())
            hi = float(torch.tensor(finite_estimates).quantile(0.95).item())
        else:
            lo, hi = 0.0, float("inf")

        return {
            "noise_scale": point_estimate,
            "confidence_interval": (lo, hi),
            "num_samples": len(self.grad_buffer),
        }

    def plot_noise_scale_history(self) -> Dict[str, Any]:
        """Return data for plotting noise scale over training steps.

        Returns:
            ``{'steps': list[int], 'noise_scale': list[float],
               'snr': list[float]}``
        """
        steps = list(range(len(self.noise_scale_history)))
        return {
            "steps": steps,
            "noise_scale": list(self.noise_scale_history),
            "snr": list(self.signal_to_noise_ratio_history),
        }

    def detect_phase_transition(self) -> Dict[str, Any]:
        """Detect rapid changes in noise scale — indicates a learning phase transition.

        A phase transition is declared when the noise scale changes by more
        than a factor of 2 within a short window (20 steps).

        Returns:
            ``{'has_transition': bool, 'transition_step': int,
               'old_noise': float, 'new_noise': float}``
        """
        window = 20
        history = self.noise_scale_history
        if len(history) < window * 2:
            return {
                "has_transition": False,
                "transition_step": -1,
                "old_noise": 0.0,
                "new_noise": 0.0,
            }

        for i in range(window, len(history)):
            recent = history[i - window : i]
            older = history[i - window * 2 : i - window]
            recent_mean = sum(recent) / len(recent) if recent else 0.0
            older_mean = sum(older) / len(older) if older else 0.0

            if older_mean > 1e-10 and recent_mean > 1e-10:
                ratio = recent_mean / older_mean
                if ratio > 2.0 or ratio < 0.5:
                    return {
                        "has_transition": True,
                        "transition_step": i,
                        "old_noise": older_mean,
                        "new_noise": recent_mean,
                    }

        return {
            "has_transition": False,
            "transition_step": -1,
            "old_noise": 0.0,
            "new_noise": 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch(batch: Any) -> Tuple[Any, Any]:
        """Parse a batch into (inputs, targets)."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        if isinstance(batch, dict):
            return batch.get("input_ids", batch.get("inputs")), batch.get("labels", batch.get("targets"))
        raise ValueError(f"Cannot parse batch of type {type(batch)}")


# ---------------------------------------------------------------------------
# 4. Critical Batch Size Estimation
# ---------------------------------------------------------------------------

class CriticalBatchSizeEstimator:
    """Estimates the critical batch size for training.

    The **critical batch size** B_crit is the point at which increasing the
    batch size further gives diminishing returns.  It is determined by the
    gradient noise scale:

    - B < B_crit: linear scaling — doubling the batch ≈ halving the steps.
    - B > B_crit: diminishing returns — doubling the batch gives < 2× speedup.
    - B = B_crit: optimal trade-off between throughput and step efficiency.

    Reference: "Measuring the Effects of Data Parallelism on Neural Network
    Training" (Shallue et al., 2018).

    Args:
        model: The neural network model.
        loss_fn: Loss function callable.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.device = device

    def measure_gradient_variance(
        self,
        dataloader: DataLoader,
        batch_sizes: Optional[List[int]] = None,
        num_trials: int = 5,
    ) -> Dict[int, Dict[str, float]]:
        """Measure gradient variance at different batch sizes.

        For each batch size B:
            1. Create sub-batches of size B from the full batch
            2. Compute gradient g_k for each sub-batch
            3. Compute statistics:
               - mean_grad_norm = ||ḡ||
               - grad_variance = E[||g_k − ḡ||²]
               - snr = ||ḡ||² / E[||g_k − ḡ||²]

        Args:
            dataloader: DataLoader providing data.
            batch_sizes: List of batch sizes to test.  If ``None``, uses
                         a geometric progression [8, 16, 32, 64, 128, 256].
            num_trials: Number of mini-batches to sample per batch size.

        Returns:
            Dict mapping batch_size → ``{mean_grad_norm, grad_variance, snr}``.
        """
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64, 128, 256]

        results: Dict[int, Dict[str, float]] = {}
        full_batch = next(iter(dataloader))
        inputs_full, targets_full = self._parse_batch(full_batch)
        device = _get_device(self.model)
        inputs_full = inputs_full.to(device)
        targets_full = targets_full.to(device)
        N = inputs_full.shape[0] if hasattr(inputs_full, "shape") else len(inputs_full)

        for bs in batch_sizes:
            if bs > N:
                continue

            grad_snapshots: List[torch.Tensor] = []
            for _ in range(num_trials):
                indices = torch.randint(0, N, (bs,))
                inputs_sub = inputs_full[indices]
                targets_sub = targets_full[indices]

                self.model.zero_grad()
                self.model.train()
                outputs = self.model(inputs_sub)
                loss = self.loss_fn(outputs, targets_sub)
                loss.backward()

                flat_grad_parts: List[torch.Tensor] = []
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        flat_grad_parts.append(param.grad.detach().flatten())
                if flat_grad_parts:
                    grad_snapshots.append(torch.cat(flat_grad_parts, dim=0))

            if len(grad_snapshots) < 2:
                continue

            grads = torch.stack(grad_snapshots, dim=0)
            mean_grad = grads.mean(dim=0)
            mean_grad_norm = mean_grad.norm().item()
            mean_grad_norm_sq = mean_grad_norm ** 2

            devs = grads - mean_grad
            variance = (devs.norm(dim=1) ** 2).mean().item()

            snr = mean_grad_norm_sq / variance if variance > 1e-30 else float("inf")

            results[bs] = {
                "mean_grad_norm": mean_grad_norm,
                "grad_variance": variance,
                "snr": snr,
            }

        return results

    def estimate_critical_batch_size(
        self,
        batch_sizes: Optional[List[int]] = None,
        num_trials: int = 5,
    ) -> Dict[str, Any]:
        """Estimate the critical batch size B_crit.

        The noise scale model predicts:
            B_noise(B) ≈ B + B_crit

        We fit this linear model to the measured (batch_size, noise_scale)
        data and extract B_crit from the intercept.

        The knee-point is identified as the batch size where the noise scale
        curve bends — indicating diminishing returns.

        Args:
            batch_sizes: Batch sizes to evaluate.
            num_trials: Number of gradient samples per batch size.

        Returns:
            ``{'critical_batch_size': float, 'optimal_batch_size': int,
               'plot_data': dict}``
        """
        variance_data = self.measure_gradient_variance(batch_sizes, num_trials)

        if len(variance_data) < 2:
            return {
                "critical_batch_size": 32.0,
                "optimal_batch_size": 32,
                "plot_data": {"batch_sizes": [], "noise_scales": []},
            }

        sorted_bs = sorted(variance_data.keys())
        noise_scales = []
        for bs in sorted_bs:
            d = variance_data[bs]
            if d["mean_grad_norm"] > 1e-30:
                ns = d["grad_variance"] / (d["mean_grad_norm"] ** 2)
                noise_scales.append(ns)
            else:
                noise_scales.append(0.0)

        # Linear fit: noise_scale = slope * batch_size + intercept
        # B_crit ≈ intercept / slope (where noise_scale ≈ batch_size)
        if len(sorted_bs) >= 2 and len(noise_scales) >= 2:
            x = torch.tensor(sorted_bs, dtype=torch.float64)
            y = torch.tensor(noise_scales, dtype=torch.float64)

            # Simple least squares: y = a*x + b
            x_mean = x.mean()
            y_mean = y.mean()
            slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
            intercept = y_mean - slope * x_mean

            # B_crit: point where noise_scale = batch_size
            # slope * B + intercept = B → B = intercept / (1 - slope)
            if abs(1.0 - slope.item()) > 1e-10:
                b_crit = intercept / (1.0 - slope)
                b_crit = max(1.0, b_crit.item())
            else:
                b_crit = float(sorted_bs[-1])

            # Optimal batch size ≈ 0.5 * B_crit (conservative)
            optimal_bs = max(1, int(b_crit * 0.5))
        else:
            b_crit = float(sorted_bs[-1]) if sorted_bs else 32.0
            optimal_bs = max(1, int(b_crit * 0.5))

        return {
            "critical_batch_size": b_crit,
            "optimal_batch_size": optimal_bs,
            "plot_data": {
                "batch_sizes": sorted_bs,
                "noise_scales": noise_scales,
            },
        }

    def optimal_batch_size(
        self,
        noise_scale: float,
        gradient_norm: float,
        target_reduction_factor: float = 0.9,
    ) -> Dict[str, Any]:
        """Compute the batch size achieving a target fraction of optimal loss reduction.

        Theory: the per-sample loss reduction per step scales as:

            ΔL_per_sample ∝ 1 / (1 + B_noise / B)

        The maximum (B → ∞) is proportional to 1.  We find B such that
        ΔL_per_sample(B) = target_reduction_factor.

            B = B_noise / (1/target_reduction_factor − 1)

        Args:
            noise_scale: Current gradient noise scale B_noise.
            gradient_norm: Current gradient norm (unused in formula but kept for API).
            target_reduction_factor: Target fraction of maximum efficiency (0-1).

        Returns:
            ``{'optimal_batch_size': int, 'efficiency': float, 'noise_scale': float}``
        """
        if noise_scale <= 0:
            return {
                "optimal_batch_size": 1,
                "efficiency": 0.0,
                "noise_scale": noise_scale,
            }

        target_efficiency = max(0.01, min(0.99, target_reduction_factor))
        b_opt = noise_scale / (1.0 / target_efficiency - 1.0)
        b_opt = max(1, int(round(b_opt)))

        return {
            "optimal_batch_size": b_opt,
            "efficiency": target_efficiency,
            "noise_scale": noise_scale,
        }

    def compute_dynamics(
        self,
        dataloader: DataLoader,
        num_batches: int = 20,
        batch_size_range: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Comprehensive analysis of loss reduction dynamics at different batch sizes.

        Simulates training at multiple batch sizes and tracks how the effective
        noise scale and expected convergence rate vary.

        Args:
            dataloader: DataLoader for evaluation.
            num_batches: Number of batches to simulate.
            batch_size_range: Batch sizes to evaluate.

        Returns:
            ``{'batch_sizes': list, 'noise_scales': list, 'expected_steps': list,
               'efficiency': list}``
        """
        if batch_size_range is None:
            batch_size_range = [8, 16, 32, 64, 128, 256]

        variance_data = self.measure_gradient_variance(dataloader, batch_size_range, num_trials=5)
        sorted_bs = sorted(variance_data.keys())

        noise_scales: List[float] = []
        expected_steps: List[float] = []
        efficiency: List[float] = []

        # Reference: smallest batch
        if sorted_bs:
            ref_bs = sorted_bs[0]
            ref_data = variance_data[ref_bs]
            ref_noise = ref_data["grad_variance"] / (ref_data["mean_grad_norm"] ** 2) if ref_data["mean_grad_norm"] > 1e-30 else 1.0
        else:
            ref_noise = 1.0

        for bs in sorted_bs:
            d = variance_data[bs]
            if d["mean_grad_norm"] > 1e-30:
                ns = d["grad_variance"] / (d["mean_grad_norm"] ** 2)
            else:
                ns = 0.0
            noise_scales.append(ns)

            # Expected steps to converge scales linearly with noise scale for B < B_crit
            exp_steps = ns * ref_bs if ref_noise > 1e-30 else 1.0
            expected_steps.append(exp_steps)

            # Efficiency: step reduction per sample
            eff = 1.0 / (1.0 + ns / max(bs, 1))
            efficiency.append(eff)

        return {
            "batch_sizes": sorted_bs,
            "noise_scales": noise_scales,
            "expected_steps": expected_steps,
            "efficiency": efficiency,
        }

    def compare_batch_strategies(
        self,
        small_batch_lr: float,
        large_batch_lr: float,
        scaling_factor: float = 2.0,
        num_steps: int = 100,
    ) -> Dict[str, Any]:
        """Compare small-batch vs large-batch training strategies.

        Strategy 1: Small batch with original LR for *num_steps* steps.
        Strategy 2: Large batch with scaled LR for ``num_steps / scaling_factor`` steps.

        The comparison is based on the expected loss reduction, which scales as:

            ΔL ∝ lr² × B × ||∇L||² / (B + B_noise)

        Args:
            small_batch_lr: Learning rate for the small-batch strategy.
            large_batch_lr: Learning rate for the large-batch strategy.
            scaling_factor: Ratio of large batch to small batch.
            num_steps: Total steps for the small-batch strategy.

        Returns:
            ``{'small_batch_loss': float, 'large_batch_loss': float,
               'efficiency_ratio': float, 'recommendation': str}``
        """
        # Estimate noise scale from a single batch
        try:
            variance_data = self.measure_gradient_variance(
                batch_sizes=[16, 32], num_trials=3,
            )
            bs_values = sorted(variance_data.keys())
            if bs_values:
                bs_small = bs_values[0]
                d = variance_data[bs_small]
                if d["mean_grad_norm"] > 1e-30:
                    noise_scale = d["grad_variance"] / (d["mean_grad_norm"] ** 2)
                else:
                    noise_scale = 32.0
            else:
                noise_scale = 32.0
        except Exception:
            noise_scale = 32.0

        # Expected loss reduction per step
        small_B = 32.0
        large_B = small_B * scaling_factor
        grad_norm_sq = 1.0  # normalized

        # ΔL ∝ lr² × B / (B + B_noise)
        small_reduction_per_step = (small_batch_lr ** 2) * small_B / (small_B + noise_scale)
        large_reduction_per_step = (large_batch_lr ** 2) * large_B / (large_B + noise_scale)

        small_batch_loss = 1.0 - small_reduction_per_step * num_steps
        large_batch_loss = 1.0 - large_reduction_per_step * (num_steps / scaling_factor)

        # Clamp to sensible range
        small_batch_loss = max(0.0, min(1.0, small_batch_loss))
        large_batch_loss = max(0.0, min(1.0, large_batch_loss))

        # Efficiency: large batch total compute / small batch total compute
        # Both use the same number of samples (large_batch uses scaling_factor fewer steps)
        small_total_reduction = small_reduction_per_step * num_steps
        large_total_reduction = large_reduction_per_step * (num_steps / scaling_factor)

        if small_total_reduction > 1e-30:
            efficiency_ratio = large_total_reduction / small_total_reduction
        else:
            efficiency_ratio = 1.0

        if efficiency_ratio > 1.0:
            recommendation = (
                f"Large batch is {efficiency_ratio:.2f}x more efficient. "
                f"Use batch={int(large_B)} with lr={large_batch_lr:.6f}."
            )
        elif efficiency_ratio > 0.8:
            recommendation = (
                "Strategies are roughly equivalent. "
                "Use large batch for throughput or small batch for better generalization."
            )
        else:
            recommendation = (
                f"Small batch is {1.0 / efficiency_ratio:.2f}x more efficient. "
                f"Use batch={int(small_B)} with lr={small_batch_lr:.6f}."
            )

        return {
            "small_batch_loss": small_batch_loss,
            "large_batch_loss": large_batch_loss,
            "efficiency_ratio": efficiency_ratio,
            "recommendation": recommendation,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_batch(batch: Any) -> Tuple[Any, Any]:
        """Parse a batch into (inputs, targets)."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch
        if isinstance(batch, dict):
            return batch.get("input_ids", batch.get("inputs")), batch.get("labels", batch.get("targets"))
        raise ValueError(f"Cannot parse batch of type {type(batch)}")


# ---------------------------------------------------------------------------
# 5. Training Dynamics Monitor
# ---------------------------------------------------------------------------

class TrainingDynamicsMonitor:
    """High-level monitor that combines all convergence analysis tools.

    Provides a unified interface for running comprehensive analyses of the
    training dynamics, detecting plateaus, estimating convergence, and
    recommending hyper-parameter adjustments.

    Args:
        model: The neural network model.
        loss_fn: Loss function callable.
        eval_dataloader: DataLoader for evaluation.
        config: Optional dictionary of configuration overrides:
            - ``window_size``: gradient buffer window size (default 100)
            - ``plateau_window``: window for plateau detection (default 100)
            - ``plateau_threshold``: loss change threshold (default 0.001)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable,
        eval_dataloader: DataLoader,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        device = config.get("device", "cuda")

        self.landscape = LossLandscapeAnalyzer(model, loss_fn, device=device)
        self.saddle = SaddlePointDetector(model, loss_fn, device=device)
        self.noise = GradientNoiseTracker(model, window_size=config.get("window_size", 100))
        self.batch_estimator = CriticalBatchSizeEstimator(model, loss_fn, device=device)
        self.model = model
        self.loss_fn = loss_fn
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.metrics_history: Dict[str, List[Any]] = {}

    def run_full_analysis(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """Run all convergence analyses and return a comprehensive report.

        Performs:
            1. Loss landscape sharpness analysis
            2. Saddle point detection
            3. Hessian curvature estimation
            4. Gradient noise scale tracking
            5. Critical batch size estimation

        Args:
            train_dataloader: DataLoader for training data.
            eval_dataloader: Optional separate eval DataLoader.
            step: Current training step (for logging).

        Returns:
            Comprehensive report dictionary with all analysis results.
        """
        eval_dl = eval_dataloader or self.eval_dataloader
        report: Dict[str, Any] = {"step": step}

        # 1. Noise scale
        try:
            self.noise.accumulate_gradient(train_dataloader, num_samples=5)
            noise_scale = self.noise.compute_noise_scale()
            snr = self.noise.compute_signal_to_noise_ratio()
            report["noise_scale"] = noise_scale
            report["signal_to_noise_ratio"] = snr
            report["phase_transition"] = self.noise.detect_phase_transition()
        except Exception as e:
            report["noise_scale"] = None
            report["noise_scale_error"] = str(e)

        # 2. Sharpness
        try:
            sharpness_result = self.landscape.compute_sharpness(eval_dl, epsilon=1e-3)
            report["sharpness"] = sharpness_result["sharpness"]
            report["original_loss"] = sharpness_result["original_loss"]
        except Exception as e:
            report["sharpness"] = None
            report["sharpness_error"] = str(e)

        # 3. Saddle point check
        try:
            saddle_result = self.saddle.is_saddle_point(eval_dl)
            report["is_saddle_point"] = saddle_result["is_saddle"]
            report["num_negative_eigenvalues"] = saddle_result["num_negative_eigenvalues"]
            report["hessian_trace"] = saddle_result["hessian_trace"]
        except Exception as e:
            report["is_saddle_point"] = None
            report["saddle_error"] = str(e)

        # 4. Curvature
        try:
            curvature = self.landscape.compute_loss_curvature(eval_dl, num_samples=5)
            report["hessian_trace_estimate"] = curvature["trace_estimate"]
            report["mean_curvature"] = curvature["mean_curvature"]
        except Exception as e:
            report["mean_curvature"] = None
            report["curvature_error"] = str(e)

        # 5. Batch size recommendation
        try:
            batch_result = self.batch_estimator.estimate_critical_batch_size(
                num_trials=3,
            )
            report["critical_batch_size"] = batch_result["critical_batch_size"]
            report["optimal_batch_size"] = batch_result["optimal_batch_size"]
        except Exception as e:
            report["critical_batch_size"] = None
            report["batch_error"] = str(e)

        # Store in history
        for key, value in report.items():
            if key == "step":
                continue
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        return report

    def detect_plateau(
        self,
        loss_history: List[float],
        window: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Detect if training has plateaued based on loss history.

        A plateau is declared when the loss change over the last *window*
        steps is below *threshold*:

            |L_{t} − L_{t−w}| / w < threshold

        Uses linear regression on the loss window for robustness.

        Args:
            loss_history: List of loss values indexed by step.
            window: Window size (from config or default 100).
            threshold: Threshold (from config or default 0.001).

        Returns:
            ``{'is_plateaued': bool, 'loss_slope': float, 'window_mean': float,
               'window_std': float, 'recommendation': str}``
        """
        window = window or self.config.get("plateau_window", 100)
        threshold = threshold or self.config.get("plateau_threshold", 0.001)

        if len(loss_history) < window:
            return {
                "is_plateaued": False,
                "loss_slope": 0.0,
                "window_mean": sum(loss_history) / len(loss_history) if loss_history else 0.0,
                "window_std": 0.0,
                "recommendation": "Not enough data to detect plateau.",
            }

        recent = loss_history[-window:]

        # Linear regression
        x = torch.arange(len(recent), dtype=torch.float64)
        y = torch.tensor(recent, dtype=torch.float64)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / ((x - x_mean) ** 2).sum()
        slope = slope.item()

        window_mean = y_mean.item()
        window_std = y.std().item()

        is_plateaued = abs(slope) < threshold

        if is_plateaued:
            if slope > 0:
                recommendation = (
                    "Loss is plateaued with slight increase. Consider: "
                    "reducing learning rate, increasing batch size, or "
                    "applying saddle point escape."
                )
            else:
                recommendation = (
                    "Loss is plateaued with very slow decrease. Consider: "
                    "reducing learning rate or applying cosine annealing."
                )
        else:
            recommendation = "Training is progressing normally."

        return {
            "is_plateaued": is_plateaued,
            "loss_slope": slope,
            "window_mean": window_mean,
            "window_std": window_std,
            "recommendation": recommendation,
        }

    def estimate_convergence_step(
        self,
        loss_history: List[float],
        target_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Estimate when training will converge to a target loss.

        Fits an exponential decay model L(t) = a · exp(−b · t) + c to the
        recent loss history and extrapolates.

        The convergence step is the step where L(t) = target_loss.

        Args:
            loss_history: List of loss values indexed by step.
            target_loss: Target loss value.  If ``None``, uses
                         min(loss_history) × 0.5.

        Returns:
            ``{'estimated_step': int, 'current_loss': float,
               'target_loss': float, 'rate': float, 'confidence': str}``
        """
        if len(loss_history) < 10:
            return {
                "estimated_step": -1,
                "current_loss": loss_history[-1] if loss_history else 0.0,
                "target_loss": target_loss,
                "rate": 0.0,
                "confidence": "insufficient_data",
            }

        current_loss = loss_history[-1]
        if target_loss is None:
            target_loss = min(loss_history) * 0.5

        if current_loss <= target_loss:
            return {
                "estimated_step": len(loss_history),
                "current_loss": current_loss,
                "target_loss": target_loss,
                "rate": 0.0,
                "confidence": "already_converged",
            }

        # Fit exponential decay to last N points
        N = min(len(loss_history), 100)
        recent = loss_history[-N:]
        y = torch.tensor(recent, dtype=torch.float64)
        t = torch.arange(len(recent), dtype=torch.float64)

        # Log-linear fit: log(L - c) ≈ log(a) - b*t
        # Estimate c as minimum (asymptote)
        c = y.min().item() * 0.99
        log_y = torch.log(y - c + 1e-30)

        t_mean = t.mean()
        log_y_mean = log_y.mean()
        slope = -((t - t_mean) * (log_y - log_y_mean)).sum() / ((t - t_mean) ** 2).sum()
        rate = max(slope.item(), 1e-10)

        # Extrapolate: target_loss = a * exp(-rate * t_est) + c
        # log((target - c) / (current - c)) = -rate * (t_est - current_t)
        current_t = len(loss_history) - 1
        if current_loss - c > 1e-30 and target_loss - c > 1e-30:
            log_ratio = math.log((current_loss - c) / (target_loss - c + 1e-30))
            steps_to_converge = log_ratio / rate
            estimated_step = int(current_t + steps_to_converge)

            confidence = "high" if steps_to_converge < len(loss_history) * 5 else "low"
        else:
            estimated_step = -1
            confidence = "unreliable"

        return {
            "estimated_step": estimated_step,
            "current_loss": current_loss,
            "target_loss": target_loss,
            "rate": rate,
            "confidence": confidence,
        }

    def compute_effective_learning_rate(
        self,
        gradient_norm: float,
        loss: float,
    ) -> Dict[str, Any]:
        """Compute the effective step size: lr × ||∇L||.

        The effective learning rate determines how much the parameters change
        per step.  Useful for diagnosing training stability:

            effective_lr = lr × ||∇L||

        Very large effective_lr → unstable training.
        Very small effective_lr → slow progress.

        Args:
            gradient_norm: Current gradient L2 norm.
            loss: Current loss value.

        Returns:
            ``{'effective_lr': float, 'relative_step': float, 'status': str}``
        """
        if loss < 1e-30:
            return {
                "effective_lr": 0.0,
                "relative_step": 0.0,
                "status": "converged",
            }

        relative_step = gradient_norm / (loss + 1e-30)

        if relative_step > 1.0:
            status = "unstable — gradient too large relative to loss"
        elif relative_step > 0.1:
            status = "active — healthy gradient magnitude"
        elif relative_step > 0.01:
            status = "moderate — consider slightly larger LR"
        else:
            status = "slow — consider increasing LR or check for vanishing gradients"

        return {
            "effective_lr": gradient_norm,
            "relative_step": relative_step,
            "status": status,
        }

    def should_adjust_batch_size(
        self,
        noise_scale: float,
        current_batch_size: int,
    ) -> Dict[str, Any]:
        """Recommend batch size adjustment based on current noise scale.

        Logic:
        - If B_noise >> current_batch: gradients are very noisy → increase batch.
        - If B_noise << current_batch: gradients are very clean → can decrease batch
          for faster wall-clock steps.
        - If B_noise ≈ current_batch: near-optimal.

        Args:
            noise_scale: Current gradient noise scale B_noise.
            current_batch_size: Current batch size.

        Returns:
            ``{'should_adjust': bool, 'direction': str, 'suggested_size': int,
               'reasoning': str}``
        """
        if noise_scale <= 0:
            return {
                "should_adjust": False,
                "direction": "none",
                "suggested_size": current_batch_size,
                "reasoning": "Noise scale is zero or negative — cannot determine.",
            }

        ratio = current_batch_size / noise_scale

        if ratio < 0.25:
            # Batch much smaller than noise scale — increase for efficiency
            suggested = int(noise_scale * 0.5)
            return {
                "should_adjust": True,
                "direction": "increase",
                "suggested_size": max(suggested, current_batch_size * 2),
                "reasoning": (
                    f"Batch ({current_batch_size}) is much smaller than noise scale "
                    f"({noise_scale:.1f}). Increasing batch will improve "
                    "gradient quality with minimal loss in step efficiency."
                ),
            }
        elif ratio > 4.0:
            # Batch much larger than noise scale — decrease for faster steps
            suggested = int(noise_scale * 0.5)
            return {
                "should_adjust": True,
                "direction": "decrease",
                "suggested_size": max(1, min(suggested, current_batch_size // 2)),
                "reasoning": (
                    f"Batch ({current_batch_size}) is much larger than noise scale "
                    f"({noise_scale:.1f}). Reducing batch will increase steps "
                    "per second with minimal loss in sample efficiency."
                ),
            }
        else:
            return {
                "should_adjust": False,
                "direction": "none",
                "suggested_size": current_batch_size,
                "reasoning": (
                    f"Batch ({current_batch_size}) is near the noise scale "
                    f"({noise_scale:.1f}). Current batch size is near-optimal."
                ),
            }
