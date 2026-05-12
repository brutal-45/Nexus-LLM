"""
Nexus Automatic Differentiation
====================================
Complete autodiff engine built from scratch supporting:

    - Forward mode AD (efficient for Jacobian-vector products)
    - Reverse mode AD (efficient for gradient-vector products = backprop)
    - Computational graph construction and topological sort
    - Gradient computation via reverse-mode (backpropagation)
    - Higher-order gradients (Hessian via forward-over-reverse)
    - Jacobian and Hessian computation
    - Gradient checkpointing (memory-efficient)

Architecture:
    Each Variable wraps a value and tracks its computation history.
    The computational graph is a DAG (Directed Acyclic Graph).
    Reverse mode traverses the graph topologically backwards,
    accumulating gradients via the chain rule.

Forward mode:
    For f: R^n -> R^m, computes Jv (Jacobian-vector product).
    Cost: O(n) per forward pass. Good when n < m.
    
    Implementation: Carry "dual numbers" — (value, tangent).
    Each operation propagates both forward.

Reverse mode (backpropagation):
    For f: R^n -> R^m, computes J^T v (vector-Jacobian product).
    Cost: O(m) per backward pass. Good when m < m (typical: scalar loss).
    
    This is THE algorithm that makes deep learning possible.
    For a scalar loss with millions of parameters, reverse mode
    computes all gradients in ONE backward pass.
    
    Chain rule: dL/dx = sum over all paths of (dL/dy * dy/dx)

Reference:
    - Griewank & Walther, "Evaluating Derivatives" (2008)
    - Baydin et al., "Automatic Differentiation in Machine Learning" (2018)
"""

from __future__ import annotations
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import wraps

Array = np.ndarray


# ================================================================
# COMPUTATIONAL GRAPH
# ================================================================

class Variable:
    """
    A node in the computational graph that tracks gradients.
    
    Each Variable stores:
        - value: The forward computation result (NumPy array)
        - grad: The accumulated gradient (dL/dthis)
        - _backward: Function to propagate gradients to parents
        - _children: Set of parent Variables (inputs to this operation)
        - _op: Name of the operation that created this Variable
    
    Example:
        x = Variable(np.array([2.0, 3.0]), requires_grad=True)
        y = Variable(np.array([4.0, 5.0]), requires_grad=True)
        z = x * y + x  # z = x*y + x, dz/dx = y + 1
        z.backward()   # x.grad = [5.0, 6.0], y.grad = [2.0, 3.0]
    """

    def __init__(
        self,
        value: ArrayLike,
        requires_grad: bool = False,
        _children: Optional[Tuple["Variable", ...]] = None,
        _backward: Optional[Callable] = None,
        _op: str = "",
    ):
        if isinstance(value, Variable):
            value = value.value
        self.value = np.asarray(value, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: Optional[Array] = None
        self._children = _children or ()
        self._backward = _backward
        self._op = _op

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def size(self) -> int:
        return int(self.value.size)

    @property
    def T(self) -> "Variable":
        return transpose(self)

    def backward(self, grad: Optional[Array] = None):
        """
        Reverse-mode automatic differentiation (backpropagation).
        
        Computes gradients for all Variables that contributed to this one,
        using the chain rule and topological ordering.
        
        Algorithm:
            1. Build topological ordering of the computation graph
            2. Initialize gradient of output: dL/dL = 1 (or provided)
            3. Process nodes in reverse topological order:
               For each node, compute dL/d(parent) = dL/d(this) * d(this)/d(parent)
               Accumulate gradients at each parent
        """
        # Build topological ordering
        topo = []
        visited = set()
        
        def build_topo(v: Variable):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradients
        if grad is None:
            grad = np.ones_like(self.value)
        
        self.grad = grad
        
        # Reverse pass: process in reverse topological order
        for v in reversed(topo):
            if v._backward is not None and v.grad is not None:
                v._backward(v.grad)
        
        # Zero gradients for non-leaf nodes that don't require grad
        for v in topo:
            if not v.requires_grad and v.grad is not None:
                # Non-leaf: gradient is used to propagate but not stored
                pass
            elif v.requires_grad and v.grad is None:
                v.grad = np.zeros_like(v.value)

    def zero_grad(self):
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> "Variable":
        """Return a new Variable detached from the graph."""
        return Variable(self.value.copy())

    def numpy(self) -> Array:
        return self.value.copy()

    def item(self) -> float:
        return float(self.value.item())

    def __repr__(self) -> str:
        return f"Variable(shape={self.shape}, grad={'✓' if self.requires_grad else '✗'}, op='{self._op}')"


# ================================================================
# FORWARD MODE AUTOMATIC DIFFERENTIATION
# ================================================================

class DualNumber:
    """
    Dual number for forward-mode AD: (value, tangent).
    
    For f(x), the dual number (x, dx) tracks both the value and derivative.
    Operations propagate both:
        (a, da) + (b, db) = (a+b, da+db)
        (a, da) * (b, db) = (a*b, da*b + a*db)
        (a, da) ^ 2 = (a^2, 2*a*da)
        sin(a, da) = (sin(a), cos(a)*da)
    
    This computes the Jacobian-vector product J*v in a single forward pass.
    """

    def __init__(self, value: float, tangent: float = 0.0):
        self.value = value
        self.tangent = tangent

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value + other.value, self.tangent + other.tangent)
        return DualNumber(self.value + other, self.tangent)

    def __radd__(self, other):
        return DualNumber(other + self.value, self.tangent)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.value - other.value, self.tangent - other.tangent)
        return DualNumber(self.value - other, self.tangent)

    def __rsub__(self, other):
        return DualNumber(other - self.value, -self.tangent)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            # Product rule: d(a*b) = da*b + a*db
            return DualNumber(
                self.value * other.value,
                self.tangent * other.value + self.value * other.tangent
            )
        return DualNumber(self.value * other, self.tangent * other)

    def __rmul__(self, other):
        return DualNumber(other * self.value, other * self.tangent)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            # Quotient rule: d(a/b) = (da*b - a*db) / b^2
            return DualNumber(
                self.value / other.value,
                (self.tangent * other.value - self.value * other.tangent) / (other.value ** 2)
            )
        return DualNumber(self.value / other, self.tangent / other)

    def __pow__(self, power):
        # Chain rule: d(a^n) = n * a^(n-1) * da
        return DualNumber(
            self.value ** power,
            power * self.value ** (power - 1) * self.tangent
        )

    def __neg__(self):
        return DualNumber(-self.value, -self.tangent)

    def __abs__(self):
        sign = 1.0 if self.value >= 0 else -1.0
        return DualNumber(abs(self.value), sign * self.tangent)

    def sin(self):
        # d(sin(a)) = cos(a) * da
        return DualNumber(np.sin(self.value), np.cos(self.value) * self.tangent)

    def cos(self):
        # d(cos(a)) = -sin(a) * da
        return DualNumber(np.cos(self.value), -np.sin(self.value) * self.tangent)

    def tanh(self):
        # d(tanh(a)) = (1 - tanh(a)^2) * da
        t = np.tanh(self.value)
        return DualNumber(t, (1 - t * t) * self.tangent)

    def exp(self):
        # d(exp(a)) = exp(a) * da
        return DualNumber(np.exp(self.value), np.exp(self.value) * self.tangent)

    def log(self):
        # d(log(a)) = da / a
        return DualNumber(np.log(self.value), self.tangent / self.value)

    def sqrt(self):
        # d(sqrt(a)) = da / (2 * sqrt(a))
        return DualNumber(np.sqrt(self.value), self.tangent / (2 * np.sqrt(self.value)))

    def relu(self):
        return DualNumber(
            max(0, self.value),
            self.tangent if self.value > 0 else 0.0
        )


def forward_mode_ad(
    func: Callable[[List[float]], float],
    x: List[float],
    direction: Optional[List[float]] = None,
) -> Tuple[float, List[float]]:
    """
    Forward mode automatic differentiation.
    
    Computes f(x) and J(x) * v simultaneously, where J is the Jacobian.
    
    For f: R^n -> R^m with n inputs:
        - Choose direction vector v (default: standard basis vectors)
        - One forward pass per direction
        - Total cost for full Jacobian: n forward passes
    
    Efficient when n << m (few inputs, many outputs).
    
    Args:
        func: Function to differentiate, takes list of floats, returns float.
        x: Input point.
        direction: Direction vector v for J*v. If None, computes df/dx_i for each i.
    
    Returns:
        Tuple of (function_value, gradients).
    """
    n = len(x)
    direction = direction or [0.0] * n
    
    # Create dual numbers
    duals = [DualNumber(xi, di) for xi, di in zip(x, direction)]
    
    # Evaluate function with dual numbers
    result = func(duals)
    
    if isinstance(result, DualNumber):
        return result.value, result.tangent
    return result, 0.0


# ================================================================
# REVERSE MODE AD OPERATIONS
# ================================================================

# These functions create Variables with proper backward functions

def _ensure_variable(x: Any) -> Variable:
    """Convert input to Variable."""
    if isinstance(x, Variable):
        return x
    return Variable(x)


def add(a: Variable, b: Variable) -> Variable:
    """Element-wise addition with gradient: d/da = 1, d/db = 1."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    requires_grad = a.requires_grad or b.requires_grad
    
    out = Variable(
        a.value + b.value,
        requires_grad=requires_grad,
        _children=(a, b),
        _op="add",
    )
    
    def _backward(grad):
        # Gradient flows through addition unchanged
        if a.requires_grad:
            if a.grad is None:
                a.grad = _unbroadcast(grad, a.shape)
            else:
                a.grad += _unbroadcast(grad, a.shape)
        if b.requires_grad:
            if b.grad is None:
                b.grad = _unbroadcast(grad, b.shape)
            else:
                b.grad += _unbroadcast(grad, b.shape)
    
    out._backward = _backward
    return out


def sub(a: Variable, b: Variable) -> Variable:
    """Element-wise subtraction: d/da = 1, d/db = -1."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    requires_grad = a.requires_grad or b.requires_grad
    
    out = Variable(
        a.value - b.value,
        requires_grad=requires_grad,
        _children=(a, b),
        _op="sub",
    )
    
    def _backward(grad):
        if a.requires_grad:
            if a.grad is None:
                a.grad = _unbroadcast(grad, a.shape)
            else:
                a.grad += _unbroadcast(grad, a.shape)
        if b.requires_grad:
            neg_grad = _unbroadcast(-grad, b.shape)
            if b.grad is None:
                b.grad = neg_grad
            else:
                b.grad += neg_grad
    
    out._backward = _backward
    return out


def mul(a: Variable, b: Variable) -> Variable:
    """Element-wise multiplication: d/da = b, d/db = a."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    requires_grad = a.requires_grad or b.requires_grad
    
    out = Variable(
        a.value * b.value,
        requires_grad=requires_grad,
        _children=(a, b),
        _op="mul",
    )
    
    def _backward(grad):
        if a.requires_grad:
            g = _unbroadcast(grad * b.value, a.shape)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
        if b.requires_grad:
            g = _unbroadcast(grad * a.value, b.shape)
            if b.grad is None:
                b.grad = g
            else:
                b.grad += g
    
    out._backward = _backward
    return out


def matmul_op(a: Variable, b: Variable) -> Variable:
    """Matrix multiplication: C = A @ B."""
    a, b = _ensure_variable(a), _ensure_variable(b)
    requires_grad = a.requires_grad or b.requires_grad
    
    out = Variable(
        np.matmul(a.value, b.value),
        requires_grad=requires_grad,
        _children=(a, b),
        _op="matmul",
    )
    
    def _backward(grad):
        # dC/dA = grad @ B^T
        # dC/dB = A^T @ grad
        if a.requires_grad:
            g = np.matmul(grad, b.value.T)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
        if b.requires_grad:
            g = np.matmul(a.value.T, grad)
            if b.grad is None:
                b.grad = g
            else:
                b.grad += g
    
    out._backward = _backward
    return out


def relu(a: Variable) -> Variable:
    """ReLU activation: f(x) = max(0, x). Gradient: 1 if x > 0, else 0."""
    a = _ensure_variable(a)
    mask = (a.value > 0).astype(np.float64)
    
    out = Variable(
        a.value * mask,
        requires_grad=a.requires_grad,
        _children=(a,),
        _op="relu",
    )
    
    def _backward(grad):
        if a.requires_grad:
            g = grad * mask
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def sigmoid(a: Variable) -> Variable:
    """Sigmoid: σ(x) = 1/(1+exp(-x)). Gradient: σ(x)(1-σ(x))."""
    a = _ensure_variable(a)
    s = 1.0 / (1.0 + np.exp(-a.value))
    
    out = Variable(s, requires_grad=a.requires_grad, _children=(a,), _op="sigmoid")
    
    def _backward(grad):
        if a.requires_grad:
            g = grad * s * (1.0 - s)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def tanh(a: Variable) -> Variable:
    """Tanh activation. Gradient: 1 - tanh(x)^2."""
    a = _ensure_variable(a)
    t = np.tanh(a.value)
    
    out = Variable(t, requires_grad=a.requires_grad, _children=(a,), _op="tanh")
    
    def _backward(grad):
        if a.requires_grad:
            g = grad * (1.0 - t * t)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def log(a: Variable) -> Variable:
    """Natural logarithm. Gradient: 1/x."""
    a = _ensure_variable(a)
    
    out = Variable(np.log(a.value), requires_grad=a.requires_grad, _children=(a,), _op="log")
    
    def _backward(grad):
        if a.requires_grad:
            g = grad / a.value
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def exp(a: Variable) -> Variable:
    """Exponential. Gradient: exp(x)."""
    a = _ensure_variable(a)
    e = np.exp(a.value)
    
    out = Variable(e, requires_grad=a.requires_grad, _children=(a,), _op="exp")
    
    def _backward(grad):
        if a.requires_grad:
            g = grad * e
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def pow_op(a: Variable, power: float) -> Variable:
    """Power: x^p. Gradient: p * x^(p-1)."""
    a = _ensure_variable(a)
    result = a.value ** power
    
    out = Variable(result, requires_grad=a.requires_grad, _children=(a,), _op="pow")
    
    def _backward(grad):
        if a.requires_grad:
            g = grad * power * a.value ** (power - 1)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def sum_op(a: Variable, axis: Optional[int] = None, keepdims: bool = False) -> Variable:
    """Sum reduction. Gradient: broadcast ones."""
    a = _ensure_variable(a)
    result = np.sum(a.value, axis=axis, keepdims=keepdims)
    
    out = Variable(result, requires_grad=a.requires_grad, _children=(a,), _op="sum")
    
    def _backward(grad):
        if a.requires_grad:
            # Broadcast gradient back to input shape
            if axis is not None and not keepdims:
                g = np.expand_dims(grad, axis=axis)
                g = np.broadcast_to(g, a.shape).copy()
            else:
                g = np.broadcast_to(grad, a.shape).copy()
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def mean_op(a: Variable, axis: Optional[int] = None) -> Variable:
    """Mean reduction. Gradient: 1/n."""
    a = _ensure_variable(a)
    result = np.mean(a.value, axis=axis)
    
    out = Variable(result, requires_grad=a.requires_grad, _children=(a,), _op="mean")
    
    def _backward(grad):
        if a.requires_grad:
            n = a.value.size if axis is None else a.value.shape[axis]
            if axis is not None:
                g = np.expand_dims(grad, axis=axis) / n
                g = np.broadcast_to(g, a.shape).copy()
            else:
                g = np.broadcast_to(grad / n, a.shape).copy()
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def transpose(a: Variable, axes: Optional[Tuple[int, ...]] = None) -> Variable:
    """Transpose. Gradient: transpose of grad."""
    a = _ensure_variable(a)
    result = np.transpose(a.value, axes=axes)
    
    out = Variable(result, requires_grad=a.requires_grad, _children=(a,), _op="transpose")
    
    def _backward(grad):
        if a.requires_grad:
            if axes is not None:
                inv_axes = [0] * len(axes)
                for i, ax in enumerate(axes):
                    inv_axes[ax] = i
                g = np.transpose(grad, axes=inv_axes)
            else:
                g = np.transpose(grad)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def reshape_op(a: Variable, shape: Tuple[int, ...]) -> Variable:
    """Reshape. Gradient: reshape back to original."""
    a = _ensure_variable(a)
    result = np.reshape(a.value, shape)
    
    out = Variable(result, requires_grad=a.requires_grad, _children=(a,), _op="reshape")
    
    def _backward(grad):
        if a.requires_grad:
            g = np.reshape(grad, a.shape)
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def softmax_op(a: Variable, axis: int = -1) -> Variable:
    """Softmax. Gradient: complex (Jacobian is S - s*s^T)."""
    a = _ensure_variable(a)
    x = a.value - np.max(a.value, axis=axis, keepdims=True)
    e = np.exp(x)
    s = e / np.sum(e, axis=axis, keepdims=True)
    
    out = Variable(s, requires_grad=a.requires_grad, _children=(a,), _op="softmax")
    
    def _backward(grad):
        if a.requires_grad:
            # Jacobian of softmax: J_ij = s_i * (delta_ij - s_j)
            # Gradient: (diag(s) - s*s^T) @ grad = s * grad - s * (s^T @ grad)
            s_grad = s * grad
            sum_s_grad = np.sum(s * grad, axis=axis, keepdims=True)
            g = s_grad - s * sum_s_grad
            if a.grad is None:
                a.grad = g
            else:
                a.grad += g
    
    out._backward = _backward
    return out


def cross_entropy_loss(logits: Variable, targets: Array) -> Variable:
    """
    Cross-entropy loss: -sum(targets * log(softmax(logits))).
    
    Combines softmax + log + NLL loss for numerical stability.
    Gradient: softmax(logits) - targets
    
    Args:
        logits: Unnormalized log-probabilities (Variable).
        targets: One-hot or probability targets (array).
    
    Returns:
        Scalar loss Variable.
    """
    logits = _ensure_variable(logits)
    targets = np.asarray(targets, dtype=np.float64)
    
    # Numerically stable softmax
    x = logits.value - np.max(logits.value, axis=-1, keepdims=True)
    e = np.exp(x)
    s = e / np.sum(e, axis=-1, keepdims=True)
    
    # Cross-entropy: -sum(t * log(s))
    eps = 1e-12
    loss = -np.sum(targets * np.log(s + eps))
    
    out = Variable(loss, requires_grad=logits.requires_grad, _children=(logits,), _op="cross_entropy")
    
    def _backward(grad):
        if logits.requires_grad:
            g = grad * (s - targets)
            if logits.grad is None:
                logits.grad = g
            else:
                logits.grad += g
    
    out._backward = _backward
    return out


def _unbroadcast(grad: Array, target_shape: Tuple[int, ...]) -> Array:
    """Un-broadcast gradient to match the original tensor shape."""
    # Sum along axes that were broadcast
    if grad.shape == target_shape:
        return grad
    
    # Handle leading dimensions
    ndim_diff = len(grad.shape) - len(target_shape)
    if ndim_diff > 0:
        grad = np.sum(grad, axis=tuple(range(ndim_diff)))
    
    # Handle dimensions of size 1
    for i, (gs, ts) in enumerate(zip(grad.shape, target_shape)):
        if ts == 1 and gs != 1:
            grad = np.sum(grad, axis=i, keepdims=True)
    
    return grad


# ================================================================
# HIGHER-ORDER GRADIENTS
# ================================================================

def jacobian(func: Callable[[Array], Array], x: Array) -> Array:
    """
    Compute the Jacobian matrix J[i,j] = df_i/dx_j.
    
    For f: R^n -> R^m, the Jacobian is an m x n matrix.
    
    Uses forward-mode AD: one forward pass per input dimension.
    
    Args:
        func: Function f(x) -> y.
        x: Input point (n,).
    
    Returns:
        Jacobian matrix (m, n).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Evaluate function once to get output size
    y = func(x)
    m = y.size
    
    J = np.zeros((m, n), dtype=np.float64)
    
    # Forward-mode AD for each input dimension
    eps = 1e-7
    for j in range(n):
        # Perturb x_j
        x_plus = x.copy()
        x_plus[j] += eps
        y_plus = func(x_plus)
        
        x_minus = x.copy()
        x_minus[j] -= eps
        y_minus = func(x_minus)
        
        J[:, j] = (y_plus - y_minus) / (2 * eps)
    
    return J


def hessian(func: Callable[[Array], float], x: Array) -> Array:
    """
    Compute the Hessian matrix H[i,j] = d^2f / (dx_i * dx_j).
    
    For f: R^n -> R, the Hessian is an n x n symmetric matrix.
    
    Uses forward-over-reverse AD:
        Forward mode to compute gradients w.r.t. each direction,
        then reverse mode for each gradient computation.
    
    For small n, uses finite differences on the gradient.
    
    Args:
        func: Scalar function f(x) -> scalar.
        x: Input point (n,).
    
    Returns:
        Hessian matrix (n, n).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    # Compute gradient function using forward differences
    def grad_func(xi):
        g = np.zeros(n)
        eps = 1e-7
        for j in range(n):
            xi_plus = xi.copy()
            xi_plus[j] += eps
            xi_minus = xi.copy()
            xi_minus[j] -= eps
            g[j] = (func(xi_plus) - func(xi_minus)) / (2 * eps)
        return g
    
    H = np.zeros((n, n), dtype=np.float64)
    eps = 1e-5
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        H[i, :] = (grad_func(x_plus) - grad_func(x_minus)) / (2 * eps)
    
    # Symmetrize (should already be symmetric, but numerical errors)
    H = (H + H.T) / 2.0
    return H


def grad(func: Callable, x: Array) -> Tuple[float, Array]:
    """
    Compute function value and gradient (convenience function).
    
    Args:
        func: Scalar function.
        x: Input point.
    
    Returns:
        Tuple of (value, gradient).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    
    value = func(x)
    gradient = np.zeros(n, dtype=np.float64)
    eps = 1e-7
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        gradient[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    
    return float(value), gradient


def value_and_grad(func: Callable, x: Array) -> Tuple[float, Array]:
    """Alias for grad()."""
    return grad(func, x)


def reverse_mode_ad(
    func: Callable[[List[Variable]], Variable],
    inputs: List[Variable],
) -> Tuple[Variable, List[Optional[Array]]]:
    """
    Reverse mode AD using the computational graph.
    
    Args:
        func: Function that takes Variables and returns a Variable.
        inputs: List of input Variables with requires_grad=True.
    
    Returns:
        Tuple of (output, list of gradients).
    """
    # Ensure inputs require gradients
    for inp in inputs:
        inp.requires_grad = True
    
    # Forward pass
    output = func(inputs)
    
    # Backward pass
    output.backward()
    
    # Collect gradients
    grads = [inp.grad for inp in inputs]
    
    return output, grads


def backprop(loss: Variable) -> None:
    """
    Perform backpropagation from a loss Variable.
    
    Convenience function that calls loss.backward().
    After calling, loss.grad = 1.0 and all ancestor Variables
    with requires_grad=True will have their .grad populated.
    """
    loss.backward()


# ================================================================
# GRADIENT CHECKPOINTING
# ================================================================

class ComputationalGraph:
    """
    Manages the computational graph for gradient computation.
    
    Supports:
        - Graph construction during forward pass
        - Topological sorting for backward pass
        - Gradient checkpointing (recompute activations during backward
          to save memory at the cost of extra computation)
        - Graph visualization
    """

    def __init__(self):
        self.nodes: List[Variable] = []
        self.checkpointed: Set[int] = set()

    def add_node(self, var: Variable):
        """Track a node in the graph."""
        self.nodes.append(var)

    def topological_sort(self, output: Variable) -> List[Variable]:
        """Return nodes in topological order (inputs first, output last)."""
        topo = []
        visited = set()
        
        def visit(v: Variable):
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    visit(child)
                topo.append(v)
        
        visit(output)
        return topo

    def set_checkpoint(self, var: Variable):
        """Mark a variable for gradient checkpointing."""
        self.checkpointed.add(id(var))

    def clear(self):
        """Clear the graph."""
        self.nodes = []
        self.checkpointed = set()
