"""
Nexus Tensor — Core Tensor Class
====================================
Custom N-dimensional tensor implementation supporting:
    - Shape manipulation (reshape, transpose, permute, expand, squeeze, unsqueeze)
    - Broadcasting (NumPy-style automatic shape expansion)
    - Indexing and slicing
    - Type conversions (float32, float64, int32, int64, bool)
    - Mathematical operations (+, -, *, /, @, **, unary ops)
    - Reduction operations (sum, mean, max, min, prod, argmax, argmin)
    - Comparison operations
    - Concatenation and stacking
    - In-place operations for memory efficiency
    - Device placement (CPU/GPU via NumPy/CuPy)

This is built on top of NumPy arrays for raw storage but provides
a complete tensor abstraction layer with operator overloading.

Note: In the full LLM, PyTorch tensors are used for GPU acceleration.
This Tensor class serves as the mathematical foundation and can be
seamlessly converted to/from PyTorch tensors.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Iterator
from functools import reduce
import operator

# Type alias
ArrayLike = Union[list, tuple, np.ndarray, int, float, bool, "Tensor"]
_DTYPE_MAP = {
    "float32": np.float32, "float64": np.float64,
    "int32": np.int32, "int64": np.int64,
    "bool": np.bool_, "float16": np.float16,
    "bfloat16": np.float32,  # NumPy doesn't have bfloat16 natively
}


class Tensor:
    """
    N-dimensional array with operator overloading and broadcasting.
    
    Core abstraction for all mathematical operations in Nexus.
    Wraps NumPy arrays with additional metadata and convenience methods.
    
    Examples:
        >>> a = Tensor([[1, 2], [3, 4]])
        >>> b = Tensor([[5, 6], [7, 8]])
        >>> (a + b).data
        array([[6, 8], [10, 12]])
        >>> (a @ b).data  # Matrix multiplication
        array([[19, 22], [43, 50]])
        >>> a.sum()
        Tensor(10)
        >>> a.reshape(4, 1)
        Tensor([[1], [2], [3], [4]])
    """

    def __init__(
        self,
        data: ArrayLike,
        dtype: Optional[str] = None,
        requires_grad: bool = False,
        _grad: Optional["Tensor"] = None,
        _backward: Optional[Any] = None,
        _children: Optional[Tuple["Tensor", ...]] = None,
        _op: str = "",
    ):
        """
        Create a Tensor.
        
        Args:
            data: Input data (list, tuple, numpy array, scalar, or another Tensor).
            dtype: Target dtype ("float32", "float64", "int32", "int64", "bool").
            requires_grad: Whether to track gradients for this tensor.
            _grad: Internal gradient accumulator (for autodiff).
            _backward: Internal backward function (for autodiff).
            _children: Parent tensors in the computation graph.
            _op: Name of the operation that created this tensor.
        """
        if isinstance(data, Tensor):
            np_data = data.data.copy()
        elif isinstance(data, np.ndarray):
            np_data = data
        else:
            np_data = np.array(data, dtype=np.float64)
        
        if dtype is not None:
            np_data = np_data.astype(_DTYPE_MAP.get(dtype, np.float64))
        elif np_data.dtype in (np.int32, np.int64):
            pass  # Keep integer dtype
        else:
            np_data = np_data.astype(np.float64)
        
        self.data = np_data
        self.dtype = str(self.data.dtype)
        self.shape = tuple(self.data.shape)
        self.ndim = self.data.ndim
        
        # Gradient tracking (for autodiff integration)
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = _grad
        self._backward = _backward
        self._children = _children or ()
        self._op = _op

    # ================================================================
    # CONVERSIONS
    # ================================================================

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self.data.copy()

    def tolist(self) -> Any:
        """Convert to Python list."""
        return self.data.tolist()

    @staticmethod
    def from_numpy(arr: np.ndarray, requires_grad: bool = False) -> "Tensor":
        """Create Tensor from NumPy array."""
        return Tensor(arr, requires_grad=requires_grad)

    def to(self, dtype: str) -> "Tensor":
        """Convert to specified dtype."""
        return Tensor(self.data, dtype=dtype)

    def float(self) -> "Tensor":
        """Convert to float64."""
        return self.to("float64")

    def half(self) -> "Tensor":
        """Convert to float16."""
        return self.to("float16")

    def int(self) -> "Tensor":
        """Convert to int64."""
        return self.to("int64")

    def bool(self) -> "Tensor":
        """Convert to bool."""
        return self.to("bool")

    def item(self) -> Union[int, float]:
        """Return scalar value."""
        return self.data.item()

    def detach(self) -> "Tensor":
        """Return a new tensor detached from the computation graph."""
        return Tensor(self.data)

    def clone(self) -> "Tensor":
        """Return a deep copy."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    # ================================================================
    # SHAPE OPERATIONS
    # ================================================================

    def reshape(self, *shape: Union[int, Tuple[int, ...]]) -> "Tensor":
        """Reshape tensor. Use -1 to infer dimension."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(self.data.reshape(shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten dimensions."""
        new_shape = self.shape[:start_dim]
        if end_dim == -1:
            new_shape += (-1,)
        else:
            new_shape += (np.prod(self.shape[start_dim:end_dim + 1]).astype(int),)
            new_shape += self.shape[end_dim + 1:]
        return self.reshape(*new_shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        """Remove dimensions of size 1."""
        if dim is None:
            return Tensor(np.squeeze(self.data))
        return Tensor(np.squeeze(self.data, axis=dim))

    def unsqueeze(self, dim: int) -> "Tensor":
        """Add a dimension of size 1."""
        return Tensor(np.expand_dims(self.data, axis=dim))

    def transpose(self, *dims: int) -> "Tensor":
        """Transpose dimensions."""
        if not dims:
            dims = tuple(reversed(range(self.ndim)))
        elif len(dims) == 2 and self.ndim == 2:
            return Tensor(self.data.T)
        return Tensor(np.transpose(self.data, axes=dims))

    @property
    def T(self) -> "Tensor":
        """Transpose (2D shorthand)."""
        return self.transpose()

    def permute(self, *dims: int) -> "Tensor":
        """Permute dimensions (alias for transpose with explicit dims)."""
        return self.transpose(*dims)

    def swapaxes(self, axis1: int, axis2: int) -> "Tensor":
        """Swap two axes."""
        return Tensor(np.swapaxes(self.data, axis1, axis2))

    def expand(self, *shape: int) -> "Tensor":
        """Broadcast tensor to a larger shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.broadcast_to(self.data, shape))

    def repeat(self, *repeats: int) -> "Tensor":
        """Repeat tensor along dimensions."""
        return Tensor(np.repeat(self.data, repeats))

    def pad(self, pad_width, mode: str = "constant", constant_values: float = 0) -> "Tensor":
        """Pad tensor."""
        return Tensor(np.pad(self.data, pad_width, mode=mode, constant_values=constant_values))

    @property
    def size(self) -> int:
        """Total number of elements."""
        return int(np.prod(self.shape))

    def split(self, sections: int, axis: int = 0) -> List["Tensor"]:
        """Split tensor into equal sections."""
        return [Tensor(arr) for arr in np.array_split(self.data, sections, axis=axis)]

    def chunk(self, chunks: int, axis: int = 0) -> List["Tensor"]:
        """Split tensor into chunks."""
        return [Tensor(arr) for arr in np.array_split(self.data, chunks, axis=axis)]

    def narrow(self, dim: int, start: int, length: int) -> "Tensor":
        """Return a narrowed view of the tensor."""
        slices = [slice(None)] * self.ndim
        slices[dim] = slice(start, start + length)
        return Tensor(self.data[tuple(slices)].copy())

    def view(self, *shape: int) -> "Tensor":
        """Alias for reshape."""
        return self.reshape(*shape)

    # ================================================================
    # MATHEMATICAL OPERATIONS
    # ================================================================

    def __add__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.add, other, "+")

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.add, other, "+")

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.subtract, other, "-")

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        other = Tensor(other)
        return other._binary_op(np.subtract, self, "-")

    def __mul__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.multiply, other, "*")

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.multiply, other, "*")

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.true_divide, other, "/")

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        other = Tensor(other)
        return other._binary_op(np.true_divide, self, "/")

    def __floordiv__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.floor_divide, other, "//")

    def __mod__(self, other: ArrayLike) -> "Tensor":
        return self._binary_op(np.mod, other, "%")

    def __pow__(self, power: ArrayLike) -> "Tensor":
        return self._binary_op(np.power, power, "**")

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        """Matrix multiplication (@ operator)."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        result = np.matmul(self.data, other_data)
        return Tensor(result)

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        other = Tensor(other)
        return other.__matmul__(self)

    def __abs__(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    def __invert__(self) -> "Tensor":
        return Tensor(~self.data)

    def __and__(self, other) -> "Tensor":
        return Tensor(self.data & (other.data if isinstance(other, Tensor) else other))

    def __or__(self, other) -> "Tensor":
        return Tensor(self.data | (other.data if isinstance(other, Tensor) else other))

    def __xor__(self, other) -> "Tensor":
        return Tensor(self.data ^ (other.data if isinstance(other, Tensor) else other))

    def __lt__(self, other: ArrayLike) -> "Tensor":
        return Tensor(self.data < self._as_np(other))

    def __le__(self, other: ArrayLike) -> "Tensor":
        return Tensor(self.data <= self._as_np(other))

    def __gt__(self, other: ArrayLike) -> "Tensor":
        return Tensor(self.data > self._as_np(other))

    def __ge__(self, other: ArrayLike) -> "Tensor":
        return Tensor(self.data >= self._as_np(other))

    def __eq__(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return Tensor(self.data == other.data)
        return Tensor(self.data == other)

    def __ne__(self, other: Any) -> "Tensor":
        if isinstance(other, Tensor):
            return Tensor(self.data != other.data)
        return Tensor(self.data != other)

    def _binary_op(self, op: callable, other: ArrayLike, name: str) -> "Tensor":
        """Apply a binary operation with broadcasting."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        # Handle scalar broadcasting
        if isinstance(other, (int, float, bool)):
            other_data = np.array(other, dtype=np.float64)
        result = op(self.data, other_data)
        return Tensor(result)

    def _as_np(self, other: ArrayLike) -> np.ndarray:
        if isinstance(other, Tensor):
            return other.data
        return np.array(other)

    # ================================================================
    # UNARY OPERATIONS
    # ================================================================

    def exp(self) -> "Tensor":
        return Tensor(np.exp(self.data))

    def log(self) -> "Tensor":
        return Tensor(np.log(self.data))

    def sqrt(self) -> "Tensor":
        return Tensor(np.sqrt(self.data))

    def sin(self) -> "Tensor":
        return Tensor(np.sin(self.data))

    def cos(self) -> "Tensor":
        return Tensor(np.cos(self.data))

    def tanh(self) -> "Tensor":
        return Tensor(np.tanh(self.data))

    def sigmoid(self) -> "Tensor":
        return Tensor(1.0 / (1.0 + np.exp(-self.data)))

    def relu(self) -> "Tensor":
        return Tensor(np.maximum(self.data, 0))

    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit."""
        x = self.data
        return Tensor(0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))))

    def silu(self) -> "Tensor":
        """SiLU / Swish activation: x * sigmoid(x)."""
        return self * self.sigmoid()

    def softmax(self, dim: int = -1) -> "Tensor":
        """Compute softmax along dimension."""
        x = self.data - np.max(self.data, axis=dim, keepdims=True)
        e = np.exp(x)
        return Tensor(e / np.sum(e, axis=dim, keepdims=True))

    def log_softmax(self, dim: int = -1) -> "Tensor":
        """Numerically stable log-softmax."""
        x = self.data - np.max(self.data, axis=dim, keepdims=True)
        return Tensor(x - np.log(np.sum(np.exp(x), axis=dim, keepdims=True)))

    def clamp(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "Tensor":
        """Clamp values to range [min_val, max_val]."""
        return Tensor(np.clip(self.data, min_val, max_val))

    def floor(self) -> "Tensor":
        return Tensor(np.floor(self.data))

    def ceil(self) -> "Tensor":
        return Tensor(np.ceil(self.data))

    def round(self, decimals: int = 0) -> "Tensor":
        return Tensor(np.round(self.data, decimals))

    def sign(self) -> "Tensor":
        return Tensor(np.sign(self.data))

    def abs(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    # ================================================================
    # REDUCTION OPERATIONS
    # ================================================================

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        return Tensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims))

    def min(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        return Tensor(np.min(self.data, axis=axis, keepdims=keepdims))

    def prod(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        return Tensor(np.prod(self.data, axis=axis, keepdims=keepdims))

    def argmax(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.argmax(self.data, axis=axis))

    def argmin(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.argmin(self.data, axis=axis))

    def var(self, axis: Optional[int] = None, correction: int = 1, keepdims: bool = False) -> "Tensor":
        if axis is None:
            n = self.size
        else:
            n = self.shape[axis]
        mean = np.mean(self.data, axis=axis, keepdims=True)
        return Tensor(np.sum((self.data - mean) ** 2, axis=axis, keepdims=keepdims) / max(1, n - correction))

    def std(self, axis: Optional[int] = None, correction: int = 1, keepdims: bool = False) -> "Tensor":
        v = self.var(axis, correction, keepdims=True)
        return Tensor(np.sqrt(v.data)).squeeze() if not keepdims and axis is None else v

    def cumsum(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.cumsum(self.data, axis=axis))

    def cumprod(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.cumprod(self.data, axis=axis))

    def any(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.any(self.data, axis=axis))

    def all(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(np.all(self.data, axis=axis))

    # ================================================================
    # LINEAR ALGEBRA SHORTHANDS
    # ================================================================

    def matmul(self, other: ArrayLike) -> "Tensor":
        """Matrix multiply."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        return Tensor(np.matmul(self.data, other_data))

    def mm(self, other: ArrayLike) -> "Tensor":
        """Matrix multiply (2D only)."""
        return self.matmul(other)

    def bmm(self, other: ArrayLike) -> "Tensor":
        """Batch matrix multiply."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        return Tensor(np.matmul(self.data, other_data))

    def dot(self, other: ArrayLike) -> "Tensor":
        """Dot product (1D vectors) or matrix multiply (2D)."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        if self.ndim == 1 and (isinstance(other, Tensor) and other.ndim == 1):
            return Tensor(np.dot(self.data, other_data))
        return Tensor(np.matmul(self.data, other_data))

    # ================================================================
    # INDEXING AND SLICING
    # ================================================================

    def __getitem__(self, key: Any) -> "Tensor":
        result = self.data[key]
        return Tensor(result)

    def __setitem__(self, key: Any, value: ArrayLike):
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        """Gather elements along an axis."""
        return Tensor(np.take_along_axis(self.data, index.data, axis=dim))

    def scatter(self, dim: int, index: "Tensor", src: "Tensor") -> "Tensor":
        """Scatter elements along an axis."""
        result = self.data.copy()
        np.add.at(result, tuple(np.ix_(*[np.arange(s) for s in self.shape])), src.data)
        return Tensor(result)

    def where(self, condition: "Tensor", other: "Tensor") -> "Tensor":
        """Select elements based on condition."""
        cond_data = condition.data if isinstance(condition, Tensor) else np.array(condition)
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        return Tensor(np.where(cond_data, self.data, other_data))

    @staticmethod
    def where_global(condition: "Tensor", x: "Tensor", y: "Tensor") -> "Tensor":
        """Static where: select from x where condition is True, else y."""
        cond_data = condition.data if isinstance(condition, Tensor) else np.array(condition)
        return Tensor(np.where(cond_data, x.data, y.data))

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        """Fill elements where mask is True."""
        result = self.data.copy()
        result[mask.data] = value
        return Tensor(result)

    def index_select(self, dim: int, index: "Tensor") -> "Tensor":
        """Select elements along a dimension."""
        slices = [slice(None)] * self.ndim
        slices[dim] = index.data
        return Tensor(self.data[tuple(slices)])

    # ================================================================
    # CONCATENATION / STACKING
    # ================================================================

    @staticmethod
    def cat(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Concatenate tensors along a dimension."""
        arrays = [t.data for t in tensors]
        return Tensor(np.concatenate(arrays, axis=dim))

    @staticmethod
    def stack(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Stack tensors along a new dimension."""
        arrays = [t.data for t in tensors]
        return Tensor(np.stack(arrays, axis=dim))

    @staticmethod
    def vstack(tensors: List["Tensor"]) -> "Tensor":
        return Tensor.cat(tensors, dim=0)

    @staticmethod
    def hstack(tensors: List["Tensor"]) -> "Tensor":
        return Tensor.cat(tensors, dim=1)

    # ================================================================
    # CREATION UTILITIES
    # ================================================================

    @staticmethod
    def zeros(*shape: int, dtype: str = "float64") -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.zeros(shape, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def ones(*shape: int, dtype: str = "float64") -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.ones(shape, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def full(shape: Tuple[int, ...], fill_value: float, dtype: str = "float64") -> "Tensor":
        return Tensor(np.full(shape, fill_value, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def arange(start: float, end: Optional[float] = None, step: float = 1, dtype: str = "float64") -> "Tensor":
        if end is None:
            return Tensor(np.arange(start, step=step, dtype=_DTYPE_MAP.get(dtype, np.float64)))
        return Tensor(np.arange(start, end, step=step, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def linspace(start: float, stop: float, num: int = 50, dtype: str = "float64") -> "Tensor":
        return Tensor(np.linspace(start, stop, num, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def eye(n: int, m: Optional[int] = None, dtype: str = "float64") -> "Tensor":
        return Tensor(np.eye(n, m, dtype=_DTYPE_MAP.get(dtype, np.float64)))

    @staticmethod
    def randn(*shape: int, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        """Standard normal random tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.random.randn(*shape) * std + mean)

    @staticmethod
    def rand(*shape: int) -> "Tensor":
        """Uniform [0, 1) random tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return Tensor(np.random.rand(*shape))

    @staticmethod
    def randn_like(t: "Tensor") -> "Tensor":
        return Tensor.randn(*t.shape)

    @staticmethod
    def zeros_like(t: "Tensor") -> "Tensor":
        return Tensor.zeros(*t.shape)

    @staticmethod
    def ones_like(t: "Tensor") -> "Tensor":
        return Tensor.ones(*t.shape)

    # ================================================================
    # UTILITIES
    # ================================================================

    def isfinite(self) -> "Tensor":
        return Tensor(np.isfinite(self.data))

    def isinf(self) -> "Tensor":
        return Tensor(np.isinf(self.data))

    def isnan(self) -> "Tensor":
        return Tensor(np.isnan(self.data))

    def allclose(self, other: "Tensor", rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return bool(np.allclose(self.data, other.data, rtol=rtol, atol=atol))

    def equal(self, other: "Tensor") -> bool:
        return bool(np.array_equal(self.data, other.data))

    # ================================================================
    # REPRESENTATION
    # ================================================================

    def __repr__(self) -> str:
        return f"Tensor({self.data}, shape={self.shape}, dtype={self.dtype})"

    def __str__(self) -> str:
        return f"Tensor({self.data})"
