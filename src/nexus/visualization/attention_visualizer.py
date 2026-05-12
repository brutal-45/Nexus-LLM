"""
Attention Visualizer - Attention Pattern Analysis
=================================================

Comprehensive tools for extracting, analyzing, and visualizing
attention patterns from transformer models.

All visualizations are text-based using Unicode characters.
"""

import math
import json
import os
import time
import hashlib
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Sequence, Set
)
from enum import Enum


# ============================================================================
# Constants and Unicode Characters
# ============================================================================

BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_H = "─"
BOX_V = "│"
BOX_LT = "├"
BOX_RT = "┤"
BOX_BT = "┬"
BOX_BB = "┴"
BOX_CROSS = "┼"

# Heatmap intensity characters (from light to dark)
HEAT_CHARS = [" ", "·", "░", "▒", "▓", "█"]
HEAT_CHARS_8 = [" ", "·", "▫", "▒", "▓", "█", "▇", "█"]
HEAT_CHARS_10 = [" ", " ", "·", "░", "▒", "▓", "█", "▇", "▆", "█"]

# Role markers
ROLE_SYNTAX = "S"
ROLE_SEMANTIC = "E"
ROLE_POSITIONAL = "P"
ROLE_GLOBAL = "G"
ROLE_LOCAL = "L"
ROLE_SCATTERED = "X"
ROLE_UNKNOWN = "?"

STATUS_OK = "✓"
STATUS_WARN = "⚠"
STATUS_ERROR = "✗"
STATUS_INFO = "ℹ"


class AttentionPattern(Enum):
    """Types of attention patterns."""
    DIAGONAL = "diagonal"
    LOCAL = "local"
    GLOBAL = "global"
    SCATTERED = "scattered"
    HEAD_SPECIFIC = "head_specific"
    UNIFORM = "uniform"
    UNKNOWN = "unknown"


class HeadRole(Enum):
    """Roles that attention heads can play."""
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    POSITIONAL = "positional"
    MIXED = "mixed"
    REDUNDANT = "redundant"
    DEAD = "dead"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AttentionMap:
    """Represents a single attention map."""
    weights: List[List[float]] = field(default_factory=list)
    layer_idx: int = 0
    head_idx: int = 0
    query_tokens: List[str] = field(default_factory=list)
    key_tokens: List[str] = field(default_factory=list)
    num_heads: int = 1
    num_layers: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_rows(self) -> int:
        """Number of query positions."""
        return len(self.weights)

    @property
    def num_cols(self) -> int:
        """Number of key positions."""
        return len(self.weights[0]) if self.weights else 0

    def get_row(self, idx: int) -> List[float]:
        """Get attention weights for a specific query position.

        Args:
            idx: Query position index.

        Returns:
            List of attention weights over key positions.
        """
        if 0 <= idx < len(self.weights):
            return list(self.weights[idx])
        return []

    def get_col(self, idx: int) -> List[float]:
        """Get attention weights for a specific key position.

        Args:
            idx: Key position index.

        Returns:
            List of attention weights from query positions.
        """
        return [row[idx] for row in self.weights if idx < len(row)]

    def entropy(self) -> float:
        """Compute entropy of the attention distribution.

        Returns:
            Entropy in nats.
        """
        flat = [w for row in self.weights for w in row]
        return _entropy(flat)

    def coverage(self) -> float:
        """Compute attention coverage (fraction of positions with significant weight).

        Returns:
            Coverage fraction (0-1).
        """
        flat = [w for row in self.weights for w in row]
        threshold = 1.0 / (len(flat) * 10) if flat else 0
        significant = sum(1 for w in flat if w > threshold)
        return significant / len(flat) if flat else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": self.weights,
            "layer_idx": self.layer_idx,
            "head_idx": self.head_idx,
            "query_tokens": self.query_tokens,
            "key_tokens": self.key_tokens,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "metadata": self.metadata,
        }


@dataclass
class PatternClassification:
    """Classification of an attention pattern."""
    pattern: AttentionPattern = AttentionPattern.UNKNOWN
    confidence: float = 0.0
    entropy: float = 0.0
    coverage: float = 0.0
    diagonal_strength: float = 0.0
    local_strength: float = 0.0
    global_strength: float = 0.0
    uniformity: float = 0.0
    peak_position: Optional[Tuple[int, int]] = None
    description: str = ""


@dataclass
class HeadAnalysis:
    """Analysis result for a single attention head."""
    head_idx: int = 0
    role: HeadRole = HeadRole.UNKNOWN
    confidence: float = 0.0
    pattern: PatternClassification = field(default_factory=PatternClassification)
    avg_entropy: float = 0.0
    avg_coverage: float = 0.0
    attention_entropy: float = 0.0
    is_duplicate_of: Optional[int] = None
    duplicate_similarity: float = 0.0
    description: str = ""


@dataclass
class AttentionEvolution:
    """Tracks how attention patterns change over training steps."""
    step: int = 0
    layer_idx: int = 0
    head_idx: int = 0
    entropy: float = 0.0
    coverage: float = 0.0
    pattern: str = "unknown"
    avg_diagonal_weight: float = 0.0
    avg_local_weight: float = 0.0
    concentration: float = 0.0
    stability_score: float = 0.0
    change_from_previous: float = 0.0


# ============================================================================
# Utility Functions
# ============================================================================

def _entropy(values: List[float]) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        values: Probability values (should sum to ~1).

    Returns:
        Entropy in nats.
    """
    total = sum(values)
    if total <= 0:
        return 0.0
    normalized = [v / total for v in values if v > 0]
    if not normalized:
        return 0.0
    return -sum(p * math.log(p) for p in normalized)


def _kl_divergence(p: List[float], q: List[float]) -> float:
    """Compute KL divergence D(p || q).

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        KL divergence in nats.
    """
    if len(p) != len(q):
        return float("inf")
    total_p = sum(p) or 1.0
    total_q = sum(q) or 1.0
    kl = 0.0
    for pi, qi in zip(p, q):
        pn = pi / total_p
        qn = qi / total_q
        if pn > 0 and qn > 0:
            kl += pn * math.log(pn / qn)
        elif pn > 0:
            kl += float("inf")
    return kl


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity (-1 to 1).
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _flatten_matrix(matrix: List[List[float]]) -> List[float]:
    """Flatten a 2D matrix to a 1D list.

    Args:
        matrix: 2D list.

    Returns:
        Flattened 1D list.
    """
    return [v for row in matrix for v in row]


def _matrix_frobenius_norm(matrix: List[List[float]]) -> float:
    """Compute Frobenius norm of a matrix.

    Args:
        matrix: 2D list.

    Returns:
        Frobenius norm.
    """
    flat = _flatten_matrix(matrix)
    return math.sqrt(sum(x * x for x in flat))


def _normalize_rows(matrix: List[List[float]]) -> List[List[float]]:
    """Normalize each row of a matrix to sum to 1.

    Args:
        matrix: Input matrix.

    Returns:
        Row-normalized matrix.
    """
    result = []
    for row in matrix:
        total = sum(row)
        if total > 0:
            result.append([v / total for v in row])
        else:
            n = len(row)
            result.append([1.0 / n] * n if n > 0 else [])
    return result


def _normalize_to_range(values: List[float], min_val: float = 0.0, max_val: float = 1.0) -> List[float]:
    """Normalize values to [min_val, max_val] range.

    Args:
        values: Input values.
        min_val: Target minimum.
        max_val: Target maximum.

    Returns:
        Normalized values.
    """
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if v_max == v_min:
        return [min_val + (max_val - min_val) / 2] * len(values)
    return [min_val + (v - v_min) / (v_max - v_min) * (max_val - min_val) for v in values]


def _correlation(a: List[float], b: List[float]) -> float:
    """Compute Pearson correlation between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Pearson correlation coefficient.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    n = len(a)
    mean_a = sum(a) / n
    mean_b = sum(b) / n

    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    std_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
    std_b = math.sqrt(sum((x - mean_b) ** 2 for x in b))

    if std_a == 0 or std_b == 0:
        return 0.0
    return cov / (std_a * std_b)


# ============================================================================
# AttentionMapExtractor
# ============================================================================

class AttentionMapExtractor:
    """
    Extract attention weights from transformer models.

    Supports multiple model architectures including HuggingFace transformers,
    custom models, and models with explicit attention weight access.

    Example:
        extractor = AttentionMapExtractor()
        attention_maps = extractor.extract_attention(model, input_ids)
        for amap in attention_maps:
            print(f"Layer {amap.layer_idx}, Head {amap.head_idx}")
    """

    def __init__(
        self,
        extract_all_layers: bool = True,
        extract_all_heads: bool = True,
        normalize: bool = True,
    ):
        """Initialize the attention map extractor.

        Args:
            extract_all_layers: Whether to extract from all layers.
            extract_all_heads: Whether to extract from all heads.
            normalize: Whether to normalize attention weights per row.
        """
        self._extract_all_layers = extract_all_layers
        self._extract_all_heads = extract_all_heads
        self._normalize = normalize
        self._cache: Dict[str, List[AttentionMap]] = {}

    def extract_attention(
        self,
        model: Any,
        input_ids: Any,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        tokens: Optional[List[str]] = None,
    ) -> List[AttentionMap]:
        """Extract attention weights from a transformer model.

        Args:
            model: Transformer model.
            input_ids: Input token IDs.
            layer_idx: Specific layer to extract (None for all).
            head_idx: Specific head to extract (None for all).
            tokens: Optional token strings for labels.

        Returns:
            List of AttentionMap objects.
        """
        # Try HuggingFace model first
        attention_weights = self._try_huggingface(model, input_ids)
        if attention_weights is not None:
            return self._process_extracted(
                attention_weights, layer_idx, head_idx, tokens
            )

        # Try custom model with get_attention_weights
        if hasattr(model, "get_attention_weights"):
            try:
                weights = model.get_attention_weights(input_ids)
                if weights is not None:
                    return self._process_extracted(
                        weights, layer_idx, head_idx, tokens
                    )
            except Exception:
                pass

        # Try model with attention output
        if hasattr(model, "forward"):
            try:
                output = model.forward(
                    input_ids,
                    output_attentions=True,
                )
                if hasattr(output, "attentions") and output.attentions:
                    weights = []
                    for layer_attn in output.attentions:
                        if hasattr(layer_attn, "detach"):
                            layer_attn = layer_attn.detach()
                        if hasattr(layer_attn, "cpu"):
                            layer_attn = layer_attn.cpu()
                        if hasattr(layer_attn, "numpy"):
                            layer_attn = layer_attn.numpy()
                        if hasattr(layer_attn, "tolist"):
                            layer_attn = layer_attn.tolist()
                        weights.append(layer_attn)
                    return self._process_extracted(
                        weights, layer_idx, head_idx, tokens
                    )
            except Exception:
                pass

        # Try to extract from model layers directly
        weights = self._extract_from_layers(model, input_ids)
        if weights:
            return self._process_extracted(weights, layer_idx, head_idx, tokens)

        return []

    def extract_from_weights(
        self,
        attention_weights: List[List[List[float]]],
        layer_idx: int = 0,
        head_idx: int = 0,
        tokens: Optional[List[str]] = None,
    ) -> AttentionMap:
        """Create an AttentionMap from raw weight data.

        Args:
            attention_weights: 3D tensor [num_heads][seq_len][seq_len] or
                             2D tensor [seq_len][seq_len].
            layer_idx: Layer index.
            head_idx: Head index.
            tokens: Optional token labels.

        Returns:
            AttentionMap object.
        """
        if not attention_weights:
            return AttentionMap(layer_idx=layer_idx, head_idx=head_idx)

        # Handle 3D input (multiple heads)
        if attention_weights and isinstance(attention_weights[0], list) and (
            len(attention_weights[0]) > 0 and isinstance(attention_weights[0][0], list)
        ):
            weights = attention_weights[head_idx] if head_idx < len(attention_weights) else attention_weights[0]
        else:
            weights = attention_weights

        if self._normalize:
            weights = _normalize_rows(weights)

        return AttentionMap(
            weights=weights,
            layer_idx=layer_idx,
            head_idx=head_idx,
            query_tokens=tokens or [],
            key_tokens=tokens or [],
        )

    def aggregate_heads(
        self,
        attention_maps: List[AttentionMap],
        method: str = "mean",
    ) -> AttentionMap:
        """Aggregate attention across multiple heads.

        Args:
            attention_maps: List of AttentionMap objects from same layer.
            method: Aggregation method: "mean", "max", "min", "median".

        Returns:
            Aggregated AttentionMap.
        """
        if not attention_maps:
            return AttentionMap()

        first = attention_maps[0]
        num_rows = first.num_rows
        num_cols = first.num_cols

        if num_rows == 0 or num_cols == 0:
            return AttentionMap()

        aggregated = [[0.0] * num_cols for _ in range(num_rows)]

        for amap in attention_maps:
            for r in range(min(num_rows, amap.num_rows)):
                for c in range(min(num_cols, amap.num_cols)):
                    aggregated[r][c] += amap.weights[r][c]

        n = len(attention_maps)

        if method == "mean":
            for r in range(num_rows):
                for c in range(num_cols):
                    aggregated[r][c] /= n
        elif method == "max":
            for r in range(num_rows):
                for c in range(num_cols):
                    vals = [amap.weights[r][c] for amap in attention_maps
                            if r < amap.num_rows and c < amap.num_cols]
                    aggregated[r][c] = max(vals) if vals else 0.0
        elif method == "min":
            for r in range(num_rows):
                for c in range(num_cols):
                    vals = [amap.weights[r][c] for amap in attention_maps
                            if r < amap.num_rows and c < amap.num_cols]
                    aggregated[r][c] = min(vals) if vals else 0.0
        elif method == "median":
            for r in range(num_rows):
                for c in range(num_cols):
                    vals = sorted([amap.weights[r][c] for amap in attention_maps
                                   if r < amap.num_rows and c < amap.num_cols])
                    if vals:
                        mid = len(vals) // 2
                        aggregated[r][c] = vals[mid] if len(vals) % 2 == 1 else (
                            vals[mid - 1] + vals[mid]) / 2

        if self._normalize:
            aggregated = _normalize_rows(aggregated)

        return AttentionMap(
            weights=aggregated,
            layer_idx=first.layer_idx,
            head_idx=-1,
            query_tokens=first.query_tokens,
            key_tokens=first.key_tokens,
            num_heads=len(attention_maps),
            metadata={"aggregation_method": method},
        )

    def _try_huggingface(self, model: Any, input_ids: Any) -> Optional[List]:
        """Try to extract attention from HuggingFace model.

        Args:
            model: HuggingFace model.
            input_ids: Input token IDs.

        Returns:
            List of attention weight tensors or None.
        """
        if not hasattr(model, "config"):
            return None

        try:
            output = model(
                input_ids,
                output_attentions=True,
                return_dict=True,
            )

            if hasattr(output, "attentions") and output.attentions:
                weights = []
                for layer_attn in output.attentions:
                    if hasattr(layer_attn, "detach"):
                        layer_attn = layer_attn.detach()
                    if hasattr(layer_attn, "cpu"):
                        layer_attn = layer_attn.cpu()
                    if hasattr(layer_attn, "numpy"):
                        layer_attn = layer_attn.numpy()
                    if hasattr(layer_attn, "tolist"):
                        layer_attn = layer_attn.tolist()
                    weights.append(layer_attn)
                return weights
        except Exception:
            pass

        return None

    def _extract_from_layers(self, model: Any, input_ids: Any) -> List:
        """Try to extract attention from model layers directly.

        Args:
            model: Model object.
            input_ids: Input tensor.

        Returns:
            List of attention weight tensors.
        """
        weights = []

        # Try to find attention layers
        layers = []
        if hasattr(model, "layers"):
            layers = list(model.layers)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = list(model.encoder.layers)
        elif hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = list(model.decoder.layers)
        elif hasattr(model, "children"):
            for child in model.children():
                if hasattr(child, "attention"):
                    layers.append(child)

        for layer in layers:
            if hasattr(layer, "attention"):
                attn = layer.attention
                if hasattr(attn, "attention_weights"):
                    w = attn.attention_weights
                    if hasattr(w, "tolist"):
                        w = w.tolist()
                    if isinstance(w, list):
                        weights.append(w)

        return weights

    def _process_extracted(
        self,
        weights: List,
        layer_idx: Optional[int],
        head_idx: Optional[int],
        tokens: Optional[List[str]],
    ) -> List[AttentionMap]:
        """Process extracted attention weights into AttentionMap objects.

        Args:
            weights: Raw attention weight tensors.
            layer_idx: Specific layer filter.
            head_idx: Specific head filter.
            tokens: Token labels.

        Returns:
            List of AttentionMap objects.
        """
        results = []

        for l_idx, layer_weights in enumerate(weights):
            if layer_idx is not None and l_idx != layer_idx:
                continue

            # Convert to list if needed
            if hasattr(layer_weights, "tolist"):
                layer_weights = layer_weights.tolist()

            # Determine number of heads
            num_heads = len(layer_weights) if layer_weights else 1

            # If 3D: [heads][seq][seq], iterate over heads
            if layer_weights and isinstance(layer_weights[0], list) and (
                layer_weights[0] and isinstance(layer_weights[0][0], list)
            ):
                for h_idx in range(num_heads):
                    if head_idx is not None and h_idx != head_idx:
                        continue
                    head_weights = layer_weights[h_idx]
                    if self._normalize:
                        head_weights = _normalize_rows(head_weights)
                    results.append(AttentionMap(
                        weights=head_weights,
                        layer_idx=l_idx,
                        head_idx=h_idx,
                        query_tokens=tokens or [],
                        key_tokens=tokens or [],
                        num_heads=num_heads,
                    ))
            else:
                # 2D: [seq][seq] - single head
                w = list(layer_weights)
                if self._normalize:
                    w = _normalize_rows(w)
                results.append(AttentionMap(
                    weights=w,
                    layer_idx=l_idx,
                    head_idx=head_idx or 0,
                    query_tokens=tokens or [],
                    key_tokens=tokens or [],
                    num_heads=1,
                ))

        return results


# ============================================================================
# AttentionPatternAnalyzer
# ============================================================================

class AttentionPatternAnalyzer:
    """
    Identify and classify attention patterns.

    Detects diagonal (self-attention to same token), local window,
    global attention, scattered patterns, and head-specific behaviors.

    Example:
        analyzer = AttentionPatternAnalyzer()
        classification = analyzer.classify_pattern(attention_map)
        print(f"Pattern: {classification.pattern}, Confidence: {classification.confidence:.2f}")
    """

    def __init__(
        self,
        local_window_threshold: float = 0.3,
        diagonal_threshold: float = 0.5,
        global_threshold: float = 0.2,
        uniform_threshold: float = 0.9,
    ):
        """Initialize the pattern analyzer.

        Args:
            local_window_threshold: Threshold for local window detection.
            diagonal_threshold: Threshold for diagonal pattern detection.
            global_threshold: Threshold for global attention detection.
            uniform_threshold: Threshold for uniform distribution.
        """
        self._local_window = local_window_threshold
        self._diagonal_thresh = diagonal_threshold
        self._global_thresh = global_threshold
        self._uniform_thresh = uniform_threshold

    def classify_pattern(self, attention_map: AttentionMap) -> PatternClassification:
        """Classify the attention pattern of a map.

        Args:
            attention_map: AttentionMap to classify.

        Returns:
            PatternClassification with results.
        """
        result = PatternClassification()
        if not attention_map.weights or attention_map.num_rows == 0:
            result.description = "Empty attention map"
            return result

        weights = attention_map.weights
        n_rows = attention_map.num_rows
        n_cols = attention_map.num_cols

        # Compute entropy
        result.entropy = attention_map.entropy()
        max_entropy = math.log(max(n_cols, 1))
        result.coverage = attention_map.coverage()

        # Compute uniformity (how close to uniform distribution)
        uniform_val = 1.0 / max(n_cols, 1)
        flat = _flatten_matrix(weights)
        if flat:
            result.uniformity = 1.0 - sum(abs(w - uniform_val) for w in flat) / (len(flat) * uniform_val) if uniform_val > 0 else 0
            result.uniformity = max(0, min(1, result.uniformity))

        # Compute diagonal strength
        result.diagonal_strength = self._compute_diagonal_strength(weights)

        # Compute local window strength
        result.local_strength = self._compute_local_strength(weights)

        # Compute global attention strength
        result.global_strength = self._compute_global_strength(weights)

        # Find peak
        peak_val = 0.0
        peak_pos = (0, 0)
        for r in range(n_rows):
            for c in range(n_cols):
                if weights[r][c] > peak_val:
                    peak_val = weights[r][c]
                    peak_pos = (r, c)
        result.peak_position = peak_pos

        # Classify based on features
        scores = {
            AttentionPattern.DIAGONAL: result.diagonal_strength,
            AttentionPattern.LOCAL: result.local_strength,
            AttentionPattern.GLOBAL: result.global_strength,
            AttentionPattern.UNIFORM: result.uniformity,
            AttentionPattern.SCATTERED: 1.0 - result.coverage - result.uniformity,
        }

        best_pattern = max(scores, key=scores.get)
        best_score = scores[best_pattern]
        result.pattern = best_pattern
        result.confidence = max(0, min(1, best_score))

        # Refine classification
        if result.uniformity > self._uniform_thresh:
            result.pattern = AttentionPattern.UNIFORM
            result.confidence = result.uniformity
            result.description = "Nearly uniform attention distribution"
        elif result.diagonal_strength > self._diagonal_thresh:
            result.pattern = AttentionPattern.DIAGONAL
            result.confidence = result.diagonal_strength
            result.description = f"Strong diagonal attention (strength={result.diagonal_strength:.3f})"
        elif result.local_strength > self._local_window:
            result.pattern = AttentionPattern.LOCAL
            result.confidence = result.local_strength
            window = self._estimate_local_window(weights)
            result.description = f"Local window attention (window≈{window}, strength={result.local_strength:.3f})"
        elif result.global_strength > self._global_thresh:
            result.pattern = AttentionPattern.GLOBAL
            result.confidence = result.global_strength
            result.description = f"Global attention with {result.global_strength:.1%} of attention on few positions"
        else:
            result.pattern = AttentionPattern.SCATTERED
            result.confidence = 1.0 - max(
                result.diagonal_strength, result.local_strength,
                result.global_strength, result.uniformity
            )
            result.description = "Scattered attention with no clear pattern"

        return result

    def compute_entropy(self, attention_map: AttentionMap) -> float:
        """Compute entropy of attention distribution.

        Args:
            attention_map: Attention map.

        Returns:
            Entropy value in nats.
        """
        return attention_map.entropy()

    def compute_coverage(
        self,
        attention_map: AttentionMap,
        threshold_factor: float = 10.0,
    ) -> float:
        """Compute attention coverage.

        Coverage = fraction of positions receiving more than 1/(n*factor) weight.

        Args:
            attention_map: Attention map.
            threshold_factor: Inverse of threshold as multiple of uniform.

        Returns:
            Coverage fraction (0-1).
        """
        flat = _flatten_matrix(attention_map.weights)
        if not flat:
            return 0.0
        threshold = 1.0 / (len(flat) * threshold_factor)
        return sum(1 for w in flat if w > threshold) / len(flat)

    def compute_concentration(self, attention_map: AttentionMap) -> float:
        """Compute how concentrated the attention is (inverse of entropy, normalized).

        Args:
            attention_map: Attention map.

        Returns:
            Concentration score (0-1, higher = more concentrated).
        """
        n = attention_map.num_cols
        if n <= 1:
            return 1.0
        max_entropy = math.log(n)
        actual_entropy = attention_map.entropy()
        return 1.0 - actual_entropy / max_entropy if max_entropy > 0 else 0.0

    def compare_patterns(
        self,
        map_a: AttentionMap,
        map_b: AttentionMap,
    ) -> Dict[str, float]:
        """Compare two attention maps.

        Args:
            map_a: First attention map.
            map_b: Second attention map.

        Returns:
            Dictionary with comparison metrics.
        """
        flat_a = _flatten_matrix(map_a.weights)
        flat_b = _flatten_matrix(map_b.weights)

        if len(flat_a) != len(flat_b):
            return {"error": "Maps have different sizes"}

        # Cosine similarity
        cosine = _cosine_similarity(flat_a, flat_b)

        # KL divergence (symmetric)
        kl_ab = _kl_divergence(flat_a, flat_b)
        kl_ba = _kl_divergence(flat_b, flat_a)
        symmetric_kl = (kl_ab + kl_ba) / 2

        # L2 distance
        l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(flat_a, flat_b)))

        # Correlation
        corr = _correlation(flat_a, flat_b)

        # Entropy difference
        entropy_a = _entropy(flat_a)
        entropy_b = _entropy(flat_b)

        return {
            "cosine_similarity": cosine,
            "symmetric_kl": symmetric_kl,
            "l2_distance": l2,
            "correlation": corr,
            "entropy_a": entropy_a,
            "entropy_b": entropy_b,
            "entropy_diff": abs(entropy_a - entropy_b),
        }

    def batch_classify(
        self,
        attention_maps: List[AttentionMap],
    ) -> List[PatternClassification]:
        """Classify multiple attention maps.

        Args:
            attention_maps: List of AttentionMap objects.

        Returns:
            List of PatternClassification results.
        """
        return [self.classify_pattern(amap) for amap in attention_maps]

    def _compute_diagonal_strength(self, weights: List[List[float]]) -> float:
        """Compute how much attention is on the diagonal.

        Args:
            weights: Attention weight matrix.

        Returns:
            Diagonal strength (0-1).
        """
        total = 0.0
        diag = 0.0
        for r, row in enumerate(weights):
            for c, w in enumerate(row):
                total += w
                if abs(r - c) <= 1:
                    diag += w
        return diag / total if total > 0 else 0.0

    def _compute_local_strength(self, weights: List[List[float]]) -> float:
        """Compute local window attention strength.

        Args:
            weights: Attention weight matrix.

        Returns:
            Local strength (0-1).
        """
        n = len(weights)
        if n <= 1:
            return 1.0

        # Try different window sizes and find the best
        best_strength = 0.0
        best_window = 3

        for window_size in [3, 5, 7, 11, min(15, n)]:
            if window_size > n:
                continue
            half_w = window_size // 2
            local_total = 0.0
            all_total = 0.0
            for r, row in enumerate(weights):
                for c, w in enumerate(row):
                    all_total += w
                    if abs(r - c) <= half_w:
                        local_total += w
            strength = local_total / all_total if all_total > 0 else 0.0
            if strength > best_strength:
                best_strength = strength
                best_window = window_size

        return best_strength

    def _compute_global_strength(self, weights: List[List[float]]) -> float:
        """Compute global attention strength (attention to distant positions).

        Args:
            weights: Attention weight matrix.

        Returns:
            Global strength (0-1).
        """
        n = len(weights)
        if n <= 2:
            return 0.0

        total = 0.0
        global_total = 0.0
        for r, row in enumerate(weights):
            for c, w in enumerate(row):
                total += w
                # Global = attention to positions far from current
                if abs(r - c) > n // 4:
                    global_total += w

        return global_total / total if total > 0 else 0.0

    def _estimate_local_window(self, weights: List[List[float]]) -> int:
        """Estimate the local attention window size.

        Args:
            weights: Attention weight matrix.

        Returns:
            Estimated window size.
        """
        n = len(weights)
        if n <= 1:
            return 1

        # For each query position, find the range of positions that get 80% of attention
        windows = []
        for row in weights:
            sorted_indices = sorted(range(len(row)), key=lambda i: -row[i])
            cumulative = 0.0
            threshold = sum(row) * 0.8
            max_dist = 0
            for idx in sorted_indices:
                cumulative += row[idx]
                dist = abs(sorted_indices[0] - idx)
                max_dist = max(max_dist, dist)
                if cumulative >= threshold:
                    break
            windows.append(max_dist * 2 + 1)

        if windows:
            return int(sum(windows) / len(windows))
        return n


# ============================================================================
# HeadRoleAnalyzer
# ============================================================================

class HeadRoleAnalyzer:
    """
    Analyze attention head specialization and find duplicate heads.

    Identifies heads that focus on syntax, semantic relations,
    positional patterns, and detects redundant heads.

    Example:
        analyzer = HeadRoleAnalyzer()
        roles = analyzer.analyze_head_roles(attention_maps, tokens)
        for role in roles:
            print(f"Head {role.head_idx}: {role.role} (confidence: {role.confidence:.2f})")
    """

    def __init__(
        self,
        duplicate_threshold: float = 0.95,
        dead_threshold: float = 0.99,
        syntax_window: int = 5,
    ):
        """Initialize the head role analyzer.

        Args:
            duplicate_threshold: Cosine similarity threshold for duplicate detection.
            dead_threshold: Entropy threshold (relative to max) for dead head detection.
            syntax_window: Window size for syntax pattern detection.
        """
        self._duplicate_thresh = duplicate_threshold
        self._dead_thresh = dead_threshold
        self._syntax_window = syntax_window
        self._pattern_analyzer = AttentionPatternAnalyzer()

    def analyze_head_roles(
        self,
        attention_maps: List[AttentionMap],
        tokens: Optional[List[str]] = None,
    ) -> List[HeadAnalysis]:
        """Analyze roles for a list of attention heads.

        Args:
            attention_maps: List of AttentionMap objects (one per head).
            tokens: Optional token strings.

        Returns:
            List of HeadAnalysis results.
        """
        results = []

        for amap in attention_maps:
            analysis = HeadAnalysis(head_idx=amap.head_idx)

            # Classify pattern
            pattern = self._pattern_analyzer.classify_pattern(amap)
            analysis.pattern = pattern

            # Compute aggregate statistics
            entropies = []
            coverages = []
            for row in amap.weights:
                entropies.append(_entropy(row))
                total = sum(row)
                if total > 0:
                    threshold = total / (len(row) * 10)
                    coverages.append(sum(1 for w in row if w > threshold) / len(row))
                else:
                    coverages.append(0.0)

            analysis.avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
            analysis.avg_coverage = sum(coverages) / len(coverages) if coverages else 0.0
            analysis.attention_entropy = amap.entropy()

            # Determine role
            analysis.role, analysis.confidence = self._determine_role(
                amap, pattern, tokens
            )

            # Description
            analysis.description = self._generate_description(analysis, amap)
            results.append(analysis)

        # Check for duplicates
        self._find_duplicates(results, attention_maps)

        return results

    def find_duplicate_heads(
        self,
        attention_maps: List[AttentionMap],
    ) -> List[Tuple[int, int, float]]:
        """Find pairs of duplicate/similar heads.

        Args:
            attention_maps: List of AttentionMap objects.

        Returns:
            List of (head_a, head_b, similarity) tuples.
        """
        duplicates = []
        n = len(attention_maps)

        for i in range(n):
            flat_i = _flatten_matrix(attention_maps[i].weights)
            for j in range(i + 1, n):
                flat_j = _flatten_matrix(attention_maps[j].weights)
                if len(flat_i) != len(flat_j):
                    continue
                sim = _cosine_similarity(flat_i, flat_j)
                if sim > self._duplicate_thresh:
                    duplicates.append((i, j, sim))

        duplicates.sort(key=lambda x: -x[2])
        return duplicates

    def role_summary(self, analyses: List[HeadAnalysis]) -> str:
        """Generate a summary of head roles.

        Args:
            analyses: List of HeadAnalysis results.

        Returns:
            Multi-line summary string.
        """
        role_counts: Dict[str, int] = defaultdict(int)
        for a in analyses:
            role_counts[a.role.value] += 1

        lines = []
        lines.append(f"{BOX_TL}{'Attention Head Role Summary':^50}{BOX_TR}")
        lines.append(f"{BOX_V}{'Role':<15}{'Count':>8}{'Heads':>25}{BOX_V}")
        lines.append(f"{BOX_V}{BOX_H*15}{BOX_CROSS}{BOX_H*8}{BOX_CROSS}{BOX_H*25}{BOX_V}")

        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            heads = [str(a.head_idx) for a in analyses if a.role.value == role]
            head_str = ", ".join(heads[:10])
            if len(heads) > 10:
                head_str += f"... (+{len(heads)-10})"
            lines.append(f"{BOX_V}{role:<15}{count:>8}{head_str:>25}{BOX_V}")

        lines.append(BOX_BL + BOX_H * 50 + BOX_BR)

        # Individual head details
        lines.append("")
        lines.append("Individual Head Analysis:")
        lines.append(f"{BOX_TL}{'Head':>6}{BOX_V}{'Role':<12}{BOX_V}{'Conf':>8}{BOX_V}{'Entropy':>10}{BOX_V}{'Coverage':>10}{BOX_V}{'Pattern':<15}{BOX_V}")
        lines.append(
            f"{BOX_H*6}{BOX_BT}{BOX_H*12}{BOX_BT}{BOX_H*8}{BOX_BT}{BOX_H*10}{BOX_BT}{BOX_H*10}{BOX_BT}{BOX_H*15}{BOX_TR}"
        )

        for a in sorted(analyses, key=lambda x: x.head_idx):
            lines.append(
                f"{BOX_V}{a.head_idx:>6}{BOX_V}{a.role.value:<12}{BOX_V}"
                f"{a.confidence:>8.3f}{BOX_V}{a.avg_entropy:>10.3f}{BOX_V}"
                f"{a.avg_coverage:>10.3f}{BOX_V}{a.pattern.pattern.value:<15}{BOX_V}"
            )

        lines.append(
            f"{BOX_BL}{BOX_H*6}{BOX_BB}{BOX_H*12}{BOX_BB}{BOX_H*8}{BOX_BB}{BOX_H*10}{BOX_BB}{BOX_H*10}{BOX_BB}{BOX_H*15}{BOX_BR}"
        )

        return "\n".join(lines)

    def _determine_role(
        self,
        amap: AttentionMap,
        pattern: PatternClassification,
        tokens: Optional[List[str]],
    ) -> Tuple[HeadRole, float]:
        """Determine the role of an attention head.

        Args:
            amap: Attention map.
            pattern: Pattern classification.
            tokens: Optional tokens.

        Returns:
            Tuple of (HeadRole, confidence).
        """
        # Dead head: very high entropy (nearly uniform)
        n_cols = amap.num_cols
        if n_cols > 0:
            max_entropy = math.log(n_cols)
            relative_entropy = pattern.entropy / max_entropy if max_entropy > 0 else 0
            if relative_entropy > self._dead_thresh:
                return HeadRole.DEAD, relative_entropy

        # Pattern-based classification
        if pattern.pattern == AttentionPattern.DIAGONAL:
            return HeadRole.SYNTAX, pattern.confidence

        if pattern.pattern == AttentionPattern.LOCAL:
            return HeadRole.SYNTAX, pattern.confidence

        if pattern.pattern == AttentionPattern.UNIFORM:
            if pattern.uniformity > 0.95:
                return HeadRole.DEAD, pattern.uniformity
            return HeadRole.GLOBAL, pattern.confidence * 0.7

        if pattern.pattern == AttentionPattern.GLOBAL:
            # Check if attending to specific token types (semantic)
            if tokens and len(tokens) > 1:
                is_semantic = self._check_semantic_pattern(amap, tokens)
                if is_semantic:
                    return HeadRole.SEMANTIC, pattern.confidence
            return HeadRole.GLOBAL, pattern.confidence

        if pattern.pattern == AttentionPattern.SCATTERED:
            if tokens:
                is_semantic = self._check_semantic_pattern(amap, tokens)
                if is_semantic:
                    return HeadRole.SEMANTIC, pattern.confidence * 0.8
            # Check positional patterns
            is_positional = self._check_positional_pattern(amap)
            if is_positional:
                return HeadRole.POSITIONAL, is_positional
            return HeadRole.MIXED, pattern.confidence * 0.5

        return HeadRole.UNKNOWN, 0.3

    def _check_semantic_pattern(
        self, amap: AttentionMap, tokens: List[str]
    ) -> bool:
        """Check if attention follows semantic patterns (e.g., attending to same-type tokens).

        Args:
            amap: Attention map.
            tokens: Token strings.

        Returns:
            Whether semantic pattern detected.
        """
        if not tokens or len(tokens) < 3:
            return False

        # Group tokens by simple type heuristics
        def token_type(t: str) -> str:
            if t.lower() in ("the", "a", "an", "is", "are", "was", "were", "be", "been",
                             "do", "does", "did", "have", "has", "had", "will", "would",
                             "could", "should", "may", "might", "can", "shall"):
                return "function"
            if any(c.isdigit() for c in t):
                return "number"
            if t in (",", ".", "!", "?", ";", ":", "'", '"', "(", ")", "-", "—"):
                return "punctuation"
            if t.startswith("##") or t.startswith("▁"):
                return "subword"
            return "content"

        types = [token_type(t) for t in tokens[:amap.num_cols]]

        # Check if same-type tokens attend to each other
        same_type_attention = 0.0
        total_attention = 0.0
        for r in range(amap.num_rows):
            r_type = types[r] if r < len(types) else ""
            for c in range(amap.num_cols):
                c_type = types[c] if c < len(types) else ""
                total_attention += amap.weights[r][c]
                if c_type == r_type and c_type != "punctuation":
                    same_type_attention += amap.weights[r][c]

        if total_attention > 0:
            ratio = same_type_attention / total_attention
            uniform_ratio = sum(1 for t in types[:amap.num_cols] if t == types[0] if types else False) / max(len(types), 1)
            return ratio > uniform_ratio * 1.5

        return False

    def _check_positional_pattern(self, amap: AttentionMap) -> float:
        """Check if attention follows a positional pattern.

        Args:
            amap: Attention map.

        Returns:
            Confidence of positional pattern (0-1).
        """
        n = amap.num_cols
        if n < 4:
            return 0.0

        # Check for periodic patterns
        position_attention: Dict[int, float] = defaultdict(float)
        total = 0.0

        for r, row in enumerate(amap.weights):
            for c, w in enumerate(row):
                # Relative position
                rel_pos = c - r
                position_attention[rel_pos] += w
                total += w

        if total <= 0:
            return 0.0

        # Check if certain relative positions dominate
        sorted_positions = sorted(position_attention.items(), key=lambda x: -x[1])
        top_positions = sorted_positions[:3]
        top_weight = sum(w for _, w in top_positions)

        return top_weight / total

    def _find_duplicates(
        self,
        analyses: List[HeadAnalysis],
        attention_maps: List[AttentionMap],
    ) -> None:
        """Find and mark duplicate heads in the analyses.

        Args:
            analyses: List of HeadAnalysis (modified in place).
            attention_maps: List of AttentionMap.
        """
        n = len(analyses)
        for i in range(n):
            flat_i = _flatten_matrix(attention_maps[i].weights)
            for j in range(i + 1, n):
                flat_j = _flatten_matrix(attention_maps[j].weights)
                if len(flat_i) != len(flat_j):
                    continue
                sim = _cosine_similarity(flat_i, flat_j)
                if sim > self._duplicate_thresh:
                    analyses[j].is_duplicate_of = i
                    analyses[j].duplicate_similarity = sim
                    analyses[j].role = HeadRole.REDUNDANT
                    analyses[j].confidence = sim

    def _generate_description(self, analysis: HeadAnalysis, amap: AttentionMap) -> str:
        """Generate a human-readable description.

        Args:
            analysis: Head analysis.
            amap: Attention map.

        Returns:
            Description string.
        """
        parts = [f"Head {analysis.head_idx}:"]
        parts.append(f"role={analysis.role.value}")
        parts.append(f"entropy={analysis.avg_entropy:.3f}")
        parts.append(f"coverage={analysis.avg_coverage:.3f}")
        if analysis.pattern.pattern != AttentionPattern.UNKNOWN:
            parts.append(f"pattern={analysis.pattern.pattern.value}")
        if analysis.is_duplicate_of is not None:
            parts.append(f"duplicate_of_head={analysis.is_duplicate_of}")
        return " | ".join(parts)


# ============================================================================
# AttentionEvolutionTracker
# ============================================================================

class AttentionEvolutionTracker:
    """
    Track how attention patterns change during training.

    Records attention statistics at each checkpoint and computes
    stability metrics, trend analysis, and evolution reports.

    Example:
        tracker = AttentionEvolutionTracker(max_checkpoints=100)
        tracker.record(attention_maps, step=100)
        tracker.record(attention_maps, step=200)
        report = tracker.evolution_report()
        print(report)
    """

    def __init__(
        self,
        max_checkpoints: int = 1000,
        max_heads: int = 100,
    ):
        """Initialize the evolution tracker.

        Args:
            max_checkpoints: Maximum checkpoints to store.
            max_heads: Maximum heads per checkpoint.
        """
        self._max_checkpoints = max_checkpoints
        self._max_heads = max_heads
        self._lock = threading.Lock()

        # Per-head evolution: head_key -> list of evolution records
        self._evolution: Dict[str, List[AttentionEvolution]] = defaultdict(list)
        self._checkpoint_steps: List[int] = []

    def record(
        self,
        attention_maps: List[AttentionMap],
        step: int,
    ) -> None:
        """Record attention patterns at a training step.

        Args:
            attention_maps: List of AttentionMap objects.
            step: Training step number.
        """
        with self._lock:
            self._checkpoint_steps.append(step)
            pattern_analyzer = AttentionPatternAnalyzer()

            for amap in attention_maps[:self._max_heads]:
                key = f"layer{amap.layer_idx}_head{amap.head_idx}"

                # Compute statistics
                classification = pattern_analyzer.classify_pattern(amap)
                concentration = pattern_analyzer.compute_concentration(amap)

                evolution = AttentionEvolution(
                    step=step,
                    layer_idx=amap.layer_idx,
                    head_idx=amap.head_idx,
                    entropy=classification.entropy,
                    coverage=classification.coverage,
                    pattern=classification.pattern.value,
                    avg_diagonal_weight=classification.diagonal_strength,
                    avg_local_weight=classification.local_strength,
                    concentration=concentration,
                )

                # Compute change from previous
                if self._evolution[key]:
                    prev = self._evolution[key][-1]
                    if prev.entropy > 0:
                        evolution.change_from_previous = abs(evolution.entropy - prev.entropy) / prev.entropy
                    # Stability: inverse of normalized change
                    evolution.stability_score = 1.0 / (1.0 + evolution.change_from_previous * 10)

                self._evolution[key].append(evolution)

            # Trim old checkpoints
            if len(self._checkpoint_steps) > self._max_checkpoints:
                oldest_step = self._checkpoint_steps[0]
                self._checkpoint_steps = self._checkpoint_steps[-self._max_checkpoints:]
                for key in self._evolution:
                    self._evolution[key] = [
                        e for e in self._evolution[key] if e.step >= oldest_step
                    ]

    def get_evolution(
        self,
        layer_idx: int,
        head_idx: int,
    ) -> List[AttentionEvolution]:
        """Get evolution records for a specific head.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.

        Returns:
            List of AttentionEvolution records.
        """
        key = f"layer{layer_idx}_head{head_idx}"
        return list(self._evolution.get(key, []))

    def get_all_heads(self) -> Dict[str, List[AttentionEvolution]]:
        """Get evolution records for all tracked heads.

        Returns:
            Dictionary mapping head keys to evolution lists.
        """
        with self._lock:
            return {k: list(v) for k, v in self._evolution.items()}

    def compute_stability(
        self,
        layer_idx: int,
        head_idx: int,
        window: int = 10,
    ) -> float:
        """Compute attention stability score for a head.

        Args:
            layer_idx: Layer index.
            head_idx: Head index.
            window: Window size for stability computation.

        Returns:
            Stability score (0-1, higher = more stable).
        """
        evolution = self.get_evolution(layer_idx, head_idx)
        if len(evolution) < 2:
            return 1.0

        recent = evolution[-window:]
        changes = [e.change_from_previous for e in recent]
        avg_change = sum(changes) / len(changes)
        return 1.0 / (1.0 + avg_change * 10)

    def detect_convergence(
        self,
        threshold: float = 0.01,
        min_steps: int = 10,
    ) -> List[Tuple[str, int]]:
        """Detect heads that have converged (stable attention pattern).

        Args:
            threshold: Maximum change threshold for convergence.
            min_steps: Minimum steps to check.

        Returns:
            List of (head_key, convergence_step) tuples.
        """
        converged = []

        with self._lock:
            for key, evolution in self._evolution.items():
                if len(evolution) < min_steps:
                    continue

                for i in range(min_steps, len(evolution)):
                    recent_changes = [
                        e.change_from_previous for e in evolution[max(0, i-5):i]
                    ]
                    if recent_changes:
                        avg_change = sum(recent_changes) / len(recent_changes)
                        if avg_change < threshold:
                            converged.append((key, evolution[i].step))
                            break

        return converged

    def evolution_report(self, layer_idx: Optional[int] = None) -> str:
        """Generate an evolution report.

        Args:
            layer_idx: Specific layer to report (None for all).

        Returns:
            Multi-line report string.
        """
        with self._lock:
            lines = []
            lines.append(f"{BOX_TL}{'Attention Pattern Evolution Report':^66}{BOX_TR}")
            lines.append(f"{BOX_V} Checkpoints: {len(self._checkpoint_steps)}{BOX_V}")
            if self._checkpoint_steps:
                lines.append(
                    f"{BOX_V} Steps: {self._checkpoint_steps[0]} - {self._checkpoint_steps[-1]}{BOX_V}"
                )
            lines.append(f"{BOX_V}{BOX_H * 66}{BOX_V}")

            # Per-head summary
            lines.append(
                f"{BOX_V}{'Head':<20}{'Pattern':<12}{'Entropy':>8}{'Coverage':>10}"
                f"{'Stability':>10}{'Converged':>10}{BOX_V}"
            )
            lines.append(
                f"{BOX_V}{BOX_H*20}{BOX_CROSS}{BOX_H*12}{BOX_CROSS}{BOX_H*8}"
                f"{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*10}{BOX_CROSS}{BOX_H*10}{BOX_V}"
            )

            sorted_keys = sorted(self._evolution.keys())
            for key in sorted_keys:
                if layer_idx is not None:
                    expected_prefix = f"layer{layer_idx}_"
                    if not key.startswith(expected_prefix):
                        continue

                evolution = self._evolution[key]
                if not evolution:
                    continue

                latest = evolution[-1]
                stability = latest.stability_score

                # Check convergence
                converged_step = "-"
                for i in range(1, len(evolution)):
                    if evolution[i].change_from_previous < 0.01:
                        converged_step = str(evolution[i].step)
                        break

                lines.append(
                    f"{BOX_V}{key:<20}{latest.pattern:<12}{latest.entropy:>8.3f}"
                    f"{latest.coverage:>10.3f}{stability:>10.3f}{converged_step:>10}{BOX_V}"
                )

            lines.append(BOX_BL + BOX_H * 66 + BOX_BR)

            # Convergence summary
            all_converged = self.detect_convergence()
            if all_converged:
                lines.append(f"\nConverged heads: {len(all_converged)}/{len(self._evolution)}")
                for key, step in all_converged[:10]:
                    lines.append(f"  {key} at step {step}")

            return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save evolution data to JSON.

        Args:
            path: Output file path.
        """
        with self._lock:
            data = {
                "checkpoint_steps": self._checkpoint_steps,
                "evolution": {},
            }
            for key, records in self._evolution.items():
                data["evolution"][key] = [
                    {
                        "step": r.step,
                        "layer_idx": r.layer_idx,
                        "head_idx": r.head_idx,
                        "entropy": r.entropy,
                        "coverage": r.coverage,
                        "pattern": r.pattern,
                        "avg_diagonal_weight": r.avg_diagonal_weight,
                        "avg_local_weight": r.avg_local_weight,
                        "concentration": r.concentration,
                        "stability_score": r.stability_score,
                        "change_from_previous": r.change_from_previous,
                    }
                    for r in records
                ]
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load evolution data from JSON.

        Args:
            path: Input file path.
        """
        with self._lock:
            if not os.path.exists(path):
                return
            with open(path, "r") as f:
                data = json.load(f)

            self._checkpoint_steps = data.get("checkpoint_steps", [])
            self._evolution = defaultdict(list)
            for key, records in data.get("evolution", {}).items():
                for r in records:
                    self._evolution[key].append(AttentionEvolution(**r))


# ============================================================================
# TextAttentionRenderer
# ============================================================================

class TextAttentionRenderer:
    """
    Render attention patterns as text heatmaps using Unicode block characters.

    Creates terminal-friendly visualizations of attention weights,
    with options for highlighting, annotations, and token labels.

    Example:
        renderer = TextAttentionRenderer(width=80)
        heatmap = renderer.render(attention_map, tokens)
        print(heatmap)
    """

    def __init__(
        self,
        width: int = 80,
        show_values: bool = False,
        precision: int = 3,
        heat_chars: Optional[List[str]] = None,
    ):
        """Initialize the text attention renderer.

        Args:
            width: Maximum line width.
            show_values: Whether to show numeric values.
            precision: Decimal precision for values.
            heat_chars: Custom heatmap characters.
        """
        self._width = width
        self._show_values = show_values
        self._precision = precision
        self._heat_chars = heat_chars or HEAT_CHARS_8

    def render(
        self,
        attention_map: AttentionMap,
        tokens: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
        title: Optional[str] = None,
    ) -> str:
        """Render an attention map as a text heatmap.

        Args:
            attention_map: AttentionMap to render.
            tokens: Token labels for rows and columns.
            max_rows: Maximum rows to display.
            max_cols: Maximum columns to display.
            title: Optional title.

        Returns:
            Multi-line heatmap string.
        """
        if not attention_map.weights:
            return "(empty attention map)"

        weights = attention_map.weights
        n_rows = len(weights)
        n_cols = len(weights[0]) if weights else 0

        rows = min(n_rows, max_rows or n_rows)
        cols = min(n_cols, max_cols or n_cols)

        q_tokens = tokens or attention_map.query_tokens or [str(i) for i in range(n_rows)]
        k_tokens = tokens or attention_map.key_tokens or [str(i) for i in range(n_cols)]

        lines = []

        if title:
            lines.append(title)

        # Column header
        header = " " * 12 + " "
        col_width = 3
        for c in range(cols):
            label = q_tokens[c] if c < len(q_tokens) else str(c)
            label = label[:col_width].ljust(col_width)
            header += label
        lines.append(header)

        # Separator
        lines.append(" " * 12 + " " + "─" * (cols * col_width))

        # Rows
        for r in range(rows):
            row_label = q_tokens[r] if r < len(q_tokens) else str(r)
            row_label = row_label[:10].rjust(10)
            row_line = f"{row_label} │ "

            for c in range(cols):
                w = weights[r][c] if r < len(weights) and c < len(weights[r]) else 0.0
                char = self._weight_to_char(w)
                if self._show_values and col_width >= 5:
                    val_str = f"{w:.{self._precision}f}"
                    if len(val_str) > col_width:
                        val_str = val_str[:col_width]
                    row_line += val_str.ljust(col_width)
                else:
                    row_line += char * col_width

            lines.append(row_line)

        # Legend
        lines.append("")
        legend = " " * 12 + " " + "Low" + " " * (cols * col_width - 6) + "High"
        lines.append(legend)

        # Scale bar
        scale = " " * 12 + " "
        n_chars = len(self._heat_chars)
        for i in range(min(cols, n_chars * 3)):
            idx = int(i / (cols / n_chars))
            idx = max(0, min(n_chars - 1, idx))
            scale += self._heat_chars[idx] * col_width
        lines.append(scale)

        return "\n".join(lines)

    def render_comparison(
        self,
        map_a: AttentionMap,
        map_b: AttentionMap,
        tokens: Optional[List[str]] = None,
        title: str = "Attention Comparison",
    ) -> str:
        """Render a side-by-side comparison of two attention maps.

        Args:
            map_a: First attention map.
            map_b: Second attention map.
            tokens: Token labels.
            title: Chart title.

        Returns:
            Multi-line comparison string.
        """
        lines = []
        lines.append(title)
        lines.append("")

        half_width = self._width // 2 - 2
        renderer_a = TextAttentionRenderer(width=half_width, heat_chars=self._heat_chars)
        renderer_b = TextAttentionRenderer(width=half_width, heat_chars=self._heat_chars)

        render_a = renderer_a.render(map_a, tokens, title="Map A")
        render_b = renderer_b.render(map_b, tokens, title="Map B")

        lines_a = render_a.split("\n")
        lines_b = render_b.split("\n")

        max_lines = max(len(lines_a), len(lines_b))
        for i in range(max_lines):
            left = lines_a[i] if i < len(lines_a) else ""
            right = lines_b[i] if i < len(lines_b) else ""
            left = left[:half_width].ljust(half_width)
            right = right[:half_width].ljust(half_width)
            lines.append(f"{left} │ {right}")

        return "\n".join(lines)

    def render_multi_head(
        self,
        attention_maps: List[AttentionMap],
        tokens: Optional[List[str]] = None,
        title: str = "Multi-Head Attention",
    ) -> str:
        """Render multiple attention heads in a grid.

        Args:
            attention_maps: List of AttentionMap objects.
            tokens: Token labels.
            title: Chart title.

        Returns:
            Multi-line grid string.
        """
        if not attention_maps:
            return "(no attention maps)"

        cols_per_map = min(25, self._width // max(len(attention_maps), 1) - 2)
        cols_per_map = max(cols_per_map, 10)
        rows_per_map = 10

        lines = []
        lines.append(title)

        # Headers
        headers = []
        for amap in attention_maps:
            header = f"Head {amap.head_idx}"
            headers.append(header.ljust(cols_per_map))
        lines.append(" " * 12 + " ".join(headers))

        # Determine grid dimensions
        available = self._width - 14
        num_cols_grid = max(1, available // (cols_per_map + 1))
        num_rows_grid = (len(attention_maps) + num_cols_grid - 1) // num_cols_grid

        for grid_row in range(num_rows_grid):
            start_idx = grid_row * num_cols_grid
            end_idx = min(start_idx + num_cols_grid, len(attention_maps))

            for grid_col in range(end_idx - start_idx):
                amap = attention_maps[start_idx + grid_col]
                renderer = TextAttentionRenderer(
                    width=cols_per_map, heat_chars=self._heat_chars
                )
                render = renderer.render(
                    amap, tokens,
                    max_rows=rows_per_map,
                    max_cols=cols_per_map // 3,
                )
                lines.append(render)
            lines.append("")

        return "\n".join(lines)

    def render_aggregated(
        self,
        attention_maps: List[AttentionMap],
        tokens: Optional[List[str]] = None,
        method: str = "mean",
        title: str = "Aggregated Attention",
    ) -> str:
        """Render attention aggregated across heads.

        Args:
            attention_maps: List of AttentionMap objects.
            tokens: Token labels.
            method: Aggregation method.
            title: Chart title.

        Returns:
            Multi-line rendered string.
        """
        extractor = AttentionMapExtractor()
        aggregated = extractor.aggregate_heads(attention_maps, method=method)
        return self.render(aggregated, tokens, title=title)

    def _weight_to_char(self, weight: float) -> str:
        """Convert attention weight to heatmap character.

        Args:
            weight: Attention weight (0-1).

        Returns:
            Heatmap character.
        """
        idx = int(weight * (len(self._heat_chars) - 1))
        idx = max(0, min(len(self._heat_chars) - 1, idx))
        return self._heat_chars[idx]

    def render_heatmap_simple(
        self,
        values: List[List[float]],
        row_labels: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        title: Optional[str] = None,
    ) -> str:
        """Render a simple 2D heatmap.

        Args:
            values: 2D array of values.
            row_labels: Row labels.
            col_labels: Column labels.
            title: Title.

        Returns:
            Multi-line heatmap string.
        """
        if not values:
            return "(empty)"

        # Normalize values
        flat = [v for row in values for v in row]
        v_min = min(flat) if flat else 0
        v_max = max(flat) if flat else 1
        if v_max == v_min:
            v_max = v_min + 1.0

        lines = []
        if title:
            lines.append(title)

        # Column labels
        if col_labels:
            header = " " * 15
            for label in col_labels:
                header += label[:3].ljust(3)
            lines.append(header)

        for r, row in enumerate(values):
            row_label = row_labels[r] if row_labels and r < len(row_labels) else str(r)
            row_label = row_label[:12].rjust(12)
            line = f"{row_label} │ "
            for v in row:
                norm = (v - v_min) / (v_max - v_min)
                char = self._weight_to_char(norm)
                line += char * 3
            lines.append(line)

        # Legend
        lines.append(f"  Range: [{v_min:.4f}, {v_max:.4f}]")
        return "\n".join(lines)
