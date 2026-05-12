"""
Model Compression Module
=========================

Production-grade model compression toolkit including weight sharing, low-rank
factorization, hashed embeddings, product quantization, vocabulary pruning,
layer fusion, and compression analysis.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nexus.optimization.optimization_config import CompressionConfig

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# ModelCompressor
# =============================================================================

class ModelCompressor:
    """High-level compression pipeline for neural network models.

    Orchestrates multiple compression techniques to achieve target
    compression ratios while preserving model accuracy.
    """

    def __init__(self, config: CompressionConfig):
        """Initialize model compressor.

        Args:
            config: Compression configuration.
        """
        self.config = config
        self.device = torch.device(config.device) if config.device != "auto" else _get_device()
        self._compression_log: List[Dict[str, Any]] = []

    def compress(
        self,
        model: nn.Module,
        method: str = "low_rank",
        ratio: float = 0.5,
        dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Compress model by target ratio.

        Args:
            model: Model to compress.
            method: Compression method.
            ratio: Target compression ratio.
            dataloader: Optional calibration dataloader.

        Returns:
            Compressed model.
        """
        original_params = sum(p.numel() for p in model.parameters())
        original_size = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.info("Compressing model: %d params, %.2f MB, method=%s, ratio=%.2f",
                    original_params, original_size / (1024**2), method, ratio)

        model = model.eval().to(self.device)

        if method == "low_rank":
            compressor = LowRankFactorization()
            model = compressor.compress_model(model, ratio)
        elif method == "weight_sharing":
            compressor = WeightSharing()
            model = compressor.compress(model)
        elif method == "hash_embedding":
            compressor = HashedEmbedding()
            model = compressor.compress_model(model)
        elif method == "product_quantization":
            compressor = ProductQuantization()
            model = compressor.compress_model(model)
        elif method == "layer_fusion":
            compressor = LayerFusion()
            model = compressor.fuse_model(model)
        elif method == "pruning":
            from nexus.optimization.pruning import MagnitudePruner
            from nexus.optimization.optimization_config import PruningConfig
            prune_config = PruningConfig(sparsity=ratio)
            pruner = MagnitudePruner(prune_config)
            model = pruner.prune_model(model, ratio, dataloader)
        elif method == "quantization":
            from nexus.optimization.quantization_advanced import QuantizationSimulator
            simulator = QuantizationSimulator(bits=max(2, int(32 * (1 - ratio))))
            simulator.simulate_quantization(model)
            model = model
        else:
            logger.warning("Unknown compression method '%s', skipping", method)

        new_params = sum(p.numel() for p in model.parameters())
        actual_ratio = 1.0 - (new_params / max(1, original_params))

        self._compression_log.append({
            "method": method,
            "target_ratio": ratio,
            "actual_ratio": actual_ratio,
            "original_params": original_params,
            "new_params": new_params,
        })

        logger.info("Compression done: ratio=%.2f (%d -> %d params)",
                    actual_ratio, original_params, new_params)

        return model

    def benchmark_compressed(self, model: nn.Module) -> Dict[str, Any]:
        """Compare original vs compressed model metrics.

        Args:
            model: Compressed model.

        Returns:
            Benchmark results.
        """
        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())

        benchmark = {
            "total_parameters": total_params,
            "model_size_bytes": total_size,
            "model_size_mb": total_size / (1024 * 1024),
            "num_layers": sum(1 for _ in model.modules()),
            "compression_log": self._compression_log,
        }

        if self.device.type == "cuda":
            model.eval()
            dummy = torch.randn(1, 16, model.parameters().__next__().shape[-1] if model.parameters().__next__().dim() >= 1 else 16, device=self.device)
            try:
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(10):
                    with torch.no_grad():
                        if dummy.dim() == 2:
                            model(dummy)
                torch.cuda.synchronize()
                benchmark["avg_latency_ms"] = (time.time() - start) / 10 * 1000
            except Exception:
                benchmark["avg_latency_ms"] = -1.0

            benchmark["gpu_memory_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)

        return benchmark

    def export_compressed(
        self,
        model: nn.Module,
        format: str = "pytorch",
        save_path: str = "./compressed_model",
    ) -> str:
        """Export compressed model to various formats.

        Args:
            model: Model to export.
            format: Export format.
            save_path: Save path.

        Returns:
            Path to exported model.
        """
        save_path = os.path.abspath(os.path.expanduser(save_path))
        os.makedirs(save_path, exist_ok=True)

        if format == "pytorch":
            path = os.path.join(save_path, "model.pt")
            torch.save(model.state_dict(), path)
        elif format == "torchscript":
            model.eval()
            model.cpu()
            dummy = torch.randn(1, 16, 768)
            try:
                scripted = torch.jit.trace(model, dummy)
                path = os.path.join(save_path, "model_scripted.pt")
                scripted.save(path)
            except Exception:
                path = os.path.join(save_path, "model.pt")
                torch.save(model.state_dict(), path)
        elif format == "onnx":
            path = os.path.join(save_path, "model.onnx")
            dummy = torch.randn(1, 16, 768)
            try:
                torch.onnx.export(model, dummy, path, opset_version=14)
            except Exception as e:
                logger.warning("ONNX export failed: %s", e)
                path = os.path.join(save_path, "model.pt")
                torch.save(model.state_dict(), path)
        else:
            path = os.path.join(save_path, "model.pt")
            torch.save(model.state_dict(), path)

        logger.info("Exported model to %s", path)
        return path


# =============================================================================
# WeightSharing
# =============================================================================

class WeightSharing:
    """Share weights across layers with similar shapes and patterns.

    Identifies layers with similar weight distributions and ties their
    parameters together to reduce model size.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """Initialize weight sharing.

        Args:
            similarity_threshold: Cosine similarity threshold for sharing.
        """
        self.similarity_threshold = similarity_threshold
        self._shared_pairs: List[Tuple[str, str]] = []

    def find_similar_layers(
        self,
        model: nn.Module,
    ) -> List[Tuple[str, str, float]]:
        """Find layers with similar weight distributions.

        Args:
            model: Model to analyze.

        Returns:
            List of (layer_a, layer_b, similarity) tuples.
        """
        linear_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers[name] = module.weight.data

        similar_pairs = []
        names = sorted(linear_layers.keys())

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name_a, name_b = names[i], names[j]
                weight_a = linear_layers[name_a]
                weight_b = linear_layers[name_b]

                if weight_a.shape != weight_b.shape:
                    continue

                cos_sim = F.cosine_similarity(
                    weight_a.flatten().unsqueeze(0),
                    weight_b.flatten().unsqueeze(0),
                ).item()

                if cos_sim >= self.similarity_threshold:
                    similar_pairs.append((name_a, name_b, cos_sim))

        self._shared_pairs = [(a, b) for a, b, _ in similar_pairs]
        logger.info("Found %d pairs of similar layers", len(similar_pairs))
        return similar_pairs

    def tie_weights(self, layer_a: nn.Module, layer_b: nn.Module):
        """Tie weight tensors between two layers.

        Args:
            layer_a: First layer.
            layer_b: Second layer (will share weights with layer_a).
        """
        layer_b.weight = layer_a.weight
        if layer_a.bias is not None and layer_b.bias is not None:
            layer_b.bias = layer_a.bias

    def compress(self, model: nn.Module) -> nn.Module:
        """Apply weight sharing to model.

        Args:
            model: Model to compress.

        Returns:
            Model with shared weights.
        """
        similar = self.find_similar_layers(model)

        shared_count = 0
        for name_a, name_b, sim in similar:
            module_a = model.get_submodule(name_a)
            module_b = model.get_submodule(name_b)
            self.tie_weights(module_a, module_b)
            shared_count += 1

        logger.info("Shared weights for %d layer pairs", shared_count)
        return model


# =============================================================================
# LowRankFactorization
# =============================================================================

class LowRankFactorization:
    """Factorize weight matrices using SVD and Tucker decomposition.

    Replaces dense weight matrices with low-rank approximations,
    significantly reducing parameter count.
    """

    def __init__(self, rank_ratio: float = 0.5):
        """Initialize low-rank factorization.

        Args:
            rank_ratio: Ratio of target rank to full rank.
        """
        self.rank_ratio = rank_ratio
        self._replaced_layers: List[str] = []

    def svd_compress(self, linear: nn.Linear, rank: int) -> nn.Sequential:
        """Compress a linear layer using SVD.

        W = U @ S @ V^T ≈ U[:,:r] @ diag(S[:r]) @ V^T[:r,:]

        Args:
            linear: Linear layer to compress.
            rank: Target rank.

        Returns:
            Sequential module with two linear layers.
        """
        weight = linear.weight.data.float()
        out_features, in_features = weight.shape

        rank = min(rank, min(out_features, in_features))

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        Vh_trunc = Vh[:rank, :]

        W1 = torch.diag(S_trunc) @ Vh_trunc
        W2 = U_trunc

        layer1 = nn.Linear(in_features, rank, bias=False)
        layer1.weight.data.copy_(W1.to(linear1.weight.dtype))

        layer2 = nn.Linear(rank, out_features, bias=(linear.bias is not None))
        layer2.weight.data.copy_(W2.to(layer2.weight.dtype))
        if linear.bias is not None:
            layer2.bias.data.copy_(linear.bias.data)

        compressed = nn.Sequential(layer1, layer2)
        return compressed

    def tucker_compress(self, conv: nn.Conv2d, ranks: Tuple[int, int]) -> nn.Sequential:
        """Compress a convolutional layer using Tucker decomposition.

        Args:
            conv: Conv2d layer to compress.
            ranks: Target (output rank, input rank).

        Returns:
            Sequential module with decomposed convolutions.
        """
        weight = conv.weight.data.float()
        out_ch, in_ch, kh, kw = weight.shape

        rank_out, rank_in = ranks
        rank_out = min(rank_out, out_ch)
        rank_in = min(rank_in, in_ch)

        W = weight.reshape(out_ch, in_ch, kh * kw)
        W_2d = W.reshape(out_ch, -1)

        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        core = (U[:, :rank_out].T @ W_2d @ Vh.T[:, :rank_in]).reshape(rank_out, rank_in, kh, kw)

        conv1 = nn.Conv2d(in_ch, rank_in, kernel_size=(kh, kw), bias=False, padding=conv.padding)
        conv2 = nn.Conv2d(rank_in, rank_out, kernel_size=1, bias=False)
        conv3 = nn.Conv2d(rank_out, out_ch, kernel_size=1, bias=(conv.bias is not None))

        conv1.weight.data.copy_(Vh[:rank_in].reshape(rank_in, in_ch, kh, kw).to(conv1.weight.dtype))
        conv2.weight.data.copy_(torch.eye(rank_in, rank_out).unsqueeze(-1).unsqueeze(-1).to(conv2.weight.dtype))
        conv3.weight.data.copy_(U[:, :rank_out].reshape(out_ch, rank_out, 1, 1).to(conv3.weight.dtype))
        if conv.bias is not None:
            conv3.bias.data.copy_(conv.bias.data)

        return nn.Sequential(conv1, conv2, conv3)

    def compress_model(self, model: nn.Module, ratio: float = 0.5) -> nn.Module:
        """Compress all linear layers in the model.

        Args:
            model: Model to compress.
            ratio: Compression ratio.

        Returns:
            Compressed model.
        """
        model = model.to("cpu")
        compress_count = 0

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                out_features, in_features = module.weight.shape
                full_rank = min(out_features, in_features)
                target_rank = max(1, int(full_rank * (1.0 - ratio)))

                if target_rank >= full_rank:
                    continue

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                compressed = self.svd_compress(module, target_rank)
                setattr(parent, child_name, compressed)
                self._replaced_layers.append(name)
                compress_count += 1

        logger.info("Compressed %d layers with low-rank factorization", compress_count)
        return model


# =============================================================================
# HashedEmbedding
# =============================================================================

class HashedEmbedding(nn.Module):
    """Hash-based embedding compression.

    Uses a hash function to map a large vocabulary into a smaller
    embedding table, dramatically reducing memory.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_hashes: int,
        hash_buckets: int,
        device: Optional[torch.device] = None,
    ):
        """Initialize hashed embedding.

        Args:
            num_embeddings: Original vocabulary size.
            embedding_dim: Embedding dimension.
            num_hashes: Number of hash functions.
            hash_buckets: Number of hash buckets.
            device: Target device.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        self.hash_buckets = hash_buckets

        self.embedding_table = nn.Embedding(hash_buckets, embedding_dim)
        self.hash_weights = nn.Parameter(
            torch.randn(num_hashes, embedding_dim) * 0.01
        )

        if device is not None:
            self.to(device)

    def _hash_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute hash bucket indices for input IDs.

        Args:
            input_ids: Input token IDs.

        Returns:
            Hash bucket indices.
        """
        batch_size, seq_len = input_ids.shape
        hashes = []

        for h in range(self.num_hashes):
            seed = torch.tensor([h * 1337], dtype=torch.long, device=input_ids.device)
            hash_val = (input_ids * (h + 1) * 2654435761) % self.hash_buckets
            hashes.append(hash_val)

        stacked = torch.stack(hashes, dim=-1)
        primary_hash = stacked[:, :, 0]
        return primary_hash

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs.

        Returns:
            Embedding vectors.
        """
        indices = self._hash_indices(input_ids)
        return self.embedding_table(indices)


class HashedEmbeddingCompressor:
    """Compress embedding layers using hashing."""

    def __init__(self, hash_ratio: float = 0.25):
        """Initialize compressor.

        Args:
            hash_ratio: Ratio of hash table size to original size.
        """
        self.hash_ratio = hash_ratio

    def compress_model(self, model: nn.Module) -> nn.Module:
        """Compress all embedding layers.

        Args:
            model: Model to compress.

        Returns:
            Model with hashed embeddings.
        """
        compress_count = 0

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Embedding):
                num_emb = module.num_embeddings
                emb_dim = module.embedding_dim
                hash_buckets = max(256, int(num_emb * self.hash_ratio))
                num_hashes = 2

                hashed = HashedEmbedding(
                    num_emb, emb_dim, num_hashes, hash_buckets
                )

                hashed.embedding_table.weight.data.copy_(
                    module.weight[:hash_buckets].data
                )

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, hashed)
                compress_count += 1

        logger.info("Compressed %d embedding layers with hashing", compress_count)
        return model


# =============================================================================
# ProductQuantization
# =============================================================================

class ProductQuantization:
    """Compress embeddings using product quantization.

    Splits embedding vectors into subvectors and quantizes each
    subvector independently.
    """

    def __init__(self, num_subvectors: int = 8, codebook_size: int = 256):
        """Initialize product quantization.

        Args:
            num_subvectors: Number of subvectors per embedding.
            codebook_size: Size of each sub-codebook.
        """
        self.num_subvectors = num_subvectors
        self.codebook_size = codebook_size

    def compress_model(self, model: nn.Module) -> nn.Module:
        """Compress embedding layers with product quantization.

        Args:
            model: Model to compress.

        Returns:
            Model with PQ-compressed embeddings.
        """
        compress_count = 0

        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Embedding):
                pq_layer = ProductQuantizedEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    self.num_subvectors,
                    self.codebook_size,
                )
                pq_layer.train_codebooks(module.weight.data)

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, pq_layer)
                compress_count += 1

        logger.info("Compressed %d embedding layers with product quantization", compress_count)
        return model


class ProductQuantizedEmbedding(nn.Module):
    """Embedding layer with product quantization."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_subvectors: int,
        codebook_size: int,
    ):
        """Initialize PQ embedding.

        Args:
            num_embeddings: Number of embeddings.
            embedding_dim: Embedding dimension.
            num_subvectors: Number of subvectors.
            codebook_size: Codebook size per subvector.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_subvectors = num_subvectors
        self.codebook_size = codebook_size
        self.sub_dim = embedding_dim // num_subvectors

        self.codebooks = nn.Parameter(
            torch.randn(num_subvectors, codebook_size, self.sub_dim) * 0.01
        )
        self.codes = nn.Parameter(
            torch.randint(0, codebook_size, (num_embeddings, num_subvectors)),
            requires_grad=False,
        )

    def train_codebooks(self, original_weight: torch.Tensor, num_iterations: int = 20):
        """Train codebooks using k-means on original embeddings.

        Args:
            original_weight: Original embedding matrix.
            num_iterations: Number of k-means iterations.
        """
        weight = original_weight.float()
        n, d = weight.shape

        for s in range(self.num_subvectors):
            start = s * self.sub_dim
            end = start + self.sub_dim
            sub_weights = weight[:, start:end]

            centroids = sub_weights[torch.randperm(n)[:self.codebook_size]].clone()

            for iteration in range(num_iterations):
                dists = torch.cdist(sub_weights.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
                assignments = dists.argmin(dim=1)
                new_centroids = torch.zeros_like(centroids)

                for c in range(self.codebook_size):
                    mask = assignments == c
                    if mask.sum() > 0:
                        new_centroids[c] = sub_weights[mask].mean(dim=0)
                    else:
                        new_centroids[c] = centroids[c]

                centroids = new_centroids

            self.codebooks.data[s].copy_(centroids)
            dists = torch.cdist(sub_weights.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
            self.codes.data[:, s] = dists.argmin(dim=1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with PQ lookup.

        Args:
            input_ids: Input token IDs.

        Returns:
            Reconstructed embeddings.
        """
        codes = self.codes[input_ids.long()]
        result = torch.zeros(
            input_ids.shape[0], input_ids.shape[1], self.embedding_dim,
            device=codes.device, dtype=self.codebooks.dtype,
        )

        for s in range(self.num_subvectors):
            sub_codes = codes[:, :, s]
            sub_vectors = self.codebooks[s][sub_codes]
            start = s * self.sub_dim
            end = start + self.sub_dim
            result[:, :, start:end] = sub_vectors

        return result


# =============================================================================
# VocabularyPruning
# =============================================================================

class VocabularyPruning:
    """Reduce vocabulary size by removing rare tokens.

    Analyzes token usage and prunes rare tokens to reduce embedding
    table size.
    """

    def __init__(self, min_freq: int = 2, target_size: int = 32000):
        """Initialize vocabulary pruning.

        Args:
            min_freq: Minimum token frequency to keep.
            target_size: Target vocabulary size.
        """
        self.min_freq = min_freq
        self.target_size = target_size

    def count_token_frequencies(
        self,
        dataloader: DataLoader,
        token_key: str = "input_ids",
    ) -> Dict[int, int]:
        """Count token frequencies from a dataloader.

        Args:
            dataloader: Data to analyze.
            token_key: Key for token IDs in batch dict.

        Returns:
            Dictionary mapping token ID to frequency count.
        """
        freq = defaultdict(int)

        for batch in dataloader:
            if isinstance(batch, dict):
                tokens = batch.get(token_key)
            elif isinstance(batch, (list, tuple)):
                tokens = batch[0]
            else:
                continue

            if isinstance(tokens, torch.Tensor):
                tokens = tokens.flatten()
                for token_id in tokens.unique():
                    freq[token_id.item()] += (tokens == token_id).sum().item()

        return dict(freq)

    def prune_vocabulary(
        self,
        model: nn.Module,
        token_frequencies: Dict[int, int],
    ) -> Tuple[nn.Module, Dict[int, int]]:
        """Prune vocabulary based on token frequencies.

        Args:
            model: Model with embedding layers.
            token_frequencies: Token frequency dictionary.

        Returns:
            Tuple of (pruned model, old_to_new_id mapping).
        """
        sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
        keep_tokens = set()

        target = min(self.target_size, len(sorted_tokens))
        for i in range(target):
            if sorted_tokens[i][1] >= self.min_freq:
                keep_tokens.add(sorted_tokens[i][0])

        if not keep_tokens:
            keep_tokens = {t for t, f in sorted_tokens[:1000]}

        old_to_new = {old: new for new, old in enumerate(sorted(keep_tokens))}
        max_new_id = max(old_to_new.values()) + 1

        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                old_weight = module.weight.data
                new_num_embeddings = max_new_id
                new_weight = torch.zeros(
                    new_num_embeddings, module.embedding_dim,
                    dtype=old_weight.dtype, device=old_weight.device,
                )

                for old_id, new_id in old_to_new.items():
                    if old_id < old_weight.shape[0]:
                        new_weight[new_id] = old_weight[old_id]

                module.num_embeddings = new_num_embeddings
                module.weight = nn.Parameter(new_weight)

            elif isinstance(module, nn.Linear) and module.out_features == len(token_frequencies):
                old_weight = module.weight.data
                new_weight = torch.zeros(
                    module.out_features, module.in_features,
                    dtype=old_weight.dtype, device=old_weight.device,
                )
                for old_id, new_id in old_to_new.items():
                    if old_id < old_weight.shape[0]:
                        new_weight[new_id] = old_weight[old_id]
                module.weight = nn.Parameter(new_weight)

        logger.info("Pruned vocabulary from %d to %d tokens",
                    len(token_frequencies), max_new_id)
        return model, old_to_new


# =============================================================================
# LayerFusion
# =============================================================================

class LayerFusion:
    """Fuse consecutive layers for faster inference.

    Combines consecutive linear+activation, linear+normalization, etc.
    into single operations.
    """

    def __init__(self, merge_threshold: float = 0.95):
        """Initialize layer fusion.

        Args:
            merge_threshold: Similarity threshold for merging.
        """
        self.merge_threshold = merge_threshold
        self._fused_pairs: List[str] = []

    def fuse_linear_relu(self, linear: nn.Linear, relu: nn.ReLU) -> nn.Linear:
        """Fuse linear layer with ReLU (pre-activation) by noting ReLU can be
        absorbed into the next operation.

        Args:
            linear: Linear layer.
            relu: ReLU activation.

        Returns:
            Fused linear layer.
        """
        return linear

    def fuse_linear_batchnorm(self, linear: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
        """Fuse linear layer with BatchNorm.

        W_fused = W * diag(gamma / sqrt(var + eps))
        b_fused = gamma * (b - mu) / sqrt(var + eps) + beta

        Args:
            linear: Linear layer.
            bn: BatchNorm layer.

        Returns:
            Fused linear layer.
        """
        weight = linear.weight.data.float()
        bias = linear.bias.data.float() if linear.bias is not None else torch.zeros(linear.out_features)

        gamma = bn.weight.data.float()
        beta = bn.bias.data.float()
        mu = bn.running_mean.float()
        var = bn.running_var.float()
        eps = bn.eps

        scale = gamma / torch.sqrt(var + eps)

        if weight.dim() == 2:
            fused_weight = weight * scale.unsqueeze(1)
        else:
            fused_weight = weight * scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        fused_bias = scale * (bias - mu) + beta

        fused = nn.Linear(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None or bn.bias is not None,
        )
        fused.weight.data.copy_(fused_weight.to(linear.weight.dtype))
        fused.bias = nn.Parameter(fused_bias.to(linear.bias.dtype if linear.bias is not None else torch.float32))

        return fused

    def fuse_linear_layernorm(self, linear: nn.Linear, ln: nn.LayerNorm) -> Tuple[nn.Linear, nn.LayerNorm]:
        """Partially fuse linear + LayerNorm.

        Cannot fully fuse due to data-dependent normalization, but can
        pre-compute the weight scaling.

        Args:
            linear: Linear layer.
            ln: LayerNorm layer.

        Returns:
            Tuple of (adjusted linear, adjusted layernorm).
        """
        weight = linear.weight.data.float()
        ln_weight = ln.weight.data.float()

        if weight.dim() == 2 and ln_weight.dim() == 1:
            adjusted_weight = weight * ln_weight.unsqueeze(0)
            linear.weight.data.copy_(adjusted_weight.to(linear.weight.dtype))
            ln.weight.data.fill_(1.0)

        return linear, ln

    def fuse_model(self, model: nn.Module) -> nn.Module:
        """Fuse applicable layers in the model.

        Args:
            model: Model to fuse.

        Returns:
            Model with fused layers.
        """
        fused_count = 0

        for name, module in list(model.named_modules()):
            children = list(module.named_children())
            child_names = [n for n, _ in children]

            for i in range(len(children) - 1):
                child_name_a, child_a = children[i]
                child_name_b, child_b = children[i + 1]

                if isinstance(child_a, nn.Linear) and isinstance(child_b, nn.BatchNorm1d):
                    fused = self.fuse_linear_batchnorm(child_a, child_b)
                    setattr(module, child_name_a, fused)
                    setattr(module, child_name_b, nn.Identity())
                    self._fused_pairs.append(f"{name}.{child_name_a}+{child_name_b}")
                    fused_count += 1

                elif isinstance(child_a, nn.Linear) and isinstance(child_b, nn.ReLU):
                    setattr(module, child_name_b, nn.Identity())
                    self._fused_pairs.append(f"{name}.{child_name_a}+{child_name_b}")
                    fused_count += 1

                elif isinstance(child_a, nn.Linear) and isinstance(child_b, nn.LayerNorm):
                    self.fuse_linear_layernorm(child_a, child_b)
                    self._fused_pairs.append(f"{name}.{child_name_a}+{child_name_b}(partial)")
                    fused_count += 1

        logger.info("Fused %d layer pairs", fused_count)
        return model


# =============================================================================
# CompressionAnalyzer
# =============================================================================

class CompressionAnalyzer:
    """Analyze compression opportunities and estimate savings."""

    def __init__(self):
        """Initialize analyzer."""
        self._analysis_cache: Dict[str, Dict[str, Any]] = {}

    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """Comprehensive compression analysis.

        Args:
            model: Model to analyze.

        Returns:
            Analysis report.
        """
        layer_analysis = []
        total_params = 0
        total_compressible = 0

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                params = module.weight.numel()
                total_params += params
                if module.bias is not None:
                    params += module.bias.numel()

                rank = min(module.out_features, module.in_features)
                svd_savings = params - (module.in_features + module.out_features + 1) * rank
                svd_ratio = svd_savings / max(1, params)

                q4_size = math.ceil(module.weight.numel() * 4 / 8)
                q4_ratio = 1.0 - q4_size / max(1, module.weight.numel() * 4)

                layer_info = {
                    "name": name,
                    "type": "linear",
                    "shape": list(module.weight.shape),
                    "params": params,
                    "svd_compression_ratio": svd_ratio,
                    "q4_compression_ratio": q4_ratio,
                    "recommended": "svd" if svd_ratio > 0.5 else "quantization",
                }
                layer_analysis.append(layer_info)
                total_compressible += params

            elif isinstance(module, nn.Embedding):
                params = module.num_embeddings * module.embedding_dim
                total_params += params

                layer_info = {
                    "name": name,
                    "type": "embedding",
                    "shape": [module.num_embeddings, module.embedding_dim],
                    "params": params,
                    "pq_compression_ratio": 0.75,
                    "hash_compression_ratio": 0.9,
                    "recommended": "product_quantization",
                }
                layer_analysis.append(layer_info)
                total_compressible += params

            elif isinstance(module, nn.Conv2d):
                params = module.weight.numel()
                total_params += params

                layer_info = {
                    "name": name,
                    "type": "conv2d",
                    "shape": list(module.weight.shape),
                    "params": params,
                    "tucker_compression_ratio": 0.5,
                    "recommended": "tucker",
                }
                layer_analysis.append(layer_info)
                total_compressible += params

        total_size_mb = total_params * 4 / (1024 * 1024)
        compressible_ratio = total_compressible / max(1, total_params)

        return {
            "total_parameters": total_params,
            "total_size_mb": total_size_mb,
            "compressible_ratio": compressible_ratio,
            "layers": layer_analysis,
            "recommendations": self._generate_recommendations(layer_analysis),
        }

    def estimate_compression_savings(
        self,
        model: nn.Module,
        methods: List[str],
        ratios: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Estimate savings from different compression methods.

        Args:
            model: Model to analyze.
            methods: List of compression methods.
            ratios: Optional ratios per method.

        Returns:
            Dictionary mapping method to estimated size in MB.
        """
        if ratios is None:
            ratios = [0.5] * len(methods)

        total_params = sum(p.numel() for p in model.parameters())
        original_size = total_params * 4 / (1024 * 1024)
        savings = {"original_mb": original_size}

        for method, ratio in zip(methods, ratios):
            if method == "low_rank":
                compressed = total_params * ratio * 2
            elif method == "quantization_4bit":
                compressed = total_params * 0.5
            elif method == "quantization_8bit":
                compressed = total_params * 1.0
            elif method == "pruning":
                compressed = total_params * (1 - ratio) * 0.25 + total_params * ratio * 0.01
            elif method == "hash_embedding":
                embed_params = sum(
                    m.num_embeddings * m.embedding_dim
                    for m in model.modules()
                    if isinstance(m, nn.Embedding)
                )
                other_params = total_params - embed_params
                compressed = embed_params * ratio + other_params
            else:
                compressed = total_params * (1 - ratio)

            savings[f"{method}_mb"] = compressed / (1024 * 1024)

        return savings

    def _generate_recommendations(self, layers: List[Dict[str, Any]]) -> List[str]:
        """Generate compression recommendations.

        Args:
            layers: Layer analysis results.

        Returns:
            List of recommendation strings.
        """
        recommendations = []
        linear_layers = [l for l in layers if l["type"] == "linear"]
        embedding_layers = [l for l in layers if l["type"] == "embedding"]

        if linear_layers:
            avg_svd = sum(l["svd_compression_ratio"] for l in linear_layers) / len(linear_layers)
            if avg_svd > 0.5:
                recommendations.append("Low-rank SVD: High compression potential (avg ratio: {:.1%})".format(avg_svd))

            avg_q4 = sum(l["q4_compression_ratio"] for l in linear_layers) / len(linear_layers)
            if avg_q4 > 0.5:
                recommendations.append("4-bit quantization: Significant savings ({:.1%} size reduction)".format(avg_q4))

        if embedding_layers:
            recommendations.append(f"Product quantization: {len(embedding_layers)} embedding layers can be compressed")

        if not recommendations:
            recommendations.append("Model is already well-optimized or very small")

        return recommendations
