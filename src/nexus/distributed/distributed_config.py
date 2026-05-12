# Copyright (c) 2024 Nexus LLM Contributors
# SPDX-License-Identifier: Apache-2.0
"""Distributed training configuration module.

Provides comprehensive configuration dataclasses for all distributed training
aspects: tensor parallelism, pipeline parallelism, data parallelism (DDP/FSDP/ZeRO),
elastic scaling, fault tolerance, communication backends, networking, checkpointing,
and debugging. All configs support serialization to/from YAML, JSON, and dicts.
"""

from __future__ import annotations

import copy
import enum
import json
import os
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


# ==============================================================================
# Enums
# ==============================================================================


class ParallelBackend(enum.Enum):
    """Supported distributed communication backends."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    AUTO = "auto"


class DataParallelStrategy(enum.Enum):
    """Data parallelism implementation strategies."""
    DDP = "ddp"
    FSDP = "fsdp"
    ZERO_STAGE_1 = "zero_stage_1"
    ZERO_STAGE_2 = "zero_stage_2"
    ZERO_STAGE_3 = "zero_stage_3"
    HYBRID = "hybrid"


class ShardingStrategy(enum.Enum):
    """FSDP/ZeRO parameter sharding strategies."""
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"


class PipelineScheduleType(enum.Enum):
    """Pipeline parallelism scheduling strategies."""
    ONE_F_ONE_B = "1f1b"
    INTERLEAVED = "interleaved"
    FORWARD_ONLY = "forward_only"
    MANUAL = "manual"


class ScalingPolicy(enum.Enum):
    """Elastic scaling policies."""
    MANUAL = "manual"
    AUTO_SCALE_UP = "auto_scale_up"
    AUTO_SCALE_DOWN = "auto_scale_down"
    ADAPTIVE = "adaptive"


class RecoveryStrategy(enum.Enum):
    """Failure recovery strategies."""
    RESTART_ALL = "restart_all"
    RESTART_FAILED = "restart_failed"
    ROLLBACK = "rollback"
    CHECKPOINT_RESUME = "checkpoint_resume"


class CommunicationCompression(enum.Enum):
    """Communication compression types."""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    POW2 = "power_of_two"


class TensorParallelMode(enum.Enum):
    """Tensor parallelism splitting modes."""
    COLUMN = "column"
    ROW = "row"
    BOTH = "both"


class GradientReduceOp(enum.Enum):
    """Gradient reduction operations."""
    SUM = "sum"
    MEAN = "mean"
    AVG = "avg"


class CheckpointFormat(enum.Enum):
    """Distributed checkpoint storage formats."""
    TORCH = "torch"
    SAFETENSORS = "safetensors"
    ZARR = "zarr"
    HF = "huggingface"


class LogLevel(enum.Enum):
    """Logging verbosity levels for distributed operations."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ==============================================================================
# Network Configuration
# ==============================================================================


@dataclass
class NetworkConfig:
    """Network configuration for distributed training.

    Attributes:
        master_addr: Address of the master node for rendezvous.
        master_port: Port of the master node for rendezvous.
        interface: Network interface to bind to (e.g. 'eth0', 'ib0').
        bind_timeout: Timeout in seconds for binding to the network interface.
        reconnect_attempts: Number of reconnection attempts on failure.
        reconnect_delay: Delay in seconds between reconnection attempts.
        tcp_buffer_size: TCP buffer size in bytes for socket communication.
        rdma: Whether to enable RDMA for communication (if available).
        rdma_port: RDMA port for InfiniBand communication.
        nvlink: Whether to enable NVLink for intra-node communication.
        enable_gdr: Whether to enable GPU Direct RDMA.
        use_ib verbs: Whether to use InfiniBand verbs for NCCL.
    """
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    interface: str = ""
    bind_timeout: float = 30.0
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    tcp_buffer_size: int = 16 * 1024 * 1024  # 16 MB
    rdma: bool = False
    rdma_port: int = 18515
    nvlink: bool = True
    enable_gdr: bool = False
    use_ib_verbs: bool = False

    def __post_init__(self):
        if self.master_port < 1 or self.master_port > 65535:
            raise ValueError(
                f"master_port must be between 1 and 65535, got {self.master_port}"
            )
        if self.bind_timeout <= 0:
            raise ValueError(
                f"bind_timeout must be positive, got {self.bind_timeout}"
            )
        if self.reconnect_delay < 0:
            raise ValueError(
                f"reconnect_delay must be non-negative, got {self.reconnect_delay}"
            )

    def get_init_method(self) -> str:
        """Construct a TCP init_method URL from master address and port."""
        return f"tcp://{self.master_addr}:{self.master_port}"

    def find_free_port(self) -> int:
        """Find a free port on the current machine."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def detect_interface(self) -> str:
        """Auto-detect the best network interface for distributed communication."""
        if self.interface:
            return self.interface
        interfaces = self._get_network_interfaces()
        preferred = ["ib0", "ib1", "ib2", "ens", "enp", "eth0", "eth1", "eth2"]
        for pref in preferred:
            for iface in interfaces:
                if iface.startswith(pref):
                    return iface
        if interfaces:
            return interfaces[0]
        return "lo"

    @staticmethod
    def _get_network_interfaces() -> List[str]:
        """Get a list of active network interfaces."""
        interfaces = []
        try:
            import psutil
            addrs = psutil.net_if_addrs()
            for name, addr_list in addrs.items():
                for addr in addr_list:
                    if addr.family == socket.AF_INET and addr.address != "127.0.0.1":
                        interfaces.append(name)
                        break
        except ImportError:
            try:
                import netifaces
                for iface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(iface)
                    if netifaces.AF_INET in addrs:
                        for addr in addrs[netifaces.AF_INET]:
                            if addr.get("addr", "") != "127.0.0.1":
                                interfaces.append(iface)
                                break
            except ImportError:
                pass
        return interfaces

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NetworkConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "NetworkConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Process Group Configuration
# ==============================================================================


@dataclass
class ProcessGroupConfig:
    """Configuration for a specific torch.distributed process group.

    Attributes:
        name: Human-readable name for the process group.
        ranks: List of global ranks in this process group.
        backend: Communication backend for this group.
        timeout: Timeout for collective operations in this group.
        is_default: Whether this is the default (world) process group.
    """
    name: str = "default"
    ranks: List[int] = field(default_factory=list)
    backend: ParallelBackend = ParallelBackend.AUTO
    timeout: float = 30.0
    is_default: bool = True

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError(
                f"Process group timeout must be positive, got {self.timeout}"
            )
        if self.ranks and len(set(self.ranks)) != len(self.ranks):
            raise ValueError("Process group ranks must be unique")

    @property
    def size(self) -> int:
        """Number of ranks in this process group."""
        return len(self.ranks) if self.ranks else 0

    def contains_rank(self, rank: int) -> bool:
        """Check if a rank belongs to this process group."""
        return rank in self.ranks

    def get_local_rank(self, global_rank: int) -> int:
        """Get the local rank within this process group."""
        if global_rank not in self.ranks:
            raise ValueError(
                f"Global rank {global_rank} is not in process group {self.name}"
            )
        return self.ranks.index(global_rank)

    def get_global_rank(self, local_rank: int) -> int:
        """Get the global rank given a local rank in this group."""
        if local_rank < 0 or local_rank >= len(self.ranks):
            raise ValueError(
                f"Local rank {local_rank} is out of range for group {self.name} "
                f"with size {len(self.ranks)}"
            )
        return self.ranks[local_rank]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProcessGroupConfig":
        return cls(
            name=d.get("name", "default"),
            ranks=d.get("ranks", []),
            backend=ParallelBackend(d["backend"]) if "backend" in d else ParallelBackend.AUTO,
            timeout=d.get("timeout", 30.0),
            is_default=d.get("is_default", True),
        )


# ==============================================================================
# Tensor Parallel Configuration
# ==============================================================================


@dataclass
class TensorParallelConfig:
    """Configuration for tensor parallelism.

    Tensor parallelism splits individual weight matrices across GPUs, where
    each GPU computes a subset of the output. This is effective for large
    linear layers in transformer models.

    Attributes:
        enabled: Whether tensor parallelism is enabled.
        size: Degree of tensor parallelism (number of GPUs for TP).
        mode: Splitting mode — column splits output dim, row splits input dim.
        overlap_communication: Whether to overlap all-reduce communication with
            computation to hide latency.
        memory_fraction: Fraction of GPU memory reserved for tensor parallel
            activations and buffers.
        enable_sequence_parallelism: Whether to partition sequence dimension.
        enable_custom_all_reduce: Whether to use custom all-reduce kernels.
        all_reduce_algorithm: Algorithm for all-reduce (ring, tree, mesh).
        enable_flash_attention: Whether to use flash attention in TP attention.
        use_qkv_fusion: Whether to fuse QKV projections for efficiency.
        use_output_gelu_fusion: Whether to fuse output projection + GeLU.
        compute_dtype: Data type for TP computations.
        param_dtype: Data type for TP parameter storage.
        enable_async_tensor_parallel: Enable async TP communication.
        tp_comm_overlap_chunk_size: Chunk size for communication overlap.
        tp_comm_overlap_num_chunks: Number of chunks for overlapped communication.
    """
    enabled: bool = False
    size: int = 1
    mode: TensorParallelMode = TensorParallelMode.COLUMN
    overlap_communication: bool = True
    memory_fraction: float = 0.15
    enable_sequence_parallelism: bool = False
    enable_custom_all_reduce: bool = False
    all_reduce_algorithm: str = "ring"
    enable_flash_attention: bool = True
    use_qkv_fusion: bool = True
    use_output_gelu_fusion: bool = True
    compute_dtype: str = "float32"
    param_dtype: str = "float32"
    enable_async_tensor_parallel: bool = False
    tp_comm_overlap_chunk_size: int = 8192
    tp_comm_overlap_num_chunks: int = 4

    def __post_init__(self):
        if self.size < 1:
            raise ValueError(f"TP size must be >= 1, got {self.size}")
        if not self.enabled and self.size > 1:
            warnings.warn(
                f"TP size={self.size} > 1 but TP is not enabled. Enabling TP."
            )
            self.enabled = True
        if self.memory_fraction <= 0 or self.memory_fraction > 1.0:
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {self.memory_fraction}"
            )
        if self.tp_comm_overlap_chunk_size < 1:
            raise ValueError(
                f"tp_comm_overlap_chunk_size must be >= 1, "
                f"got {self.tp_comm_overlap_chunk_size}"
            )
        if self.tp_comm_overlap_num_chunks < 1:
            raise ValueError(
                f"tp_comm_overlap_num_chunks must be >= 1, "
                f"got {self.tp_comm_overlap_num_chunks}"
            )

    @property
    def is_column(self) -> bool:
        return self.mode == TensorParallelMode.COLUMN

    @property
    def is_row(self) -> bool:
        return self.mode == TensorParallelMode.ROW

    @property
    def effective_size(self) -> int:
        return self.size if self.enabled else 1

    def validate_model_dim(self, model_dim: int) -> None:
        """Validate that model dimensions are divisible by TP size."""
        if self.enabled and model_dim % self.size != 0:
            raise ValueError(
                f"Model dimension {model_dim} is not divisible by "
                f"TP size {self.size}"
            )

    def shard_dim(self, dim: int) -> int:
        """Calculate the sharded dimension size."""
        if not self.enabled or self.size == 1:
            return dim
        if dim % self.size != 0:
            raise ValueError(
                f"Dimension {dim} is not divisible by TP size {self.size}"
            )
        return dim // self.size

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorParallelConfig":
        d = copy.deepcopy(d)
        if "mode" in d and isinstance(d["mode"], str):
            d["mode"] = TensorParallelMode(d["mode"])
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TensorParallelConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Pipeline Parallel Configuration
# ==============================================================================


@dataclass
class PipelineParallelConfig:
    """Configuration for pipeline parallelism.

    Pipeline parallelism assigns different layers of a model to different GPUs,
    creating a pipeline where each GPU processes a stage. Micro-batching allows
    multiple mini-batches to flow through the pipeline concurrently.

    Attributes:
        enabled: Whether pipeline parallelism is enabled.
        stages: Number of pipeline stages (GPUs for PP).
        micro_batches: Number of micro-batches per mini-batch.
        schedule: Pipeline scheduling algorithm.
        gradient_accumulation: Number of gradient accumulation steps per
            optimizer update.
        enable_interleaved: Whether to use interleaved pipeline scheduling.
        interleaved_chunks: Number of virtual chunks per stage for interleaved.
        recompute_policy: Activation recomputation policy ('full', 'selective',
            'none').
        recompute_num_layers: Number of layers to recompute in selective mode.
        recompute_granularity: Recompute granularity ('full_layer', 'block',
            'activation').
        enable_comm_overlap: Whether to overlap stage-to-stage communication
            with computation.
        first_stage_id: ID of the first pipeline stage.
        last_stage_id: ID of the last pipeline stage.
        pipeline_model_parallel_size: Deprecated alias for stages.
    """
    enabled: bool = False
    stages: int = 1
    micro_batches: int = 1
    schedule: PipelineScheduleType = PipelineScheduleType.ONE_F_ONE_B
    gradient_accumulation: int = 1
    enable_interleaved: bool = False
    interleaved_chunks: int = 1
    recompute_policy: str = "selective"
    recompute_num_layers: int = 1
    recompute_granularity: str = "full_layer"
    enable_comm_overlap: bool = True
    first_stage_id: int = 0
    last_stage_id: int = 0
    pipeline_model_parallel_size: int = 0

    def __post_init__(self):
        if self.stages < 1:
            raise ValueError(f"Pipeline stages must be >= 1, got {self.stages}")
        if self.micro_batches < 1:
            raise ValueError(
                f"Micro-batches must be >= 1, got {self.micro_batches}"
            )
        if self.gradient_accumulation < 1:
            raise ValueError(
                f"Gradient accumulation must be >= 1, got {self.gradient_accumulation}"
            )
        if self.recompute_policy not in ("full", "selective", "none"):
            raise ValueError(
                f"Invalid recompute_policy: {self.recompute_policy}. "
                f"Must be 'full', 'selective', or 'none'"
            )
        if self.recompute_granularity not in ("full_layer", "block", "activation"):
            raise ValueError(
                f"Invalid recompute_granularity: {self.recompute_granularity}"
            )
        if self.interleaved_chunks < 1:
            raise ValueError(
                f"interleaved_chunks must be >= 1, got {self.interleaved_chunks}"
            )
        if self.pipeline_model_parallel_size > 0 and self.stages == 1:
            self.stages = self.pipeline_model_parallel_size
        self.last_stage_id = self.stages - 1

    @property
    def is_first_stage(self) -> bool:
        """Whether the current rank is the first pipeline stage."""
        return self.first_stage_id == 0

    @property
    def is_last_stage(self) -> bool:
        """Whether the current rank is the last pipeline stage."""
        return self.last_stage_id == self.stages - 1

    @property
    def effective_stages(self) -> int:
        return self.stages if self.enabled else 1

    @property
    def total_micro_batches(self) -> int:
        return self.micro_batches * self.gradient_accumulation

    @property
    def bubble_fraction(self) -> float:
        """Estimate the pipeline bubble fraction.

        For 1F1B schedule with p stages and m micro-batches:
        bubble = (p - 1) / (p - 1 + m)
        """
        if not self.enabled or self.stages <= 1 or self.micro_batches <= 0:
            return 0.0
        if self.schedule == PipelineScheduleType.ONE_F_ONE_B:
            p = self.stages
            m = self.micro_batches
            return (p - 1) / (p - 1 + m)
        elif self.schedule == PipelineScheduleType.INTERLEAVED:
            p = self.stages
            m = self.micro_batches
            v = self.interleaved_chunks
            return (p - 1) / (p - 1 + m * v)
        return 0.0

    def validate_model_layers(self, num_layers: int) -> None:
        """Validate that model layers can be evenly distributed."""
        if self.enabled and num_layers < self.stages:
            raise ValueError(
                f"Number of model layers ({num_layers}) must be >= "
                f"pipeline stages ({self.stages})"
            )
        if self.enable_interleaved:
            if num_layers < self.stages * self.interleaved_chunks:
                raise ValueError(
                    f"Number of layers ({num_layers}) must be >= "
                    f"stages * chunks ({self.stages * self.interleaved_chunks}) "
                    f"for interleaved pipeline"
                )

    def layers_per_stage(self, num_layers: int) -> List[int]:
        """Compute how many layers each pipeline stage should have."""
        if not self.enabled:
            return [num_layers]
        base = num_layers // self.stages
        remainder = num_layers % self.stages
        result = []
        for i in range(self.stages):
            result.append(base + (1 if i < remainder else 0))
        return result

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["schedule"] = self.schedule.value
        d["bubble_fraction"] = self.bubble_fraction
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineParallelConfig":
        d = copy.deepcopy(d)
        if "schedule" in d and isinstance(d["schedule"], str):
            d["schedule"] = PipelineScheduleType(d["schedule"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "PipelineParallelConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Data Parallel Configuration
# ==============================================================================


@dataclass
class DataParallelConfig:
    """Configuration for data parallelism.

    Supports DDP (DistributedDataParallel), FSDP (FullyShardedDataParallel),
    and ZeRO optimization stages 1-3.

    Attributes:
        enabled: Whether data parallelism is enabled.
        strategy: Data parallelism strategy to use.
        world_size: Total number of data parallel replicas.
        sharding_strategy: FSDP sharding strategy.
        offload_optimizer: Whether to offload optimizer states to CPU.
        offload_param: Whether to offload parameters to CPU.
        offload_device: Device for offloading ('cpu', 'nvme').
        nvme_path: Path to NVMe storage for NVMe offloading.
        bucket_cap_mb: Bucket capacity in MB for gradient reduction.
        gradient_predivide_factor: Pre-divide factor for gradient all-reduce.
        gradient_as_bucket_view: Whether to use gradient as bucket view for
            reduced memory.
        find_unused_parameters: Whether to find unused parameters in backward.
        static_graph: Whether the computation graph is static.
        sync_bn: Whether to use synchronized batch normalization.
        broadcast_buffers: Whether to broadcast buffers in DDP.
        use_fp32_reduce_scatter: Use FP32 for reduce-scatter in ZeRO.
        zero_offload_param_fraction: Fraction of parameters to offload in ZeRO.
        zero_sub_group_size: Sub-group size for ZeRO parameter partitioning.
        zero_contiguous_gradients: Whether to allocate contiguous gradient buffers.
        zero_overlap_communication: Whether to overlap ZeRO communication.
        overlap_grad_reduce: Whether to overlap gradient reduction with backward.
        param_sync_on_fetch: Sync params when fetched in FSDP.
        forward_prefetch: Whether to prefetch parameters in FSDP forward.
        backward_prefetch: Backward prefetch mode for FSDP.
        ignore_unused_parameters: Whether to ignore unused parameters.
        flatten_parameters: Whether to flatten parameters for DDP.
    """
    enabled: bool = False
    strategy: DataParallelStrategy = DataParallelStrategy.DDP
    world_size: int = 1
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    offload_optimizer: bool = False
    offload_param: bool = False
    offload_device: str = "cpu"
    nvme_path: str = "/tmp/nvme_offload"
    bucket_cap_mb: int = 25
    gradient_predivide_factor: float = 1.0
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    static_graph: bool = False
    sync_bn: bool = False
    broadcast_buffers: bool = True
    use_fp32_reduce_scatter: bool = True
    zero_offload_param_fraction: float = 0.0
    zero_sub_group_size: int = 1_000_000_000_000
    zero_contiguous_gradients: bool = True
    zero_overlap_communication: bool = True
    overlap_grad_reduce: bool = True
    param_sync_on_fetch: bool = False
    forward_prefetch: bool = True
    backward_prefetch: str = "backward_pre"
    ignore_unused_parameters: bool = False
    flatten_parameters: bool = True

    def __post_init__(self):
        if self.world_size < 1:
            raise ValueError(
                f"Data parallel world_size must be >= 1, got {self.world_size}"
            )
        if self.bucket_cap_mb < 1:
            raise ValueError(
                f"bucket_cap_mb must be >= 1, got {self.bucket_cap_mb}"
            )
        if self.gradient_predivide_factor <= 0:
            raise ValueError(
                f"gradient_predivide_factor must be positive, "
                f"got {self.gradient_predivide_factor}"
            )
        if self.zero_offload_param_fraction < 0 or self.zero_offload_param_fraction > 1:
            raise ValueError(
                f"zero_offload_param_fraction must be in [0, 1], "
                f"got {self.zero_offload_param_fraction}"
            )
        if self.backward_prefetch not in (
            "backward_pre", "backward_post", "none"
        ):
            raise ValueError(
                f"Invalid backward_prefetch: {self.backward_prefetch}"
            )

    @property
    def is_zero(self) -> bool:
        return self.strategy in (
            DataParallelStrategy.ZERO_STAGE_1,
            DataParallelStrategy.ZERO_STAGE_2,
            DataParallelStrategy.ZERO_STAGE_3,
        )

    @property
    def is_fsdp(self) -> bool:
        return self.strategy == DataParallelStrategy.FSDP

    @property
    def is_ddp(self) -> bool:
        return self.strategy == DataParallelStrategy.DDP

    @property
    def zero_stage(self) -> int:
        stage_map = {
            DataParallelStrategy.ZERO_STAGE_1: 1,
            DataParallelStrategy.ZERO_STAGE_2: 2,
            DataParallelStrategy.ZERO_STAGE_3: 3,
        }
        return stage_map.get(self.strategy, 0)

    def get_bucket_size_bytes(self) -> int:
        """Get bucket size in bytes."""
        return self.bucket_cap_mb * 1024 * 1024

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["strategy"] = self.strategy.value
        d["sharding_strategy"] = self.sharding_strategy.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataParallelConfig":
        d = copy.deepcopy(d)
        if "strategy" in d and isinstance(d["strategy"], str):
            d["strategy"] = DataParallelStrategy(d["strategy"])
        if "sharding_strategy" in d and isinstance(d["sharding_strategy"], str):
            d["sharding_strategy"] = ShardingStrategy(d["sharding_strategy"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DataParallelConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Elastic Configuration
# ==============================================================================


@dataclass
class ElasticConfig:
    """Configuration for elastic distributed training.

    Elastic training allows the number of workers to change dynamically during
    training, accommodating heterogeneous clusters and node failures.

    Attributes:
        enabled: Whether elastic training is enabled.
        min_nodes: Minimum number of nodes for training to proceed.
        max_nodes: Maximum number of nodes allowed.
        scaling_policy: Policy for automatic scaling.
        target_nodes: Target number of nodes (elastic will try to reach this).
        allowed_scale_delta: Maximum number of nodes that can be added/removed
            in a single scaling event.
        scale_up_threshold: Utilization threshold above which scale-up is
            considered (0.0-1.0).
        scale_down_threshold: Utilization threshold below which scale-down is
            considered (0.0-1.0).
        scale_check_interval: Interval in seconds between scaling checks.
        node_grace_period: Grace period in seconds before a missing node is
            considered failed.
        checkpoint_frequency: Number of steps between elastic checkpoints.
        max_scaling_events: Maximum number of scaling events per training run.
        preserve_batch_size: Whether to preserve total batch size when scaling.
        preserve_lr_schedule: Whether to preserve learning rate schedule when
            scaling.
        max_wait_time: Maximum time in seconds to wait for minimum nodes.
        node_timeout: Timeout for node health checks.
        rebalance_on_scale: Whether to rebalance data on scale events.
    """
    enabled: bool = False
    min_nodes: int = 1
    max_nodes: int = 16
    scaling_policy: ScalingPolicy = ScalingPolicy.MANUAL
    target_nodes: int = 4
    allowed_scale_delta: int = 2
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_check_interval: float = 60.0
    node_grace_period: float = 120.0
    checkpoint_frequency: int = 100
    max_scaling_events: int = 50
    preserve_batch_size: bool = True
    preserve_lr_schedule: bool = True
    max_wait_time: float = 600.0
    node_timeout: float = 30.0
    rebalance_on_scale: bool = True

    def __post_init__(self):
        if self.min_nodes < 1:
            raise ValueError(
                f"min_nodes must be >= 1, got {self.min_nodes}"
            )
        if self.max_nodes < self.min_nodes:
            raise ValueError(
                f"max_nodes ({self.max_nodes}) must be >= "
                f"min_nodes ({self.min_nodes})"
            )
        if self.target_nodes < self.min_nodes or self.target_nodes > self.max_nodes:
            raise ValueError(
                f"target_nodes ({self.target_nodes}) must be in "
                f"[{self.min_nodes}, {self.max_nodes}]"
            )
        if self.allowed_scale_delta < 1:
            raise ValueError(
                f"allowed_scale_delta must be >= 1, got {self.allowed_scale_delta}"
            )
        if not (0 < self.scale_up_threshold <= 1.0):
            raise ValueError(
                f"scale_up_threshold must be in (0, 1], "
                f"got {self.scale_up_threshold}"
            )
        if not (0 <= self.scale_down_threshold < 1.0):
            raise ValueError(
                f"scale_down_threshold must be in [0, 1), "
                f"got {self.scale_down_threshold}"
            )
        if self.scale_check_interval <= 0:
            raise ValueError(
                f"scale_check_interval must be positive, "
                f"got {self.scale_check_interval}"
            )
        if self.checkpoint_frequency < 1:
            raise ValueError(
                f"checkpoint_frequency must be >= 1, "
                f"got {self.checkpoint_frequency}"
            )

    @property
    def scaling_range(self) -> int:
        return self.max_nodes - self.min_nodes

    def can_scale_to(self, num_nodes: int) -> bool:
        """Check if scaling to the given number of nodes is allowed."""
        return self.min_nodes <= num_nodes <= self.max_nodes

    def compute_new_batch_size(
        self, current_batch: int, old_nodes: int, new_nodes: int
    ) -> int:
        """Compute the new per-node batch size after scaling."""
        if self.preserve_batch_size:
            total_batch = current_batch * old_nodes
            return max(1, total_batch // new_nodes)
        return current_batch

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scaling_policy"] = self.scaling_policy.value
        d["scaling_range"] = self.scaling_range
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ElasticConfig":
        d = copy.deepcopy(d)
        if "scaling_policy" in d and isinstance(d["scaling_policy"], str):
            d["scaling_policy"] = ScalingPolicy(d["scaling_policy"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ElasticConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Fault Tolerance Configuration
# ==============================================================================


@dataclass
class FaultToleranceConfig:
    """Configuration for fault tolerance in distributed training.

    Provides mechanisms for detecting failures, saving checkpoints, and
    recovering from worker/node failures.

    Attributes:
        enabled: Whether fault tolerance is enabled.
        max_retries: Maximum number of retry attempts for failed operations.
        timeout: Timeout in seconds for operations before considering them
            failed.
        checkpoint_on_failure: Whether to automatically checkpoint on failure.
        recovery_strategy: Strategy for recovering from failures.
        heartbeat_interval: Interval in seconds between heartbeat messages.
        heartbeat_timeout: Timeout in seconds before a worker is considered
            failed based on heartbeat.
        max_heartbeat_misses: Number of missed heartbeats before declaring
            a worker dead.
        enable_stragglers_detection: Whether to detect and handle stragglers.
        straggler_threshold: Multiplier over median time to be considered a
            straggler.
        checkpoint_save_interval: Interval in training steps between checkpoints.
        checkpoint_keep_last_n: Number of recent checkpoints to keep.
        checkpoint_async_save: Whether to save checkpoints asynchronously.
        checkpoint_verify: Whether to verify checkpoint integrity after saving.
        checkpoint_compression: Whether to compress checkpoints.
        checkpoint_format: Format for checkpoint storage.
        enable_restart_on_failure: Whether to automatically restart on failure.
        restart_cooldown: Cooldown in seconds before restarting after failure.
        failure_notification: Whether to send notifications on failure.
        enable_warm_restart: Whether to do warm restarts (reusing existing
            process group if possible).
        persistent_storage_path: Path for persistent checkpoint storage.
        temporary_storage_path: Path for temporary checkpoint storage.
    """
    enabled: bool = False
    max_retries: int = 3
    timeout: float = 300.0
    checkpoint_on_failure: bool = True
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.CHECKPOINT_RESUME
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    max_heartbeat_misses: int = 6
    enable_stragglers_detection: bool = True
    straggler_threshold: float = 3.0
    checkpoint_save_interval: int = 500
    checkpoint_keep_last_n: int = 3
    checkpoint_async_save: bool = True
    checkpoint_verify: bool = True
    checkpoint_compression: bool = False
    checkpoint_format: CheckpointFormat = CheckpointFormat.TORCH
    enable_restart_on_failure: bool = True
    restart_cooldown: float = 10.0
    failure_notification: bool = False
    enable_warm_restart: bool = False
    persistent_storage_path: str = "/tmp/nexus_checkpoints"
    temporary_storage_path: str = "/tmp/nexus_checkpoints_tmp"

    def __post_init__(self):
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.timeout <= 0:
            raise ValueError(
                f"timeout must be positive, got {self.timeout}"
            )
        if self.heartbeat_interval <= 0:
            raise ValueError(
                f"heartbeat_interval must be positive, "
                f"got {self.heartbeat_interval}"
            )
        if self.heartbeat_timeout <= self.heartbeat_interval:
            raise ValueError(
                f"heartbeat_timeout ({self.heartbeat_timeout}) must be > "
                f"heartbeat_interval ({self.heartbeat_interval})"
            )
        if self.max_heartbeat_misses < 1:
            raise ValueError(
                f"max_heartbeat_misses must be >= 1, "
                f"got {self.max_heartbeat_misses}"
            )
        if self.straggler_threshold <= 1.0:
            raise ValueError(
                f"straggler_threshold must be > 1.0, "
                f"got {self.straggler_threshold}"
            )
        if self.checkpoint_save_interval < 1:
            raise ValueError(
                f"checkpoint_save_interval must be >= 1, "
                f"got {self.checkpoint_save_interval}"
            )
        if self.checkpoint_keep_last_n < 1:
            raise ValueError(
                f"checkpoint_keep_last_n must be >= 1, "
                f"got {self.checkpoint_keep_last_n}"
            )
        if self.restart_cooldown < 0:
            raise ValueError(
                f"restart_cooldown must be non-negative, "
                f"got {self.restart_cooldown}"
            )

    @property
    def effective_heartbeat_timeout(self) -> float:
        """Effective timeout before a worker is considered dead."""
        return self.heartbeat_interval * self.max_heartbeat_misses

    def should_checkpoint(self, step: int) -> bool:
        """Check if a checkpoint should be saved at this step."""
        if not self.enabled:
            return False
        return step > 0 and step % self.checkpoint_save_interval == 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["recovery_strategy"] = self.recovery_strategy.value
        d["checkpoint_format"] = self.checkpoint_format.value
        d["effective_heartbeat_timeout"] = self.effective_heartbeat_timeout
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FaultToleranceConfig":
        d = copy.deepcopy(d)
        if "recovery_strategy" in d and isinstance(d["recovery_strategy"], str):
            d["recovery_strategy"] = RecoveryStrategy(d["recovery_strategy"])
        if "checkpoint_format" in d and isinstance(d["checkpoint_format"], str):
            d["checkpoint_format"] = CheckpointFormat(d["checkpoint_format"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FaultToleranceConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))


# ==============================================================================
# Checkpoint Configuration
# ==============================================================================


@dataclass
class CheckpointConfig:
    """Configuration for distributed checkpointing.

    Attributes:
        save_dir: Directory to save checkpoints.
        save_interval: Steps between automatic checkpoints.
        save_on_exit: Whether to save on training exit.
        load_latest: Whether to load the latest checkpoint on start.
        async_save: Whether to save asynchronously.
        compression: Whether to compress checkpoints.
        format: Checkpoint format.
        keep_last_n: Number of recent checkpoints to keep.
        max_checkpoints_total: Maximum total checkpoints to keep.
        sharding: Whether to shard checkpoints across ranks.
        verify_on_save: Whether to verify checkpoints after saving.
        verify_on_load: Whether to verify checkpoints before loading.
        metadata_file: Name of the metadata file.
        temp_suffix: Suffix for temporary checkpoint files during save.
        enable_ckpt_profiling: Whether to profile checkpoint save/load time.
        max_save_time: Maximum time in seconds to wait for checkpoint save.
    """
    save_dir: str = "./checkpoints"
    save_interval: int = 500
    save_on_exit: bool = True
    load_latest: bool = True
    async_save: bool = True
    compression: bool = False
    format: CheckpointFormat = CheckpointFormat.TORCH
    keep_last_n: int = 3
    max_checkpoints_total: int = 10
    sharding: bool = True
    verify_on_save: bool = True
    verify_on_load: bool = True
    metadata_file: str = "metadata.json"
    temp_suffix: str = ".tmp"
    enable_ckpt_profiling: bool = False
    max_save_time: float = 3600.0

    def __post_init__(self):
        if self.save_interval < 1:
            raise ValueError(
                f"save_interval must be >= 1, got {self.save_interval}"
            )
        if self.keep_last_n < 1:
            raise ValueError(
                f"keep_last_n must be >= 1, got {self.keep_last_n}"
            )
        if self.max_save_time <= 0:
            raise ValueError(
                f"max_save_time must be positive, got {self.max_save_time}"
            )

    def get_checkpoint_path(self, step: int, rank: int = 0) -> Path:
        """Get the path for a checkpoint at a given step."""
        base = Path(self.save_dir)
        if self.sharding:
            return base / f"step_{step:08d}" / f"shard_{rank:04d}.pt"
        return base / f"step_{step:08d}" / "model.pt"

    def get_metadata_path(self, step: int) -> Path:
        """Get the path for checkpoint metadata."""
        return Path(self.save_dir) / f"step_{step:08d}" / self.metadata_file

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["format"] = self.format.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CheckpointConfig":
        d = copy.deepcopy(d)
        if "format" in d and isinstance(d["format"], str):
            d["format"] = CheckpointFormat(d["format"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ==============================================================================
# Debug Configuration
# ==============================================================================


@dataclass
class DebugConfig:
    """Configuration for debugging distributed training.

    Attributes:
        enabled: Whether debug mode is enabled.
        log_level: Logging verbosity level.
        log_dir: Directory for debug logs.
        profile_communication: Whether to profile communication operations.
        profile_memory: Whether to profile memory usage.
        profile_compute: Whether to profile computation time.
        dump_activations: Whether to dump intermediate activations.
        dump_gradients: Whether to dump gradients.
        dump_check_interval: Interval in steps between dumps.
        max_dump_size_mb: Maximum size in MB for activation/gradient dumps.
        trace_all_ranks: Whether to trace all ranks (vs. just rank 0).
        detect_anomalies: Whether to enable autograd anomaly detection.
        sync_debug_operations: Whether to synchronize debug operations across
            ranks.
        hang_detection: Whether to enable hang detection.
        hang_timeout: Timeout in seconds before declaring a hang.
        enable_nccl_debug: Whether to set NCCL_DEBUG environment variable.
        nccl_debug_level: NCCL debug log level (INFO, WARN, TRACE).
        enable_cudnn_benchmark: Whether to enable cuDNN benchmarking.
    """
    enabled: bool = False
    log_level: LogLevel = LogLevel.INFO
    log_dir: str = "./logs/distributed"
    profile_communication: bool = False
    profile_memory: bool = False
    profile_compute: bool = False
    dump_activations: bool = False
    dump_gradients: bool = False
    dump_check_interval: int = 100
    max_dump_size_mb: int = 100
    trace_all_ranks: bool = False
    detect_anomalies: bool = False
    sync_debug_operations: bool = True
    hang_detection: bool = True
    hang_timeout: float = 300.0
    enable_nccl_debug: bool = False
    nccl_debug_level: str = "INFO"
    enable_cudnn_benchmark: bool = True

    def __post_init__(self):
        if self.dump_check_interval < 1:
            raise ValueError(
                f"dump_check_interval must be >= 1, got {self.dump_check_interval}"
            )
        if self.max_dump_size_mb < 1:
            raise ValueError(
                f"max_dump_size_mb must be >= 1, got {self.max_dump_size_mb}"
            )
        if self.hang_timeout <= 0:
            raise ValueError(
                f"hang_timeout must be positive, got {self.hang_timeout}"
            )
        if self.nccl_debug_level not in ("INFO", "WARN", "TRACE", "VERSION"):
            raise ValueError(
                f"Invalid nccl_debug_level: {self.nccl_debug_level}"
            )

    def setup_environment(self) -> None:
        """Set up environment variables based on debug configuration."""
        if self.enabled:
            if self.enable_nccl_debug:
                os.environ["NCCL_DEBUG"] = self.nccl_debug_level
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
            if self.detect_anomalies:
                import torch
                torch.autograd.set_detect_anomaly(True)
            if self.enable_cudnn_benchmark:
                os.environ["CUDNN_BENCHMARK"] = "1"
            if self.log_dir:
                os.makedirs(self.log_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["log_level"] = self.log_level.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DebugConfig":
        d = copy.deepcopy(d)
        if "log_level" in d and isinstance(d["log_level"], str):
            d["log_level"] = LogLevel(d["log_level"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ==============================================================================
# Main Distributed Configuration
# ==============================================================================


@dataclass
class DistributedRuntimeConfig:
    """Runtime-specific distributed configuration computed from main configs.

    These values are derived and not directly set by the user.

    Attributes:
        rank: Global rank of this process.
        local_rank: Local rank on the current node.
        world_size: Total number of processes.
        local_world_size: Number of processes on the current node.
        node_id: ID of the current node (for multi-node setups).
        num_nodes: Total number of nodes.
        tp_rank: Tensor parallel rank within the TP group.
        tp_world_size: Tensor parallel world size.
        pp_rank: Pipeline parallel rank within the PP group.
        pp_world_size: Pipeline parallel world size.
        dp_rank: Data parallel rank within the DP group.
        dp_world_size: Data parallel world size.
        is_main_process: Whether this is the main (rank 0) process.
        is_main_node: Whether this process is on the main node (node 0).
        device: Device for this process.
        cuda_device_index: CUDA device index for this process.
    """
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    node_id: int = 0
    num_nodes: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
    pp_rank: int = 0
    pp_world_size: int = 1
    dp_rank: int = 0
    dp_world_size: int = 1
    is_main_process: bool = True
    is_main_node: bool = True
    device: str = "cpu"
    cuda_device_index: int = 0

    @property
    def global_id(self) -> str:
        return f"rank{self.rank}_node{self.node_id}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributedRuntimeConfig":
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


@dataclass
class DistributedConfig:
    """Top-level distributed training configuration.

    Aggregates all sub-configurations into a single, unified configuration
    object. Provides validation, serialization, and environment setup.

    Attributes:
        backend: Primary distributed communication backend.
        rank: Global rank of this process (0-indexed).
        world_size: Total number of processes.
        init_method: URL or file path for process group initialization.
        timeout: Timeout for collective operations.
        port: Port for distributed communication.
        tensor_parallel: Tensor parallelism configuration.
        pipeline_parallel: Pipeline parallelism configuration.
        data_parallel: Data parallelism configuration.
        elastic: Elastic training configuration.
        fault_tolerance: Fault tolerance configuration.
        network: Network configuration.
        checkpoint: Checkpoint configuration.
        debug: Debug configuration.
        seed: Random seed for reproducibility (set per rank for diversity).
        base_seed: Base seed that is combined with rank.
    """
    backend: ParallelBackend = ParallelBackend.AUTO
    rank: int = 0
    world_size: int = 1
    init_method: str = "env://"
    timeout: float = 1800.0
    port: int = 29500
    tensor_parallel: TensorParallelConfig = field(default_factory=TensorParallelConfig)
    pipeline_parallel: PipelineParallelConfig = field(default_factory=PipelineParallelConfig)
    data_parallel: DataParallelConfig = field(default_factory=DataParallelConfig)
    elastic: ElasticConfig = field(default_factory=ElasticConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    seed: int = 42
    base_seed: int = 42

    def __post_init__(self):
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
        if self.world_size < 1:
            raise ValueError(
                f"world_size must be >= 1, got {self.world_size}"
            )
        if self.rank >= self.world_size:
            raise ValueError(
                f"rank ({self.rank}) must be < world_size ({self.world_size})"
            )
        if self.timeout <= 0:
            raise ValueError(
                f"timeout must be positive, got {self.timeout}"
            )
        if self.port < 1 or self.port > 65535:
            raise ValueError(
                f"port must be between 1 and 65535, got {self.port}"
            )
        self.network.master_port = self.port

    def validate(self) -> List[str]:
        """Validate the entire configuration and return a list of errors."""
        errors = []
        tp_size = self.tensor_parallel.effective_size
        pp_size = self.pipeline_parallel.effective_stages
        dp_size = self.data_parallel.world_size

        if tp_size * pp_size * dp_size > self.world_size:
            errors.append(
                f"TP size ({tp_size}) * PP stages ({pp_size}) * "
                f"DP world size ({dp_size}) = {tp_size * pp_size * dp_size} "
                f"> total world_size ({self.world_size})"
            )

        if tp_size * pp_size != 0 and self.world_size % (tp_size * pp_size) != 0:
            errors.append(
                f"world_size ({self.world_size}) is not divisible by "
                f"TP*PP ({tp_size * pp_size})"
            )

        effective_dp = self.world_size // (tp_size * pp_size) if tp_size * pp_size > 0 else self.world_size
        if effective_dp != dp_size and dp_size > 1:
            errors.append(
                f"Effective DP size ({effective_dp}) != configured DP world_size ({dp_size})"
            )

        if self.elastic.enabled and self.fault_tolerance.enabled:
            if (self.elastic.min_nodes * 8) > self.world_size:
                errors.append(
                    "Elastic min_nodes * 8 GPUs per node exceeds world_size"
                )

        if self.tensor_parallel.enabled and self.pipeline_parallel.enabled:
            if tp_size * pp_size > self.world_size:
                errors.append(
                    f"TP*PP ({tp_size * pp_size}) exceeds world_size ({self.world_size})"
                )

        return errors

    def compute_runtime_config(self) -> DistributedRuntimeConfig:
        """Compute runtime configuration from the main config."""
        tp_size = self.tensor_parallel.effective_size
        pp_size = self.pipeline_parallel.effective_stages

        tp_pp_size = tp_size * pp_size
        if tp_pp_size == 0:
            dp_world_size = self.world_size
            tp_world_size = 1
            pp_world_size = 1
        else:
            dp_world_size = self.world_size // tp_pp_size
            tp_world_size = tp_size
            pp_world_size = pp_size

        tp_rank = self.rank % tp_world_size if tp_world_size > 0 else 0
        pp_rank = (self.rank // tp_world_size) % pp_world_size if pp_world_size > 0 else 0
        dp_rank = self.rank // (tp_world_size * pp_world_size) if tp_world_size * pp_world_size > 0 else self.rank

        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{self.rank % torch.cuda.device_count()}"
                cuda_idx = self.rank % torch.cuda.device_count()
            else:
                device = "cpu"
                cuda_idx = 0
        except ImportError:
            device = "cpu"
            cuda_idx = 0

        local_world_size = 8  # Default; will be set by environment
        local_rank = self.rank % local_world_size
        num_nodes = max(1, self.world_size // local_world_size)
        node_id = self.rank // local_world_size

        return DistributedRuntimeConfig(
            rank=self.rank,
            local_rank=local_rank,
            world_size=self.world_size,
            local_world_size=local_world_size,
            node_id=node_id,
            num_nodes=num_nodes,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            pp_rank=pp_rank,
            pp_world_size=pp_world_size,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            is_main_process=(self.rank == 0),
            is_main_node=(node_id == 0),
            device=device,
            cuda_device_index=cuda_idx,
        )

    def setup_environment(self) -> None:
        """Set up environment variables for distributed training."""
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ.setdefault("LOCAL_RANK", str(self.rank % 8))
        os.environ["MASTER_ADDR"] = self.network.master_addr
        os.environ["MASTER_PORT"] = str(self.port)

        if self.backend == ParallelBackend.NCCL:
            os.environ.setdefault("NCCL_DEBUG", "WARN")
            if self.debug.enabled and self.debug.enable_nccl_debug:
                os.environ["NCCL_DEBUG"] = self.debug.nccl_debug_level
                os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
            if self.network.rdma:
                os.environ.setdefault("NCCL_IB_DISABLE", "0")
                os.environ.setdefault("NCCL_SOCKET_IFNAME", self.network.interface)
            if self.network.enable_gdr:
                os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
        elif self.backend == ParallelBackend.GLOO:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", self.network.interface)

        if self.tensor_parallel.enable_flash_attention:
            os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "TRUE")

        seed = self.seed + self.rank
        os.environ["NEXUS_SEED"] = str(seed)

        self.debug.setup_environment()

    def get_effective_batch_size(self, per_gpu_batch: int) -> int:
        """Compute the effective global batch size."""
        dp_size = self.world_size // (
            self.tensor_parallel.effective_size * self.pipeline_parallel.effective_stages
        )
        pp_micro = self.pipeline_parallel.micro_batches
        grad_accum = self.pipeline_parallel.gradient_accumulation
        return per_gpu_batch * dp_size * pp_micro * grad_accum

    @property
    def total_parallel_degree(self) -> int:
        """Total degree of parallelism (TP * PP * DP)."""
        tp = self.tensor_parallel.effective_size
        pp = self.pipeline_parallel.effective_stages
        dp = self.data_parallel.world_size
        return tp * pp * dp

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is active."""
        return self.world_size > 1

    @property
    def effective_init_method(self) -> str:
        """Get the effective init method."""
        if self.init_method and self.init_method != "env://":
            return self.init_method
        return self.network.get_init_method()

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "backend": self.backend.value,
            "rank": self.rank,
            "world_size": self.world_size,
            "init_method": self.init_method,
            "timeout": self.timeout,
            "port": self.port,
            "tensor_parallel": self.tensor_parallel.to_dict(),
            "pipeline_parallel": self.pipeline_parallel.to_dict(),
            "data_parallel": self.data_parallel.to_dict(),
            "elastic": self.elastic.to_dict(),
            "fault_tolerance": self.fault_tolerance.to_dict(),
            "network": self.network.to_dict(),
            "checkpoint": self.checkpoint.to_dict(),
            "debug": self.debug.to_dict(),
            "seed": self.seed,
            "base_seed": self.base_seed,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DistributedConfig":
        d = copy.deepcopy(d)
        if "backend" in d and isinstance(d["backend"], str):
            d["backend"] = ParallelBackend(d["backend"])
        if "tensor_parallel" in d:
            d["tensor_parallel"] = TensorParallelConfig.from_dict(d["tensor_parallel"])
        if "pipeline_parallel" in d:
            d["pipeline_parallel"] = PipelineParallelConfig.from_dict(d["pipeline_parallel"])
        if "data_parallel" in d:
            d["data_parallel"] = DataParallelConfig.from_dict(d["data_parallel"])
        if "elastic" in d:
            d["elastic"] = ElasticConfig.from_dict(d["elastic"])
        if "fault_tolerance" in d:
            d["fault_tolerance"] = FaultToleranceConfig.from_dict(d["fault_tolerance"])
        if "network" in d:
            d["network"] = NetworkConfig.from_dict(d["network"])
        if "checkpoint" in d:
            d["checkpoint"] = CheckpointConfig.from_dict(d["checkpoint"])
        if "debug" in d:
            d["debug"] = DebugConfig.from_dict(d["debug"])
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "DistributedConfig":
        return cls.from_dict(json.loads(json_str))

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "DistributedConfig":
        return cls.from_dict(yaml.safe_load(yaml_str))

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        content = ""
        if suffix in (".yaml", ".yml"):
            content = self.to_yaml()
        elif suffix == ".json":
            content = self.to_json()
        else:
            content = self.to_json()
        path.write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DistributedConfig":
        """Load configuration from a file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(content)
        elif suffix == ".json":
            return cls.from_json(content)
        else:
            return cls.from_json(content)

    @classmethod
    def from_env(cls) -> "DistributedConfig":
        """Create configuration from environment variables."""
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        init_method = os.environ.get("INIT_METHOD", "env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        config = cls(
            rank=rank,
            world_size=world_size,
            init_method=init_method,
            port=master_port,
            network=NetworkConfig(
                master_addr=master_addr,
                master_port=master_port,
            ),
        )
        return config

    @classmethod
    def auto(cls, world_size: Optional[int] = None) -> "DistributedConfig":
        """Auto-configure distributed training based on available resources.

        Detects available GPUs and configures tensor parallelism optimally.
        """
        num_gpus = 0
        try:
            import torch
            num_gpus = torch.cuda.device_count()
        except ImportError:
            pass

        effective_world = world_size or int(os.environ.get("WORLD_SIZE", str(max(1, num_gpus))))
        rank = int(os.environ.get("RANK", "0"))

        config = cls(rank=rank, world_size=effective_world)

        if num_gpus > 0 and effective_world <= num_gpus:
            if num_gpus >= 8 and effective_world >= 8:
                config.tensor_parallel = TensorParallelConfig(
                    enabled=True, size=8, mode=TensorParallelMode.BOTH
                )
            elif num_gpus >= 4 and effective_world >= 4:
                config.tensor_parallel = TensorParallelConfig(
                    enabled=True, size=4, mode=TensorParallelMode.BOTH
                )
            elif num_gpus >= 2 and effective_world >= 2:
                config.tensor_parallel = TensorParallelConfig(
                    enabled=True, size=2, mode=TensorParallelMode.COLUMN
                )

        config.backend = ParallelBackend.NCCL if num_gpus > 0 else ParallelBackend.GLOO
        config.setup_environment()
        return config

    def copy(self) -> "DistributedConfig":
        """Create a deep copy of this configuration."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        tp = self.tensor_parallel.effective_size
        pp = self.pipeline_parallel.effective_stages
        dp = self.world_size // (tp * pp) if tp * pp > 0 else self.world_size
        return (
            f"DistributedConfig(rank={self.rank}, world_size={self.world_size}, "
            f"backend={self.backend.value}, TP={tp}, PP={pp}, DP={dp})"
        )
