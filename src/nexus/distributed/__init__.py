# Copyright (c) 2024 Nexus LLM Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Nexus LLM Distributed Training Module
# =====================================
# Production-grade distributed training framework providing tensor parallelism,
# pipeline parallelism, data parallelism, fault tolerance, elastic scaling,
# and RPC-based communication for large-scale LLM training.
#
# Backends: NCCL (GPU), Gloo (CPU), MPI (HPC)
# Supported parallelism: TP, PP, DP, FSDP, ZeRO (stages 1-3)
# Fault tolerance: heartbeat monitoring, checkpoint/recovery, elastic scaling

"""Nexus LLM Distributed Training Module.

This module provides a comprehensive distributed training framework for
large language models. It supports:

- **Tensor Parallelism (TP)**: Shard model weights across GPUs with
  column-parallel and row-parallel linear layers, Megatron-LM style
  attention, and overlapping communication.

- **Pipeline Parallelism (PP)**: Split model layers across GPUs with
  1F1B and interleaved scheduling, automatic load balancing, and
  activation recomputation.

- **Data Parallelism (DP)**: Distribute training data across GPUs with
  DDP, FSDP, and ZeRO stages 1-3 with gradient bucketing and overlap.

- **Fault Tolerance**: Heartbeat-based health monitoring, distributed
  checkpoint save/load, automatic failure recovery, and elastic scaling.

- **Communication**: NCCL/Gloo backends, collective operations with
  gradient support, communication-computation overlap, and compression.

Quick Start::

    from nexus.distributed import (
        init_distributed, DistributedConfig, ColumnParallelLinear,
        RowParallelLinear, TensorParallelAttention, PipelineStage,
        OneForwardOneBackwardSchedule, ShardedDataParallel,
        FullyShardedParallel, GradientAccumulator, CollectiveOps,
        CheckpointManager, HealthChecker, ElasticDriver, RPCService,
    )

    config = DistributedConfig(backend="nccl", rank=0, world_size=4)
    init_distributed(config)
"""

from nexus.distributed.parallel import (
    # --- Configuration ---
    DistributedConfig,
    TensorParallelConfig,
    PipelineParallelConfig,
    DataParallelConfig,
    ParallelBackend,
    TensorParallelMode,
    DataParallelStrategy,
    ShardingStrategy,
    PipelineScheduleType,
    # --- Tensor Parallelism ---
    ColumnParallelLinear,
    RowParallelLinear,
    TensorParallelAttention,
    TensorParallelTransformerLayer,
    TensorParallelOptimizer,
    SequenceParallelism,
    MegatronStyleTP,
    TensorParallelEngine,
    TPShardingStrategy,
    # --- Pipeline Parallelism ---
    PipelineStage,
    OneForwardOneBackwardSchedule,
    InterleavedPipelineSchedule,
    PipelineBalancer,
    PipelineMicroBatch,
    PipelineRecompute,
    PipelineCommOverlap,
    PipelineEngine,
    PipelineState,
    RecomputePolicy,
    StageTransition,
    MicroBatchId,
    ActivationHandle,
    # --- Data Parallelism ---
    ShardedDataParallel,
    ShardedGradParallel,
    FullyShardedParallel,
    GradientAccumulator,
    BucketedReduction,
    DistributedSampler,
    SyncBatchNorm,
    # --- Initialization ---
    init_distributed,
    initialize_tensor_parallel,
    initialize_pipeline_parallel,
    destroy_distributed,
)

from nexus.distributed.communication import (
    # --- Collective Operations ---
    CollectiveOps,
    AllGatherWithGradient,
    ReduceScatterWithGradient,
    OverlapCommComputation,
    CommunicationOptimizer,
    CommunicationCompressionType,
    ReduceOp,
    # --- RPC Framework ---
    RPCService,
    RemoteModule,
    ParameterServer,
    RPCProfiler,
    RPCRetryPolicy,
    RPCMessage,
    RPCResponse,
    RPCError,
    RPCTimeoutError,
    RPCConnectionError,
    RemoteCallable,
    RPCMetrics,
    ProcessGroupRPC,
    RPCBackend,
    # --- Fault Tolerance ---
    HealthChecker,
    FailureDetector,
    CheckpointManager,
    RecoveryManager,
    WorkerInfo,
    WorkerState,
    CheckpointMetadata,
    CheckpointState,
    FailureRecord,
    FailureType,
    HeartbeatMonitor,
    CheckpointStorage,
    RecoveryResult,
    # --- Elastic Training ---
    ElasticDriver,
    ElasticScaler,
    NodeManager,
    ResourceManager,
    Rendezvous,
    NodeInfo,
    NodeState,
    ResourceAllocation,
    ResourceRequest,
    ScalingEvent,
    RendezvousConfig,
    ElasticJobConfig,
    # --- Topology & Profiling ---
    TopologyManager,
    CommProfiler,
    TagManager,
    NCCLBackend,
    GlooBackend,
    CommBackend,
    TopologyType,
    CommStats,
    BandwidthEstimator,
    LatencyEstimator,
    CommunicationTag,
    # --- Batched / P2P Communication ---
    BatchedComm,
    P2PComm,
)

__all__ = [
    # --- Configuration ---
    "DistributedConfig",
    "TensorParallelConfig",
    "PipelineParallelConfig",
    "DataParallelConfig",
    "ParallelBackend",
    "TensorParallelMode",
    "DataParallelStrategy",
    "ShardingStrategy",
    "PipelineScheduleType",
    # --- Tensor Parallelism ---
    "ColumnParallelLinear",
    "RowParallelLinear",
    "TensorParallelAttention",
    "TensorParallelTransformerLayer",
    "TensorParallelOptimizer",
    "SequenceParallelism",
    "MegatronStyleTP",
    "TensorParallelEngine",
    "TPShardingStrategy",
    # --- Pipeline Parallelism ---
    "PipelineStage",
    "OneForwardOneBackwardSchedule",
    "InterleavedPipelineSchedule",
    "PipelineBalancer",
    "PipelineMicroBatch",
    "PipelineRecompute",
    "PipelineCommOverlap",
    "PipelineEngine",
    "PipelineState",
    "RecomputePolicy",
    "StageTransition",
    "MicroBatchId",
    "ActivationHandle",
    # --- Data Parallelism ---
    "ShardedDataParallel",
    "ShardedGradParallel",
    "FullyShardedParallel",
    "GradientAccumulator",
    "BucketedReduction",
    "DistributedSampler",
    "SyncBatchNorm",
    # --- Initialization ---
    "init_distributed",
    "initialize_tensor_parallel",
    "initialize_pipeline_parallel",
    "destroy_distributed",
    # --- Collective Operations ---
    "CollectiveOps",
    "AllGatherWithGradient",
    "ReduceScatterWithGradient",
    "OverlapCommComputation",
    "CommunicationOptimizer",
    "CommunicationCompressionType",
    "ReduceOp",
    # --- RPC Framework ---
    "RPCService",
    "RemoteModule",
    "ParameterServer",
    "RPCProfiler",
    "RPCRetryPolicy",
    "RPCMessage",
    "RPCResponse",
    "RPCError",
    "RPCTimeoutError",
    "RPCConnectionError",
    "RemoteCallable",
    "RPCMetrics",
    "ProcessGroupRPC",
    "RPCBackend",
    # --- Fault Tolerance ---
    "HealthChecker",
    "FailureDetector",
    "CheckpointManager",
    "RecoveryManager",
    "WorkerInfo",
    "WorkerState",
    "CheckpointMetadata",
    "CheckpointState",
    "FailureRecord",
    "FailureType",
    "HeartbeatMonitor",
    "CheckpointStorage",
    "RecoveryResult",
    # --- Elastic Training ---
    "ElasticDriver",
    "ElasticScaler",
    "NodeManager",
    "ResourceManager",
    "Rendezvous",
    "NodeInfo",
    "NodeState",
    "ResourceAllocation",
    "ResourceRequest",
    "ScalingEvent",
    "RendezvousConfig",
    "ElasticJobConfig",
    # --- Topology & Profiling ---
    "TopologyManager",
    "CommProfiler",
    "TagManager",
    "NCCLBackend",
    "GlooBackend",
    "CommBackend",
    "TopologyType",
    "CommStats",
    "BandwidthEstimator",
    "LatencyEstimator",
    "CommunicationTag",
    # --- Batched / P2P Communication ---
    "BatchedComm",
    "P2PComm",
]

__version__ = "0.1.0"
__author__ = "Nexus LLM Contributors"
