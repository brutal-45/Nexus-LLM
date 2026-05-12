"""
Distributed Training Setup
============================
Handles initialization of distributed training with NCCL backend,
3D parallelism (Tensor/Pipeline/Data), and FSDP configuration.

Supports multiple launch methods:
    - torchrun (recommended)
    - torch.distributed.launch (legacy)
    - Single GPU (no distributed)

3D Parallelism:
    TP (Tensor Parallelism): Splits individual weight matrices across GPUs.
       Each GPU computes a slice of the matrix multiply.
       Best for large models where a single layer doesn't fit on one GPU.
    
    PP (Pipeline Parallelism): Splits layers across GPUs.
       GPU 0 computes layers 0-19, GPU 1 computes layers 20-39, etc.
       Adds bubble overhead but enables very large models.
    
    DP (Data Parallelism): Replicates the model and splits data.
       Most straightforward, scales linearly with GPU count.
       Combined with FSDP/ZeRO for memory efficiency.

FSDP (Fully Sharded Data Parallel):
    - FULL_SHARD (ZeRO-3): Shards parameters, gradients, and optimizer states
    - SHARD_GRAD_OP (ZeRO-2): Shards gradients and optimizer states
    - NO_SHARD (ZeRO-1): Shards optimizer states only
"""

from __future__ import annotations
import os
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


@dataclass
class DistributedInfo:
    """Information about the distributed training environment."""
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    is_distributed: bool = False
    
    # Parallelism dimensions
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def setup_distributed(backend: str = "nccl") -> DistributedInfo:
    """
    Initialize distributed training environment.
    
    Automatically detects the launch method and configures accordingly.
    Supports torchrun, torch.distributed.launch, and environment variables.
    
    Returns:
        DistributedInfo with rank, world_size, local_rank, etc.
    """
    info = DistributedInfo()
    
    # Check if distributed training is requested
    if "RANK" in os.environ:
        # torchrun sets these
        info.rank = int(os.environ["RANK"])
        info.world_size = int(os.environ["WORLD_SIZE"])
        info.local_rank = int(os.environ["LOCAL_RANK"])
        info.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        info.is_distributed = True
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        # OpenMPI
        info.rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        info.world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        info.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        info.local_world_size = info.world_size  # Approximation
        info.is_distributed = True
    else:
        # Single GPU mode
        info.rank = 0
        info.world_size = 1
        info.local_rank = 0
        info.local_world_size = 1
        info.is_distributed = False
        return info
    
    # Initialize process group
    if info.is_distributed:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(info.local_rank)
        
        # Synchronize all processes
        dist.barrier()
    
    return info


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_world_info() -> DistributedInfo:
    """Get current distributed training info."""
    info = DistributedInfo()
    if dist.is_initialized():
        info.rank = dist.get_rank()
        info.world_size = dist.get_world_size()
        info.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        info.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        info.is_distributed = True
    return info


def configure_fsdp(
    model: nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    backward_prefetch: bool = True,
    cpu_offload: bool = False,
    wrapping_policy: str = "transformer_auto_wrap",
) -> FSDP:
    """
    Configure and wrap model with Fully Sharded Data Parallel.
    
    Args:
        model: The model to wrap.
        sharding_strategy: "FULL_SHARD" (ZeRO-3), "SHARD_GRAD_OP" (ZeRO-2),
                          or "NO_SHARD" (ZeRO-1).
        backward_prefetch: Whether to prefetch parameters for backward pass.
        cpu_offload: Whether to offload parameters/gradients to CPU.
        wrapping_policy: "transformer_auto_wrap" or "size_based_auto_wrap".
    
    Returns:
        FSDP-wrapped model ready for distributed training.
    """
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    
    # Build auto wrap policy
    if wrapping_policy == "transformer_auto_wrap":
        # Wrap each transformer block as a unit
        # This ensures all parameters within a block are on the same GPU
        from ..model.transformer import TransformerBlock
        wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={TransformerBlock},
        )
    else:
        wrap_policy = size_based_auto_wrap_policy(
            min_num_params=1_000_000_000,  # Wrap layers > 1B params
        )
    
    # Configure CPU offload
    cpu_offload_config = CPUOffload(offload_params=cpu_offload) if cpu_offload else None
    
    # Create FSDP wrapper
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy_map[sharding_strategy],
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE if backward_prefetch else BackwardPrefetch.BACKWARD_POST,
        cpu_offload=cpu_offload_config,
        auto_wrap_policy=wrap_policy,
        device_id=torch.cuda.current_device() if torch.cuda.is_available() else None,
        sync_module_states=True,  # Sync parameters across ranks
        use_orig_params=True,     # Allow access to original parameters
    )
    
    return fsdp_model


class TensorParallel(nn.Module):
    """
    Tensor Parallelism wrapper for splitting linear layers.
    
    Splits weight matrices across GPUs along the output dimension:
        W_full = [W_1, W_2, ..., W_n] (column parallel)
    
    For attention: QKV projections are split column-wise.
    For FFN: gate/up/down projections are split column-wise (gate, up) or row-wise (down).
    
    This is integrated into the model architecture via Megatron-style parallelism.
    In practice, use FSDP for simpler setup or DeepSpeed for tensor parallelism.
    """

    def __init__(self, module: nn.Module, world_size: int, rank: int):
        super().__init__()
        self.module = module
        self.world_size = world_size
        self.rank = rank

    def forward(self, *args, **kwargs):
        # In practice, the actual parallelism is handled by modifying
        # the linear layers to only compute their shard of the output
        return self.module(*args, **kwargs)
