"""Distributed training: DDP, FSDP setup, process group management, gradient sync."""

import os
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


class DistributedManager:
    """Manages distributed training setup including DDP and FSDP."""

    def __init__(
        self,
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        timeout_minutes: int = 30,
        use_fsdp: bool = False,
        fsdp_config: Optional[Dict[str, Any]] = None,
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.timeout_minutes = timeout_minutes
        self.use_fsdp = use_fsdp
        self.fsdp_config = fsdp_config or {}
        self._is_initialized = False

    @property
    def is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    @property
    def is_main_process(self) -> bool:
        if not self.is_distributed:
            return True
        return self.rank == 0

    def setup(self):
        """Initialize the distributed process group."""
        if self._is_initialized:
            logger.warning("Distributed manager already initialized.")
            return

        if not dist.is_available():
            logger.warning("torch.distributed is not available. Running in single-process mode.")
            self._is_initialized = True
            return

        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.world_size is None or self.world_size <= 1:
            logger.info("World size is 1. Running in single-process mode.")
            self._is_initialized = True
            return

        if self.backend is None:
            self.backend = self._detect_backend()

        timeout_td = datetime.timedelta(minutes=self.timeout_minutes) if hasattr(datetime, 'timedelta') else None

        init_kwargs = {
            "backend": self.backend,
            "timeout": timeout_td if timeout_td else None,
        }

        if self.init_method:
            init_kwargs["init_method"] = self.init_method
        elif "MASTER_ADDR" in os.environ:
            pass  # torchrun sets these automatically
        else:
            init_kwargs["init_method"] = "tcp://localhost:29500"

        if self.rank is not None:
            init_kwargs["rank"] = self.rank
        if self.world_size is not None:
            init_kwargs["world_size"] = self.world_size

        if init_kwargs.get("timeout") is None:
            del init_kwargs["timeout"]

        dist.init_process_group(**init_kwargs)
        self._is_initialized = True

        if self.local_rank is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        logger.info(
            f"Distributed setup complete: rank={self.rank}, "
            f"world_size={self.world_size}, backend={self.backend}"
        )

    def _detect_backend(self) -> str:
        """Detect the best available distributed backend."""
        if torch.cuda.is_available() and dist.is_nccl_available():
            return "nccl"
        elif dist.is_gloo_available():
            return "gloo"
        elif dist.is_mpi_available():
            return "mpi"
        return "gloo"

    def wrap_model_ddp(
        self,
        model: torch.nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ) -> DDP:
        """Wrap a model with DistributedDataParallel."""
        if not self.is_distributed:
            logger.warning("Not in distributed mode. Returning unwrapped model.")
            return model

        if device_ids is None and self.local_rank is not None and torch.cuda.is_available():
            device_ids = [self.local_rank]
            output_device = self.local_rank

        ddp_model = DDP(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        logger.info("Model wrapped with DistributedDataParallel.")
        return ddp_model

    def wrap_model_fsdp(
        self,
        model: torch.nn.Module,
        shard_strategy: str = "FULL_SHARD",
        mixed_precision: Optional[str] = None,
        cpu_offload: bool = False,
    ) -> Any:
        """Wrap a model with FullyShardedDataParallel."""
        if not self.is_distributed:
            logger.warning("Not in distributed mode. Returning unwrapped model.")
            return model

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        except ImportError:
            logger.error("FSDP is not available in this PyTorch version. Falling back to DDP.")
            return self.wrap_model_ddp(model)

        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }

        sharding_strategy = strategy_map.get(
            shard_strategy, ShardingStrategy.FULL_SHARD
        )

        mp_policy = None
        if mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        cpu_offload_cfg = None
        if cpu_offload:
            try:
                from torch.distributed.fsdp import CPUOffload
                cpu_offload_cfg = CPUOffload(offload_params=True)
            except ImportError:
                logger.warning("CPU offload not available in this PyTorch version.")

        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload_cfg,
        )

        logger.info("Model wrapped with FullyShardedDataParallel.")
        return fsdp_model

    def create_distributed_sampler(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> Optional[DistributedSampler]:
        """Create a DistributedSampler if in distributed mode."""
        if not self.is_distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        collate_fn=None,
        seed: int = 42,
    ) -> DataLoader:
        """Create a DataLoader with appropriate sampler for distributed training."""
        sampler = self.create_distributed_sampler(dataset, shuffle=shuffle, seed=seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )

    def all_reduce_tensor(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce a tensor across all processes."""
        if not self.is_distributed:
            return tensor
        dist.all_reduce(tensor, op=op)
        return tensor

    def all_reduce_dict(self, data: Dict[str, float], op=dist.ReduceOp.SUM) -> Dict[str, float]:
        """All-reduce a dictionary of scalars across all processes."""
        if not self.is_distributed:
            return data

        keys = sorted(data.keys())
        values = torch.tensor([data[k] for k in keys], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(values, op=op)
        return {k: values[i].item() / self.world_size for i, k in enumerate(keys)}

    def barrier(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """Broadcast a Python object from src to all processes."""
        if not self.is_distributed:
            return obj
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]

    def gather_object(self, obj: Any, dst: int = 0) -> Optional[List[Any]]:
        """Gather Python objects from all processes to dst."""
        if not self.is_distributed:
            return [obj]
        obj_list = [None] * self.world_size
        dist.all_gather_object(obj_list, obj)
        return obj_list

    def cleanup(self):
        """Clean up the distributed process group."""
        if self.is_distributed:
            dist.destroy_process_group()
            self._is_initialized = False
            logger.info("Distributed process group destroyed.")

    def get_world_size(self) -> int:
        """Get the world size."""
        if self.is_distributed:
            return dist.get_world_size()
        return 1

    def get_rank(self) -> int:
        """Get the current process rank."""
        if self.is_distributed:
            return dist.get_rank()
        return 0

    def get_local_rank(self) -> int:
        """Get the local rank within the node."""
        return self.local_rank if self.local_rank is not None else 0
