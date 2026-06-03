"""Tests for distributed training."""
import pytest


class DistributedInfo:
    """Simulated distributed training info."""
    def __init__(self, rank=0, world_size=1, local_rank=0):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def is_distributed(self):
        return self.world_size > 1


def test_distributed_single_process():
    """Test single-process distributed info."""
    info = DistributedInfo(rank=0, world_size=1)
    assert info.is_main_process is True
    assert info.is_distributed is False


def test_distributed_multi_process():
    """Test multi-process distributed info."""
    info = DistributedInfo(rank=2, world_size=4)
    assert info.is_main_process is False
    assert info.is_distributed is True


def test_distributed_rank():
    """Test rank assignment."""
    info = DistributedInfo(rank=3, world_size=8)
    assert info.rank == 3
    assert info.world_size == 8


def test_distributed_main_process():
    """Test main process detection."""
    for rank in range(4):
        info = DistributedInfo(rank=rank, world_size=4)
        assert info.is_main_process == (rank == 0)
