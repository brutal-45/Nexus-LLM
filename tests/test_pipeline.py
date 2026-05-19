"""Tests for pipeline."""
import pytest
import torch
import torch.nn as nn


class PipelineStage(nn.Module):
    """A single pipeline stage."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    
    def forward(self, x):
        return self.linear(x)


class SimplePipeline:
    """Simple sequential pipeline for testing."""
    def __init__(self, stages):
        self.stages = stages
    
    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x
    
    def __len__(self):
        return len(self.stages)


@pytest.fixture
def pipeline():
    return SimplePipeline([PipelineStage(64) for _ in range(3)])


def test_pipeline_creation(pipeline):
    """Test creating a pipeline."""
    assert len(pipeline) == 3


def test_pipeline_forward(pipeline):
    """Test pipeline forward pass."""
    x = torch.randn(2, 64)
    out = pipeline.forward(x)
    assert out.shape == (2, 64)


def test_pipeline_single_stage():
    """Test pipeline with a single stage."""
    p = SimplePipeline([PipelineStage(32)])
    x = torch.randn(1, 32)
    out = p.forward(x)
    assert out.shape == (1, 32)


def test_pipeline_gradient_flow():
    """Test that gradients flow through pipeline stages."""
    p = SimplePipeline([PipelineStage(16) for _ in range(2)])
    x = torch.randn(1, 16, requires_grad=True)
    out = p.forward(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
