"""Tests for Nexus LLM model components."""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nexus.model.config import ModelConfig
from nexus.model.transformer import NexusTransformer, TransformerBlock, TransformerOutput
from nexus.model.attention import GroupedQueryAttention
from nexus.model.ffn import SwiGLUFFN
from nexus.model.norm import RMSNorm
from nexus.model.rope import RotaryEmbedding, apply_rotary_pos_emb
from nexus.model.embeddings import Embedding


class TestModelConfig:
    """Test ModelConfig creation and validation."""

    def test_default_config(self):
        config = ModelConfig()
        assert config.hidden_size == 12288
        assert config.num_hidden_layers == 80
        assert config.name == "Nexus-100B"

    def test_custom_config(self):
        config = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        assert config.hidden_size == 256
        assert config.head_dim == 64
        assert config.num_kv_groups == 2

    def test_validation_error(self):
        with pytest.raises(AssertionError):
            ModelConfig(hidden_size=100, num_attention_heads=7)  # 100 % 7 != 0

    def test_parameter_estimation(self):
        config = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
                             num_hidden_layers=2, intermediate_size=512, vocab_size=256)
        params = config.total_params
        assert params > 0

    def test_to_dict_and_from_dict(self):
        config = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        d = config.to_dict()
        config2 = ModelConfig.from_dict(d)
        assert config2.hidden_size == config.hidden_size

    def test_repr(self):
        config = ModelConfig(hidden_size=256, num_attention_heads=4, num_key_value_heads=2)
        repr_str = repr(config)
        assert "Nexus" in repr_str


class TestRMSNorm:
    """Test RMSNorm layer."""

    def test_output_shape(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 16, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalization(self):
        norm = RMSNorm(128)
        x = torch.randn(2, 16, 128)
        out = norm(x)
        # Output should have similar magnitude
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_weight_initialization(self):
        norm = RMSNorm(128)
        assert torch.allclose(norm.weight, torch.ones(128))


class TestSwiGLUFFN:
    """Test SwiGLU FFN layer."""

    def test_output_shape(self, tiny_model_config):
        ffn = SwiGLUFFN(tiny_model_config)
        x = torch.randn(2, 16, tiny_model_config.hidden_size)
        out = ffn(x)
        assert out.shape == x.shape

    def test_no_nans(self, tiny_model_config):
        ffn = SwiGLUFFN(tiny_model_config)
        x = torch.randn(2, 16, tiny_model_config.hidden_size)
        out = ffn(x)
        assert not torch.isnan(out).any()


class TestGroupedQueryAttention:
    """Test GQA attention layer."""

    def test_output_shape(self, tiny_model_config):
        attn = GroupedQueryAttention(tiny_model_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_model_config.hidden_size)
        out, weights, kv = attn(x, use_cache=True)
        assert out.shape == (2, 16, tiny_model_config.hidden_size)

    def test_kv_cache(self, tiny_model_config):
        attn = GroupedQueryAttention(tiny_model_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_model_config.hidden_size)
        _, _, kv = attn(x, use_cache=True)
        assert kv is not None
        k, v = kv
        assert k.shape[1] == tiny_model_config.num_key_value_heads


class TestRotaryEmbedding:
    """Test Rotary Position Embedding."""

    def test_cos_sin_shape(self, tiny_model_config):
        rope = RotaryEmbedding(
            dim=tiny_model_config.head_dim,
            max_position_embeddings=tiny_model_config.max_position_embeddings,
        )
        input_ids = torch.randint(0, 100, (1, 16))
        cos, sin = rope(input_ids, seq_len=16)
        assert cos.shape[0] == 1
        assert sin.shape[0] == 1

    def test_rotation_preserves_shape(self):
        rope = RotaryEmbedding(dim=32, max_position_embeddings=256)
        x = torch.randn(1, 4, 8, 32)  # batch, heads, seq, head_dim
        y = torch.randn(1, 4, 8, 32)
        input_ids = torch.randint(0, 100, (1, 8))
        cos, sin = rope(input_ids, seq_len=8)
        # apply_rotary_pos_emb expects (batch, heads, seq, head_dim) shape
        cos_expanded = cos[:, :, :8, :].expand_as(x)
        sin_expanded = sin[:, :, :8, :].expand_as(x)
        x_rot, y_rot = apply_rotary_pos_emb(x, y, cos_expanded, sin_expanded)
        assert x_rot.shape == x.shape
        assert y_rot.shape == y.shape


class TestTransformerBlock:
    """Test single transformer block."""

    def test_forward_shape(self, tiny_model_config):
        block = TransformerBlock(tiny_model_config, layer_idx=0)
        x = torch.randn(2, 16, tiny_model_config.hidden_size)
        rope = RotaryEmbedding(
            dim=tiny_model_config.head_dim,
            max_position_embeddings=tiny_model_config.max_position_embeddings,
        )
        input_ids = torch.randint(0, 100, (2, 16))
        cos, sin = rope(input_ids, seq_len=16)
        out, kv, _ = block(x, rope_cos=cos, rope_sin=cos)
        assert out.shape == x.shape


class TestNexusTransformer:
    """Test the full NexusTransformer model."""

    def test_forward_output_shape(self, tiny_model, sample_input_ids):
        output = tiny_model(sample_input_ids)
        assert isinstance(output, TransformerOutput)
        assert output.logits.shape == (2, 32, 256)  # batch, seq, vocab

    def test_forward_with_labels(self, tiny_model, sample_input_ids):
        labels = sample_input_ids.clone()
        output = tiny_model(sample_input_ids, labels=labels)
        assert output.loss is not None
        assert output.loss.item() > 0

    def test_forward_with_attention_mask(self, tiny_model, sample_input_ids):
        mask = torch.ones_like(sample_input_ids)
        output = tiny_model(sample_input_ids, attention_mask=mask)
        assert output.logits.shape[0] == 2

    def test_no_nans_in_output(self, tiny_model, sample_input_ids):
        output = tiny_model(sample_input_ids)
        assert not torch.isnan(output.logits).any()

    def test_gradient_flow(self, tiny_model, sample_input_ids):
        labels = sample_input_ids.clone()
        output = tiny_model(sample_input_ids, labels=labels)
        output.loss.backward()
        # Check gradients exist
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_num_parameters(self, tiny_model):
        num_params = tiny_model.num_parameters()
        assert num_params > 0

    def test_save_and_load(self, tiny_model, tmp_path):
        save_path = str(tmp_path / "test_model")
        tiny_model.save_pretrained(save_path)
        loaded = NexusTransformer.from_pretrained(save_path)
        # Check parameters match
        for (n1, p1), (n2, p2) in zip(
            tiny_model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2, atol=1e-6)

    def test_gradient_checkpointing(self, tiny_model):
        tiny_model.enable_gradient_checkpointing()
        assert tiny_model.gradient_checkpointing is True
        tiny_model.disable_gradient_checkpointing()
        assert tiny_model.gradient_checkpointing is False

    def test_generate(self, tiny_model, sample_input_ids):
        # Use only first batch item for generation
        output_ids = tiny_model.generate(
            sample_input_ids[:1],
            max_new_tokens=5,
            temperature=1.0,
        )
        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] > sample_input_ids.shape[1]
