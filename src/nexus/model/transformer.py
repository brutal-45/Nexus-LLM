"""
Nexus Transformer - Decoder-Only Architecture
==================================================
Full implementation of the 100B+ parameter decoder-only transformer.

Architecture per layer (Pre-Norm):
    x' = x + Attention(RMSNorm(x))
    x'' = x' + FFN(RMSNorm(x'))

Final:
    output = RMSNorm(x'') @ embedding.T  (weight tying optional)

Key features:
    - Pre-normalization (RMSNorm before attention and FFN)
    - RoPE positional encoding (applied inside attention)
    - Grouped Query Attention (GQA) for efficient KV caching
    - SwiGLU feed-forward activation
    - No bias in linear projections (improves training stability)
    - Supports gradient checkpointing for memory efficiency
    - Supports FSDP wrapping for distributed training

This is a complete, self-contained implementation with no dependencies
on external model libraries — every component is built from scratch.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

from .config import ModelConfig
from .attention import GroupedQueryAttention
from .ffn import SwiGLUFFN
from .embeddings import Embedding
from .norm import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class TransformerOutput:
    """Output dataclass for the transformer forward pass."""
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    loss: Optional[torch.Tensor] = None


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with pre-normalization.
    
    Structure:
        x' = x + Attention(RMSNorm(x))   [with RoPE, GQA]
        x'' = x' + FFN(RMSNorm(x'))      [with SwiGLU]
    
    Pre-norm is preferred over post-norm because:
        - More stable training (gradients don't explode)
        - No need for learning rate warmup (though we still use it)
        - Better performance in practice (proven by GPT-2, LLaMA)
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Grouped Query Attention
        self.self_attn = GroupedQueryAttention(
            config, layer_idx=layer_idx
        )

        # Pre-FFN normalization
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # SwiGLU Feed-Forward Network
        self.mlp = SwiGLUFFN(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass through one transformer block.
        
        Args:
            hidden_states: Input (batch, seq_len, hidden_size).
            attention_mask: Causal mask.
            position_ids: Position indices for RoPE.
            past_key_value: KV cache from previous step.
            output_attentions: Return attention weights.
            use_cache: Update and return KV cache.
            rope_cos, rope_sin: Precomputed RoPE tensors.
        
        Returns:
            Tuple of (hidden_states, present_kv, attn_weights).
        """
        residual = hidden_states

        # === Self-Attention Block ===
        # Pre-norm
        normed_hidden = self.input_layernorm(hidden_states)

        # GQA attention with RoPE
        attn_output, attn_weights, present_kv = self.self_attn(
            hidden_states=normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # === Feed-Forward Block ===
        residual = hidden_states

        # Pre-norm
        normed_hidden = self.post_attention_layernorm(hidden_states)

        # SwiGLU FFN
        ff_output = self.mlp(normed_hidden)

        # Residual connection
        hidden_states = residual + ff_output

        return hidden_states, present_kv, attn_weights


class NexusTransformer(nn.Module):
    """
    Nexus Decoder-Only Transformer.
    
    A complete, production-grade transformer language model supporting:
        - 100B+ parameter scale
        - Grouped Query Attention (GQA) with Flash Attention
        - Rotary Position Embeddings (RoPE)
        - SwiGLU feed-forward activation
        - RMSNorm pre-normalization
        - KV caching for efficient generation
        - Gradient checkpointing
        - FSDP-compatible (auto-wrap policy friendly)
    
    Usage:
        config = ModelConfig(hidden_size=12288, num_hidden_layers=80, ...)
        model = NexusTransformer(config)
        logits = model(input_ids).logits  # (batch, seq_len, vocab_size)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = None
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embed_tokens = Embedding(config)

        # RoPE for positional encoding (shared across all layers)
        self.rope = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=None,  # Will be set on first forward
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language modeling head
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Tie embeddings if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Gradient checkpointing flag
        self.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights following the LLaMA initialization scheme.
        
        - Linear layers: Normal(0, initializer_range)
        - RMSNorm: weight = 1.0
        - Embeddings: Normal(0, initializer_range)
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _prepare_rope(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare position information and RoPE cos/sin tensors.
        """
        seq_len = input_ids.shape[1]

        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                seq_len + past_key_values_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)

        # Get RoPE cos/sin
        cos, sin = self.rope(input_ids, seq_len=seq_len + past_key_values_length)
        cos = cos[:, :, past_key_values_length:seq_len + past_key_values_length, :]
        sin = sin[:, :, past_key_values_length:seq_len + past_key_values_length, :]

        return position_ids, cos, sin

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TransformerOutput:
        """
        Forward pass of the full transformer model.
        
        Args:
            input_ids: Token indices (batch, seq_len).
            attention_mask: Optional mask (batch, seq_len). 1=attend, 0=ignore.
            position_ids: Optional position indices (batch, seq_len).
            past_key_values: List of (key, value) tuples per layer for KV cache.
            inputs_embeds: Optional embedded inputs (bypasses embed_tokens).
            labels: Optional target token indices for computing loss.
                    Shifted internally: predict next token.
            use_cache: Whether to use and return KV cache.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
        
        Returns:
            TransformerOutput with logits, optional loss, and caching info.
        """
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else False

        # Get sequence length and past KV length
        if past_key_values is not None:
            past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        else:
            past_len = 0

        # === Embedding ===
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # === RoPE Setup ===
        position_ids, rope_cos, rope_sin = self._prepare_rope(
            input_ids, position_ids, past_len
        )

        # === Prepare attention mask ===
        # Create causal mask if not provided
        if attention_mask is not None:
            # Expand mask: (batch, 1, seq_len, total_seq_len)
            # Where total_seq_len = seq_len + past_len (for KV cache)
            bsz, seq_len = input_ids.shape
            # For simplicity, create 4D causal mask
            causal_mask = torch.triu(
                torch.ones(
                    seq_len, seq_len + past_len,
                    device=input_ids.device, dtype=torch.bool
                ),
                diagonal=past_len + 1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask & (~causal_mask)
            attention_mask = attention_mask.to(dtype=inputs_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(inputs_embeds.dtype).min

        # === Process through all layers ===
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Gradient checkpointing for memory efficiency
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values[idx] if past_key_values is not None else None,
                    output_attentions,
                    use_cache,
                    rope_cos,
                    rope_sin,
                )
            else:
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values[idx] if past_key_values is not None else None,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache = next_cache + (layer_outputs[1],)

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[2],)

        # === Final normalization ===
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # === Language Modeling Head ===
        logits = self.lm_head(hidden_states)

        # === Compute loss if labels provided ===
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # logits[:, :-1, :] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
            )

        return TransformerOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            past_key_values=next_cache,
            attentions=all_self_attns,
            loss=loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        use_cache: bool = True,
        stop_token_ids: Optional[List[int]] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Autoregressive text generation with KV caching.
        
        Supports:
            - Temperature sampling
            - Top-p (nucleus) sampling
            - Top-k filtering
            - Repetition penalty
            - Stop token detection
        
        Args:
            input_ids: Prompt tokens (batch, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (<1.0 = more deterministic).
            top_p: Nucleus sampling threshold.
            top_k: Top-k filtering threshold.
            repetition_penalty: Penalty for repeating tokens (>1.0 = penalize).
            use_cache: Whether to use KV caching for efficiency.
            stop_token_ids: List of token IDs that stop generation.
            eos_token_id: End-of-sequence token ID.
        
        Returns:
            Generated token sequence (batch, prompt_len + gen_len).
        """
        self.eval()
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_key_values = None

        for step in range(max_new_tokens):
            # Forward pass (with KV cache after first step)
            outputs = self.forward(
                input_ids=generated[:, -1:] if past_key_values is not None else generated,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get logits for the next token
            next_token_logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=-1)
            past_key_values = outputs.past_key_values

            # Check for stop tokens
            if stop_token_ids is not None:
                if any(t in stop_token_ids for t in next_token.view(-1).tolist()):
                    break
            if eos_token_id is not None:
                if eos_token_id in next_token.view(-1).tolist():
                    break

        return generated

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count total parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        import json
        import os
        os.makedirs(path, exist_ok=True)
        # Save config
        self.config.save_yaml(os.path.join(path, "config.yaml"))
        # Save weights as safetensors if available
        try:
            from safetensors.torch import save_file
            save_file(
                {k: v for k, v in self.state_dict().items()},
                os.path.join(path, "model.safetensors"),
            )
        except ImportError:
            torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> NexusTransformer:
        """Load model from checkpoint."""
        config = ModelConfig.from_yaml(os.path.join(path, "config.yaml"))
        model = cls(config)

        # Try safetensors first, then fall back to pytorch_model.bin
        safetensors_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No checkpoint found in {path}")

        model.load_state_dict(state_dict)
        return model.to(device)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.config.hidden_size}, "
            f"num_layers={self.config.num_hidden_layers}, "
            f"num_heads={self.config.num_attention_heads}, "
            f"num_kv_heads={self.config.num_key_value_heads}"
        )


# Need os for from_pretrained
import os
