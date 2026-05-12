"""
Text Generator - Autoregressive Generation Engine
====================================================
Handles autoregressive text generation with KV caching, multiple
sampling strategies, and batching support.

Sampling strategies:
    - Greedy: Always pick the most probable token (deterministic)
    - Temperature: Scale logits before softmax (< 1.0 = sharper, > 1.0 = more random)
    - Top-K: Only consider the K most probable tokens
    - Top-P (Nucleus): Consider the smallest set of tokens with cumulative prob >= P
    - Beam Search: Maintain K hypotheses, expand all, prune to top K
    - Repetition Penalty: Reduce probability of recently seen tokens

KV Cache:
    During autoregressive generation, the key-value pairs for previous tokens
    are cached. This avoids recomputing attention for the entire sequence at
    each step, reducing generation time from O(n^2) to O(n).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from ..model.transformer import NexusTransformer, TransformerOutput


class SamplingStrategy(Enum):
    GREEDY = "greedy"
    TEMPERATURE = "temperature"
    TOP_K = "top_k"
    TOP_P = "top_p"
    BEAM_SEARCH = "beam_search"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 4096
    min_new_tokens: int = 0
    
    # Sampling
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    # Beam search
    num_beams: int = 1
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    early_stopping: bool = False
    
    # Stopping
    stop_token_ids: List[int] = field(default_factory=list)
    eos_token_id: Optional[int] = 2  # Default EOS
    ignore_eos: bool = False
    
    # Generation
    use_cache: bool = True


@dataclass
class GenerationResult:
    """Result from text generation."""
    generated_ids: torch.Tensor        # (batch, total_len)
    generated_text: List[str]           # Decoded text per batch item
    finish_reason: List[str]            # "eos", "stop_token", "max_length"
    num_generated_tokens: int
    sequences_scores: Optional[torch.Tensor] = None  # For beam search


class TextGenerator:
    """
    High-level text generation interface for Nexus models.
    
    Wraps the model's generate method with additional features:
        - Batch generation
        - Multiple sampling strategies
        - Streaming generation (yield tokens as they're produced)
        - Stop token detection
        - Custom callbacks
    """

    def __init__(
        self,
        model: NexusTransformer,
        tokenizer,
        device: str = "cuda",
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text from one or more prompts.
        
        Args:
            prompts: Single prompt string or list of prompts.
            config: Generation configuration. Uses defaults if None.
        
        Returns:
            GenerationResult with generated IDs, text, and metadata.
        """
        config = config or GenerationConfig()
        
        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Tokenize
        input_ids_list = [
            self.tokenizer.encode(p, add_bos=True, add_eos=False)
            for p in prompts
        ]
        
        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        padded = [
            ids + [self.tokenizer.special_tokens["<pad>"]] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.tensor(
            [[1 if i < len(ids) else 0 for i in range(max_len)] for ids in input_ids_list],
            dtype=torch.long,
            device=self.device,
        )
        
        # Generate based on strategy
        if config.num_beams > 1:
            result = self._beam_search(input_ids, attention_mask, config)
        else:
            result = self._sample(input_ids, attention_mask, config)
        
        # Decode
        generated_texts = []
        for i in range(len(prompts)):
            gen_ids = result.generated_ids[i].cpu().tolist()
            text = self.tokenizer.decode(gen_ids, skip_special=True)
            generated_texts.append(text)
        
        result.generated_text = generated_texts
        return result

    def _sample(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> GenerationResult:
        """Autoregressive sampling with KV cache."""
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        past_kv = None
        finish_reasons = ["max_length"] * batch_size
        finished = [False] * batch_size
        
        for step in range(config.max_new_tokens):
            if all(finished):
                break
            
            # Forward pass
            if past_kv is not None:
                # Only feed the last generated token
                model_input = generated[:, -1:]
                model_mask = attention_mask[:, -1:]
            else:
                model_input = generated
                model_mask = attention_mask
            
            outputs = self.model(
                input_ids=model_input,
                attention_mask=model_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            
            # Get next token logits
            next_logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated, config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature > 0 and config.do_sample:
                next_logits = next_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0 and config.do_sample:
                next_logits = self._apply_top_k(next_logits, config.top_k)
            
            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0 and config.do_sample:
                next_logits = self._apply_top_p(next_logits, config.top_p)
            
            # Sample or greedy
            if config.do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Handle finished sequences
            for i in range(batch_size):
                if finished[i]:
                    next_token[i, 0] = self.tokenizer.special_tokens.get("<pad>", 0)
                    continue
                
                # Check stop tokens
                if not config.ignore_eos:
                    if next_token[i, 0].item() == config.eos_token_id:
                        finish_reasons[i] = "eos"
                        finished[i] = True
                    elif next_token[i, 0].item() in config.stop_token_ids:
                        finish_reasons[i] = "stop_token"
                        finished[i] = True
            
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(batch_size, 1, dtype=torch.long, device=self.device)
            ], dim=-1)
        
        num_generated = min(config.max_new_tokens, generated.shape[1] - input_ids.shape[1])
        
        return GenerationResult(
            generated_ids=generated,
            generated_text=[],  # Decoded later
            finish_reason=finish_reasons,
            num_generated_tokens=num_generated,
        )

    def _beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        config: GenerationConfig,
    ) -> GenerationResult:
        """
        Beam search generation.
        
        Maintains `num_beams` hypotheses, expands each with all vocab tokens,
        scores by log probability, and prunes to top beams.
        
        For efficiency, we use a flat beam representation:
            beam_batch_size = batch_size * num_beams
        """
        batch_size = input_ids.shape[0]
        num_beams = config.num_beams
        beam_batch_size = batch_size * num_beams
        
        # Expand input for all beams
        expanded_ids = input_ids.unsqueeze(1).expand(-1, num_beams, -1).reshape(
            batch_size * num_beams, -1
        )
        expanded_mask = attention_mask.unsqueeze(1).expand(-1, num_beams, -1).reshape(
            batch_size * num_beams, -1
        )
        
        # Beam scores (log probabilities)
        beam_scores = torch.zeros(batch_size, num_beams, device=self.device)
        beam_scores[:, 1:] = float("-inf")  # Only first beam is active
        
        beam_scores = beam_scores.view(-1)
        finished_beams = [False] * batch_size
        
        for step in range(config.max_new_tokens):
            if all(finished_beams):
                break
            
            outputs = self.model(
                input_ids=expanded_ids,
                attention_mask=expanded_mask,
                use_cache=True,
            )
            
            next_logits = outputs.logits[:, -1, :]
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            
            vocab_size = next_log_probs.shape[-1]
            
            # Add beam scores
            next_scores = beam_scores.unsqueeze(-1) + next_log_probs
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            
            # Get top 2*num_beams candidates
            next_scores, next_tokens = next_scores.topk(2 * num_beams, dim=-1)
            
            # Convert flat indices to beam and token indices
            next_beam_indices = next_tokens // vocab_size
            next_token_indices = next_tokens % vocab_size
            
            # Select top num_beams
            next_beam_indices = next_beam_indices[:, :num_beams]
            next_token_indices = next_token_indices[:, :num_beams]
            next_scores = next_scores[:, :num_beams]
            
            # Reorder beams
            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, num_beams)
            expanded_ids = expanded_ids[
                (batch_indices * num_beams + next_beam_indices).view(-1)
            ]
            expanded_ids = torch.cat([
                expanded_ids,
                next_token_indices.view(-1, 1)
            ], dim=-1)
            
            expanded_mask = torch.cat([
                expanded_mask,
                torch.ones(beam_batch_size, 1, dtype=torch.long, device=self.device)
            ], dim=-1)
            
            beam_scores = next_scores.view(-1)
        
        # Reshape output
        generated = expanded_ids.view(batch_size, num_beams, -1)
        
        # Select best beam per batch
        best_beam_indices = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
        best_outputs = generated[
            torch.arange(batch_size), best_beam_indices
        ]
        
        return GenerationResult(
            generated_ids=best_outputs,
            generated_text=[],
            finish_reason=["eos"] * batch_size,
            num_generated_tokens=best_outputs.shape[1] - input_ids.shape[1],
            sequences_scores=beam_scores.view(batch_size, num_beams),
        )

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        callback=None,
    ):
        """
        Stream generation, yielding tokens one at a time.
        
        Args:
            prompt: Input prompt string.
            config: Generation configuration.
            callback: Optional callback function(token_text, is_finished).
        
        Yields:
            Generated text chunks as they're produced.
        """
        config = config or GenerationConfig()
        config.num_beams = 1  # No beam search in streaming
        
        result = ""
        token_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        generated = input_ids.clone()
        past_kv = None
        
        for step in range(config.max_new_tokens):
            if past_kv is not None:
                model_input = generated[:, -1:]
            else:
                model_input = generated
            
            outputs = self.model(
                input_ids=model_input,
                attention_mask=attention_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            
            next_logits = outputs.logits[:, -1, :]
            past_kv = outputs.past_key_values
            
            # Apply sampling
            if config.repetition_penalty != 1.0:
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated, config.repetition_penalty
                )
            if config.temperature > 0:
                next_logits = next_logits / config.temperature
            if config.top_k > 0:
                next_logits = self._apply_top_k(next_logits, config.top_k)
            if config.top_p < 1.0:
                next_logits = self._apply_top_p(next_logits, config.top_p)
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check stop tokens
            token_id = next_token[0, 0].item()
            is_finished = False
            if token_id == config.eos_token_id or token_id in config.stop_token_ids:
                is_finished = True
            
            # Decode and yield
            token_text = self.tokenizer.decode([token_id], skip_special=True)
            result += token_text
            
            if callback:
                callback(token_text, is_finished)
            
            yield token_text
            
            if is_finished:
                break
            
            generated = torch.cat([generated, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, dtype=torch.long, device=self.device)
            ], dim=-1)
    
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply repetition penalty to previously generated tokens."""
        for i in range(generated.shape[0]):
            unique_tokens = set(generated[i].tolist())
            for token_id in unique_tokens:
                if logits[i, token_id] > 0:
                    logits[i, token_id] /= penalty
                else:
                    logits[i, token_id] *= penalty
        return logits

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filter out tokens outside the top-k most probable."""
        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
        return logits

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filter out low-probability tokens (nucleus sampling)."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
        return logits
