"""Inference Engine - Core text generation with streaming support."""

import logging
import time
import threading
from typing import Generator, Dict, Any, Optional, List, Callable
from queue import Queue

import torch
from transformers import TextIteratorStreamer

from backend.model_manager import ModelManager
from backend.tokenizer_utils import TokenizerManager

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Core inference engine that handles text generation with support
    for streaming, batching, and various generation strategies.
    """

    def __init__(self, model_manager: ModelManager, tokenizer_manager: TokenizerManager):
        self.model_manager = model_manager
        self.tokenizer_manager = tokenizer_manager
        self._generation_count = 0
        self._total_tokens_generated = 0
        self._is_generating = False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
        num_beams: int = 1,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt (non-streaming).

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeated tokens
            max_new_tokens: Maximum tokens to generate
            num_beams: Number of beams for beam search
            stop_sequences: Optional list of strings to stop generation

        Returns:
            Dict with generated text, token counts, and timing info
        """
        start_time = time.time()

        # Ensure model is loaded
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer

        # Encode the prompt
        inputs = self.tokenizer_manager.encode(
            prompt,
            max_length=self.model_manager.model.max_position_embeddings
            if hasattr(self.model_manager.model, "max_position_embeddings")
            else 1024,
        )

        # Move inputs to device
        input_ids = inputs["input_ids"].to(self.model_manager.device)
        attention_mask = inputs["attention_mask"].to(self.model_manager.device)
        input_length = input_ids.shape[1]

        # Build generation config
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "num_beams": num_beams,
            "do_sample": temperature > 0 and num_beams == 1,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        # Generate
        self._is_generating = True
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

        self._is_generating = False

        # Decode only the new tokens
        new_tokens = output_ids[0][input_length:]
        generated_text = self.tokenizer_manager.decode(new_tokens.tolist())

        # Handle stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated_text:
                    generated_text = generated_text[: generated_text.index(seq)]

        # Calculate stats
        generation_time = time.time() - start_time
        tokens_generated = len(new_tokens)
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0

        self._generation_count += 1
        self._total_tokens_generated += tokens_generated

        return {
            "text": generated_text.strip(),
            "input_tokens": input_length,
            "output_tokens": tokens_generated,
            "total_tokens": input_length + tokens_generated,
            "generation_time": round(generation_time, 3),
            "tokens_per_second": round(tokens_per_second, 1),
        }

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate text with streaming - yields tokens one at a time.

        Args:
            prompt: Input text prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeated tokens
            max_new_tokens: Maximum tokens to generate
            on_token: Optional callback for each token

        Yields:
            Individual generated token strings
        """
        # Ensure model is loaded
        model = self.model_manager.model
        tokenizer = self.model_manager.tokenizer

        # Encode the prompt
        inputs = self.tokenizer_manager.encode(prompt)
        input_ids = inputs["input_ids"].to(self.model_manager.device)
        attention_mask = inputs["attention_mask"].to(self.model_manager.device)

        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # Build generation config
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Run generation in a separate thread
        self._is_generating = True
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        # Yield tokens as they come
        try:
            for new_text in streamer:
                if not self._is_generating:
                    break
                if new_text:
                    if on_token:
                        on_token(new_text)
                    yield new_text
        finally:
            thread.join(timeout=5.0)
            self._is_generating = False

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """
        Generate a chat response given conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for the assistant
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with response, token counts, and timing
        """
        # Format conversation into prompt
        prompt = self.tokenizer_manager.format_conversation(
            messages=messages,
            system_prompt=system_prompt,
        )

        return self.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop_sequences=["User:", "\n\nUser:", "Human:"],
        )

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 512,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream a chat response given conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt for the assistant
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            max_new_tokens: Maximum tokens to generate
            on_token: Optional callback for each token

        Yields:
            Individual generated token strings
        """
        prompt = self.tokenizer_manager.format_conversation(
            messages=messages,
            system_prompt=system_prompt,
        )

        yield from self.generate_stream(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            on_token=on_token,
        )

    def stop_generation(self) -> None:
        """Stop the current generation."""
        self._is_generating = False

    @property
    def is_generating(self) -> bool:
        """Check if currently generating."""
        return self._is_generating

    @property
    def stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "total_generations": self._generation_count,
            "total_tokens_generated": self._total_tokens_generated,
            "is_generating": self._is_generating,
        }
