"""Inference engine for Nexus-LLM.

Provides single-shot and streaming generation for both raw text completions
and conversational (chat) interactions.  Thread-safe via ``threading.Lock``.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Any

import torch

from nexus_llm.backend.model_manager import ModelManager
from nexus_llm.backend.tokenizer_utils import TokenizerManager
from nexus_llm.core.exceptions import InferenceError, ModelLoadError

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Generation stats
# ------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Accumulated generation statistics."""
    generation_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    @property
    def avg_tokens_per_second(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.total_tokens / self.total_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation_count": self.generation_count,
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time, 4),
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 2),
        }


# ------------------------------------------------------------------
# InferenceEngine
# ------------------------------------------------------------------

class InferenceEngine:
    """High-level inference engine that orchestrates model and tokenizer.

    Features:
    * ``generate()`` — single-shot text completion
    * ``generate_stream()`` — streaming text completion via ``TextIteratorStreamer``
    * ``chat()`` — single-shot conversational generation
    * ``chat_stream()`` — streaming conversational generation
    * ``stop_generation()`` — interrupt an in-progress generation
    * Thread-safe via an internal ``threading.Lock``
    * Accumulated generation statistics
    """

    def __init__(
        self,
        model_manager: Optional[ModelManager] = None,
        tokenizer_manager: Optional[TokenizerManager] = None,
    ) -> None:
        self._model_manager = model_manager or ModelManager()
        self._tokenizer_manager = tokenizer_manager or TokenizerManager()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stats = GenerationStats()
        self._generation_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    @property
    def tokenizer_manager(self) -> TokenizerManager:
        return self._tokenizer_manager

    @property
    def stats(self) -> GenerationStats:
        return self._stats

    @property
    def is_ready(self) -> bool:
        return self._model_manager.is_loaded and self._tokenizer_manager.is_loaded

    # ------------------------------------------------------------------
    # Convenience: load / unload
    # ------------------------------------------------------------------

    def load_model(
        self,
        model_id: str,
        device: str = "auto",
        precision: str = "fp32",
        cache_dir: Optional[str] = None,
    ) -> None:
        """Load both the model and its tokenizer atomically.

        If the model loads but the tokenizer fails the model is unloaded so
        the engine stays in a consistent state.

        Raises:
            ModelNotFoundError: If *model_id* is not in the catalogue.
            ModelLoadError: If the model or tokenizer fails to load.
        """
        with self._lock:
            self._model_manager.load(model_id, device=device, precision=precision, cache_dir=cache_dir)
            try:
                self._tokenizer_manager.load(model_id, cache_dir=cache_dir)
            except Exception:
                self._model_manager.unload()
                raise

    def unload_model(self) -> None:
        """Unload both model and tokenizer."""
        with self._lock:
            self._model_manager.unload()
            self._tokenizer_manager.unload()

    # ------------------------------------------------------------------
    # Stop generation
    # ------------------------------------------------------------------

    def stop_generation(self) -> None:
        """Signal the current generation to stop as soon as possible."""
        self._stop_event.set()
        logger.info("Generation stop requested")

    def _clear_stop(self) -> None:
        self._stop_event.clear()

    # ------------------------------------------------------------------
    # Generate (single-shot completion)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        num_beams: int = 1,
        do_sample: bool = True,
        **extra_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a single-shot completion.

        Args:
            prompt: The input prompt text.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling parameter.
            repetition_penalty: Repetition penalty factor.
            num_beams: Number of beams for beam search.
            do_sample: Whether to sample (True) or use greedy decoding.
            **extra_kwargs: Additional kwargs passed to ``model.generate()``.

        Returns:
            Dict with keys ``text``, ``prompt_tokens``, ``generated_tokens``,
            ``total_tokens``, and ``generation_time_s``.

        Raises:
            InferenceError: If the engine is not ready or generation fails.
        """
        self._ensure_ready()

        with self._lock:
            self._clear_stop()
            start = time.monotonic()

            try:
                tokenizer = self._tokenizer_manager.tokenizer
                model = self._model_manager.model

                inputs = tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
                input_length = inputs["input_ids"].shape[1]

                # Move inputs to the model's device
                device = next(model.parameters()).device  # type: ignore[union-attr]
                inputs = {k: v.to(device) for k, v in inputs.items()}

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "num_beams": num_beams,
                    "do_sample": do_sample,
                    "pad_token_id": self._tokenizer_manager.eos_token_id,
                    "eos_token_id": self._tokenizer_manager.eos_token_id,
                    **extra_kwargs,
                }

                output_ids = model.generate(**inputs, **gen_kwargs)  # type: ignore[union-attr]

                # Slice out only the new tokens
                new_tokens = output_ids[0][input_length:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)  # type: ignore[union-attr]
                generated_count = len(new_tokens)

                elapsed = time.monotonic() - start
                self._update_stats(generated_count, elapsed)

                return {
                    "text": generated_text,
                    "prompt_tokens": input_length,
                    "generated_tokens": generated_count,
                    "total_tokens": input_length + generated_count,
                    "generation_time_s": round(elapsed, 4),
                }

            except Exception as exc:
                raise InferenceError(f"Generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Generate (streaming)
    # ------------------------------------------------------------------

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **extra_kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream generated tokens one by one.

        Yields each token as a string as it is produced.  Internally uses
        HuggingFace's ``TextIteratorStreamer`` on a background thread.

        Raises:
            InferenceError: If the engine is not ready or generation fails.
        """
        self._ensure_ready()

        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer_manager.tokenizer
        model = self._model_manager.model

        inputs = tokenizer(prompt, return_tensors="pt")  # type: ignore[union-attr]
        input_length = inputs["input_ids"].shape[1]

        device = next(model.parameters()).device  # type: ignore[union-attr]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(
            tokenizer,  # type: ignore[arg-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs: Dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": True,
            "pad_token_id": self._tokenizer_manager.eos_token_id,
            "eos_token_id": self._tokenizer_manager.eos_token_id,
            **extra_kwargs,
        }

        self._clear_stop()
        start = time.monotonic()
        total_generated = 0

        # Run generation on a background thread
        thread = threading.Thread(
            target=model.generate,  # type: ignore[union-attr]
            kwargs=gen_kwargs,
        )
        thread.start()

        try:
            for token_text in streamer:
                if self._stop_event.is_set():
                    break
                total_generated += 1
                yield token_text
        finally:
            thread.join(timeout=5.0)
            elapsed = time.monotonic() - start
            self._update_stats(total_generated, elapsed)

    # ------------------------------------------------------------------
    # Chat (single-shot)
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **extra_kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a conversational (chat) response.

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            max_new_tokens: Maximum new tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling.
            repetition_penalty: Repetition penalty.
            **extra_kwargs: Additional kwargs for ``model.generate()``.

        Returns:
            Same dict shape as ``generate()``, plus ``messages_in``.

        Raises:
            InferenceError: If the engine is not ready or generation fails.
        """
        self._ensure_ready()

        prompt = self._tokenizer_manager.format_conversation(messages)
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **extra_kwargs,
        )
        result["messages_in"] = len(messages)
        return result

    # ------------------------------------------------------------------
    # Chat (streaming)
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **extra_kwargs: Any,
    ) -> Generator[str, None, None]:
        """Stream a conversational (chat) response token by token.

        Yields each token as it is produced.

        Raises:
            InferenceError: If the engine is not ready or generation fails.
        """
        self._ensure_ready()

        prompt = self._tokenizer_manager.format_conversation(messages)
        yield from self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **extra_kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        """Raise InferenceError if the engine is not ready for inference."""
        if not self._model_manager.is_loaded:
            raise InferenceError("No model is loaded. Call load_model() first.")
        if not self._tokenizer_manager.is_loaded:
            raise InferenceError("No tokenizer is loaded. Call load_model() first.")

    def _update_stats(self, num_tokens: int, elapsed: float) -> None:
        """Thread-safe update of accumulated stats."""
        self._stats.generation_count += 1
        self._stats.total_tokens += num_tokens
        self._stats.total_time += elapsed

    # ------------------------------------------------------------------
    # Public stats accessor
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return accumulated generation statistics as a dict."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset accumulated generation statistics."""
        self._stats = GenerationStats()
