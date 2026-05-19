"""Inference engine for Nexus-LLM backend.

Provides generate(), stream_generate(), and batch_generate() with full support
for temperature, top_p, top_k, repetition_penalty, beam search, and
num_return_sequences.
"""

import torch
import asyncio
from typing import List, Optional, Dict, Any, AsyncGenerator, Generator, Callable
from dataclasses import dataclass
import time
import logging

from .generation import GenerationConfig, GenerationPresets
from .logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsProcessor, TopKLogitsProcessor, TopPLogitsProcessor,
    MinLengthLogitsProcessor, MinNewTokensLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
)
from .stopping import (
    StoppingCriteriaBuilder, MaxLengthCriteria, MaxNewTokensCriteria,
    EosTokenCriteria, TimeLimitCriteria, CompositeStoppingCriteria,
)
from .sampling import SamplingConfig, CombinedSampling

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str
    token_ids: List[int]
    score: float = 0.0
    num_tokens: int = 0
    num_prompt_tokens: int = 0
    generation_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    finish_reason: str = "length"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.num_tokens == 0:
            self.num_tokens = len(self.token_ids)
        if self.generation_time_seconds > 0 and self.num_tokens > 0:
            self.tokens_per_second = self.num_tokens / self.generation_time_seconds


class InferenceEngine:
    """Complete inference engine supporting synchronous, streaming, and batch generation."""

    def __init__(self, model_manager=None):
        self._model_manager = model_manager
        self._default_config = GenerationConfig()

    def _get_model_and_tokenizer(self, model_id: Optional[str] = None):
        """Get model and tokenizer from the model manager."""
        if self._model_manager is None:
            raise RuntimeError("No model manager configured for InferenceEngine")
        model = self._model_manager.get_model(model_id)
        tokenizer = self._model_manager.get_tokenizer(model_id)
        return model, tokenizer

    def _build_logits_processors(self, config: GenerationConfig, prompt_length: int) -> LogitsProcessorList:
        """Build the logits processor chain from a generation config."""
        processors = LogitsProcessorList()

        if config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(config.repetition_penalty))

        if config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size))

        if config.min_length > 0 and config.eos_token_id is not None:
            processors.append(MinLengthLogitsProcessor(config.min_length, config.eos_token_id))

        if config.min_new_tokens > 0 and config.eos_token_id is not None:
            processors.append(MinNewTokensLogitsProcessor(prompt_length, config.min_new_tokens, config.eos_token_id))

        if config.temperature != 1.0 and config.do_sample:
            processors.append(TemperatureLogitsProcessor(config.temperature))

        if config.top_k > 0 and config.do_sample:
            processors.append(TopKLogitsProcessor(config.top_k))

        if config.top_p < 1.0 and config.do_sample:
            processors.append(TopPLogitsProcessor(config.top_p))

        return processors

    def _build_stopping_criteria(
        self,
        config: GenerationConfig,
        prompt_length: int,
        eos_token_id: Optional[int] = None,
    ) -> CompositeStoppingCriteria:
        """Build stopping criteria from a generation config."""
        builder = StoppingCriteriaBuilder()

        if config.max_length is not None:
            builder.max_length(config.max_length)

        builder.max_new_tokens(prompt_length, config.max_new_tokens)

        if eos_token_id is not None:
            eos_ids = [eos_token_id]
            if config.eos_token_id is not None:
                if isinstance(config.eos_token_id, list):
                    eos_ids.extend(config.eos_token_id)
                else:
                    eos_ids.append(config.eos_token_id)
            builder.eos_token(eos_token_id=eos_token_id, eos_token_ids=eos_ids)

        return builder.build(mode="any")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model_id: Optional[str] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text from a prompt synchronously.

        Args:
            prompt: Input text prompt.
            config: Generation configuration. Uses defaults if None.
            model_id: Specific model to use. Uses active model if None.
            stop_strings: Optional strings that trigger stopping.

        Returns:
            GenerationResult with generated text and metadata.
        """
        model, tokenizer = self._get_model_and_tokenizer(model_id)
        gen_config = config or self._default_config

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_length = input_ids.shape[-1]
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        logits_processors = self._build_logits_processors(gen_config, prompt_length)
        stopping_criteria = self._build_stopping_criteria(gen_config, prompt_length, eos_token_id)

        start_time = time.time()

        with torch.no_grad():
            if gen_config.num_beams > 1:
                output = model.generate(
                    input_ids,
                    max_new_tokens=gen_config.max_new_tokens,
                    num_beams=gen_config.num_beams,
                    num_return_sequences=gen_config.num_return_sequences,
                    length_penalty=gen_config.length_penalty,
                    early_stopping=gen_config.early_stopping,
                    num_beam_groups=gen_config.num_beam_groups,
                    diversity_penalty=gen_config.diversity_penalty,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    repetition_penalty=gen_config.repetition_penalty,
                    no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                    do_sample=gen_config.do_sample,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=gen_config.use_cache,
                    output_scores=gen_config.output_scores,
                )
            else:
                output = model.generate(
                    input_ids,
                    max_new_tokens=gen_config.max_new_tokens,
                    do_sample=gen_config.do_sample,
                    temperature=gen_config.temperature,
                    top_p=gen_config.top_p,
                    top_k=gen_config.top_k,
                    repetition_penalty=gen_config.repetition_penalty,
                    no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    use_cache=gen_config.use_cache,
                    output_scores=gen_config.output_scores,
                )

        generation_time = time.time() - start_time
        generated_ids = output[0, prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        if stop_strings:
            for stop_str in stop_strings:
                idx = generated_text.find(stop_str)
                if idx != -1:
                    generated_text = generated_text[:idx]

        num_generated = len(generated_ids)
        finish_reason = "stop" if (eos_token_id is not None and generated_ids[-1].item() == eos_token_id) else "length"

        return GenerationResult(
            text=generated_text,
            token_ids=generated_ids.tolist(),
            num_tokens=num_generated,
            num_prompt_tokens=prompt_length,
            generation_time_seconds=generation_time,
            finish_reason=finish_reason,
        )

    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream generated text token by token.

        Yields partial text strings as they are generated.
        """
        from threading import Thread
        from transformers import TextIteratorStreamer

        model, tokenizer = self._get_model_and_tokenizer(model_id)
        gen_config = config or self._default_config

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": gen_config.max_new_tokens,
            "do_sample": gen_config.do_sample,
            "temperature": gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "repetition_penalty": gen_config.repetition_penalty,
            "no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
            "use_cache": gen_config.use_cache,
        }

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            if text:
                yield text

        thread.join()

    async def async_stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        model_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Async version of stream_generate."""
        loop = asyncio.get_event_loop()
        sync_gen = self.stream_generate(prompt, config, model_id)

        for text in sync_gen:
            yield text
            await asyncio.sleep(0)

    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        model_id: Optional[str] = None,
    ) -> List[GenerationResult]:
        """Generate text for a batch of prompts.

        Uses efficient batching by padding inputs to the same length.
        """
        if not prompts:
            return []

        model, tokenizer = self._get_model_and_tokenizer(model_id)
        gen_config = config or self._default_config

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=gen_config.max_length or tokenizer.model_max_length,
        )

        device = next(model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        start_time = time.time()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_config.max_new_tokens,
                do_sample=gen_config.do_sample,
                temperature=gen_config.temperature,
                top_p=gen_config.top_p,
                top_k=gen_config.top_k,
                repetition_penalty=gen_config.repetition_penalty,
                no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
                num_beams=gen_config.num_beams,
                num_return_sequences=gen_config.num_return_sequences,
                length_penalty=gen_config.length_penalty,
                early_stopping=gen_config.early_stopping,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=gen_config.use_cache,
            )

        generation_time = time.time() - start_time

        results = []
        for i, (out_seq, p_len) in enumerate(zip(output, prompt_lengths)):
            generated_ids = out_seq[p_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            num_generated = len(generated_ids)
            finish_reason = "stop" if (eos_token_id is not None and generated_ids[-1].item() == eos_token_id) else "length"

            results.append(GenerationResult(
                text=generated_text,
                token_ids=generated_ids.tolist(),
                num_tokens=num_generated,
                num_prompt_tokens=p_len,
                generation_time_seconds=generation_time / len(prompts),
                finish_reason=finish_reason,
            ))

        return results

    def generate_with_preset(
        self,
        prompt: str,
        preset: str = "balanced",
        model_id: Optional[str] = None,
        **overrides,
    ) -> GenerationResult:
        """Generate using a named preset with optional overrides."""
        config = GenerationPresets.get_preset(preset)
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return self.generate(prompt, config=config, model_id=model_id)
