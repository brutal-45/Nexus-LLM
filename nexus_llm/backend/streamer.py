"""Output streaming for Nexus-LLM backend.

Provides TextIteratorStreamer, callback streamer, and async streamer
for streaming generated text token by token.
"""

import threading
import asyncio
import time
from typing import Optional, Callable, Any, List, Generator, AsyncGenerator
from queue import Queue, Empty
import logging

logger = logging.getLogger(__name__)


class TextIteratorStreamer:
    """Stream generated text as an iterator, yielding partial strings.

    Thread-safe: generation runs in a separate thread while the
    main thread iterates over generated text.
    """

    def __init__(
        self,
        tokenizer: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        decode_kwargs: Optional[dict] = None,
        timeout: float = 30.0,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.decode_kwargs = decode_kwargs or {}
        self.timeout = timeout

        self._text_queue: Queue = Queue()
        self._stop_signal = None
        self._token_cache: List[int] = []
        self._prompt_length: Optional[int] = None
        self._generated_tokens: int = 0
        self._start_time: Optional[float] = None
        self._on_finalized_text: Optional[Callable] = None

    def put(self, value: Any) -> None:
        """Receive new tokens from the model and queue decoded text.

        Called by the generation thread.
        """
        if self._start_time is None:
            self._start_time = time.time()

        if isinstance(value, dict):
            value = value.get("input_ids", value)

        if hasattr(value, "shape"):
            if len(value.shape) == 2:
                value = value[0]
            value = value.tolist()

        if isinstance(value, list):
            if self.skip_prompt and self._prompt_length is None:
                self._prompt_length = len(value)
                return

            if self.skip_prompt and self._prompt_length is not None:
                new_tokens = value[self._prompt_length:]
                self._prompt_length = len(value)
            else:
                new_tokens = value if isinstance(value[0], int) else value[-1:]
                if not new_tokens:
                    return

            self._token_cache.extend(new_tokens)
            self._generated_tokens += len(new_tokens)

            decoded = self.tokenizer.decode(
                self._token_cache,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                **self.decode_kwargs,
            )

            new_text = decoded
            if hasattr(self, "_prev_decoded"):
                new_text = decoded[len(self._prev_decoded):]
            self._prev_decoded = decoded

            if new_text:
                self._text_queue.put(new_text)
        else:
            self._text_queue.put(str(value))

    def end(self) -> None:
        """Signal that generation is complete."""
        if self._token_cache:
            decoded = self.tokenizer.decode(
                self._token_cache,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                **self.decode_kwargs,
            )
            if hasattr(self, "_prev_decoded"):
                remaining = decoded[len(self._prev_decoded):]
                if remaining:
                    self._text_queue.put(remaining)

        self._text_queue.put(self._stop_signal)

    def __iter__(self) -> Generator[str, None, None]:
        """Iterate over generated text chunks."""
        while True:
            try:
                value = self._text_queue.get(timeout=self.timeout)
            except Empty:
                logger.warning("Streamer timeout - no text received")
                break

            if value == self._stop_signal:
                break
            yield value

    @property
    def generated_tokens(self) -> int:
        """Number of tokens generated so far."""
        return self._generated_tokens

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since generation started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def tokens_per_second(self) -> float:
        """Tokens generated per second."""
        elapsed = self.elapsed_time
        if elapsed <= 0:
            return 0.0
        return self._generated_tokens / elapsed


class CallbackStreamer:
    """Stream generated text via callbacks instead of iteration.

    Invokes a callback function for each generated text chunk.
    """

    def __init__(
        self,
        tokenizer: Any,
        callback: Callable[[str], None],
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        on_complete: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.tokenizer = tokenizer
        self.callback = callback
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.on_complete = on_complete
        self.on_error = on_error

        self._token_cache: List[int] = []
        self._prompt_length: Optional[int] = None
        self._prev_decoded: str = ""
        self._generated_tokens: int = 0
        self._is_finished: bool = False

    def put(self, value: Any) -> None:
        """Process new tokens and invoke callback with decoded text."""
        try:
            if isinstance(value, dict):
                value = value.get("input_ids", value)

            if hasattr(value, "shape"):
                if len(value.shape) == 2:
                    value = value[0]
                value = value.tolist()

            if isinstance(value, list):
                if self.skip_prompt and self._prompt_length is None:
                    self._prompt_length = len(value)
                    return

                if self.skip_prompt and self._prompt_length is not None:
                    new_tokens = value[self._prompt_length:]
                    self._prompt_length = len(value)
                else:
                    new_tokens = value if isinstance(value[0], int) else value[-1:]
                    if not new_tokens:
                        return

                self._token_cache.extend(new_tokens)
                self._generated_tokens += len(new_tokens)

                decoded = self.tokenizer.decode(
                    self._token_cache,
                    skip_special_tokens=self.skip_special_tokens,
                    clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
                )

                new_text = decoded[len(self._prev_decoded):]
                self._prev_decoded = decoded

                if new_text:
                    self.callback(new_text)

        except Exception as e:
            if self.on_error:
                self.on_error(e)
            else:
                logger.error(f"CallbackStreamer error: {e}")

    def end(self) -> None:
        """Signal completion and flush any remaining text."""
        if self._token_cache:
            decoded = self.tokenizer.decode(
                self._token_cache,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            remaining = decoded[len(self._prev_decoded):]
            if remaining:
                self.callback(remaining)

        self._is_finished = True
        if self.on_complete:
            self.on_complete()

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens


class AsyncStreamer:
    """Async streamer for use with asyncio-based generation.

    Yields text chunks as an async generator.
    """

    def __init__(
        self,
        tokenizer: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces

        self._queue: asyncio.Queue = asyncio.Queue()
        self._token_cache: List[int] = []
        self._prompt_length: Optional[int] = None
        self._prev_decoded: str = ""
        self._generated_tokens: int = 0
        self._stop_signal = object()

    def put(self, value: Any) -> None:
        """Process new tokens (called from generation thread)."""
        if isinstance(value, dict):
            value = value.get("input_ids", value)

        if hasattr(value, "shape"):
            if len(value.shape) == 2:
                value = value[0]
            value = value.tolist()

        if isinstance(value, list):
            if self.skip_prompt and self._prompt_length is None:
                self._prompt_length = len(value)
                return

            if self.skip_prompt and self._prompt_length is not None:
                new_tokens = value[self._prompt_length:]
                self._prompt_length = len(value)
            else:
                new_tokens = value if isinstance(value[0], int) else value[-1:]
                if not new_tokens:
                    return

            self._token_cache.extend(new_tokens)
            self._generated_tokens += len(new_tokens)

            decoded = self.tokenizer.decode(
                self._token_cache,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )

            new_text = decoded[len(self._prev_decoded):]
            self._prev_decoded = decoded

            if new_text:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._queue.put(new_text))
                    else:
                        loop.run_until_complete(self._queue.put(new_text))
                except RuntimeError:
                    pass

    def end(self) -> None:
        """Signal that generation is complete."""
        if self._token_cache:
            decoded = self.tokenizer.decode(
                self._token_cache,
                skip_special_tokens=self.skip_special_tokens,
                clean_up_tokenization_spaces=self.clean_up_tokenization_spaces,
            )
            remaining = decoded[len(self._prev_decoded):]
            if remaining:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.ensure_future(self._queue.put(remaining))
                    else:
                        loop.run_until_complete(self._queue.put(remaining))
                except RuntimeError:
                    pass

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._queue.put(self._stop_signal))
            else:
                loop.run_until_complete(self._queue.put(self._stop_signal))
        except RuntimeError:
            pass

    async def stream(self) -> AsyncGenerator[str, None]:
        """Async generator that yields text chunks."""
        while True:
            value = await self._queue.get()
            if value is self._stop_signal:
                break
            yield value

    @property
    def generated_tokens(self) -> int:
        return self._generated_tokens


def create_streamer(
    streamer_type: str = "iterator",
    tokenizer: Any = None,
    callback: Optional[Callable] = None,
    skip_prompt: bool = True,
    skip_special_tokens: bool = True,
    **kwargs,
) -> Any:
    """Factory function to create the appropriate streamer.

    Args:
        streamer_type: One of 'iterator', 'callback', or 'async'.
        tokenizer: The tokenizer to use for decoding.
        callback: Required for 'callback' type streamer.
        skip_prompt: Whether to skip the prompt tokens.
        skip_special_tokens: Whether to skip special tokens in output.

    Returns:
        A streamer instance.
    """
    common_kwargs = {
        "tokenizer": tokenizer,
        "skip_prompt": skip_prompt,
        "skip_special_tokens": skip_special_tokens,
    }

    if streamer_type == "iterator":
        return TextIteratorStreamer(**common_kwargs, **kwargs)
    elif streamer_type == "callback":
        if callback is None:
            raise ValueError("callback is required for CallbackStreamer")
        return CallbackStreamer(callback=callback, **common_kwargs, **kwargs)
    elif streamer_type == "async":
        return AsyncStreamer(**common_kwargs, **kwargs)
    else:
        raise ValueError(f"Unknown streamer type: {streamer_type}")
