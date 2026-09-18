"""Output streaming for Nexus-LLM backend.

Provides TextIteratorStreamer, callback streamer, and async streamer
for streaming generated text token by token.

All three share :class:`_TokenStreamDecoder`, whose semantics match
``transformers``' own streamers: :meth:`put` is called by ``model.generate()``
with the *newly produced* tokens on every decoding step (a few call styles are
supported, see :meth:`_TokenStreamDecoder._feed`), text is decoded from the
accumulated cache, and only the portion that will not change any more is
emitted.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Generator
from queue import Empty, Queue
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _flatten_tokens(value: Any) -> list[int]:
    """Coerce a tensor / dict / nested list of token ids into a flat int list."""
    if isinstance(value, dict):
        value = value.get("input_ids", next(iter(value.values()), value))

    if hasattr(value, "tolist"):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        return []

    # ``[[1, 2, 3]]`` (batch of one sequence) or ``[[1], [2]]`` (per-step)
    if value and isinstance(value[0], (list, tuple)):
        flat: list[int] = []
        for row in value:
            flat.extend(_flatten_tokens(row))
        return flat
    return [int(tok) for tok in value if isinstance(tok, (int, float))]


def _is_cjk(code_point: int) -> bool:
    """Whether ``code_point`` is a CJK character (safe to print immediately)."""
    return any(
        start <= code_point <= end
        for start, end in (
            (0x4E00, 0x9FFF),
            (0x3400, 0x4DBF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B820, 0x2CEAF),
            (0xF900, 0xFAFF),
            (0x2F800, 0x2FA1F),
        )
    )


class _TokenStreamDecoder:
    """Incremental decode + emit logic shared by every streamer.

    Subclasses call :meth:`_feed` from their ``put`` and :meth:`_flush`` from
    their ``end``, and forward whatever text is returned to their sink (queue,
    callback, async queue, ...).
    """

    def _init_decoder(
        self,
        tokenizer: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        decode_kwargs: dict | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.decode_kwargs = dict(decode_kwargs or {})

        self._token_cache: list[int] = []
        self._print_len = 0
        self._consumed_tokens = 0
        self._next_tokens_are_prompt = True
        self._generated_tokens = 0
        self._start_time: float | None = None

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode(self, token_ids: list[int]) -> str:
        """Decode ``token_ids``, tolerating tokenizers with narrower signatures."""
        attempts = (
            {
                "skip_special_tokens": self.skip_special_tokens,
                "clean_up_tokenization_spaces": self.clean_up_tokenization_spaces,
                **self.decode_kwargs,
            },
            {"skip_special_tokens": self.skip_special_tokens, **self.decode_kwargs},
            dict(self.decode_kwargs),
            {},
        )
        for kwargs in attempts:
            try:
                return str(self.tokenizer.decode(token_ids, **kwargs))
            except TypeError:
                continue
            except (ValueError, IndexError) as exc:
                logger.warning("Tokenizer decode failed: %s", exc)
                return ""
        return ""

    def _emit(self, decoded: str, *, final: bool) -> str:
        """Return the part of ``decoded`` that is safe to show right now.

        Mirrors the ``transformers`` heuristic: a trailing partial word may be
        re-written by the next token, so it is held back until a space follows
        (or generation ends).  CJK characters have no such ambiguity and are
        released immediately.
        """
        pending = decoded[self._print_len :]
        if not pending:
            return ""
        if final:
            self._print_len = len(decoded)
            return pending

        cut = decoded.rfind(" ")
        if cut > self._print_len:
            text = decoded[self._print_len : cut + 1]
        elif pending and _is_cjk(ord(pending[-1])):
            text = pending
        else:
            return ""
        self._print_len += len(text)
        return text

    def _feed(self, value: Any) -> list[str]:
        """Ingest one ``put`` and return the text chunks to forward.

        Supports both call styles seen in the wild: ``model.generate()``
        handing over only the freshly generated ids, and callers feeding the
        whole growing ``input_ids`` sequence each step.
        """
        if self._start_time is None:
            self._start_time = time.time()

        tokens = _flatten_tokens(value)
        if not tokens:
            return []

        if self.skip_prompt and self._next_tokens_are_prompt:
            # First payload is the prompt: remember its length, render nothing.
            self._next_tokens_are_prompt = False
            self._consumed_tokens = len(tokens)
            return []

        if len(tokens) > self._consumed_tokens:
            # Caller re-sent the full sequence; keep only the new tail.
            new_tokens = tokens[self._consumed_tokens :]
            self._consumed_tokens = len(tokens)
        else:
            new_tokens = tokens
            self._consumed_tokens += len(tokens)

        if not new_tokens:
            return []

        self._token_cache.extend(new_tokens)
        self._generated_tokens += len(new_tokens)
        decoded = self._decode(self._token_cache)
        chunk = self._emit(decoded, final=False)
        return [chunk] if chunk else []

    def _flush(self) -> list[str]:
        """Release any text held back by the word-boundary heuristic."""
        if not self._token_cache:
            return []
        chunk = self._emit(self._decode(self._token_cache), final=True)
        return [chunk] if chunk else []

    def _finish(self) -> None:
        """Reset per-generation state so the streamer can be reused."""
        self._token_cache = []
        self._print_len = 0
        self._consumed_tokens = 0
        self._next_tokens_are_prompt = True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

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


class TextIteratorStreamer(_TokenStreamDecoder):
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
        decode_kwargs: dict | None = None,
        timeout: float = 30.0,
    ):
        self._init_decoder(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            decode_kwargs=decode_kwargs,
        )
        self.timeout = timeout
        self._text_queue: Queue = Queue()
        # A unique sentinel: ``None`` could legitimately appear in a queue.
        self._stop_signal = object()
        self._on_finalized_text: Callable | None = None

    def put(self, value: Any) -> None:
        """Receive new tokens from the model and queue decoded text.

        Called by the generation thread.
        """
        for chunk in self._feed(value):
            self._text_queue.put(chunk)

    def end(self) -> None:
        """Signal that generation is complete, flushing any held-back text."""
        for chunk in self._flush():
            self._text_queue.put(chunk)
        self._finish()
        self._text_queue.put(self._stop_signal)

    def __iter__(self) -> Generator[str, None, None]:
        """Iterate over generated text chunks."""
        while True:
            try:
                value = self._text_queue.get(timeout=self.timeout)
            except Empty:
                logger.warning("Streamer timeout - no text received")
                break

            if value is self._stop_signal:
                break
            if value:
                yield value

    def get(self, timeout: float | None = None) -> str:
        """Pop a single chunk (useful for tests and non-iterating consumers)."""
        return self._text_queue.get(timeout=timeout if timeout is not None else self.timeout)


class CallbackStreamer(_TokenStreamDecoder):
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
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        self._init_decoder(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        self.callback = callback
        self.on_complete = on_complete
        self.on_error = on_error
        self._is_finished = False

    def put(self, value: Any) -> None:
        """Process new tokens and invoke callback with decoded text."""
        try:
            for chunk in self._feed(value):
                self.callback(chunk)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(exc)
            else:
                logger.error("CallbackStreamer error: %s", exc)

    def end(self) -> None:
        """Signal completion and flush any remaining text."""
        try:
            for chunk in self._flush():
                self.callback(chunk)
        except Exception as exc:
            if self.on_error is not None:
                self.on_error(exc)
            else:
                logger.error("CallbackStreamer error: %s", exc)
        finally:
            self._finish()
            self._is_finished = True
            if self.on_complete is not None:
                self.on_complete()

    @property
    def is_finished(self) -> bool:
        return self._is_finished


class AsyncStreamer(_TokenStreamDecoder):
    """Async streamer for use with asyncio-based generation.

    Yields text chunks as an async generator. :meth:`put` is safe to call from
    a synchronous generation loop; it schedules the queue write on the running
    event loop when there is one.
    """

    def __init__(
        self,
        tokenizer: Any,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
        maxsize: int = 0,
    ):
        self._init_decoder(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._stop_signal = object()
        self._pending: list[Any] = []
        self._finished = False

    def _push(self, item: Any) -> None:
        """Queue an item, whether or not an event loop is currently running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.call_soon_threadsafe(self._queue.put_nowait, item)
        else:
            # No loop yet (e.g. producer started before the consumer): buffer
            # so nothing is lost, and drain when the consumer attaches.
            self._pending.append(item)

    def put(self, value: Any) -> None:
        """Process new tokens (called from generation thread)."""
        for chunk in self._feed(value):
            self._push(chunk)

    def end(self) -> None:
        """Signal that generation is complete."""
        for chunk in self._flush():
            self._push(chunk)
        self._finish()
        self._finished = True
        self._push(self._stop_signal)

    async def stream(self) -> AsyncGenerator[str, None]:
        """Async generator that yields text chunks."""
        while True:
            if self._pending:
                value = self._pending.pop(0)
            else:
                value = await self._queue.get()
            if value is self._stop_signal:
                break
            if value:
                yield value

    def __aiter__(self) -> AsyncGenerator[str, None]:
        return self.stream()

    @property
    def is_finished(self) -> bool:
        return self._finished


def create_streamer(
    streamer_type: str = "iterator",
    tokenizer: Any = None,
    callback: Callable | None = None,
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

    Raises:
        ValueError: If ``streamer_type`` is unknown, or ``callback`` is missing
            for the 'callback' type.
    """
    common_kwargs = {
        "tokenizer": tokenizer,
        "skip_prompt": skip_prompt,
        "skip_special_tokens": skip_special_tokens,
    }

    if streamer_type == "iterator":
        return TextIteratorStreamer(**common_kwargs, **kwargs)
    if streamer_type == "callback":
        if callback is None:
            raise ValueError("callback is required for CallbackStreamer")
        return CallbackStreamer(callback=callback, **common_kwargs, **kwargs)
    if streamer_type == "async":
        return AsyncStreamer(**common_kwargs, **kwargs)
    raise ValueError(f"Unknown streamer type: {streamer_type}")
