"""Tokenizer management for Nexus-LLM.

Handles loading tokenizers, managing chat templates, and encoding/decoding.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Sequence

from transformers import AutoTokenizer

from nexus_llm.core.exceptions import ModelLoadError, ModelNotFoundError
from nexus_llm.core.model_catalog import get_model_info

logger = logging.getLogger(__name__)


class ChatTemplate(str, Enum):
    """Supported chat template formats."""

    GPT2 = "gpt2"
    DIALOGPT = "dialogpt"
    LLAMA = "llama"
    CHATML = "chatml"
    PHI = "phi"
    QWEN = "qwen"
    GEMMA = "gemma"
    DEFAULT = "default"


# Mapping from model category to chat template
_CATEGORY_TEMPLATE_MAP: dict[str, ChatTemplate] = {
    "gpt2": ChatTemplate.GPT2,
    "dialogpt": ChatTemplate.DIALOGPT,
    "llama": ChatTemplate.LLAMA,
    "phi": ChatTemplate.PHI,
    "qwen": ChatTemplate.QWEN,
    "gemma": ChatTemplate.GEMMA,
    "smollm": ChatTemplate.CHATML,
    "stablelm": ChatTemplate.CHATML,
    "opt": ChatTemplate.GPT2,
    "pythia": ChatTemplate.GPT2,
    "bloom": ChatTemplate.GPT2,
    "flan-t5": ChatTemplate.DEFAULT,
    "mamba": ChatTemplate.GPT2,
}


class TokenizerManager:
    """Manages tokenizer loading, chat templates, and token operations."""

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None
        self._model_id: str | None = None
        self._template: ChatTemplate = ChatTemplate.DEFAULT
        self._pad_token_set: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_id: str, cache_dir: str | None = None) -> None:
        """Load a tokenizer for the given model ID.

        Args:
            model_id: Short model ID from the catalog (e.g. "gpt2-medium").
            cache_dir: Optional HuggingFace cache directory.

        Raises:
            ModelNotFoundError: If the model_id is not in the catalog.
            ModelLoadError: If the tokenizer fails to load.
        """
        try:
            info = get_model_info(model_id)
        except ModelNotFoundError:
            raise

        hf_id = info.hf_id
        logger.info("Loading tokenizer for %s (%s)", model_id, hf_id)

        try:
            kwargs: dict = {"trust_remote_code": True}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir

            self._tokenizer = AutoTokenizer.from_pretrained(hf_id, **kwargs)

            # Ensure pad token is set for causal LMs that lack one
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                    self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
                    self._pad_token_set = True
                    logger.debug("Set pad_token to eos_token for %s", hf_id)

            self._model_id = model_id
            self._template = self._detect_template(model_id, info.category)
            logger.info(
                "Tokenizer loaded successfully (template=%s, vocab_size=%d)",
                self._template.value,
                len(self._tokenizer),
            )

        except ModelNotFoundError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load tokenizer for '{model_id}': {exc}") from exc

    def unload(self) -> None:
        """Unload the current tokenizer and release resources."""
        self._tokenizer = None
        self._model_id = None
        self._template = ChatTemplate.DEFAULT
        self._pad_token_set = False
        logger.info("Tokenizer unloaded")

    @property
    def is_loaded(self) -> bool:
        """Whether a tokenizer is currently loaded."""
        return self._tokenizer is not None

    @property
    def model_id(self) -> str | None:
        """The model ID of the currently loaded tokenizer."""
        return self._model_id

    @property
    def tokenizer(self) -> AutoTokenizer | None:
        """The underlying HuggingFace tokenizer."""
        return self._tokenizer

    @property
    def template(self) -> ChatTemplate:
        """The active chat template."""
        return self._template

    # ------------------------------------------------------------------
    # Template detection
    # ------------------------------------------------------------------

    def _detect_template(self, model_id: str, category: str) -> ChatTemplate:
        """Auto-detect the best chat template based on model metadata.

        First tries the tokenizer's built-in chat_template attribute, then
        falls back to a category-based lookup.
        """
        # Prefer the tokenizer's own chat_template if available
        if (
            self._tokenizer is not None
            and hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
        ):
            tmpl_str = str(self._tokenizer.chat_template).lower()
            if "llama" in tmpl_str or ("<|im_start|>" not in tmpl_str and "system" in tmpl_str):
                return ChatTemplate.LLAMA
            if "<|im_start|>" in tmpl_str:
                return ChatTemplate.CHATML
            if "phi" in tmpl_str:
                return ChatTemplate.PHI
            if "qwen" in tmpl_str:
                return ChatTemplate.QWEN
            if "gemma" in tmpl_str:
                return ChatTemplate.GEMMA

        return _CATEGORY_TEMPLATE_MAP.get(category, ChatTemplate.DEFAULT)

    # ------------------------------------------------------------------
    # Chat formatting
    # ------------------------------------------------------------------

    def format_conversation(
        self,
        messages: list[dict[str, str]],
        template: ChatTemplate | None = None,
    ) -> str:
        """Format a list of chat messages into a single prompt string.

        Each message dict should have 'role' (system/user/assistant) and
        'content' keys.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            template: Override the auto-detected template.

        Returns:
            Formatted prompt string ready for the model.

        Raises:
            ModelLoadError: If no tokenizer is loaded.
        """
        if not self.is_loaded:
            raise ModelLoadError("No tokenizer loaded. Call load() first.")

        tmpl = template or self._template

        # Try the tokenizer's native apply_chat_template first for supported templates
        if tmpl in (
            ChatTemplate.LLAMA,
            ChatTemplate.CHATML,
            ChatTemplate.PHI,
            ChatTemplate.QWEN,
            ChatTemplate.GEMMA,
        ):
            try:
                if hasattr(self._tokenizer, "apply_chat_template"):
                    formatted = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if formatted:
                        return formatted
            except Exception:
                pass  # Fall through to manual formatting

        # Manual formatting fallback
        formatters = {
            ChatTemplate.GPT2: self._format_gpt2,
            ChatTemplate.DIALOGPT: self._format_dialogpt,
            ChatTemplate.LLAMA: self._format_llama,
            ChatTemplate.CHATML: self._format_chatml,
            ChatTemplate.PHI: self._format_phi,
            ChatTemplate.QWEN: self._format_qwen,
            ChatTemplate.GEMMA: self._format_gemma,
            ChatTemplate.DEFAULT: self._format_default,
        }

        formatter = formatters.get(tmpl, self._format_default)
        return formatter(messages)

    # -- Individual template formatters --------------------------------

    @staticmethod
    def render_plain(
        messages: Sequence[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render *messages* without a chat template.

        Used as the fallback when the wrapped tokenizer has no
        ``chat_template`` (typical for base models such as ``gpt2``), and by
        :meth:`TokenizerWrapper.apply_chat_template`.
        """
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("assistant:")
        return "\n".join(lines) + ("\n" if add_generation_prompt else "")

    @staticmethod
    def _format_gpt2(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"Instructions: {content}\n\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant: ")
        return "".join(parts)

    @staticmethod
    def _format_dialogpt(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(
                    f"{content}{AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium').eos_token if False else ''}"
                )
            elif role == "assistant":
                parts.append(content)
        # DialoGPT uses eos_token between turns
        return "<|endoftext|>".join(parts) + "<|endoftext|>"

    @staticmethod
    def _format_llama(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST] ")
            elif role == "assistant":
                parts.append(f"{content} ")
        return "".join(parts)

    @staticmethod
    def _format_chatml(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _format_phi(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"{content}\n\n")
            elif role == "user":
                parts.append(f"Instruct: {content}\n")
            elif role == "assistant":
                parts.append(f"Output: {content}\n")
        parts.append("Output: ")
        return "".join(parts)

    @staticmethod
    def _format_qwen(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _format_gemma(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    @staticmethod
    def _format_default(messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}\n\n")
            elif role == "user":
                parts.append(f"Human: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant: ")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: The text to encode.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token IDs.

        Raises:
            ModelLoadError: If no tokenizer is loaded.
        """
        if not self.is_loaded:
            raise ModelLoadError("No tokenizer loaded. Call load() first.")
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)  # type: ignore[union-attr]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.

        Raises:
            ModelLoadError: If no tokenizer is loaded.
        """
        if not self.is_loaded:
            raise ModelLoadError("No tokenizer loaded. Call load() first.")
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)  # type: ignore[union-attr]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens.

        Raises:
            ModelLoadError: If no tokenizer is loaded.
        """
        if not self.is_loaded:
            raise ModelLoadError("No tokenizer loaded. Call load() first.")
        return len(self._tokenizer.encode(text, add_special_tokens=False))  # type: ignore[union-attr]

    def truncate_conversation(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        keep_system: bool = True,
    ) -> list[dict[str, str]]:
        """Truncate a conversation to fit within a token budget.

        Removes the oldest messages first while optionally preserving the
        system prompt.

        Args:
            messages: List of message dicts.
            max_tokens: Maximum number of tokens to allow.
            keep_system: Whether to always keep the system message.

        Returns:
            Truncated list of message dicts.

        Raises:
            ModelLoadError: If no tokenizer is loaded.
        """
        if not self.is_loaded:
            raise ModelLoadError("No tokenizer loaded. Call load() first.")

        # Separate system messages if they must be kept
        system_msgs: list[dict[str, str]] = []
        chat_msgs: list[dict[str, str]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_msgs.append(msg)
            else:
                chat_msgs.append(msg)

        # Calculate system message tokens
        system_text = self.format_conversation(system_msgs) if system_msgs else ""
        system_token_count = self.count_tokens(system_text) if system_text else 0

        remaining_budget = max_tokens - system_token_count
        if remaining_budget <= 0:
            # Even system prompt exceeds budget; return system only
            return system_msgs if keep_system else []

        # Add chat messages from newest to oldest until budget is exhausted
        result_msgs: list[dict[str, str]] = []
        used_tokens = 0

        for msg in reversed(chat_msgs):
            msg_tokens = self.count_tokens(msg["content"])
            if used_tokens + msg_tokens > remaining_budget:
                break
            result_msgs.insert(0, msg)
            used_tokens += msg_tokens

        return system_msgs + result_msgs if keep_system else result_msgs

    # ------------------------------------------------------------------
    # Special token helpers
    # ------------------------------------------------------------------

    @property
    def eos_token(self) -> str | None:
        """The end-of-sequence token, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.eos_token  # type: ignore[union-attr]

    @property
    def eos_token_id(self) -> int | None:
        """The end-of-sequence token ID, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.eos_token_id  # type: ignore[union-attr]

    @property
    def pad_token(self) -> str | None:
        """The padding token, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.pad_token  # type: ignore[union-attr]

    @property
    def pad_token_id(self) -> int | None:
        """The padding token ID, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.pad_token_id  # type: ignore[union-attr]

    @property
    def vocab_size(self) -> int:
        """The vocabulary size."""
        if not self.is_loaded:
            return 0
        return len(self._tokenizer)  # type: ignore[arg-type]

    def get_info(self) -> dict[str, object]:
        """Return a summary dict of the current tokenizer state."""
        return {
            "model_id": self._model_id,
            "template": self._template.value,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "is_loaded": self.is_loaded,
        }


class TokenizerWrapper:
    """Adapt a HuggingFace tokenizer to the interface Nexus pipelines expect.

    :class:`TokenizerManager` *loads* tokenizers and formats conversations;
    this class is the thin, stateless adapter the pipelines use at inference
    time.  It guarantees the small set of methods every pipeline relies on
    (:meth:`encode`, :meth:`encode_with_padding`, :meth:`decode`,
    :meth:`batch_decode`, :meth:`apply_chat_template`) regardless of whether
    the wrapped object is a fast or slow tokenizer, and it never loses the
    underlying object: attribute access falls through to it.

    Args:
        tokenizer: A HuggingFace tokenizer (fast or slow), or an existing
            :class:`TokenizerWrapper` (in which case it is reused as-is).
        pad_to_multiple_of: Optional padding multiple for batch encoding.

    Example::

        wrapper = TokenizerWrapper(AutoTokenizer.from_pretrained("gpt2"))
        batch = wrapper.encode_with_padding(["hello", "hello world"], return_tensors="pt")
        assert batch["input_ids"].shape[0] == 2
    """

    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        if isinstance(tokenizer, TokenizerWrapper):
            tokenizer = tokenizer.tokenizer
        if tokenizer is None:
            raise ValueError("TokenizerWrapper requires a tokenizer instance")
        self._tokenizer = tokenizer
        self._pad_to_multiple_of = pad_to_multiple_of

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def tokenizer(self) -> Any:
        """The wrapped HuggingFace tokenizer."""
        return self._tokenizer

    @property
    def name(self) -> str:
        """Class name of the wrapped tokenizer."""
        return type(self._tokenizer).__name__

    @property
    def vocab_size(self) -> int:
        """Size of the tokenizer vocabulary."""
        try:
            return len(self._tokenizer)
        except TypeError:  # pragma: no cover - exotic tokenizers
            return int(self._tokenizer.vocab_size)

    @property
    def pad_token(self) -> Any:
        """The padding token string."""
        return getattr(self._tokenizer, "pad_token", None)

    @pad_token.setter
    def pad_token(self, value: Any) -> None:
        self._tokenizer.pad_token = value

    @property
    def pad_token_id(self) -> int | None:
        """The padding token id, falling back to ``eos_token_id``."""
        token_id = getattr(self._tokenizer, "pad_token_id", None)
        if token_id is None:
            token_id = getattr(self._tokenizer, "eos_token_id", None)
        return token_id

    @property
    def eos_token(self) -> Any:
        """The end-of-sequence token string."""
        return getattr(self._tokenizer, "eos_token", None)

    @property
    def eos_token_id(self) -> int | None:
        """The end-of-sequence token id."""
        return getattr(self._tokenizer, "eos_token_id", None)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, return_tensors: str | None = None, **kwargs: Any) -> Any:
        """Tokenize a single string.

        Args:
            text: Input text.
            return_tensors: ``"pt"`` for a torch tensor, ``"np"`` for numpy,
                or ``None`` for a plain list of ids.
            **kwargs: Forwarded to the tokenizer (e.g. ``add_special_tokens``).

        Returns:
            Token ids in the requested container.
        """
        encoded = self._tokenizer(text, return_tensors=return_tensors, **kwargs)
        # Fast tokenizers return a BatchEncoding; callers of encode() want ids.
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return encoded["input_ids"]
        return encoded

    def encode_with_padding(
        self,
        texts: Sequence[str],
        return_tensors: str | None = "pt",
        padding: bool | str = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Tokenize a batch with padding and an attention mask.

        Args:
            texts: The batch of strings.
            return_tensors: Framework for the returned tensors.
            padding: Padding strategy (``True`` pads to the longest row).
            truncation: Whether to truncate to ``max_length``.
            max_length: Optional maximum sequence length.
            **kwargs: Forwarded to the tokenizer.

        Returns:
            A plain ``dict`` (never a ``BatchEncoding``) containing at least
            ``input_ids`` and ``attention_mask``.
        """
        if isinstance(texts, str):
            texts = [texts]
        kwargs.setdefault("padding", padding)
        if max_length is not None:
            kwargs.setdefault("max_length", max_length)
            kwargs.setdefault("truncation", True)
        elif truncation:
            kwargs.setdefault("truncation", True)
        if self._pad_to_multiple_of is not None:
            kwargs.setdefault("pad_to_multiple_of", self._pad_to_multiple_of)

        encoded = self._tokenizer(list(texts), return_tensors=return_tensors, **kwargs)
        if not isinstance(encoded, dict):  # very old / custom tokenizers
            encoded = {"input_ids": encoded}
        result = dict(encoded)
        if "attention_mask" not in result:
            result["attention_mask"] = self._build_attention_mask(
                result["input_ids"], return_tensors
            )
        return result

    def _build_attention_mask(self, input_ids: Any, return_tensors: str | None) -> Any:
        """Create an attention mask from padded ``input_ids``."""
        pad_id = self.pad_token_id
        if return_tensors == "pt" and pad_id is not None:
            import torch

            return (input_ids != pad_id).to(torch.long)
        if pad_id is None:
            pad_id = 0
        if return_tensors == "np":
            import numpy as np

            return np.array(input_ids != pad_id, dtype=np.int64)
        return [[1 if token != pad_id else 0 for token in row] for row in input_ids]

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: Any, skip_special_tokens: bool = True, **kwargs: Any) -> str:
        """Decode a single sequence of ids into text."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def batch_decode(
        self,
        sequences: Any,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Decode a batch of id sequences into a list of strings."""
        decoded = self._tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens, **kwargs
        )
        return list(decoded)

    def convert_tokens_to_ids(self, tokens: Any) -> Any:
        """Map token strings to their ids."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: Any) -> Any:
        """Map ids back to token strings."""
        return self._tokenizer.convert_ids_to_tokens(ids)

    # ------------------------------------------------------------------
    # Chat templates
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Render ``messages`` using the tokenizer's chat template.

        Falls back to a plain ``role: content`` rendering when the tokenizer
        ships no template (e.g. a base model such as ``gpt2``), so chat
        features degrade gracefully instead of raising.
        """
        messages = list(messages)
        method = getattr(self._tokenizer, "apply_chat_template", None)
        if method is not None and getattr(self._tokenizer, "chat_template", None):
            try:
                return method(
                    messages,
                    tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
            except (ValueError, TypeError) as exc:
                logger.warning("apply_chat_template failed (%s); using fallback", exc)

        rendered = TokenizerManager.render_plain(
            messages, add_generation_prompt=add_generation_prompt
        )
        if not tokenize:
            return rendered
        return self.encode(rendered, tokenize=False)

    # ------------------------------------------------------------------
    # Dunders
    # ------------------------------------------------------------------

    def __call__(self, text: Any, return_tensors: str | None = None, **kwargs: Any) -> Any:
        """Tokenize like the wrapped HF tokenizer would.

        ``model.generate()`` callers and pipeline code frequently treat a
        tokenizer as callable, so the wrapper has to forward that too (special
        methods are resolved on the class, never through ``__getattr__``).
        """
        return self._tokenizer(text, return_tensors=return_tensors, **kwargs)

    def __getattr__(self, item: str) -> Any:
        """Fall through to the wrapped tokenizer for anything not wrapped."""
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._tokenizer, item)

    def __repr__(self) -> str:
        return f"TokenizerWrapper({self.name}, vocab_size={self.vocab_size})"
