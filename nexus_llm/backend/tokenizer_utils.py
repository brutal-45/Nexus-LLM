"""Tokenizer management for Nexus-LLM.

Handles loading tokenizers, managing chat templates, and encoding/decoding.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer

from nexus_llm.core.exceptions import ModelLoadError, ModelNotFoundError
from nexus_llm.core.model_catalog import MODEL_CATALOG, get_model_info

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
_CATEGORY_TEMPLATE_MAP: Dict[str, ChatTemplate] = {
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
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model_id: Optional[str] = None
        self._template: ChatTemplate = ChatTemplate.DEFAULT
        self._pad_token_set: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, model_id: str, cache_dir: Optional[str] = None) -> None:
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
            kwargs: Dict = {"trust_remote_code": True}
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
    def model_id(self) -> Optional[str]:
        """The model ID of the currently loaded tokenizer."""
        return self._model_id

    @property
    def tokenizer(self) -> Optional[AutoTokenizer]:
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
            if "llama" in tmpl_str or "<|im_start|>" not in tmpl_str and "system" in tmpl_str:
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
        messages: List[Dict[str, str]],
        template: Optional[ChatTemplate] = None,
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
        if tmpl in (ChatTemplate.LLAMA, ChatTemplate.CHATML, ChatTemplate.PHI,
                    ChatTemplate.QWEN, ChatTemplate.GEMMA):
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
    def _format_gpt2(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
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
    def _format_dialogpt(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"{content}{AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium').eos_token if False else ''}")
            elif role == "assistant":
                parts.append(content)
        # DialoGPT uses eos_token between turns
        return "<|endoftext|>".join(parts) + "<|endoftext|>"

    @staticmethod
    def _format_llama(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
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
    def _format_chatml(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _format_phi(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
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
    def _format_qwen(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _format_gemma(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)

    @staticmethod
    def _format_default(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
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

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
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

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
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
        messages: List[Dict[str, str]],
        max_tokens: int,
        keep_system: bool = True,
    ) -> List[Dict[str, str]]:
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
        system_msgs: List[Dict[str, str]] = []
        chat_msgs: List[Dict[str, str]] = []

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
        result_msgs: List[Dict[str, str]] = []
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
    def eos_token(self) -> Optional[str]:
        """The end-of-sequence token, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.eos_token  # type: ignore[union-attr]

    @property
    def eos_token_id(self) -> Optional[int]:
        """The end-of-sequence token ID, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.eos_token_id  # type: ignore[union-attr]

    @property
    def pad_token(self) -> Optional[str]:
        """The padding token, or None if not loaded."""
        if not self.is_loaded:
            return None
        return self._tokenizer.pad_token  # type: ignore[union-attr]

    @property
    def pad_token_id(self) -> Optional[int]:
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

    def get_info(self) -> Dict[str, object]:
        """Return a summary dict of the current tokenizer state."""
        return {
            "model_id": self._model_id,
            "template": self._template.value,
            "vocab_size": self.vocab_size,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
            "is_loaded": self.is_loaded,
        }
