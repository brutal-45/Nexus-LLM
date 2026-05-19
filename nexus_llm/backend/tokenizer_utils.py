"""Tokenizer utilities for Nexus-LLM backend.

Handles tokenizer loading, encoding/decoding, chat template formatting,
token counting, and clean_up_tokenization_spaces fix.
"""

from typing import List, Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Wrapper around HuggingFace tokenizers with enhanced functionality."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._chat_template = getattr(tokenizer, "chat_template", None)

    @property
    def tokenizer(self) -> Any:
        """Access the underlying tokenizer."""
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self._tokenizer)

    @property
    def pad_token_id(self) -> Optional[int]:
        """Return the pad token ID."""
        return self._tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """Return the end-of-sequence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Return the beginning-of-sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def model_max_length(self) -> int:
        """Return the maximum sequence length the tokenizer supports."""
        return self._tokenizer.model_max_length

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Any]:
        """Encode text to token IDs."""
        kwargs = {
            "text": text,
            "add_special_tokens": add_special_tokens,
            "truncation": truncation,
        }
        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = True
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors

        return self._tokenizer.encode(**kwargs)

    def decode(
        self,
        token_ids: Union[List[int], Any],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        The clean_up_tokenization_spaces parameter fixes the common issue
        where tokenizers produce extra spaces around punctuation.
        """
        result = self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return result

    def batch_decode(
        self,
        token_ids_batch: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        """Decode a batch of token ID lists to text."""
        return self._tokenizer.batch_decode(
            token_ids_batch,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        return len(self.encode(text, add_special_tokens=False))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        return [self.count_tokens(text) for text in texts]

    def truncate_to_max_length(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to fit within max_length tokens."""
        limit = max_length or self.model_max_length
        token_ids = self.encode(text, add_special_tokens=True)
        if len(token_ids) <= limit:
            return text
        truncated_ids = token_ids[:limit]
        return self.decode(truncated_ids, skip_special_tokens=False)

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply the tokenizer's chat template to a list of messages.

        Messages should be in the format: [{"role": "user/assistant/system", "content": "..."}]
        """
        if not hasattr(self._tokenizer, "apply_chat_template"):
            return self._format_chat_fallback(messages, add_generation_prompt)

        result = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            truncation=truncation,
            max_length=max_length,
            **kwargs,
        )
        return result

    def _format_chat_fallback(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
        """Fallback chat formatting when no template is available."""
        formatted_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                formatted_parts.append(f"<|system|>\n{content}</s>")
            elif role == "user":
                formatted_parts.append(f"<|user|>\n{content}</s>")
            elif role == "assistant":
                formatted_parts.append(f"<|assistant|)\n{content}</s>")

        if add_generation_prompt:
            formatted_parts.append("<|assistant|)\n")

        return "\n".join(formatted_parts)

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """Format a conversation into a single string for the model.

        Prepends a system prompt if provided.
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        result = self.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        return result if isinstance(result, str) else self.decode(result)

    def get_special_tokens(self) -> Dict[str, int]:
        """Return a mapping of special token names to their IDs."""
        special = {}
        if self.bos_token_id is not None:
            special["bos_token"] = self.bos_token_id
        if self.eos_token_id is not None:
            special["eos_token"] = self.eos_token_id
        if self.pad_token_id is not None:
            special["pad_token"] = self.pad_token_id

        for attr_name in dir(self._tokenizer):
            if attr_name.endswith("_token_id") and attr_name not in (
                "bos_token_id", "eos_token_id", "pad_token_id"
            ):
                token_id = getattr(self._tokenizer, attr_name)
                if isinstance(token_id, int):
                    name = attr_name.replace("_token_id", "")
                    special[name] = token_id
        return special

    def encode_with_padding(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, Any]:
        """Encode a batch of texts with padding and return as tensor dict."""
        return self._tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count without full tokenization (rough: ~4 chars per token)."""
        return max(1, len(text) // 4)


def load_tokenizer(
    model_path: str,
    trust_remote_code: bool = False,
    use_fast: bool = True,
    padding_side: str = "left",
    clean_up_tokenization_spaces: bool = True,
) -> TokenizerWrapper:
    """Load a tokenizer from a model path and return a TokenizerWrapper.

    Handles the clean_up_tokenization_spaces fix by ensuring the tokenizer
    is configured to properly clean up spaces after decoding.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = padding_side
    tokenizer.clean_up_tokenization_spaces = clean_up_tokenization_spaces

    if hasattr(tokenizer, "init_kwargs"):
        tokenizer.init_kwargs["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

    wrapper = TokenizerWrapper(tokenizer)
    logger.info(f"Tokenizer loaded from '{model_path}' (vocab_size={wrapper.vocab_size})")
    return wrapper


def count_message_tokens(
    tokenizer: TokenizerWrapper,
    messages: List[Dict[str, str]],
) -> int:
    """Count the total number of tokens in a list of chat messages."""
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    if isinstance(formatted, str):
        return tokenizer.count_tokens(formatted)
    return len(formatted)


def truncate_messages(
    tokenizer: TokenizerWrapper,
    messages: List[Dict[str, str]],
    max_tokens: int,
    keep_system: bool = True,
) -> List[Dict[str, str]]:
    """Truncate a message list to fit within max_tokens, optionally preserving system messages."""
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    while non_system:
        test_messages = (system_messages if keep_system else []) + non_system
        token_count = count_message_tokens(tokenizer, test_messages)
        if token_count <= max_tokens:
            break
        non_system.pop(0)

    return (system_messages if keep_system else []) + non_system
