"""Tokenizer utilities for managing tokenization, encoding, and special tokens."""

import logging
from typing import List, Dict, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages tokenizer operations including encoding, decoding,
    special token handling, and conversation formatting.
    """

    # Special token templates for different model families
    CHAT_TEMPLATES = {
        "gpt2": {
            "system": "{content}\n",
            "user": "User: {content}\n",
            "assistant": "Assistant: {content}\n",
            "separator": "\n",
        },
        "dialogpt": {
            "user": "{content}{eos_token}",
            "assistant": "{content}{eos_token}",
            "separator": "",
        },
        "llama": {
            "system": "<<SYS>>\n{content}\n<</SYS>>\n\n",
            "user": "[INST] {content} [/INST]\n",
            "assistant": "{content}</s>\n",
            "separator": "",
        },
        "chatml": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "separator": "",
        },
    }

    def __init__(self, tokenizer, model_name: str = "gpt2-medium"):
        self.tokenizer = tokenizer
        self.model_name = model_name.lower()
        self._template = self._detect_template()

    def _detect_template(self) -> Dict[str, str]:
        """Detect the appropriate chat template based on model name."""
        if "llama" in self.model_name:
            return self.CHAT_TEMPLATES["llama"]
        elif "dialogpt" in self.model_name:
            return self.CHAT_TEMPLATES["dialogpt"]
        elif "chatml" in self.model_name or "hermes" in self.model_name:
            return self.CHAT_TEMPLATES["chatml"]
        else:
            return self.CHAT_TEMPLATES["gpt2"]

    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a list of messages into a single prompt string.

        Args:
            messages: List of dicts with 'role' and 'content' keys
            system_prompt: Optional system prompt to prepend

        Returns:
            Formatted conversation string
        """
        parts = []

        # Add system prompt if provided
        if system_prompt:
            system_text = self._template.get("system", "{content}\n").format(
                content=system_prompt,
                eos_token=self.tokenizer.eos_token or "",
            )
            parts.append(system_text)

        # Add conversation messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            template_key = role

            if template_key in self._template:
                formatted = self._template[template_key].format(
                    content=content,
                    eos_token=self.tokenizer.eos_token or "",
                )
                parts.append(formatted)
            else:
                # Fallback
                parts.append(f"{role.capitalize()}: {content}\n")

        # Add assistant prefix for generation
        assistant_prefix = self._template.get("assistant", "Assistant: {content}\n")
        # Extract just the prefix part before {content}
        if "{content}" in assistant_prefix:
            prefix = assistant_prefix.split("{content}")[0]
            parts.append(prefix)

        return "".join(parts)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> Dict[str, Any]:
        """
        Encode text into token IDs with attention mask.

        Args:
            text: Input text to encode
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            truncation: Whether to truncate to max_length

        Returns:
            Dict with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            return_tensors="pt",
            padding=False,
        )
        return encoded

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        """
        Decode token IDs back to text.

        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces

        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens across all messages."""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.get("content", ""))
            # Add overhead for role formatting (~4 tokens per message)
            total += 4
        return total

    def truncate_conversation(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Truncate conversation to fit within max_tokens by removing
        oldest messages while preserving system prompt context.

        Args:
            messages: Conversation messages
            max_tokens: Maximum token budget
            system_prompt: System prompt that uses tokens

        Returns:
            Truncated list of messages
        """
        # Calculate system prompt tokens
        system_tokens = 0
        if system_prompt:
            system_tokens = self.count_tokens(system_prompt) + 4

        # Calculate remaining budget
        remaining_budget = max_tokens - system_tokens - 256  # Reserve 256 for generation

        # Work backwards - keep most recent messages
        kept_messages = []
        current_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self.count_message_tokens([msg])
            if current_tokens + msg_tokens <= remaining_budget:
                kept_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break

        return kept_messages

    def get_special_tokens(self) -> Dict[str, Any]:
        """Get information about special tokens."""
        return {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
            "vocab_size": self.tokenizer.vocab_size,
            "model_max_length": self.tokenizer.model_max_length,
        }

    def set_chat_template(self, template_name: str) -> None:
        """Set a specific chat template by name."""
        if template_name in self.CHAT_TEMPLATES:
            self._template = self.CHAT_TEMPLATES[template_name]
            logger.info(f"Chat template set to: {template_name}")
        else:
            available = ", ".join(self.CHAT_TEMPLATES.keys())
            logger.warning(
                f"Unknown template '{template_name}'. Available: {available}"
            )
