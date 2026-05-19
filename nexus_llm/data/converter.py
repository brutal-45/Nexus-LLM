"""Nexus-LLM Format Converter Module.

Converts between common LLM training data formats:

- **Alpaca** → **ChatML**: Instruction/input/output to message list.
- **ChatML** → **Alpaca**: Message list back to instruction format.
- **ShareGPT** → **HuggingFace**: Multi-turn conversations to HF datasets.
- **Auto-detection**: Automatically infer the source format from keys.
- **Custom field mapping**: Remap arbitrary field names.

Supported formats:
    - ``alpaca``: ``{instruction, input, output}``
    - ``chatml``: ``{messages: [{role, content}, ...]}``
    - ``sharegpt``: ``{conversations: [{from, value}, ...]}``
    - ``hf``: HuggingFace datasets format with ``conversations`` field.

Example::

    from nexus_llm.data.converter import FormatConverter

    converter = FormatConverter()
    chatml = converter.alpaca_to_chatml(alpaca_sample)
    detected = converter.detect_format(sample)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

# Key signatures for auto-detection
_ALPACA_KEYS = {"instruction", "output"}
_CHATML_KEYS = {"messages"}
_SHAREGPT_KEYS = {"conversations"}
_HF_CONVERSATION_KEYS = {"conversations", "id"}


def detect_format(sample: Dict[str, Any]) -> str:
    """Auto-detect the data format of a single sample.

    Inspects the top-level keys and values to classify the sample.

    Args:
        sample: A single row dict.

    Returns:
        One of ``"alpaca"``, ``"chatml"``, ``"sharegpt"``, ``"hf"``,
        or ``"unknown"``.
    """
    keys = set(sample.keys())

    if _ALPACA_KEYS.issubset(keys):
        return "alpaca"
    if _CHATML_KEYS.issubset(keys):
        messages = sample.get("messages", [])
        if isinstance(messages, list) and messages and "role" in messages[0]:
            return "chatml"
    if _SHAREGPT_KEYS.issubset(keys):
        convs = sample.get("conversations", [])
        if isinstance(convs, list) and convs and "from" in convs[0]:
            return "sharegpt"
    if _HF_CONVERSATION_KEYS.issubset(keys):
        return "hf"
    # Heuristic: if there's an "instruction" key but no "output"
    if "instruction" in keys and "output" not in keys:
        return "alpaca"
    return "unknown"


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def alpaca_to_chatml(
    sample: Dict[str, Any],
    system_prompt: str = "You are a helpful assistant.",
    include_input: bool = True,
) -> Dict[str, Any]:
    """Convert an Alpaca-format sample to ChatML.

    Args:
        sample: Dict with at least ``instruction`` and ``output`` keys.
            May also contain ``input``.
        system_prompt: System message content.
        include_input: Whether to append the ``input`` field to the user
            message when it is non-empty.

    Returns:
        Dict with a ``messages`` key containing a list of message dicts.
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    user_content = instruction
    if include_input and input_text.strip():
        user_content = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    result: Dict[str, Any] = {"messages": messages}
    # Carry over any extra keys
    for key, value in sample.items():
        if key not in ("instruction", "input", "output"):
            result[key] = value
    return result


def chatml_to_alpaca(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a ChatML-format sample to Alpaca.

    Extracts the first user message as ``instruction`` and the first
    assistant message as ``output``.  System messages are dropped.

    Args:
        sample: Dict with a ``messages`` key.

    Returns:
        Dict with ``instruction``, ``input``, and ``output`` keys.
    """
    messages = sample.get("messages", [])
    instruction = ""
    output = ""
    extra_input = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user" and not instruction:
            # Try to split on "### Input:" pattern
            if "### Input:" in content:
                parts = content.split("### Input:", 1)
                instruction = parts[0].replace("### Instruction:", "").strip()
                extra_input = parts[1].strip()
            else:
                instruction = content
        elif role == "assistant" and not output:
            output = content

    result: Dict[str, Any] = {
        "instruction": instruction,
        "input": extra_input,
        "output": output,
    }
    for key, value in sample.items():
        if key != "messages":
            result[key] = value
    return result


def sharegpt_to_hf(
    sample: Dict[str, Any],
    role_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert a ShareGPT-format sample to HuggingFace conversation format.

    ShareGPT uses ``{from, value}`` pairs inside a ``conversations`` list.
    HF format uses ``{role, content}`` inside a ``conversations`` list.

    Args:
        sample: Dict with a ``conversations`` key.
        role_map: Optional mapping from ShareGPT ``from`` values to
            HF ``role`` values.  Defaults to
            ``{"human": "user", "gpt": "assistant", "system": "system"}``.

    Returns:
        Dict with ``conversations`` in HF format and preserved metadata.
    """
    default_map = {"human": "user", "gpt": "assistant", "system": "system"}
    if role_map is not None:
        default_map.update(role_map)

    conversations = sample.get("conversations", [])
    hf_conversations = []
    for turn in conversations:
        from_role = turn.get("from", "")
        value = turn.get("value", "")
        hf_role = default_map.get(from_role, from_role)
        hf_conversations.append({"role": hf_role, "content": value})

    result: Dict[str, Any] = {"conversations": hf_conversations}
    for key, value in sample.items():
        if key != "conversations":
            result[key] = value
    return result


def hf_to_sharegpt(
    sample: Dict[str, Any],
    role_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Convert HuggingFace conversation format to ShareGPT.

    Args:
        sample: Dict with ``conversations`` in HF format.
        role_map: Inverse role mapping.  Defaults to
            ``{"user": "human", "assistant": "gpt", "system": "system"}``.

    Returns:
        Dict with ``conversations`` in ShareGPT format.
    """
    default_map = {"user": "human", "assistant": "gpt", "system": "system"}
    if role_map is not None:
        default_map.update(role_map)

    conversations = sample.get("conversations", [])
    sg_conversations = []
    for turn in conversations:
        role = turn.get("role", "")
        content = turn.get("content", "")
        from_role = default_map.get(role, role)
        sg_conversations.append({"from": from_role, "value": content})

    result: Dict[str, Any] = {"conversations": sg_conversations}
    for key, value in sample.items():
        if key != "conversations":
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Custom field mapping
# ---------------------------------------------------------------------------

def remap_fields(
    sample: Dict[str, Any],
    mapping: Dict[str, str],
) -> Dict[str, Any]:
    """Rename fields in *sample* according to *mapping*.

    Args:
        sample: Source row dict.
        mapping: ``{old_key: new_key, ...}`` mapping.

    Returns:
        New dict with renamed keys.  Unmapped keys are preserved.
    """
    result: Dict[str, Any] = {}
    for key, value in sample.items():
        new_key = mapping.get(key, key)
        result[new_key] = value
    return result


def remap_fields_batch(
    data: List[Dict[str, Any]],
    mapping: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Apply :func:`remap_fields` to every row in *data*."""
    return [remap_fields(row, mapping) for row in data]


# ---------------------------------------------------------------------------
# FormatConverter class
# ---------------------------------------------------------------------------

class FormatConverter:
    """High-level format conversion with auto-detection and chaining.

    Provides both individual conversion methods and a unified
    :meth:`convert` entry point that auto-detects the source format
    and converts to the requested target.

    Args:
        system_prompt: Default system prompt for ChatML outputs.
        alpaca_include_input: Whether to include the ``input`` field
            when converting Alpaca → ChatML.

    Example::

        converter = FormatConverter()
        result = converter.convert(sample, target="chatml")
    """

    # Registry of known conversions: (source, target) → callable
    _CONVERSIONS: Dict[Tuple[str, str], Any] = {}

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        alpaca_include_input: bool = True,
    ) -> None:
        self._system_prompt = system_prompt
        self._alpaca_include_input = alpaca_include_input

        # Build conversion registry
        self._CONVERSIONS = {
            ("alpaca", "chatml"): self._alpaca_to_chatml_impl,
            ("chatml", "alpaca"): self._chatml_to_alpaca_impl,
            ("sharegpt", "hf"): self._sharegpt_to_hf_impl,
            ("hf", "sharegpt"): self._hf_to_sharegpt_impl,
        }

    # -- Delegating helpers -------------------------------------------------

    def _alpaca_to_chatml_impl(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return alpaca_to_chatml(
            sample,
            system_prompt=self._system_prompt,
            include_input=self._alpaca_include_input,
        )

    @staticmethod
    def _chatml_to_alpaca_impl(sample: Dict[str, Any]) -> Dict[str, Any]:
        return chatml_to_alpaca(sample)

    @staticmethod
    def _sharegpt_to_hf_impl(sample: Dict[str, Any]) -> Dict[str, Any]:
        return sharegpt_to_hf(sample)

    @staticmethod
    def _hf_to_sharegpt_impl(sample: Dict[str, Any]) -> Dict[str, Any]:
        return hf_to_sharegpt(sample)

    # -- Public API ---------------------------------------------------------

    def detect_format(self, sample: Dict[str, Any]) -> str:
        """Auto-detect the format of *sample*."""
        return detect_format(sample)

    def convert(
        self,
        sample: Dict[str, Any],
        target: str = "chatml",
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert *sample* to *target* format.

        If *source* is ``None``, the format is auto-detected.

        Args:
            sample: Input row dict.
            target: Target format name (``"chatml"``, ``"alpaca"``,
                ``"hf"``, ``"sharegpt"``).
            source: Explicit source format.  Auto-detected if ``None``.

        Returns:
            Converted sample dict.

        Raises:
            ValueError: If the conversion is not supported.
        """
        src = source or self.detect_format(sample)
        if src == "unknown":
            raise ValueError(
                f"Cannot detect source format. Keys: {list(sample.keys())}"
            )
        key = (src, target)
        converter = self._CONVERSIONS.get(key)
        if converter is None:
            raise ValueError(
                f"No converter registered for {src!r} → {target!r}"
            )
        return converter(sample)

    def convert_batch(
        self,
        data: List[Dict[str, Any]],
        target: str = "chatml",
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert a list of samples.

        The source format is detected from the first sample (or uses
        the explicit *source* argument) and applied uniformly.
        """
        if not data:
            return []
        src = source or self.detect_format(data[0])
        return [self.convert(row, target=target, source=src) for row in data]

    def alpaca_to_chatml(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert Alpaca → ChatML."""
        return alpaca_to_chatml(
            sample,
            system_prompt=self._system_prompt,
            include_input=self._alpaca_include_input,
        )

    def chatml_to_alpaca(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert ChatML → Alpaca."""
        return chatml_to_alpaca(sample)

    def sharegpt_to_hf(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert ShareGPT → HuggingFace."""
        return sharegpt_to_hf(sample)

    def hf_to_sharegpt(
        self, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert HuggingFace → ShareGPT."""
        return hf_to_sharegpt(sample)

    def remap(
        self,
        sample: Dict[str, Any],
        mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """Apply custom field mapping to a sample."""
        return remap_fields(sample, mapping)

    def remap_batch(
        self,
        data: List[Dict[str, Any]],
        mapping: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Apply custom field mapping to a batch of samples."""
        return remap_fields_batch(data, mapping)

    # -- Introspection ------------------------------------------------------

    @classmethod
    def supported_conversions(cls) -> List[Tuple[str, str]]:
        """Return a list of ``(source, target)`` conversion pairs."""
        return list(cls._CONVERSIONS.keys())

    def __repr__(self) -> str:
        return (
            f"FormatConverter(system_prompt={self._system_prompt!r}, "
            f"alpaca_include_input={self._alpaca_include_input})"
        )
