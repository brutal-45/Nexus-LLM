"""Text processing: token counting, word counting, sentence splitting, truncation, clean text."""

import re
import logging
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
    """Count the number of tokens in text.

    Args:
        text: Input text.
        tokenizer: Optional tokenizer for accurate counting. Falls back to word count.

    Returns:
        Number of tokens.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass

    # Approximate: ~1.3 tokens per word for English text
    words = len(text.split())
    return int(words * 1.3)


def count_words(text: str) -> int:
    """Count the number of words in text.

    Args:
        text: Input text.

    Returns:
        Number of words.
    """
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """Count the number of characters in text.

    Args:
        text: Input text.
        include_spaces: Whether to include spaces in the count.

    Returns:
        Number of characters.
    """
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Uses regex-based sentence boundary detection that handles
    common abbreviations and edge cases.

    Args:
        text: Input text.

    Returns:
        List of sentence strings.
    """
    # Handle common abbreviations
    abbreviations = [
        r"Mr\.", r"Mrs\.", r"Ms\.", r"Dr\.", r"Prof\.", r"Sr\.", r"Jr\.",
        r"vs\.", r"etc\.", r"i\.e\.", r"e\.g\.", r"cf\.", r"al\.",
    ]

    processed = text
    placeholder_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        processed = re.sub(abbr, placeholder, processed)
        placeholder_map[placeholder] = abbr.replace(r"\.", ".")

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', processed.strip())

    # Restore abbreviations
    result = []
    for sentence in sentences:
        for placeholder, original in placeholder_map.items():
            sentence = sentence.replace(placeholder, original)
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)

    return result


def truncate_text(
    text: str,
    max_length: int,
    truncation_side: str = "right",
    ellipsis: str = "...",
    by: str = "characters",
) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Input text.
        max_length: Maximum length.
        truncation_side: 'right' to keep the start, 'left' to keep the end.
        ellipsis: String to append/prepend to indicate truncation.
        by: Unit of max_length: 'characters', 'words', or 'tokens'.

    Returns:
        Truncated text string.
    """
    if by == "words":
        words = text.split()
        if len(words) <= max_length:
            return text
        if truncation_side == "right":
            return " ".join(words[:max_length]) + ellipsis
        else:
            return ellipsis + " ".join(words[-max_length:])

    elif by == "tokens":
        # Approximate token truncation
        words = text.split()
        approx_tokens = 0
        if truncation_side == "right":
            kept_words = []
            for word in words:
                approx_tokens += max(1, len(word) // 4 + 1)
                if approx_tokens > max_length:
                    break
                kept_words.append(word)
            return " ".join(kept_words) + ellipsis
        else:
            kept_words = []
            for word in reversed(words):
                approx_tokens += max(1, len(word) // 4 + 1)
                if approx_tokens > max_length:
                    break
                kept_words.insert(0, word)
            return ellipsis + " ".join(kept_words)

    else:  # characters
        if len(text) <= max_length:
            return text
        if truncation_side == "right":
            return text[:max_length - len(ellipsis)] + ellipsis
        else:
            return ellipsis + text[len(text) - max_length + len(ellipsis):]


def clean_text(
    text: str,
    remove_urls: bool = False,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    remove_emojis: bool = False,
    lowercase: bool = False,
) -> str:
    """Clean and normalize text.

    Args:
        text: Input text.
        remove_urls: Whether to remove URLs.
        remove_html: Whether to remove HTML tags.
        normalize_whitespace: Whether to collapse multiple spaces/newlines.
        remove_emojis: Whether to remove emoji characters.
        lowercase: Whether to convert to lowercase.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)

    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

    if remove_emojis:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)

    if normalize_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

    if lowercase:
        text = text.lower()

    return text.strip()


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text.

    Args:
        text: Input text.

    Returns:
        List of extracted numbers.
    """
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches if m]


def remove_duplicate_lines(text: str) -> str:
    """Remove duplicate lines from text.

    Args:
        text: Input text.

    Returns:
        Text with duplicate lines removed.
    """
    seen = set()
    lines = text.split("\n")
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)
    return "\n".join(unique_lines)


def pad_text(
    text: str,
    target_length: int,
    pad_token: str = " ",
    side: str = "right",
) -> str:
    """Pad text to a target length.

    Args:
        text: Input text.
        target_length: Desired length.
        pad_token: Padding character/string.
        side: 'right' or 'left'.

    Returns:
        Padded text string.
    """
    if len(text) >= target_length:
        return text

    pad_length = target_length - len(text)
    padding = pad_token * (pad_length // len(pad_token))

    if side == "right":
        return text + padding
    return padding + text
