"""Nexus-LLM Text Analyzer.

Provides the Analyzer class for extracting structured insights from text,
including sentiment, readability, keyword extraction, and statistical analysis.
"""

import logging
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result from text analysis.

    Attributes:
        text: Original analyzed text (may be truncated).
        char_count: Number of characters.
        word_count: Number of words.
        sentence_count: Number of sentences.
        paragraph_count: Number of paragraphs.
        avg_word_length: Average word length in characters.
        avg_sentence_length: Average sentence length in words.
        vocabulary_size: Number of unique words.
        lexical_diversity: Ratio of unique words to total words.
        readability_score: Flesch reading ease score (approximate).
        sentiment: Estimated sentiment label.
        sentiment_score: Numerical sentiment score (-1.0 to 1.0).
        top_keywords: Most frequent meaningful words.
        language_hint: Detected language hint.
        metadata: Additional analysis metadata.
    """

    text: str = ""
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    vocabulary_size: int = 0
    lexical_diversity: float = 0.0
    readability_score: float = 0.0
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    top_keywords: List[str] = field(default_factory=list)
    language_hint: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


# Simple stop words list for keyword extraction
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "me",
    "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "what", "which", "who", "whom", "when", "where", "why",
    "how", "not", "no", "nor", "if", "then", "than", "too", "very",
    "so", "just", "about", "also", "up", "out", "into", "over", "after",
}

# Simple sentiment word lists
_POSITIVE_WORDS: Set[str] = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "brilliant", "outstanding", "superb", "terrific", "happy", "love",
    "best", "perfect", "beautiful", "impressive", "remarkable", "joy",
    "delightful", "magnificent", "pleased", "satisfied", "enjoy",
}

_NEGATIVE_WORDS: Set[str] = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
    "ugly", "disappointing", "dreadful", "miserable", "annoying",
    "frustrating", "unpleasant", "disgusting", "inferior", "sad",
    "angry", "upset", "dislike", "boring", "waste", "fail", "wrong",
}


class Analyzer:
    """Text analyzer for extracting insights and statistics.

    The Analyzer performs statistical, structural, and sentiment analysis
    on text input without requiring external NLP libraries.

    Attributes:
        max_keywords: Maximum number of keywords to extract.
    """

    def __init__(self, max_keywords: int = 10) -> None:
        self._max_keywords = max_keywords
        logger.debug("Analyzer initialized with max_keywords=%d", max_keywords)

    @property
    def max_keywords(self) -> int:
        """Maximum number of keywords to extract."""
        return self._max_keywords

    def analyze(self, text: str) -> AnalysisResult:
        """Perform full analysis on the given text.

        Args:
            text: The text to analyze.

        Returns:
            An AnalysisResult with all computed metrics.
        """
        if not text or not text.strip():
            return AnalysisResult()

        result = AnalysisResult(text=text[:500])  # Store truncated

        # Basic counts
        result.char_count = len(text)
        words = self._tokenize(text)
        result.word_count = len(words)
        sentences = self._split_sentences(text)
        result.sentence_count = len(sentences) if sentences else 1
        result.paragraph_count = len([p for p in text.split("\n\n") if p.strip()]) or 1

        # Averages
        if words:
            result.avg_word_length = sum(len(w) for w in words) / len(words)
        if result.sentence_count > 0:
            result.avg_sentence_length = result.word_count / result.sentence_count

        # Vocabulary
        lower_words = [w.lower() for w in words]
        unique_words = set(lower_words)
        result.vocabulary_size = len(unique_words)
        result.lexical_diversity = len(unique_words) / len(lower_words) if lower_words else 0.0

        # Readability (Flesch Reading Ease approximation)
        result.readability_score = self._flesch_reading_ease(
            result.avg_sentence_length, self._avg_syllables_per_word(words)
        )

        # Sentiment
        result.sentiment_score = self._sentiment_score(words)
        if result.sentiment_score > 0.05:
            result.sentiment = "positive"
        elif result.sentiment_score < -0.05:
            result.sentiment = "negative"
        else:
            result.sentiment = "neutral"

        # Keywords
        result.top_keywords = self._extract_keywords(words)

        # Language hint (very basic)
        result.language_hint = self._detect_language_hint(text)

        return result

    def quick_stats(self, text: str) -> Dict[str, Any]:
        """Return quick statistics without full analysis.

        Args:
            text: The text to analyze.

        Returns:
            Dictionary with basic stats.
        """
        words = self._tokenize(text)
        return {
            "char_count": len(text),
            "word_count": len(words),
            "line_count": text.count("\n") + 1,
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.translate(str.maketrans("", "", string.punctuation))
        return [w for w in text.split() if w]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _avg_syllables_per_word(self, words: List[str]) -> float:
        """Estimate average syllables per word."""
        if not words:
            return 0.0
        total = sum(self._count_syllables(w) for w in words)
        return total / len(words)

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)
        word = re.sub(r"^y", "", word)
        matches = re.findall(r"[aeiouy]{1,2}", word)
        return max(1, len(matches))

    @staticmethod
    def _flesch_reading_ease(avg_sentence_len: float, avg_syllables: float) -> float:
        """Compute Flesch Reading Ease score."""
        return 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)

    def _sentiment_score(self, words: List[str]) -> float:
        """Compute a simple sentiment score."""
        if not words:
            return 0.0
        lower_words = [w.lower() for w in words]
        pos = sum(1 for w in lower_words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in lower_words if w in _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract top keywords by frequency, excluding stop words."""
        lower_words = [w.lower() for w in words if len(w) > 2]
        filtered = [w for w in lower_words if w not in _STOP_WORDS]
        counter = Counter(filtered)
        return [word for word, _ in counter.most_common(self._max_keywords)]

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        """Provide a very basic language detection hint."""
        # Check for CJK characters
        cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', text))
        latin_count = len(re.findall(r'[a-zA-Z]', text))
        if cjk_count > latin_count:
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
                return "ja"
            return "zh"
        if latin_count > 0:
            return "en"
        return "unknown"
