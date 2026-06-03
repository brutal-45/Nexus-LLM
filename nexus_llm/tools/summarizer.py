"""Nexus-LLM Summarization Tool.

Provides the SummarizerTool for generating text summaries using
extractive methods (no external model required).
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Set

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

# Simple stop words for English
_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "was", "are", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "not",
    "no", "nor", "if", "then", "than", "too", "very", "so", "just",
}


class SummarizerTool(BaseTool):
    """Extractive text summarization tool.

    Uses frequency-based sentence scoring to extract the most
    important sentences from a text. No external model is required.

    Supports methods: frequency, position, combined.

    Example::

        summarizer = SummarizerTool()
        result = summarizer.execute(text="Long article text...", ratio=0.3)
    """

    def __init__(self) -> None:
        super().__init__(name="summarizer", description="Extractive text summarization")

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="text", type=ParameterType.STRING, description="Text to summarize", required=True),
            ToolParameter(name="ratio", type=ParameterType.FLOAT, description="Summary ratio (0.0-1.0)", required=False, default=0.3),
            ToolParameter(name="method", type=ParameterType.STRING, description="Summarization method", required=False,
                          default="combined", choices=["frequency", "position", "combined"]),
            ToolParameter(name="max_sentences", type=ParameterType.INTEGER, description="Maximum sentences in summary", required=False, default=5),
        ]

    def execute(
        self,
        text: str = "",
        ratio: float = 0.3,
        method: str = "combined",
        max_sentences: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """Generate an extractive summary.

        Args:
            text: Input text to summarize.
            ratio: Fraction of original sentences to keep (0.0-1.0).
            method: Summarization method.
            max_sentences: Maximum number of sentences.

        Returns:
            ToolResult with the summary.
        """
        if not text or not text.strip():
            return ToolResult(tool_name=self.name, success=False, error="No text provided")

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return ToolResult(
                tool_name=self.name, success=True, output=text,
                metadata={"method": method, "original_sentences": len(sentences), "summary_sentences": len(sentences)},
            )

        # Determine number of summary sentences
        target = max(1, min(int(len(sentences) * ratio), max_sentences))

        if method == "frequency":
            scores = self._frequency_scores(sentences)
        elif method == "position":
            scores = self._position_scores(sentences)
        else:  # combined
            f_scores = self._frequency_scores(sentences)
            p_scores = self._position_scores(sentences)
            scores = {
                i: f_scores.get(i, 0) * 0.7 + p_scores.get(i, 0) * 0.3
                for i in range(len(sentences))
            }

        # Select top sentences, preserving original order
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in ranked[:target]])
        summary_sentences = [sentences[i] for i in selected_indices]
        summary = " ".join(summary_sentences)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output=summary,
            metadata={
                "method": method,
                "original_sentences": len(sentences),
                "summary_sentences": len(summary_sentences),
                "ratio": ratio,
                "compression": round(len(summary) / max(len(text), 1), 2),
            },
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _frequency_scores(self, sentences: List[str]) -> Dict[int, float]:
        """Score sentences by word frequency importance."""
        # Build word frequency table (excluding stop words)
        word_freq: Counter = Counter()
        for sent in sentences:
            words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
            filtered = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
            word_freq.update(filtered)

        if not word_freq:
            return {i: 0.0 for i in range(len(sentences))}

        max_freq = max(word_freq.values())
        normalized = {w: f / max_freq for w, f in word_freq.items()}

        scores: Dict[int, float] = {}
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b[a-zA-Z]+\b', sent.lower())
            scores[i] = sum(normalized.get(w, 0) for w in words) / max(len(words), 1)

        return scores

    @staticmethod
    def _position_scores(sentences: List[str]) -> Dict[int, float]:
        """Score sentences by position (first and last sentences score higher)."""
        n = len(sentences)
        scores: Dict[int, float] = {}
        for i in range(n):
            # First sentence gets highest, then decays
            if i == 0:
                scores[i] = 1.0
            elif i == n - 1:
                scores[i] = 0.5
            else:
                scores[i] = 1.0 / (1.0 + i * 0.3)
        return scores
