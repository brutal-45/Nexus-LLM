"""Curriculum learning: difficulty ranking, pacing functions, staged training."""

import math
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: str = "predetermined"
    pacing_function: str = "linear"
    difficulty_metric: str = "length"
    initial_fraction: float = 0.25
    final_fraction: float = 1.0
    num_stages: int = 4
    warmup_steps: int = 0
    total_steps: int = 1000
    seed: int = 42
    reverse: bool = False


class DifficultyRanker:
    """Ranks data samples by difficulty using various metrics."""

    def __init__(self, metric: str = "length", tokenizer: Optional[Any] = None):
        self.metric = metric
        self.tokenizer = tokenizer

    def rank(
        self,
        data: List[Dict[str, Any]],
        text_field: str = "text",
        ascending: bool = True,
    ) -> List[Tuple[int, float]]:
        """Rank data samples by difficulty.

        Args:
            data: List of data dictionaries.
            text_field: Field containing the text.
            ascending: If True, easiest samples come first.

        Returns:
            List of (original_index, difficulty_score) tuples, sorted by difficulty.
        """
        scores = []
        for idx, item in enumerate(data):
            score = self._compute_difficulty(item, text_field)
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=not ascending)
        return scores

    def _compute_difficulty(self, item: Dict[str, Any], text_field: str) -> float:
        """Compute a difficulty score for a single data item."""
        text = item.get(text_field, "")

        if self.metric == "length":
            return float(len(text))
        elif self.metric == "word_count":
            return float(len(text.split()))
        elif self.metric == "token_count":
            if self.tokenizer:
                return float(len(self.tokenizer.encode(text)))
            return float(len(text.split()))
        elif self.metric == "vocabulary_rarity":
            return self._vocabulary_rarity(text)
        elif self.metric == "sentence_complexity":
            return self._sentence_complexity(text)
        elif self.metric == "random":
            return random.random()
        else:
            return float(len(text))

    @staticmethod
    def _vocabulary_rarity(text: str) -> float:
        """Estimate vocabulary rarity based on word frequency heuristics."""
        common_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "may", "might", "must", "can", "could", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "this", "that", "these", "those",
            "and", "or", "but", "if", "then", "so", "as", "not", "no", "in", "on",
            "at", "to", "for", "of", "with", "by", "from", "up", "about", "into",
        }
        words = text.lower().split()
        if not words:
            return 0.0
        rare_count = sum(1 for w in words if w not in common_words)
        return rare_count / len(words)

    @staticmethod
    def _sentence_complexity(text: str) -> float:
        """Estimate sentence complexity based on structural features."""
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0

        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        num_clauses = text.count(",") + text.count(";") + text.count(":")
        return avg_sentence_len + num_clauses * 0.5


class PacingFunction:
    """Determines the pacing schedule for curriculum learning."""

    def __init__(
        self,
        pacing_type: str = "linear",
        initial_fraction: float = 0.25,
        final_fraction: float = 1.0,
        total_steps: int = 1000,
        warmup_steps: int = 0,
        **kwargs,
    ):
        self.pacing_type = pacing_type
        self.initial_fraction = initial_fraction
        self.final_fraction = final_fraction
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = warmup_steps
        self.kwargs = kwargs

    def get_fraction(self, step: int) -> float:
        """Get the fraction of data to use at a given step.

        Args:
            step: Current training step.

        Returns:
            Fraction of data to include (between initial_fraction and final_fraction).
        """
        if step < self.warmup_steps:
            return self.initial_fraction

        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)

        if self.pacing_type == "linear":
            fraction = self._linear_pacing(progress)
        elif self.pacing_type == "root":
            fraction = self._root_pacing(progress)
        elif self.pacing_type == "quadratic":
            fraction = self._quadratic_pacing(progress)
        elif self.pacing_type == "step":
            fraction = self._step_pacing(progress)
        elif self.pacing_type == "exponential":
            fraction = self._exponential_pacing(progress)
        elif self.pacing_type == "sigmoid":
            fraction = self._sigmoid_pacing(progress)
        else:
            fraction = self._linear_pacing(progress)

        return max(self.initial_fraction, min(fraction, self.final_fraction))

    def _linear_pacing(self, progress: float) -> float:
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * progress

    def _root_pacing(self, progress: float) -> float:
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * math.sqrt(progress)

    def _quadratic_pacing(self, progress: float) -> float:
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * (progress ** 2)

    def _step_pacing(self, progress: float) -> float:
        num_stages = self.kwargs.get("num_stages", 4)
        stage_size = 1.0 / num_stages
        stage = min(int(progress / stage_size), num_stages - 1)
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * (stage + 1) / num_stages

    def _exponential_pacing(self, progress: float) -> float:
        rate = self.kwargs.get("rate", 3.0)
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * (
            (math.exp(rate * progress) - 1) / (math.exp(rate) - 1)
        )

    def _sigmoid_pacing(self, progress: float) -> float:
        midpoint = self.kwargs.get("midpoint", 0.5)
        steepness = self.kwargs.get("steepness", 10.0)
        sigmoid = 1.0 / (1.0 + math.exp(-steepness * (progress - midpoint)))
        sigmoid_normalized = (sigmoid - 1.0 / (1.0 + math.exp(steepness * midpoint))) / (
            1.0 / (1.0 + math.exp(-steepness * (1.0 - midpoint)))
            - 1.0 / (1.0 + math.exp(steepness * midpoint))
        )
        return self.initial_fraction + (self.final_fraction - self.initial_fraction) * sigmoid_normalized


class CurriculumLearner:
    """Orchestrates curriculum learning by managing difficulty ranking and pacing."""

    def __init__(
        self,
        data: List[Dict[str, Any]],
        config: Optional[CurriculumConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.data = data
        self.config = config or CurriculumConfig()
        self.ranker = DifficultyRanker(
            metric=self.config.difficulty_metric,
            tokenizer=tokenizer,
        )
        self.pacer = PacingFunction(
            pacing_type=self.config.pacing_function,
            initial_fraction=self.config.initial_fraction,
            final_fraction=self.config.final_fraction,
            total_steps=self.config.total_steps,
            warmup_steps=self.config.warmup_steps,
            num_stages=self.config.num_stages,
        )
        self._ranked_indices: Optional[List[int]] = None
        self.current_step = 0

    def setup(self, text_field: str = "text"):
        """Rank all data samples by difficulty."""
        ranked = self.ranker.rank(
            self.data,
            text_field=text_field,
            ascending=not self.config.reverse,
        )
        self._ranked_indices = [idx for idx, score in ranked]
        logger.info(
            f"Curriculum learning setup complete. "
            f"Ranked {len(self.data)} samples by {self.config.difficulty_metric}."
        )

    def get_current_data(self, step: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get the subset of data appropriate for the current step.

        Args:
            step: Current training step. Uses internal counter if None.

        Returns:
            List of data items for this step.
        """
        if self._ranked_indices is None:
            self.setup()

        step = step if step is not None else self.current_step
        fraction = self.pacer.get_fraction(step)
        num_samples = max(1, int(len(self.data) * fraction))

        selected_indices = self._ranked_indices[:num_samples]
        return [self.data[idx] for idx in selected_indices]

    def get_current_dataset(
        self,
        step: Optional[int] = None,
        dataset_class: Optional[type] = None,
        **dataset_kwargs,
    ) -> Any:
        """Get a dataset object with the current curriculum subset.

        Args:
            step: Current training step.
            dataset_class: Dataset class to instantiate.
            **dataset_kwargs: Additional arguments for the dataset class.

        Returns:
            Dataset instance with curriculum-filtered data.
        """
        current_data = self.get_current_data(step)

        if dataset_class is not None:
            return dataset_class(current_data, **dataset_kwargs)

        from nexus_llm.training.dataset import TextDataset
        return TextDataset(current_data, **dataset_kwargs)

    def step(self) -> List[Dict[str, Any]]:
        """Advance to the next step and return the appropriate data subset."""
        data = self.get_current_data(self.current_step)
        self.current_step += 1
        return data

    def get_staged_data(self, num_stages: Optional[int] = None) -> List[List[Dict[str, Any]]]:
        """Split data into stages for staged curriculum learning.

        Args:
            num_stages: Number of stages. Defaults to config value.

        Returns:
            List of data lists, one per stage.
        """
        if self._ranked_indices is None:
            self.setup()

        num_stages = num_stages or self.config.num_stages
        stage_size = len(self._ranked_indices) // num_stages

        stages = []
        for i in range(num_stages):
            start = 0
            end = min((i + 1) * stage_size, len(self._ranked_indices))
            stage_indices = self._ranked_indices[:end]
            stages.append([self.data[idx] for idx in stage_indices])

        logger.info(f"Created {len(stages)} curriculum stages with sizes: {[len(s) for s in stages]}")
        return stages

    def get_difficulty_distribution(self) -> Dict[str, Any]:
        """Get statistics about the difficulty distribution."""
        if self._ranked_indices is None:
            self.setup()

        ranked = self.ranker.rank(self.data, ascending=not self.config.reverse)
        scores = [score for _, score in ranked]

        if not scores:
            return {}

        return {
            "num_samples": len(scores),
            "min_difficulty": min(scores),
            "max_difficulty": max(scores),
            "mean_difficulty": sum(scores) / len(scores),
            "median_difficulty": sorted(scores)[len(scores) // 2],
        }

    def reset(self):
        """Reset the curriculum to the beginning."""
        self.current_step = 0
