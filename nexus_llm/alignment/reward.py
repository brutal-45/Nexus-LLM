"""Reward model for alignment training.

Provides :class:`RewardModel` — a model that scores prompt-response
pairs and ranks candidate responses.  In a full RLHF pipeline this
would be a trained scalar-head model; here the implementation supports
both a mock scoring mode (for testing and pipeline development) and
a pluggable backend for real reward models.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RewardModel:
    """Score and rank responses using a reward model.

    The default implementation uses a **deterministic mock** scorer
    based on string hashing, which is useful for pipeline development
    and integration testing.  A real model can be plugged in by
    passing a callable ``scorer_fn`` at construction time.

    Usage::

        rm = RewardModel()
        score = rm.score("What is AI?", "AI is the simulation of intelligence.")
        ranked = rm.rank_responses("What is AI?", ["Good answer", "Bad answer"])
    """

    def __init__(
        self,
        scorer_fn: Optional[Any] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialise the RewardModel.

        Args:
            scorer_fn: Optional callable with signature
                ``(prompt: str, response: str) -> float`` that replaces
                the mock scorer.
            model: An optional HuggingFace ``PreTrainedModel`` or
                PyTorch ``nn.Module`` used as the reward backbone.
                When provided, ``scorer_fn`` is ignored and the model's
                forward output is used as the reward signal.
            device: Target device for the model.
        """
        self._scorer_fn = scorer_fn
        self._model = model
        self._device = device
        self._trained = False
        self._preference_buffer: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, prompt: str, response: str) -> float:
        """Score a single prompt-response pair.

        Uses the custom scorer function if provided, otherwise the model
        if available, otherwise the deterministic mock scorer.

        Args:
            prompt: The input prompt / context.
            response: The model response to evaluate.

        Returns:
            A scalar reward score (higher is better).
        """
        # Custom scorer function
        if self._scorer_fn is not None:
            try:
                return float(self._scorer_fn(prompt, response))
            except Exception as exc:
                logger.warning("scorer_fn failed: %s — falling back to mock", exc)

        # Real model
        if self._model is not None:
            return self._score_with_model(prompt, response)

        # Mock scorer
        return self._mock_score(prompt, response)

    def rank_responses(
        self,
        prompt: str,
        responses: List[str],
    ) -> List[Tuple[str, float]]:
        """Score and rank a list of candidate responses.

        Args:
            prompt: The shared prompt / context.
            responses: List of candidate response strings.

        Returns:
            List of ``(response, score)`` tuples sorted in descending
            order of score (best first).
        """
        scored = [(resp, self.score(prompt, resp)) for resp in responses]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Training from preferences (mock)
    # ------------------------------------------------------------------

    def train_from_preferences(
        self,
        preferences: List[Dict[str, str]],
        epochs: int = 1,
        learning_rate: float = 1e-5,
    ) -> Dict[str, Any]:
        """Train the reward model from preference data (mock).

        In a full implementation this would fine-tune the reward model
        using a Bradley-Terry preference loss.  The mock version
        buffers the preferences and updates an internal heuristic.

        Args:
            preferences: List of dicts with keys ``prompt``, ``chosen``,
                and ``rejected``.
            epochs: Number of training epochs.
            learning_rate: Learning rate (unused in mock).

        Returns:
            Dict with training metadata.
        """
        logger.info(
            "Mock reward model training: %d preferences, %d epochs",
            len(preferences),
            epochs,
        )

        # Buffer preferences for use by the mock scorer
        for pref in preferences:
            self._preference_buffer.append(
                {
                    "prompt": pref["prompt"],
                    "chosen": pref["chosen"],
                    "rejected": pref["rejected"],
                }
            )

        # Simulate training
        for epoch in range(1, epochs + 1):
            # In a real implementation: forward pass + Bradley-Terry loss + backward
            mock_loss = max(0.1, 1.0 / (1.0 + len(self._preference_buffer) * 0.01))
            logger.debug(
                "Mock training epoch %d/%d — loss=%.4f",
                epoch,
                epochs,
                mock_loss,
            )

        self._trained = True

        return {
            "status": "completed",
            "num_preferences": len(self._preference_buffer),
            "epochs": epochs,
            "final_loss": mock_loss,
            "mode": "mock",
        }

    @property
    def is_trained(self) -> bool:
        """Whether the reward model has been trained."""
        return self._trained

    # ------------------------------------------------------------------
    # Internal – model-based scoring
    # ------------------------------------------------------------------

    def _score_with_model(self, prompt: str, response: str) -> float:
        """Score using the loaded neural reward model."""
        import torch

        try:
            # Tokenise (requires tokenizer attribute or external setup)
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer is None:
                logger.warning("No tokenizer on model — falling back to mock score")
                return self._mock_score(prompt, response)

            text = f"{prompt}\n{response}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self._device or "cpu") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Assume the model outputs a scalar reward as the last element
            if hasattr(outputs, "logits"):
                reward = outputs.logits[:, -1].item()
            else:
                reward = float(outputs)

            return reward

        except Exception as exc:
            logger.warning("Model scoring failed: %s — falling back to mock", exc)
            return self._mock_score(prompt, response)

    # ------------------------------------------------------------------
    # Internal – mock scorer
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_score(prompt: str, response: str) -> float:
        """Deterministic mock scorer based on content hashing.

        Produces a score in the range [0.0, 1.0] that is:

        * Deterministic — same input always yields the same score.
        * Correlated with response length (longer ≈ higher, with
          diminishing returns) — mimics real reward models that tend
          to favour more informative responses.
        """
        # Hash the combined text for determinism
        text = f"{prompt}|||{response}"
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hash_val = int(h[:8], 16) / 0xFFFFFFFF  # [0, 1)

        # Length bonus: longer responses get a slight boost
        length_bonus = min(len(response) / 500.0, 0.3)

        # Combine: base score from hash + length bonus, clamped to [0, 1]
        score = min(hash_val * 0.7 + length_bonus, 1.0)
        return round(score, 6)
