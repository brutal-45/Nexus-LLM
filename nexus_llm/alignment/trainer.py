"""Alignment trainer for Nexus-LLM.

Implements preference-based alignment using Direct Preference
Optimization (DPO) and an RLHF-style training loop.  DPO directly
optimises the policy from preference pairs without a separate reward
model, while RLHF uses the reward model as a signal source.
"""

import copy
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_llm.alignment.config import RLHFConfig
from nexus_llm.alignment.preference import PreferenceDataset, PreferenceSample
from nexus_llm.alignment.reward import RewardModel

logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """Train a language model to align with human preferences.

    Supports two training modes controlled by
    :class:`RLHFConfig`.method:

    * **DPO** – Direct Preference Optimization.  Uses the
      ``compute_preference_loss`` method to compute the DPO loss from
      chosen/rejected log-probability ratios and updates the policy
      directly.
    * **RLHF** – Uses a :class:`RewardModel` to score responses and
      a simplified PPO-style policy gradient to update the model.

    Usage::

        trainer = AlignmentTrainer()
        config = RLHFConfig(method="dpo", beta=0.1)
        aligned_model = trainer.train(model, preference_dataset, config)
    """

    def __init__(
        self,
        reward_model: Optional[RewardModel] = None,
        reference_model: Optional[Any] = None,
        checkpoint_dir: str = "checkpoints/alignment",
    ) -> None:
        """Initialise the AlignmentTrainer.

        Args:
            reward_model: A :class:`RewardModel` instance (required for
                RLHF mode, optional for DPO).
            reference_model: The frozen reference policy used in DPO to
                compute log-probability ratios.  If *None*, a deep copy
                of the training model is made at train time.
            checkpoint_dir: Directory for checkpoint saves.
        """
        self._reward_model = reward_model or RewardModel()
        self._reference_model = reference_model
        self._checkpoint_dir = Path(checkpoint_dir)

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        model: Any,
        dataset: PreferenceDataset,
        config: Optional[RLHFConfig] = None,
    ) -> Any:
        """Align *model* with the preference data.

        Args:
            model: The language model to align (modified in-place).
            dataset: A :class:`PreferenceDataset` containing chosen and
                rejected response pairs.
            config: Alignment configuration.  Defaults to
                ``RLHFConfig()``.

        Returns:
            The aligned model (same object as *model*).
        """
        cfg = config or RLHFConfig()
        logger.info(
            "Starting alignment training: method=%s, beta=%.2f, "
            "lr=%.2e, batch_size=%d, epochs=%d",
            cfg.method,
            cfg.beta,
            cfg.learning_rate,
            cfg.batch_size,
            cfg.epochs,
        )

        if cfg.method == "dpo":
            return self._train_dpo(model, dataset, cfg)
        elif cfg.method == "rlhf":
            return self._train_rlhf(model, dataset, cfg)
        else:
            raise ValueError(f"Unsupported alignment method: {cfg.method}")

    # ------------------------------------------------------------------
    # DPO training
    # ------------------------------------------------------------------

    def _train_dpo(
        self,
        model: Any,
        dataset: PreferenceDataset,
        config: RLHFConfig,
    ) -> Any:
        """Run DPO-style alignment training.

        The DPO loss optimises the policy so that the log-odds ratio
        between chosen and rejected responses increases relative to the
        reference policy.

        .. math::

            \\mathcal{L}_{\\text{DPO}} = -\\mathbb{E}\\left[
            \\log\\sigma\\left(
            \\beta \\left(
            \\log\\frac{\\pi_\\theta(y_w|x)}
            {\\pi_{\\text{ref}}(y_w|x)}
            - \\log\\frac{\\pi_\\theta(y_l|x)}
            {\\pi_{\\text{ref}}(y_l|x)}
            \\right)
            \\right)\\right]
        """
        import torch
        import torch.nn.functional as F

        # Create frozen reference model
        ref_model = self._reference_model
        if ref_model is None:
            logger.info("Creating reference model (deep copy of policy)")
            ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        device = self._resolve_device(model)
        model = model.to(device)
        ref_model = ref_model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        num_samples = dataset.size()
        num_batches = max(1, num_samples // config.batch_size)
        start_time = time.perf_counter()

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                batch = dataset.get_batch(config.batch_size)

                total_loss = torch.tensor(0.0, device=device)
                for sample in batch:
                    # Tokenise chosen and rejected responses
                    chosen_inputs = self._tokenise(sample.prompt, sample.chosen, device)
                    rejected_inputs = self._tokenise(sample.prompt, sample.rejected, device)

                    # Policy log-probs
                    chosen_logp = self._get_sequence_logp(model, chosen_inputs)
                    rejected_logp = self._get_sequence_logp(model, rejected_inputs)

                    # Reference log-probs (no grad)
                    with torch.no_grad():
                        ref_chosen_logp = self._get_sequence_logp(ref_model, chosen_inputs)
                        ref_rejected_logp = self._get_sequence_logp(ref_model, rejected_inputs)

                    # DPO loss
                    loss = self.compute_preference_loss(
                        chosen_logp - ref_chosen_logp,
                        rejected_logp - ref_rejected_logp,
                        beta=config.beta,
                    )
                    total_loss = total_loss + loss

                # Average over the batch
                batch_loss = total_loss / max(len(batch), 1)

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info("DPO epoch %d/%d — avg_loss=%.4f", epoch, config.epochs, avg_loss)

        elapsed = time.perf_counter() - start_time
        logger.info("DPO training complete — %.1fs", elapsed)

        model.eval()
        return model

    # ------------------------------------------------------------------
    # RLHF training
    # ------------------------------------------------------------------

    def _train_rlhf(
        self,
        model: Any,
        dataset: PreferenceDataset,
        config: RLHFConfig,
    ) -> Any:
        """Run RLHF-style alignment using the reward model.

        For each prompt in the preference dataset, generates a response
        from the current policy, scores it with the reward model, and
        applies a simplified REINFORCE-style policy gradient update.
        """
        import torch

        device = self._resolve_device(model)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        num_samples = dataset.size()
        num_batches = max(1, num_samples // config.batch_size)
        start_time = time.perf_counter()

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                batch = dataset.get_batch(config.batch_size)

                batch_loss = torch.tensor(0.0, device=device)
                for sample in batch:
                    # Use chosen response as the "generated" response
                    # (in a full pipeline, model.generate() would be called)
                    inputs = self._tokenise(sample.prompt, sample.chosen, device)
                    logp = self._get_sequence_logp(model, inputs)

                    # Get reward from the reward model
                    reward = self._reward_model.score(sample.prompt, sample.chosen)
                    reward_tensor = torch.tensor(reward, device=device, dtype=torch.float32)

                    # REINFORCE: -logp * reward  (negative because we minimise)
                    loss = -logp * reward_tensor
                    batch_loss = batch_loss + loss

                batch_loss = batch_loss / max(len(batch), 1)

                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info("RLHF epoch %d/%d — avg_loss=%.4f", epoch, config.epochs, avg_loss)

        elapsed = time.perf_counter() - start_time
        logger.info("RLHF training complete — %.1fs", elapsed)

        model.eval()
        return model

    # ------------------------------------------------------------------
    # DPO preference loss
    # ------------------------------------------------------------------

    @staticmethod
    def compute_preference_loss(
        chosen_logps: Any,
        rejected_logps: Any,
        beta: float = 0.1,
    ) -> Any:
        """Compute the DPO preference loss.

        Given the log-probability ratios (policy - reference) for the
        chosen and rejected responses, the DPO loss is:

        .. math::

            \\mathcal{L} = -\\log\\sigma\\left(
            \\beta (\\log r_w - \\log r_l)
            \\right)

        where :math:`r_w` and :math:`r_l` are the chosen/rejected
        likelihood ratios.

        Args:
            chosen_logps: Log-probability difference (policy - ref)
                for the chosen response.  Can be a scalar or tensor.
            rejected_logps: Log-probability difference (policy - ref)
                for the rejected response.
            beta: Inverse temperature / KL penalty strength.

        Returns:
            Scalar loss tensor.
        """
        import torch

        chosen = chosen_logps if isinstance(chosen_logps, torch.Tensor) else torch.tensor(chosen_logps)
        rejected = rejected_logps if isinstance(rejected_logps, torch.Tensor) else torch.tensor(rejected_logps)

        logits = beta * (chosen - rejected)
        loss = -torch.nn.functional.logsigmoid(logits)
        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(prompt: str, response: str, device: str) -> Dict[str, Any]:
        """Tokenise a prompt + response pair.

        Falls back to a simple whitespace tokeniser if no HuggingFace
        tokenizer is available.
        """
        import torch

        text = f"{prompt}\n{response}"

        # Try using the model's tokenizer
        try:
            tokenizer = torch.hub.load("huggingface/pytorch-transformers", "tokenizer", "gpt2")
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            return {k: v.to(device) for k, v in enc.items()}
        except Exception:
            pass

        # Fallback: simple character-level tokenisation
        token_ids = [ord(c) % 256 for c in text][:512]
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def _get_sequence_logp(model: Any, inputs: Dict[str, Any]) -> Any:
        """Compute the log-probability of a token sequence under *model*.

        Returns the mean log-probability across all tokens.
        """
        import torch
        import torch.nn.functional as F

        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Shift for next-token prediction
        input_ids = inputs["input_ids"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Log-softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather the log-prob of the actual token at each position
        per_token_logp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Mask padding
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
        shift_mask = attention_mask[:, 1:].contiguous()
        per_token_logp = per_token_logp * shift_mask

        # Mean log-prob over non-padding tokens
        num_tokens = shift_mask.sum().clamp(min=1)
        seq_logp = per_token_logp.sum() / num_tokens
        return seq_logp

    @staticmethod
    def _resolve_device(model: Any) -> str:
        """Determine the device a model lives on."""
        try:
            import torch
            return str(next(model.parameters()).device)
        except Exception:
            return "cpu"
