"""Knowledge distiller for Nexus-LLM.

Implements the core distillation logic: computing teacher logits,
calculating the distillation loss (KL-divergence between softened
distributions), and combining it with the hard-label loss.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.distillation.config import DistillationConfig

logger = logging.getLogger(__name__)


class Distiller:
    """Orchestrate knowledge distillation from teacher to student.

    The Distiller is responsible for the *mechanics* of distillation —
    computing teacher logits, forming the distillation loss, and
    returning a trained student.  For a higher-level training loop with
    logging and checkpointing, see :class:`DistillationTrainer`.

    Usage::

        distiller = Distiller()
        student = distiller.distill(
            teacher_model=teacher,
            student_model=student,
            dataset=train_data,
            config=DistillationConfig(temperature=4.0, alpha=0.7),
        )
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """Initialise the Distiller.

        Args:
            device: Target device string (e.g. ``"cuda:0"``).  When
                *None*, the device is inferred from the model.
        """
        self._device = device

    # ------------------------------------------------------------------
    # Main distillation entry point
    # ------------------------------------------------------------------

    def distill(
        self,
        teacher_model: Any,
        student_model: Any,
        dataset: List[Dict[str, Any]],
        config: DistillationConfig,
    ) -> Any:
        """Distill knowledge from *teacher_model* into *student_model*.

        This is a convenience wrapper that runs a simple training loop
        using :meth:`compute_teacher_logits` and
        :meth:`distillation_loss` at each step.

        Args:
            teacher_model: The teacher model (frozen during distillation).
            student_model: The student model (updated in-place).
            dataset: Training data — a list of dicts, each containing
                at least ``"input_ids"`` and ``"labels"`` tensors.
            config: Distillation hyper-parameters.

        Returns:
            The trained student model (same object as *student_model*).
        """
        import torch
        import torch.nn.functional as F

        logger.info(
            "Starting distillation: temperature=%.1f, alpha=%.2f, "
            "lr=%.2e, batch_size=%d, epochs=%d",
            config.temperature,
            config.alpha,
            config.learning_rate,
            config.batch_size,
            config.epochs,
        )

        # Move models to device
        device = self._resolve_device(teacher_model)
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        teacher_model.eval()

        # Optimiser for student only
        optimizer = torch.optim.AdamW(
            student_model.parameters(), lr=config.learning_rate
        )

        num_batches = max(1, len(dataset) // config.batch_size)
        global_step = 0

        for epoch in range(1, config.epochs + 1):
            student_model.train()
            epoch_loss = 0.0

            for batch_idx in range(num_batches):
                # Slice batch
                start = batch_idx * config.batch_size
                end = start + config.batch_size
                batch = dataset[start:end]

                # Collate inputs
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.as_tensor(s["input_ids"]) for s in batch],
                    batch_first=True,
                ).to(device)
                labels = torch.cat(
                    [torch.as_tensor(s["labels"]) for s in batch]
                ).to(device)
                attention_mask = (input_ids != 0).long()

                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = self.compute_teacher_logits(
                        teacher_model, {"input_ids": input_ids, "attention_mask": attention_mask}
                    )

                # Student forward
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs

                # Compute combined loss
                loss = self.distillation_loss(
                    student_logits, teacher_logits, config.temperature
                )

                # Add hard-label loss
                hard_loss = F.cross_entropy(
                    student_logits.view(-1, student_logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = config.alpha * loss + (1.0 - config.alpha) * hard_loss

                # Backprop
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                global_step += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                "Epoch %d/%d — avg_loss=%.4f, global_step=%d",
                epoch,
                config.epochs,
                avg_loss,
                global_step,
            )

        student_model.eval()
        logger.info("Distillation complete — total steps=%d", global_step)
        return student_model

    # ------------------------------------------------------------------
    # Teacher logit computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_teacher_logits(teacher: Any, inputs: Dict[str, Any]) -> Any:
        """Compute teacher logits without updating teacher weights.

        Args:
            teacher: The teacher model (should already be in eval mode).
            inputs: Dict of tensors (``input_ids``, ``attention_mask``,
                etc.) that the teacher's forward method accepts.

        Returns:
            Teacher logits tensor of shape ``(batch, seq_len, vocab)``.
        """
        import torch

        teacher.eval()
        with torch.no_grad():
            outputs = teacher(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits

    # ------------------------------------------------------------------
    # Distillation loss
    # ------------------------------------------------------------------

    @staticmethod
    def distillation_loss(
        student_logits: Any,
        teacher_logits: Any,
        temperature: float = 2.0,
    ) -> Any:
        """Compute the KL-divergence distillation loss.

        The loss is the KL divergence between the softened teacher
        distribution and the softened student distribution, multiplied
        by ``temperature ** 2`` to ensure gradient magnitudes remain
        consistent across temperature choices.

        .. math::

            L_{\\text{distill}} = T^2 \\cdot
            D_{\\mathrm{KL}}(p_T \\| p_S)

        where :math:`p_T = \\sigma(z_T / T)` and
        :math:`p_S = \\sigma(z_S / T)`.

        Args:
            student_logits: Raw logits from the student model.
            teacher_logits: Raw logits from the teacher model.
            temperature: Softmax temperature.

        Returns:
            Scalar loss tensor.
        """
        import torch
        import torch.nn.functional as F

        if student_logits.shape != teacher_logits.shape:
            # Trim to matching length if sequence dimensions differ
            min_seq = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq, :]
            teacher_logits = teacher_logits[:, :min_seq, :]

        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        # KL divergence: sum over vocab, mean over batch & seq
        kl = F.kl_div(
            log_soft_student,
            soft_targets,
            reduction="batchmean",
        )

        # Scale by T^2 to keep gradient magnitudes consistent
        loss = (temperature ** 2) * kl
        return loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_device(self, model: Any) -> str:
        """Determine the device a model lives on."""
        if self._device is not None:
            return self._device
        try:
            import torch
            return str(next(model.parameters()).device)
        except Exception:
            return "cpu"
