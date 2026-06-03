"""Loss functions: cross-entropy, label smoothing, focal loss, KL divergence, custom losses."""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Label smoothing prevents the model from becoming overconfident by
    distributing a small portion of probability mass across all tokens.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            logits: Model output logits of shape (batch, seq_len, vocab_size) or (batch, vocab_size).
            targets: Target token IDs of shape (batch, seq_len) or (batch,).

        Returns:
            Computed loss scalar.
        """
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

        vocab_size = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (vocab_size - 1))
            mask = targets != self.ignore_index
            valid_targets = targets.clone()
            valid_targets[~mask] = 0
            smooth_targets.scatter_(1, valid_targets.unsqueeze(1), 1.0 - self.smoothing)
            smooth_targets[~mask] = 0.0

        loss = -smooth_targets * log_probs
        loss = loss.sum(dim=-1)

        if mask.any():
            loss = loss[mask]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Model output logits.
            targets: Target labels.

        Returns:
            Computed loss scalar.
        """
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)

        ce_loss = F.cross_entropy(
            logits, targets, ignore_index=self.ignore_index, reduction="none"
        )

        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * (targets != self.ignore_index).float()
            focal_loss = alpha_t * focal_loss

        mask = targets != self.ignore_index
        focal_loss = focal_loss[mask] if mask.any() else focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class KLDivergenceLoss(nn.Module):
    """KL Divergence loss for knowledge distillation."""

    def __init__(
        self,
        temperature: float = 2.0,
        reduction: str = "batchmean",
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence between student and teacher distributions.

        Args:
            student_logits: Student model logits.
            teacher_logits: Teacher model logits.
            mask: Optional mask to ignore certain positions.

        Returns:
            KL divergence loss scalar.
        """
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="none")
        kl_loss = kl_loss * (self.temperature ** 2)

        if mask is not None:
            kl_loss = kl_loss * mask.unsqueeze(-1)
            kl_loss = kl_loss.sum() / max(mask.sum() * kl_loss.size(-1), 1)
            return kl_loss

        if self.reduction == "batchmean":
            return kl_loss.mean()
        elif self.reduction == "sum":
            return kl_loss.sum()
        return kl_loss


class CombinedLoss(nn.Module):
    """Combined loss that merges multiple loss functions with weights."""

    def __init__(
        self,
        losses: Optional[list] = None,
        weights: Optional[list] = None,
    ):
        """Initialize combined loss.

        Args:
            losses: List of (loss_fn, name) tuples.
            weights: List of weights for each loss function.
        """
        super().__init__()
        self.losses = losses or []
        self.weights = weights or [1.0] * len(self.losses)

        if len(self.losses) != len(self.weights):
            raise ValueError("Number of losses must match number of weights.")

    def add_loss(self, loss_fn: nn.Module, weight: float = 1.0, name: str = ""):
        """Add a loss function with weight."""
        self.losses.append((loss_fn, name))
        self.weights.append(weight)

    def forward(self, **kwargs) -> torch.Tensor:
        """Compute the combined loss.

        Kwargs are passed to all loss functions. Each loss function
        should accept the relevant kwargs.
        """
        total_loss = torch.tensor(0.0, device=kwargs.get("logits", kwargs.get("student_logits", torch.tensor(0.0))).device if kwargs else "cpu")
        for (loss_fn, name), weight in zip(self.losses, self.weights):
            try:
                loss = loss_fn(**kwargs)
                total_loss = total_loss + weight * loss
            except TypeError as e:
                logger.warning(f"Loss {name} failed with kwargs: {e}")
        return total_loss


class PerplexityLoss(nn.Module):
    """Loss that returns perplexity instead of cross-entropy."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute perplexity from logits and targets."""
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return torch.exp(ce_loss)


class CosineEmbeddingLoss(nn.Module):
    """Cosine embedding loss for contrastive learning."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine embedding loss.

        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            labels: 1 for similar pairs, -1 for dissimilar.

        Returns:
            Cosine embedding loss.
        """
        return F.cosine_embedding_loss(
            embeddings1, embeddings2, labels,
            margin=self.margin, reduction=self.reduction,
        )


class MaskedLanguageModelLoss(nn.Module):
    """Loss specifically for masked language modeling."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute MLM loss only on masked positions.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size).
            labels: Target labels (batch, seq_len).
            mask: Optional binary mask (batch, seq_len), 1 for positions to compute loss.

        Returns:
            MLM loss scalar.
        """
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        loss = F.cross_entropy(
            logits_flat, labels_flat, ignore_index=self.ignore_index, reduction="none"
        )
        loss = loss.reshape(batch_size, seq_len)

        if mask is not None:
            loss = loss * mask
            num_masked = mask.sum()
            if num_masked > 0:
                return loss.sum() / num_masked
            return loss.sum() * 0.0

        active_loss = labels_flat != self.ignore_index
        num_active = active_loss.sum()
        if num_active > 0:
            return loss.reshape(-1)[active_loss].mean()
        return loss.sum() * 0.0
