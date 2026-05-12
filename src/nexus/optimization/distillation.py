"""
Knowledge Distillation Module
==============================

Production-grade knowledge distillation methods for training compact student
models from larger teacher models. Includes response-based, feature-based,
attention-based, multi-teacher, progressive, TinyBERT, and MiniLM-style
distillation.

References:
    - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    - TinyBERT: "TinyBERT: Distilling BERT for Natural Language Understanding", 2020
    - MiniLM: "MiniLM: Deep Self-Attention Distillation", 2020
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nexus.optimization.optimization_config import DistillationConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _compute_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Compute KL divergence between student and teacher softened logits.

    Args:
        student_logits: Student model output logits.
        teacher_logits: Teacher model output logits.
        temperature: Softmax temperature for softening.

    Returns:
        KL divergence loss scalar.
    """
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_soft, teacher_soft, reduction="batchmean"
    ) * (temperature ** 2)

    return kl_loss


def _compute_cosine_similarity_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity based loss between features.

    Args:
        student_feat: Student feature tensor.
        teacher_feat: Teacher feature tensor.

    Returns:
        Mean cosine distance loss.
    """
    if student_feat.shape != teacher_feat.shape:
        min_dim = min(student_feat.shape[-1], teacher_feat.shape[-1])
        student_feat = student_feat[..., :min_dim]
        teacher_feat = teacher_feat[..., :min_dim]

    student_flat = student_feat.float().flatten(1)
    teacher_flat = teacher_feat.float().flatten(1)

    cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
    loss = 1.0 - cos_sim.mean()
    return loss


def _compute_mse_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss between features with optional dimension projection.

    Args:
        student_feat: Student feature tensor.
        teacher_feat: Teacher feature tensor.

    Returns:
        MSE loss scalar.
    """
    if student_feat.shape != teacher_feat.shape:
        if student_feat.shape[-1] > teacher_feat.shape[-1]:
            student_feat = student_feat[..., :teacher_feat.shape[-1]]
        else:
            teacher_feat = teacher_feat[..., :student_feat.shape[-1]]

    return F.mse_loss(student_feat.float(), teacher_feat.float())


def _project_features(
    source: torch.Tensor,
    target_dim: int,
    projection: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, nn.Module]:
    """Project features to a target dimension.

    Args:
        source: Source feature tensor.
        target_dim: Target dimension.
        projection: Optional pre-existing projection layer.

    Returns:
        Tuple of (projected_features, projection_layer).
    """
    source_dim = source.shape[-1]

    if source_dim == target_dim:
        return source, projection

    if projection is None:
        projection = nn.Linear(source_dim, target_dim, bias=False)
        nn.init.orthogonal_(projection.weight)

    return projection(source.float()), projection


def _align_attention_maps(
    student_attn: torch.Tensor,
    teacher_attn: torch.Tensor,
) -> torch.Tensor:
    """Align attention maps between student and teacher.

    Handles different head counts by averaging over heads.

    Args:
        student_attn: Student attention of shape (batch, heads_s, seq, seq).
        teacher_attn: Teacher attention of shape (batch, heads_t, seq, seq).

    Returns:
        Aligned student attention.
    """
    if student_attn.shape == teacher_attn.shape:
        return student_attn

    student_avg = student_attn.mean(dim=1)
    teacher_avg = teacher_attn.mean(dim=1)

    if student_avg.shape != teacher_avg.shape:
        min_seq = min(student_avg.shape[-1], teacher_avg.shape[-1])
        student_avg = student_avg[:, :min_seq, :min_seq]
        teacher_avg = teacher_avg[:, :min_seq, :min_seq]

    return student_avg


def _get_named_intermediate_outputs(
    model: nn.Module,
    input_data: Any,
    target_layers: Optional[List[str]] = None,
    device: torch.device = None,
) -> Dict[str, torch.Tensor]:
    """Extract intermediate outputs from specific layers.

    Args:
        model: Model to extract from.
        input_data: Input data for forward pass.
        target_layers: Layer names to extract from.
        device: Target device.

    Returns:
        Dictionary of layer name to output tensor.
    """
    if device is None:
        device = _get_device()

    outputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    outputs[name] = output[0].detach()
        return hook_fn

    for name, module in model.named_modules():
        if target_layers is None or name in target_layers:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    model.eval()
    with torch.no_grad():
        if isinstance(input_data, (list, tuple)):
            inputs = [item.to(device) if isinstance(item, torch.Tensor) else item for item in input_data]
            model(*inputs)
        elif isinstance(input_data, dict):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in input_data.items()}
            model(**inputs)
        else:
            model(input_data.to(device) if isinstance(input_data, torch.Tensor) else input_data)

    for h in hooks:
        h.remove()

    return outputs


# =============================================================================
# Base Distiller
# =============================================================================

class BaseDistiller(ABC):
    """Abstract base class for knowledge distillation."""

    def __init__(self, config: DistillationConfig):
        """Initialize the base distiller.

        Args:
            config: Distillation configuration.
        """
        self.config = config
        self.device = torch.device(config.device) if config.device != "auto" else _get_device()
        self.temperature = config.temperature
        self.alpha = config.alpha
        self._history: List[Dict[str, Any]] = []
        self._feature_projections: Dict[str, nn.Module] = {}

    @abstractmethod
    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run the distillation process.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        ...


# =============================================================================
# DistillationTrainer
# =============================================================================

class DistillationTrainer(BaseDistiller):
    """Main trainer for knowledge distillation.

    Supports response-based distillation with KL divergence loss,
    combined with standard cross-entropy loss on hard labels.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize distillation trainer.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.optimizer = None
        self.scheduler = None
        self._best_eval_loss = float("inf")
        self._patience_counter = 0

    def forward(
        self,
        student: nn.Module,
        teacher: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Joint forward pass through teacher and student.

        Args:
            student: Student model.
            teacher: Teacher model.
            inputs: Input batch.

        Returns:
            Dictionary with student_logits, teacher_logits, and labels.
        """
        teacher.eval()
        student.train()

        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        teacher_inputs = {}
        for k, v in inputs.items():
            if k != "labels" and isinstance(v, torch.Tensor):
                teacher_inputs[k] = v.to(self.device)

        with torch.no_grad():
            teacher_outputs = teacher(**teacher_inputs)

        student_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                student_inputs[k] = v.to(self.device)

        student_outputs = student(**student_inputs)

        teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs
        student_logits = student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs

        if labels is not None:
            labels = labels.to(self.device)

        return {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "labels": labels,
        }

    def compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute KL divergence loss between softened logits.

        Args:
            student_logits: Student output logits.
            teacher_logits: Teacher output logits.
            temperature: Softmax temperature (uses config default if None).

        Returns:
            KL divergence loss.
        """
        if temperature is None:
            temperature = self.temperature

        t = max(temperature, 0.01)

        student_soft = F.log_softmax(student_logits.float() / t, dim=-1)
        teacher_soft = F.softmax(teacher_logits.float() / t, dim=-1)

        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        kl_loss = kl_loss * (t * t)

        return kl_loss

    def compute_hard_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard cross-entropy loss on hard labels.

        Args:
            student_logits: Student output logits.
            labels: Ground truth labels.

        Returns:
            Cross-entropy loss.
        """
        if labels is None:
            return torch.tensor(0.0, device=student_logits.device)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        return loss_fn(student_logits.float(), labels)

    def compute_feature_loss(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute MSE loss on intermediate features.

        Args:
            student_features: Student intermediate features.
            teacher_features: Teacher intermediate features.

        Returns:
            Feature distillation loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for name in teacher_features:
            if name in student_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name]

                if s_feat.shape != t_feat.shape:
                    if s_feat.shape[-1] != t_feat.shape[-1]:
                        min_dim = min(s_feat.shape[-1], t_feat.shape[-1])
                        s_feat = s_feat[..., :min_dim]
                        t_feat = t_feat[..., :min_dim]

                    if s_feat.shape != t_feat.shape:
                        min_shape = [min(a, b) for a, b in zip(s_feat.shape, t_feat.shape)]
                        s_slice = tuple(slice(0, s) for s in min_shape)
                        t_slice = tuple(slice(0, s) for s in min_shape)
                        s_feat = s_feat[s_slice]
                        t_feat = t_feat[t_slice]

                if s_feat.numel() > 0 and t_feat.numel() > 0:
                    layer_loss = F.mse_loss(s_feat.float(), t_feat.float())
                    total_loss = total_loss + layer_loss
                    count += 1

        if count > 0:
            total_loss = total_loss / count * self.config.intermediate_loss_weight

        return total_loss

    def compute_attention_loss(
        self,
        student_attn: Dict[str, torch.Tensor],
        teacher_attn: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute attention map distillation loss.

        Args:
            student_attn: Student attention maps.
            teacher_attn: Teacher attention maps.

        Returns:
            Attention distillation loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0

        for name in teacher_attn:
            if name in student_attn:
                s_attn = student_attn[name]
                t_attn = teacher_attn[name]

                s_aligned = _align_attention_maps(s_attn, t_attn)
                t_avg = t_attn.mean(dim=1)

                if s_aligned.shape == t_avg.shape:
                    loss = F.mse_loss(s_aligned.float(), t_avg.float())
                    total_loss = total_loss + loss
                    count += 1

        if count > 0:
            total_loss = total_loss / count * self.config.intermediate_loss_weight

        return total_loss

    def compute_combined_loss(
        self,
        losses: Dict[str, torch.Tensor],
        alpha: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute weighted combination of all distillation losses.

        Args:
            losses: Dictionary of loss_name -> loss_tensor.
            alpha: Weight for distillation vs hard loss.

        Returns:
            Combined loss.
        """
        if alpha is None:
            alpha = self.alpha

        kl_loss = losses.get("kl_loss", torch.tensor(0.0, device=self.device))
        hard_loss = losses.get("hard_loss", torch.tensor(0.0, device=self.device))
        feature_loss = losses.get("feature_loss", torch.tensor(0.0, device=self.device))
        attention_loss = losses.get("attention_loss", torch.tensor(0.0, device=self.device))

        combined = alpha * kl_loss + (1.0 - alpha) * hard_loss

        if self.config.loss_type in ("feature", "combined"):
            combined = combined + self.config.intermediate_loss_weight * feature_loss

        if self.config.loss_type in ("attention", "combined"):
            combined = combined + self.config.intermediate_loss_weight * attention_loss

        return combined

    def train_epoch(
        self,
        student: nn.Module,
        teacher: nn.Module,
        dataloader: DataLoader,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train student for one epoch.

        Args:
            student: Student model.
            teacher: Teacher model.
            dataloader: Training dataloader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        student.train()
        teacher.eval()

        total_loss = 0.0
        total_kl_loss = 0.0
        total_hard_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(dataloader):
            if not isinstance(batch, dict):
                if isinstance(batch, (list, tuple)):
                    batch = {"input_ids": batch[0], "labels": batch[-1] if len(batch) > 1 else None}
                else:
                    continue

            self.optimizer.zero_grad()

            outputs = self.forward(student, teacher, batch)
            student_logits = outputs["student_logits"]
            teacher_logits = outputs["teacher_logits"]
            labels = outputs["labels"]

            kl_loss = self.compute_kl_loss(student_logits, teacher_logits)
            hard_loss = self.compute_hard_loss(student_logits, labels)

            losses = {
                "kl_loss": kl_loss,
                "hard_loss": hard_loss,
                "feature_loss": torch.tensor(0.0, device=self.device),
                "attention_loss": torch.tensor(0.0, device=self.device),
            }

            loss = self.compute_combined_loss(losses)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss at step %d, skipping", step)
                continue

            loss.backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), self.config.max_grad_norm
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_hard_loss += hard_loss.item()
            num_batches += 1

        metrics = {
            "epoch": epoch,
            "loss": total_loss / max(1, num_batches),
            "kl_loss": total_kl_loss / max(1, num_batches),
            "hard_loss": total_hard_loss / max(1, num_batches),
            "num_batches": num_batches,
        }

        self._history.append(metrics)
        return metrics

    def evaluate(
        self,
        student: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate student model.

        Args:
            student: Student model.
            dataloader: Evaluation dataloader.

        Returns:
            Dictionary of evaluation metrics.
        """
        student.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if not isinstance(batch, dict):
                    continue

                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                outputs = student(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits.float(), labels)
                    total_loss += loss.item()

                    predictions = logits.argmax(dim=-1)
                    total_correct += (predictions == labels).sum().item()
                    total_samples += labels.numel()

                num_batches += 1

        metrics = {
            "eval_loss": total_loss / max(1, num_batches),
        }

        if total_samples > 0:
            metrics["accuracy"] = total_correct / total_samples

        return metrics

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run full knowledge distillation training.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting knowledge distillation training")

        student = student.to(self.device)
        teacher = teacher.to(self.device)
        teacher.eval()

        self.optimizer = self._create_optimizer(student)
        self.scheduler = self._create_scheduler(self.optimizer, len(train_dataloader))

        best_model_state = copy.deepcopy(student.state_dict())
        self._best_eval_loss = float("inf")
        self._patience_counter = 0

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            train_metrics = self.train_epoch(student, teacher, train_dataloader, epoch)

            epoch_time = time.time() - epoch_start
            train_metrics["epoch_time"] = epoch_time

            logger.info(
                "Epoch %d/%d: loss=%.4f, kl=%.4f, hard=%.4f, time=%.1fs",
                epoch + 1, self.config.num_epochs,
                train_metrics["loss"], train_metrics["kl_loss"],
                train_metrics["hard_loss"], epoch_time,
            )

            if eval_dataloader is not None and (epoch + 1) % max(1, self.config.eval_steps // max(1, len(train_dataloader))) == 0:
                eval_metrics = self.evaluate(student, eval_dataloader)
                logger.info("Eval: loss=%.4f, accuracy=%.4f",
                           eval_metrics.get("eval_loss", 0),
                           eval_metrics.get("accuracy", 0))

                if eval_metrics.get("eval_loss", float("inf")) < self._best_eval_loss:
                    self._best_eval_loss = eval_metrics["eval_loss"]
                    best_model_state = copy.deepcopy(student.state_dict())
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1

                if (self.config.early_stopping_patience > 0 and
                    self._patience_counter >= self.config.early_stopping_patience):
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        student.load_state_dict(best_model_state)
        logger.info("Distillation complete. Best eval loss: %.4f", self._best_eval_loss)
        return student

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create optimizer for student model.

        Args:
            model: Student model.

        Returns:
            Optimizer instance.
        """
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        params_grouped = [
            {
                "params": [p for n, p in model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params_grouped, lr=self.config.learning_rate)
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(params_grouped, lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
                momentum=0.9,
            )
        elif self.config.optimizer == "rmsprop":
            return torch.optim.RMSprop(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
            )
        else:
            return torch.optim.AdamW(params_grouped, lr=self.config.learning_rate)

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.

        Args:
            optimizer: Optimizer.
            num_training_steps: Total training steps.

        Returns:
            Learning rate scheduler.
        """
        warmup_steps = self.config.warmup_steps
        total_steps = num_training_steps * self.config.num_epochs

        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        elif self.config.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=total_steps
            )
        elif self.config.lr_scheduler == "constant":
            return None

        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)


# =============================================================================
# FeatureDistiller
# =============================================================================

class FeatureDistiller(BaseDistiller):
    """Distills intermediate representations (features) from teacher to student.

    Matches corresponding layers between teacher and student, handles dimension
    mismatches with projections, and minimizes MSE or cosine similarity loss
    on intermediate feature maps.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize feature distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)
        self._layer_mapping: Dict[str, str] = {}
        self._projections: Dict[str, nn.Module] = {}

    def _match_layers(
        self,
        student: nn.Module,
        teacher: nn.Module,
    ) -> Dict[str, str]:
        """Find corresponding layers between student and teacher.

        Maps student layers to teacher layers based on naming conventions
        or manual configuration.

        Args:
            student: Student model.
            teacher: Teacher model.

        Returns:
            Dictionary mapping student layer names to teacher layer names.
        """
        if self.config.teacher_layer_mapping is not None:
            manual_mapping = {}
            for s_idx, t_idx in self.config.teacher_layer_mapping.items():
                manual_mapping[str(s_idx)] = str(t_idx)
            if manual_mapping:
                return manual_mapping

        student_layers = {name: module for name, module in student.named_modules()
                        if isinstance(module, (nn.Linear, nn.LayerNorm))}
        teacher_layers = {name: module for name, module in teacher.named_modules()
                        if isinstance(module, (nn.Linear, nn.LayerNorm))}

        mapping = {}
        student_names = sorted(student_layers.keys())
        teacher_names = sorted(teacher_layers.keys())

        if self.config.teacher_layer_mapping is not None:
            for s_name in student_names:
                for t_name in teacher_names:
                    s_parts = s_name.replace(".", "_").split("_")
                    t_parts = t_name.replace(".", "_").split("_")
                    if any(p in t_parts for p in s_parts if len(p) > 2):
                        mapping[s_name] = t_name
                        break
        else:
            min_layers = min(len(student_names), len(teacher_names))
            if len(student_names) > 0 and len(teacher_names) > 0:
                step = max(1, len(teacher_names) // len(student_names))
                for i, s_name in enumerate(student_names):
                    t_idx = min(i * step, len(teacher_names) - 1)
                    mapping[s_name] = teacher_names[t_idx]

        self._layer_mapping = mapping
        return mapping

    def _project_features(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        layer_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle dimension mismatch between student and teacher features.

        Args:
            student_feat: Student feature tensor.
            teacher_feat: Teacher feature tensor.
            layer_name: Layer name for caching projection.

        Returns:
            Tuple of (student_projected, teacher_target).
        """
        if student_feat.shape == teacher_feat.shape:
            return student_feat.float(), teacher_feat.float()

        s_dim = student_feat.shape[-1]
        t_dim = teacher_feat.shape[-1]

        if s_dim == t_dim:
            min_shape = tuple(min(a, b) for a, b in zip(student_feat.shape, teacher_feat.shape))
            slices = tuple(slice(0, s) for s in min_shape)
            return student_feat[slices].float(), teacher_feat[slices].float()

        if layer_name not in self._projections:
            self._projections[layer_name] = nn.Linear(s_dim, t_dim, bias=False).to(self.device)
            nn.init.orthogonal_(self._projections[layer_name].weight)

        projection = self._projections[layer_name]
        projected = projection(student_feat.float().flatten(1))
        projected = projected.view(*student_feat.shape[:-1], t_dim)

        min_shape = tuple(min(a, b) for a, b in zip(projected.shape, teacher_feat.shape))
        slices = tuple(slice(0, s) for s in min_shape)

        return projected[slices].float(), teacher_feat[slices].float()

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run feature-based distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting feature distillation")
        student = student.to(self.device)
        teacher = teacher.to(self.device)

        self._match_layers(student, teacher)
        self._projections = {k: v.to(self.device) for k, v in self._projections.items()}

        self.base_trainer.optimizer = self.base_trainer._create_optimizer(student)
        self.base_trainer.scheduler = self.base_trainer._create_scheduler(
            self.base_trainer.optimizer, len(train_dataloader)
        )

        for param in teacher.parameters():
            param.requires_grad = False

        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            total_feature_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                if not isinstance(batch, dict):
                    continue

                self.base_trainer.optimizer.zero_grad()

                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                student_feats = _get_named_intermediate_outputs(
                    student, inputs, list(self._layer_mapping.keys()), self.device
                )

                with torch.no_grad():
                    teacher_feats = _get_named_intermediate_outputs(
                        teacher, inputs,
                        list(self._layer_mapping.values()), self.device
                    )

                feature_loss = torch.tensor(0.0, device=self.device)
                count = 0

                for s_name, t_name in self._layer_mapping.items():
                    if s_name in student_feats and t_name in teacher_feats:
                        s_feat, t_feat = self._project_features(
                            student_feats[s_name], teacher_feats[t_name], s_name
                        )
                        if self.config.use_cosine_loss:
                            loss = _compute_cosine_similarity_loss(s_feat, t_feat)
                        else:
                            loss = _compute_mse_loss(s_feat, t_feat)
                        feature_loss = feature_loss + loss
                        count += 1

                if count > 0:
                    feature_loss = feature_loss / count

                student_out = student(**inputs)
                student_logits = student_out.logits if hasattr(student_out, "logits") else student_out

                hard_loss = torch.tensor(0.0, device=self.device)
                if labels is not None:
                    hard_loss = F.cross_entropy(student_logits.float(), labels)

                total = (self.alpha * feature_loss + (1 - self.alpha) * hard_loss)

                if not torch.isnan(total) and not torch.isinf(total):
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), self.config.max_grad_norm
                    )
                    self.base_trainer.optimizer.step()

                total_loss += total.item()
                total_feature_loss += feature_loss.item()
                num_batches += 1

            if self.base_trainer.scheduler:
                self.base_trainer.scheduler.step()

            avg_loss = total_loss / max(1, num_batches)
            avg_feat = total_feature_loss / max(1, num_batches)
            logger.info("FeatureDistill Epoch %d/%d: loss=%.4f, feat=%.4f",
                       epoch + 1, self.config.num_epochs, avg_loss, avg_feat)

        return student


# =============================================================================
# AttentionDistiller
# =============================================================================

class AttentionDistiller(BaseDistiller):
    """Distills attention patterns from teacher to student.

    Transfers self-attention distributions to help the student learn
    similar attention patterns as the teacher.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize attention distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)
        self._attention_hooks: List[Any] = []

    def _align_attention(
        self,
        student_attn: torch.Tensor,
        teacher_attn: torch.Tensor,
    ) -> torch.Tensor:
        """Handle different head counts between student and teacher.

        Args:
            student_attn: Student attention maps.
            teacher_attn: Teacher attention maps.

        Returns:
            Aligned student attention maps.
        """
        return _align_attention_maps(student_attn, teacher_attn)

    def _attention_loss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention similarity loss.

        Args:
            student: Student attention.
            teacher: Teacher attention.

        Returns:
            Attention distillation loss.
        """
        s_aligned = self._align_attention(student, teacher)
        t_avg = teacher.mean(dim=1)

        if s_aligned.shape != t_avg.shape:
            min_seq = min(s_aligned.shape[-1], t_avg.shape[-1])
            s_aligned = s_aligned[:, :min_seq, :min_seq]
            t_avg = t_avg[:, :min_seq, :min_seq]

        loss = F.mse_loss(s_aligned.float(), t_avg.float())
        return loss

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run attention-based distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting attention distillation")
        student = student.to(self.device)
        teacher = teacher.to(self.device)

        self.base_trainer.optimizer = self.base_trainer._create_optimizer(student)
        self.base_trainer.scheduler = self.base_trainer._create_scheduler(
            self.base_trainer.optimizer, len(train_dataloader)
        )

        for param in teacher.parameters():
            param.requires_grad = False

        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                if not isinstance(batch, dict):
                    continue

                self.base_trainer.optimizer.zero_grad()

                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                s_attns = {}
                t_attns = {}
                s_hooks = []
                t_hooks = []

                attn_count = [0]

                def make_attn_hook(name, store_dict):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple) and len(output) > 1:
                            attn = output[1]
                            if isinstance(attn, torch.Tensor):
                                store_dict[f"{name}_{attn_count[0]}"] = attn.detach()
                                attn_count[0] += 1
                        elif isinstance(output, dict) and "attentions" in output:
                            for i, a in enumerate(output["attentions"]):
                                store_dict[f"{name}_{i}"] = a.detach()
                    return hook_fn

                for n, m in student.named_modules():
                    if "attn" in n.lower() or "attention" in n.lower():
                        s_hooks.append(m.register_forward_hook(make_attn_hook(n, s_attns)))

                for n, m in teacher.named_modules():
                    if "attn" in n.lower() or "attention" in n.lower():
                        t_hooks.append(m.register_forward_hook(make_attn_hook(n, t_attns)))

                try:
                    with torch.no_grad():
                        teacher(**{k: v for k, v in inputs.items() if k != "labels"})

                    student_out = student(**inputs)
                    student_logits = student_out.logits if hasattr(student_out, "logits") else student_out
                finally:
                    for h in s_hooks:
                        h.remove()
                    for h in t_hooks:
                        h.remove()

                attn_loss = torch.tensor(0.0, device=self.device)
                count = 0

                for s_name, s_attn in s_attns.items():
                    for t_name, t_attn in t_attns.items():
                        if s_attn.shape == t_attn.shape:
                            attn_loss = attn_loss + self._attention_loss(s_attn, t_attn)
                            count += 1

                if count > 0:
                    attn_loss = attn_loss / count

                hard_loss = torch.tensor(0.0, device=self.device)
                if labels is not None:
                    hard_loss = F.cross_entropy(student_logits.float(), labels)

                total = self.alpha * attn_loss + (1 - self.alpha) * hard_loss

                if not torch.isnan(total) and not torch.isinf(total):
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        student.parameters(), self.config.max_grad_norm
                    )
                    self.base_trainer.optimizer.step()

                total_loss += total.item()
                num_batches += 1

            if self.base_trainer.scheduler:
                self.base_trainer.scheduler.step()

            logger.info("AttentionDistill Epoch %d/%d: loss=%.4f",
                       epoch + 1, self.config.num_epochs, total_loss / max(1, num_batches))

        return student


# =============================================================================
# ResponseDistiller
# =============================================================================

class ResponseDistiller(BaseDistiller):
    """Distills final output logits (response) from teacher to student.

    Pure response-based distillation using softened logits and KL divergence.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize response distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run response-based distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        return self.base_trainer.distill(student, teacher, train_dataloader, eval_dataloader)


# =============================================================================
# MultiTeacherDistiller
# =============================================================================

class MultiTeacherDistiller(BaseDistuner):
    """Distill from multiple teacher models.

    Combines knowledge from multiple teacher models by weighting their
    contributions based on expertise or accuracy.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize multi-teacher distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self._teacher_weights: Optional[List[float]] = config.teacher_weights
        self.base_trainer = DistillationTrainer(config)

    def compute_teacher_weights(
        self,
        teachers: List[nn.Module],
        dataloader: DataLoader,
    ) -> List[float]:
        """Weight teachers by their expertise/performance.

        Args:
            teachers: List of teacher models.
            dataloader: Evaluation dataloader.

        Returns:
            List of teacher weights summing to 1.0.
        """
        if self._teacher_weights is not None:
            return self._teacher_weights

        accuracies = []
        for teacher in teachers:
            teacher = teacher.to(self.device)
            teacher.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in dataloader:
                    if not isinstance(batch, dict):
                        continue
                    inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    labels = inputs.pop("labels", None)
                    if labels is None:
                        continue

                    outputs = teacher(**inputs)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    preds = logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.numel()

            acc = correct / max(1, total)
            accuracies.append(acc)

        total_acc = sum(accuracies)
        if total_acc > 0:
            weights = [a / total_acc for a in accuracies]
        else:
            weights = [1.0 / len(teachers)] * len(teachers)

        self._teacher_weights = weights
        return weights

    def weighted_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        weights: List[float],
        temperature: float,
    ) -> torch.Tensor:
        """Compute weighted KL divergence loss across multiple teachers.

        Args:
            student_logits: Student output logits.
            teacher_logits_list: List of teacher logits.
            weights: Teacher weights.
            temperature: Softmax temperature.

        Returns:
            Weighted KL loss.
        """
        total_loss = torch.tensor(0.0, device=student_logits.device)

        for i, teacher_logits in enumerate(teacher_logits_list):
            w = weights[i] if i < len(weights) else 1.0 / len(teacher_logits_list)
            kl = _compute_kl_divergence(student_logits, teacher_logits, temperature)
            total_loss = total_loss + w * kl

        return total_loss

    def distill(
        self,
        student: nn.Module,
        teachers: Union[nn.Module, List[nn.Module]],
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run multi-teacher distillation.

        Args:
            student: Student model.
            teachers: Teacher model or list of teacher models.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        if isinstance(teachers, nn.Module):
            teachers = [teachers]

        logger.info("Starting multi-teacher distillation with %d teachers", len(teachers))

        student = student.to(self.device)
        for t in teachers:
            t.to(self.device)
            t.eval()
            for p in t.parameters():
                p.requires_grad = False

        weights = self.compute_teacher_weights(teachers, train_dataloader)

        optimizer = self.base_trainer._create_optimizer(student)
        scheduler = self.base_trainer._create_scheduler(optimizer, len(train_dataloader))

        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                if not isinstance(batch, dict):
                    continue

                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                teacher_logits_list = []
                for teacher in teachers:
                    with torch.no_grad():
                        t_out = teacher(**{k: v for k, v in inputs.items() if k != "labels"})
                        t_logits = t_out.logits if hasattr(t_out, "logits") else t_out
                        teacher_logits_list.append(t_logits)

                s_out = student(**inputs)
                s_logits = s_out.logits if hasattr(s_out, "logits") else s_out

                distill_loss = self.weighted_kl_loss(
                    s_logits, teacher_logits_list, weights, self.temperature
                )

                hard_loss = torch.tensor(0.0, device=self.device)
                if labels is not None:
                    hard_loss = F.cross_entropy(s_logits.float(), labels)

                loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            if scheduler:
                scheduler.step()

            logger.info("MultiTeacher Epoch %d/%d: loss=%.4f",
                       epoch + 1, self.config.num_epochs, total_loss / max(1, num_batches))

        return student


class BaseDistuner(BaseDistiller):
    """Alias to fix naming above."""
    pass


# =============================================================================
# ProgressiveDistiller
# =============================================================================

class ProgressiveDistiller(BaseDistiller):
    """Progressively increase distillation difficulty over time.

    Starts with easy examples and gradually increases the alpha weight
    for distillation loss, giving the student model time to adapt.
    """

    def __init__(self, config: DistillationConfig):
        """Initialize progressive distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)
        self.num_stages = config.progressive_stages
        self.start_alpha = config.progressive_start_alpha
        self.end_alpha = config.progressive_end_alpha

    def _get_alpha_at_stage(self, stage: int, total_stages: int) -> float:
        """Compute alpha for the current stage.

        Args:
            stage: Current stage.
            total_stages: Total number of stages.

        Returns:
            Alpha value for this stage.
        """
        progress = stage / max(1, total_stages - 1)
        return self.start_alpha + (self.end_alpha - self.start_alpha) * progress

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run progressive distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting progressive distillation with %d stages", self.num_stages)

        student = student.to(self.device)
        teacher = teacher.to(self.device)
        teacher.eval()

        epochs_per_stage = max(1, self.config.num_epochs // self.num_stages)
        total_epochs = 0

        for stage in range(self.num_stages):
            current_alpha = self._get_alpha_at_stage(stage, self.num_stages)
            logger.info("Stage %d/%d: alpha=%.3f", stage + 1, self.num_stages, current_alpha)

            original_alpha = self.base_trainer.alpha
            self.base_trainer.alpha = current_alpha

            for epoch in range(epochs_per_stage):
                if total_epochs >= self.config.num_epochs:
                    break

                metrics = self.base_trainer.train_epoch(
                    student, teacher, train_dataloader, total_epochs
                )

                if self.base_trainer.optimizer is None:
                    self.base_trainer.optimizer = self.base_trainer._create_optimizer(student)
                if self.base_trainer.scheduler is None:
                    self.base_trainer.scheduler = self.base_trainer._create_scheduler(
                        self.base_trainer.optimizer, len(train_dataloader)
                    )

                total_epochs += 1

                if (epoch + 1) % 5 == 0:
                    logger.info("Stage %d, Epoch %d: loss=%.4f", stage + 1, total_epochs, metrics["loss"])

            self.base_trainer.alpha = original_alpha

        return student


# =============================================================================
# DistillationDataAugmentor
# =============================================================================

class DistillationDataAugmentor:
    """Augment training data for better knowledge distillation.

    Applies techniques like token masking, token substitution, and
    span corruption to create harder training examples.
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        substitute_prob: float = 0.1,
        delete_prob: float = 0.05,
        vocab_size: int = 30522,
        special_token_id: int = 103,
    ):
        """Initialize data augmentor.

        Args:
            mask_prob: Probability of masking a token.
            substitute_prob: Probability of substituting a token.
            delete_prob: Probability of deleting a token.
            vocab_size: Vocabulary size.
            special_token_id: Mask token ID.
        """
        self.mask_prob = mask_prob
        self.substitute_prob = substitute_prob
        self.delete_prob = delete_prob
        self.vocab_size = vocab_size
        self.mask_token_id = special_token_id

    def random_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Randomly mask tokens.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Masked input IDs.
        """
        masked = input_ids.clone()
        mask_positions = torch.rand_like(input_ids.float()) < self.mask_prob

        if attention_mask is not None:
            mask_positions = mask_positions & (attention_mask == 1)

        masked[mask_positions] = self.mask_token_id
        return masked

    def random_substitute(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Randomly substitute tokens with random vocabulary tokens.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Augmented input IDs.
        """
        augmented = input_ids.clone()
        sub_positions = torch.rand_like(input_ids.float()) < self.substitute_prob

        if attention_mask is not None:
            sub_positions = sub_positions & (attention_mask == 1)

        random_tokens = torch.randint(0, self.vocab_size, input_ids.shape, device=input_ids.device)
        augmented[sub_positions] = random_tokens[sub_positions]
        return augmented

    def random_delete(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly delete tokens.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Tuple of (augmented_ids, augmented_mask).
        """
        keep_mask = torch.rand(input_ids.shape[:1], device=input_ids.device).unsqueeze(1)
        keep_mask = keep_mask.expand_as(input_ids) > self.delete_prob

        if attention_mask is not None:
            keep_mask = keep_mask & (attention_mask == 1)

        augmented_ids = input_ids[keep_mask].unsqueeze(0)
        augmented_mask = torch.ones_like(augmented_ids)

        return augmented_ids, augmented_mask

    def span_corruption(
        self,
        input_ids: torch.Tensor,
        span_length: int = 3,
        num_spans: int = 5,
    ) -> torch.Tensor:
        """Corrupt spans of tokens.

        Args:
            input_ids: Input token IDs.
            span_length: Length of each corrupted span.
            num_spans: Number of spans to corrupt.

        Returns:
            Corrupted input IDs.
        """
        corrupted = input_ids.clone()
        seq_len = input_ids.shape[1]

        for _ in range(num_spans):
            start = torch.randint(0, max(1, seq_len - span_length), (1,)).item()
            end = min(start + span_length, seq_len)
            corrupted[:, start:end] = self.mask_token_id

        return corrupted

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to a batch.

        Args:
            batch: Input batch.
            methods: Augmentation methods to apply.

        Returns:
            Augmented batch.
        """
        if methods is None:
            methods = ["random_mask"]

        augmented = {}
        for k, v in batch.items():
            if k == "input_ids" and isinstance(v, torch.Tensor):
                result = v.clone()
                attention_mask = batch.get("attention_mask")
                for method in methods:
                    if method == "random_mask":
                        result = self.random_mask(result, attention_mask)
                    elif method == "random_substitute":
                        result = self.random_substitute(result, attention_mask)
                    elif method == "span_corruption":
                        result = self.span_corruption(result)
                augmented[k] = result
            elif k == "labels":
                augmented[k] = v
            else:
                augmented[k] = v

        return augmented


# =============================================================================
# TinyBERTDistiller
# =============================================================================

class TinyBERTDistiller(BaseDistiller):
    """TinyBERT-style multi-level distillation.

    Distills knowledge at three levels:
    1. Attention-level: Transfer attention maps
    2. Hidden-level: Transfer hidden states
    3. Embedding-level: Transfer word embeddings

    Reference: "TinyBERT: Distilling BERT for Natural Language Understanding" (2020)
    """

    def __init__(self, config: DistillationConfig):
        """Initialize TinyBERT distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)
        self.fit_denses: Dict[str, nn.Linear] = {}

    def _create_fit_dense(self, student_dim: int, teacher_dim: int, name: str) -> nn.Module:
        """Create a fit dense layer for dimension matching.

        Args:
            student_dim: Student dimension.
            teacher_dim: Teacher dimension.
            name: Layer name for caching.

        Returns:
            Linear projection layer.
        """
        if name in self.fit_denses:
            return self.fit_denses[name]

        dense = nn.Linear(student_dim, teacher_dim, bias=False).to(self.device)
        nn.init.xavier_uniform_(dense.weight)
        self.fit_denses[name] = dense
        return dense

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run TinyBERT-style multi-level distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting TinyBERT multi-level distillation")

        student = student.to(self.device)
        teacher = teacher.to(self.device)
        teacher.eval()

        all_params = list(student.parameters())
        for name, dense in self.fit_denses.items():
            all_params.extend(dense.parameters())

        optimizer = torch.optim.AdamW(all_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs * len(train_dataloader)
        )

        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                if not isinstance(batch, dict):
                    continue

                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                with torch.no_grad():
                    t_hidden = _get_named_intermediate_outputs(teacher, inputs, device=self.device)
                    t_out = teacher(**{k: v for k, v in inputs.items() if k != "labels"})
                    t_logits = t_out.logits if hasattr(t_out, "logits") else t_out

                s_hidden = _get_named_intermediate_outputs(student, inputs, device=self.device)
                s_out = student(**inputs)
                s_logits = s_out.logits if hasattr(s_out, "logits") else s_out

                attn_loss = torch.tensor(0.0, device=self.device)
                hidden_loss = torch.tensor(0.0, device=self.device)
                embed_loss = torch.tensor(0.0, device=self.device)

                s_layers = sorted(s_hidden.keys())
                t_layers = sorted(t_hidden.keys())

                n_layers = min(len(s_layers), len(t_layers), self.config.num_layers_to_distill)
                if n_layers > 0:
                    step = max(1, len(t_layers) // max(1, len(s_layers)))
                    for i in range(n_layers):
                        s_idx = min(i, len(s_layers) - 1)
                        t_idx = min(i * step, len(t_layers) - 1)

                        s_feat = s_hidden[s_layers[s_idx]]
                        t_feat = t_hidden[t_layers[t_idx]]

                        if s_feat.shape[-1] != t_feat.shape[-1]:
                            fit_dense = self._create_fit_dense(
                                s_feat.shape[-1], t_feat.shape[-1], f"hidden_{i}"
                            )
                            s_feat_proj = fit_dense(s_feat.float().flatten(1))
                            t_flat = t_feat.float().flatten(1)
                            min_dim = min(s_feat_proj.shape[-1], t_flat.shape[-1])
                            s_feat_proj = s_feat_proj[:, :min_dim]
                            t_flat = t_flat[:, :min_dim]
                            hidden_loss = hidden_loss + F.mse_loss(s_feat_proj, t_flat)
                        else:
                            min_shape = tuple(min(a, b) for a, b in zip(s_feat.shape, t_feat.shape))
                            slices = tuple(slice(0, s) for s in min_shape)
                            hidden_loss = hidden_loss + F.mse_loss(
                                s_feat[slices].float(), t_feat[slices].float()
                            )

                    hidden_loss = hidden_loss / n_layers

                if labels is not None:
                    hard_loss = F.cross_entropy(s_logits.float(), labels, label_smoothing=self.config.label_smoothing)
                else:
                    hard_loss = torch.tensor(0.0, device=self.device)

                distill_loss = _compute_kl_divergence(s_logits, t_logits, self.temperature)

                total = (self.alpha * distill_loss +
                        (1 - self.alpha) * hard_loss +
                        self.config.intermediate_loss_weight * hidden_loss)

                if not torch.isnan(total) and not torch.isinf(total):
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                total_loss += total.item()
                num_batches += 1

            if scheduler:
                scheduler.step()

            logger.info("TinyBERT Epoch %d/%d: loss=%.4f",
                       epoch + 1, self.config.num_epochs, total_loss / max(1, num_batches))

        return student


# =============================================================================
# MiniLMStyleDistiller
# =============================================================================

class MiniLMStyleDistiller(BaseDistiller):
    """MiniLM-style self-attention transfer.

    Distills self-attention distributions from teacher to student using
    a head-level matching strategy, as described in "MiniLM: Deep
    Self-Attention Distillation for Language Modeling" (2020).
    """

    def __init__(self, config: DistillationConfig):
        """Initialize MiniLM-style distiller.

        Args:
            config: Distillation configuration.
        """
        super().__init__(config)
        self.base_trainer = DistillationTrainer(config)
        self._head_mapping: Dict[str, List[Tuple[int, int]]] = {}

    def _build_head_mapping(
        self,
        student_heads: int,
        teacher_heads: int,
    ) -> List[Tuple[int, int]]:
        """Build head mapping from student to teacher heads.

        Args:
            student_heads: Number of student attention heads.
            teacher_heads: Number of teacher attention heads.

        Returns:
            List of (student_head, teacher_head) tuples.
        """
        if student_heads == teacher_heads:
            return [(i, i) for i in range(student_heads)]

        mapping = []
        for s_h in range(student_heads):
            t_h = s_h * teacher_heads // student_heads
            t_h = min(t_h, teacher_heads - 1)
            mapping.append((s_h, t_h))

        return mapping

    def distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Run MiniLM-style self-attention distillation.

        Args:
            student: Student model.
            teacher: Teacher model.
            train_dataloader: Training dataloader.
            eval_dataloader: Optional evaluation dataloader.

        Returns:
            Trained student model.
        """
        logger.info("Starting MiniLM-style self-attention distillation")

        student = student.to(self.device)
        teacher = teacher.to(self.device)
        teacher.eval()

        optimizer = self.base_trainer._create_optimizer(student)
        scheduler = self.base_trainer._create_scheduler(optimizer, len(train_dataloader))

        for epoch in range(self.config.num_epochs):
            student.train()
            total_loss = 0.0
            total_attn_loss = 0.0
            num_batches = 0

            for batch in train_dataloader:
                if not isinstance(batch, dict):
                    continue

                optimizer.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                labels = inputs.pop("labels", None)

                s_attns = {}
                t_attns = {}
                s_hooks = []
                t_hooks = []
                hook_count = [0]

                def attn_hook(name, store_dict):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple) and len(output) > 1:
                            a = output[1]
                            if isinstance(a, torch.Tensor):
                                store_dict[f"{name}_{hook_count[0]}"] = a.detach()
                                hook_count[0] += 1
                    return hook_fn

                for n, m in student.named_modules():
                    if "attn" in n.lower() or "attention" in n.lower():
                        s_hooks.append(m.register_forward_hook(attn_hook(n, s_attns)))

                for n, m in teacher.named_modules():
                    if "attn" in n.lower() or "attention" in n.lower():
                        t_hooks.append(m.register_forward_hook(attn_hook(n, t_attns)))

                try:
                    with torch.no_grad():
                        teacher(**{k: v for k, v in inputs.items() if k != "labels"})
                    s_out = student(**inputs)
                    s_logits = s_out.logits if hasattr(s_out, "logits") else s_out
                finally:
                    for h in s_hooks:
                        h.remove()
                    for h in t_hooks:
                        h.remove()

                attn_loss = torch.tensor(0.0, device=self.device)
                attn_count = 0

                s_names = sorted(s_attns.keys())
                t_names = sorted(t_attns.keys())
                n_pairs = min(len(s_names), len(t_names))

                for i in range(n_pairs):
                    s_a = s_attns[s_names[i]]
                    t_a = t_attns[t_names[i]]

                    if s_a.dim() == 4 and t_a.dim() == 4:
                        s_h = s_a.shape[1]
                        t_h = t_a.shape[1]
                        mapping = self._build_head_mapping(s_h, t_h)

                        for s_idx, t_idx in mapping:
                            s_head = s_a[:, s_idx, :, :]
                            t_head = t_a[:, t_idx, :, :]
                            min_seq = min(s_head.shape[-1], t_head.shape[-1])
                            s_head = s_head[:, :min_seq, :min_seq]
                            t_head = t_head[:, :min_seq, :min_seq]
                            attn_loss = attn_loss + F.mse_loss(s_head.float(), t_head.float())
                            attn_count += 1

                if attn_count > 0:
                    attn_loss = attn_loss / attn_count

                hard_loss = torch.tensor(0.0, device=self.device)
                if labels is not None:
                    hard_loss = F.cross_entropy(s_logits.float(), labels)

                distill_loss = torch.tensor(0.0, device=self.device)

                t_out_final = teacher(**{k: v for k, v in inputs.items() if k != "labels"})
                t_logits_final = t_out_final.logits if hasattr(t_out_final, "logits") else t_out_final
                with torch.no_grad():
                    distill_loss = _compute_kl_divergence(s_logits, t_logits_final, self.temperature)

                total = (self.alpha * (distill_loss + attn_loss) +
                        (1 - self.alpha) * hard_loss)

                if not torch.isnan(total) and not torch.isinf(total):
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), self.config.max_grad_norm)
                    optimizer.step()

                total_loss += total.item()
                total_attn_loss += attn_loss.item()
                num_batches += 1

            if scheduler:
                scheduler.step()

            avg_loss = total_loss / max(1, num_batches)
            avg_attn = total_attn_loss / max(1, num_batches)
            logger.info("MiniLM Epoch %d/%d: loss=%.4f, attn=%.4f",
                       epoch + 1, self.config.num_epochs, avg_loss, avg_attn)

        return student
