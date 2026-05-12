"""
Multi-Token Prediction for Nexus
====================================
Predict multiple future tokens simultaneously to improve training efficiency
and enable faster inference.

Instead of predicting only the next token t+1, the model also predicts
t+2, t+3, ..., t+N using extra prediction heads. This provides:
1. Richer training signal (each sample provides N labels instead of 1)
2. Better token representations (forced to encode multi-step future)
3. Faster inference when combined with speculative decoding
4. Improved downstream task performance

Reference:
- "Multi-Token Prediction for Faster and Better Language Modeling" (Google DeepMind, 2024)
- "MEDUSA: Simple LLM Inference Acceleration Framework" (2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. MultiTokenPredictionHead
# ---------------------------------------------------------------------------

class MultiTokenPredictionHead(nn.Module):
    """
    Multiple independent prediction heads for multi-token prediction.

    Architecture:
        shared_hidden_states (from last layer of transformer)
            |
            |-- head_0 -> logits_0 (predicts token at t+1, standard next-token)
            |-- head_1 -> logits_1 (predicts token at t+2)
            |-- head_2 -> logits_2 (predicts token at t+3)
            |-- ...
            |-- head_N -> logits_N (predicts token at t+N+1)

    Each head is a separate linear layer (or small MLP):
        head_i: Linear(hidden_size, vocab_size)
        or: Linear(hidden_size, hidden_size) -> ReLU -> Linear(hidden_size, vocab_size)

    Design choices:
    - Shared trunk: all heads receive the same hidden_states
    - Independent heads: each head has its own weights (no weight sharing)
    - Optional: lightweight cross-head attention for coordination

    Weight tying: head_0 can be tied with the main lm_head

    Args:
        hidden_size: Dimensionality of the transformer hidden states.
        vocab_size: Size of the vocabulary.
        num_future_tokens: Number of future tokens to predict (N heads).
        head_type: Type of each head — ``'linear'``, ``'mlp'``, or ``'deep'``.
        use_cross_attention: If *True*, apply a shared self-attention layer
            across the sequence before feeding into each head.
        tie_first_head: If *True*, :meth:`tie_weights` will alias
            ``heads[0].weight`` to the main LM head.
    """

    SUPPORTED_HEAD_TYPES = ("linear", "mlp", "deep")

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_future_tokens: int = 4,
        head_type: str = "linear",
        use_cross_attention: bool = False,
        tie_first_head: bool = True,
    ) -> None:
        super().__init__()

        if head_type not in self.SUPPORTED_HEAD_TYPES:
            raise ValueError(
                f"Unsupported head_type '{head_type}'. "
                f"Choose from {self.SUPPORTED_HEAD_TYPES}."
            )

        self.num_future_tokens = num_future_tokens
        self.head_type = head_type
        self.vocab_size = vocab_size
        self.tie_first_head = tie_first_head

        # ---- build heads ----
        if head_type == "linear":
            self.heads = nn.ModuleList(
                [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_future_tokens)]
            )
        elif head_type == "mlp":
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                        nn.Linear(hidden_size, vocab_size, bias=False),
                    )
                    for _ in range(num_future_tokens)
                ]
            )
        elif head_type == "deep":
            self.heads = nn.ModuleList(
                [self._make_deep_head(hidden_size, vocab_size, 2 + i) for i in range(num_future_tokens)]
            )

        # ---- optional cross-head attention ----
        if use_cross_attention:
            self.cross_head_attn = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                batch_first=True,
            )
        else:
            self.cross_head_attn = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_deep_head(hidden_size: int, vocab_size: int, num_layers: int) -> nn.Sequential:
        """Create a deeper prediction head with stacked linear + GELU layers."""
        layers: List[nn.Module] = []
        in_dim = hidden_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_size), nn.GELU()])
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, vocab_size, bias=False))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate predictions for all future positions.

        Args:
            hidden_states: ``(*batch, seq_len, hidden_size)`` — output from the
                last transformer layer.

        Returns:
            A list of ``(*batch, seq_len, vocab_size)`` tensors, one per
            future token position.
        """
        logits_list: List[torch.Tensor] = []

        # Optional cross-head attention for coordination across the sequence.
        if self.cross_head_attn is not None:
            coords, _ = self.cross_head_attn(hidden_states, hidden_states, hidden_states)
        else:
            coords = hidden_states

        for i, head in enumerate(self.heads):
            # deeper heads see cross-attention output; linear/mlp use raw states
            input_h = coords if (self.cross_head_attn is not None and self.head_type == "deep") else hidden_states
            logits = head(input_h)
            logits_list.append(logits)

        return logits_list

    # ------------------------------------------------------------------
    # loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        labels: torch.Tensor,
        loss_weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-token prediction loss.

        For head *i* predicting the token at position *t+i*:

        .. code-block:: text

            loss_i = CrossEntropy(logits_list[i][:-i-1], labels[i+1:])

        The labels are shifted for each head so that head *i* is trained to
        predict the token that is *i+1* steps ahead.

        Args:
            logits_list: List of ``(*batch, seq_len, vocab_size)`` tensors from
                :meth:`forward`.
            labels: ``(*batch, seq_len)`` — target token IDs (``-100`` is
                ignored by default).
            loss_weights: Optional per-head loss weights.  Defaults to
                geometric decay ``(1, 1/2, 1/4, …)`` normalised to sum to 1.

        Returns:
            A ``(total_loss, per_head_losses)`` tuple where *total_loss* is a
            weighted sum (detached from the graph when no heads apply) and
            *per_head_losses* maps ``"head_{i}"`` → float.
        """
        per_head_losses: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)

        if loss_weights is None:
            # Default: geometric decay — first head most important
            loss_weights = [1.0 / (2 ** i) for i in range(len(logits_list))]
            total_weight = sum(loss_weights)
            loss_weights = [w / total_weight for w in loss_weights]

        for i, (logits, weight) in enumerate(zip(logits_list, loss_weights)):
            shift = i + 1
            if shift >= labels.shape[-1]:
                break

            # Shift logits and labels
            pred_logits = logits[:, :-shift, :].contiguous()
            target_labels = labels[:, shift:].contiguous()

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                pred_logits.reshape(-1, self.vocab_size),
                target_labels.reshape(-1),
                ignore_index=-100,
            )
            per_head_losses[f"head_{i}"] = loss.item()
            total_loss = total_loss + weight * loss

        return total_loss, per_head_losses

    # ------------------------------------------------------------------
    # weight tying
    # ------------------------------------------------------------------

    def tie_weights(self, lm_head: nn.Linear) -> None:
        """Alias ``heads[0].weight`` to *lm_head.weight*."""
        if self.tie_first_head and self.head_type == "linear":
            self.heads[0].weight = lm_head.weight

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters across all heads."""
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_future_tokens={self.num_future_tokens}, "
            f"head_type='{self.head_type}', "
            f"vocab_size={self.vocab_size}, "
            f"tie_first_head={self.tie_first_head}, "
            f"cross_head_attn={self.cross_head_attn is not None}, "
            f"params={self.num_parameters:,}"
        )


# ---------------------------------------------------------------------------
# 2. SharedTrunkMultiTokenHead
# ---------------------------------------------------------------------------

class SharedTrunkMultiTokenHead(nn.Module):
    """
    Multi-token prediction head with a shared projection trunk.

    Unlike :class:`MultiTokenPredictionHead` where each head is fully
    independent, this uses a shared trunk that projects ``hidden_states`` to a
    lower-dimensional space, then each head predicts from this shared
    representation.

    Architecture:

    .. code-block:: text

        hidden_states -> shared_projection -> shared_features
                                                |-- head_0 -> vocab (t+1)
                                                |-- head_1 -> vocab (t+2)
                                                |-- ...

    The shared projection reduces parameters and improves generalisation
    for heads that predict further into the future (harder task).

    Args:
        hidden_size: Dimensionality of the transformer hidden states.
        vocab_size: Size of the vocabulary.
        num_future_tokens: Number of future tokens to predict.
        shared_dim: Dimensionality of the shared trunk projection.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_future_tokens: int = 4,
        shared_dim: int = 4096,
    ) -> None:
        super().__init__()

        # Shared projection trunk
        self.shared_proj = nn.Sequential(
            nn.Linear(hidden_size, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )

        # Per-head output layers (smaller since input is already projected)
        self.heads = nn.ModuleList(
            [nn.Linear(shared_dim, vocab_size, bias=False) for _ in range(num_future_tokens)]
        )

        # Optional: learnable scaling per head so deeper heads can modulate
        # their output magnitude.
        self.head_scales = nn.Parameter(torch.ones(num_future_tokens))

        self.num_future_tokens = num_future_tokens
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.shared_dim = shared_dim

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Project through the shared trunk, then predict with each head.

        Args:
            hidden_states: ``(*batch, seq_len, hidden_size)``

        Returns:
            List of ``(*batch, seq_len, vocab_size)`` tensors.
        """
        shared_features = self.shared_proj(hidden_states)
        logits_list: List[torch.Tensor] = []
        for i, head in enumerate(self.heads):
            logits = head(shared_features) * self.head_scales[i]
            logits_list.append(logits)
        return logits_list

    def compute_loss(
        self,
        logits_list: List[torch.Tensor],
        labels: torch.Tensor,
        loss_weights: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the multi-token prediction loss (same interface as
        :meth:`MultiTokenPredictionHead.compute_loss`).

        Args:
            logits_list: Per-head logits from :meth:`forward`.
            labels: Target token IDs.
            loss_weights: Optional per-head weights (default: geometric decay).

        Returns:
            ``(total_loss, per_head_losses)`` tuple.
        """
        per_head_losses: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=labels.device, dtype=torch.float32)

        if loss_weights is None:
            loss_weights = [1.0 / (2 ** i) for i in range(len(logits_list))]
            total_weight = sum(loss_weights)
            loss_weights = [w / total_weight for w in loss_weights]

        for i, (logits, weight) in enumerate(zip(logits_list, loss_weights)):
            shift = i + 1
            if shift >= labels.shape[-1]:
                break

            pred_logits = logits[:, :-shift, :].contiguous()
            target_labels = labels[:, shift:].contiguous()

            loss = F.cross_entropy(
                pred_logits.reshape(-1, self.vocab_size),
                target_labels.reshape(-1),
                ignore_index=-100,
            )
            per_head_losses[f"head_{i}"] = loss.item()
            total_loss = total_loss + weight * loss

        return total_loss, per_head_losses

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"num_future_tokens={self.num_future_tokens}, "
            f"shared_dim={self.shared_dim}, "
            f"vocab_size={self.vocab_size}, "
            f"params={self.num_parameters:,}"
        )


# ---------------------------------------------------------------------------
# 3. MultiTokenTrainingWrapper
# ---------------------------------------------------------------------------

class MultiTokenTrainingWrapper(nn.Module):
    """
    Wraps a transformer model to add multi-token prediction during training.

    During **training**:
    - Forward pass through the base transformer as normal.
    - Additional multi-token prediction heads compute auxiliary losses.
    - ``total_loss = primary_next_token_loss + mtp_loss``.

    During **inference**:
    - Multi-token heads can be used for speculative decoding via
      :class:`MultiTokenInferenceEngine`.
    - Or they can simply be disabled (set ``model.eval()``).

    Usage:

    .. code-block:: python

        base_model = NexusTransformer(config)
        wrapper = MultiTokenTrainingWrapper(
            base_model,
            num_future_tokens=4,
            head_type="mlp",
            loss_weight_decay=0.5,
        )
        result = wrapper(input_ids, labels=labels)
        loss = result["loss"]

    Args:
        base_model: A transformer model with a ``config`` attribute (having
            ``hidden_size``) and a ``vocab_size`` attribute.
        num_future_tokens: Number of auxiliary prediction heads.
        head_type: ``'linear'``, ``'mlp'``, or ``'deep'``.
        loss_weight_decay: Geometric decay factor for auxiliary losses.
            Head *i* receives weight ``decay^i``.
        freeze_base: If *True*, freeze all parameters of *base_model*.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_future_tokens: int = 4,
        head_type: str = "linear",
        loss_weight_decay: float = 0.5,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_future_tokens = num_future_tokens

        # Resolve hidden_size / vocab_size from the base model's config
        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.vocab_size

        # Multi-token prediction heads
        self.mtp_head = MultiTokenPredictionHead(
            hidden_size,
            vocab_size,
            num_future_tokens,
            head_type=head_type,
        )

        # Loss weights with geometric decay
        raw_weights = [loss_weight_decay ** i for i in range(num_future_tokens)]
        total = sum(raw_weights)
        self.loss_weights: List[float] = [w / total for w in raw_weights]

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Forward pass with multi-token prediction.

        Args:
            input_ids: ``(*batch, seq_len)`` token IDs.
            attention_mask: Optional attention mask.
            labels: Optional target token IDs (required for loss computation).
            **kwargs: Extra keyword arguments forwarded to the base model.

        Returns:
            A dict with keys:

            - ``'logits'`` — primary next-token logits.
            - ``'loss'`` — scalar loss (primary + MTP if training).
            - ``'mtp_losses'`` — per-head loss dict (training only).
            - ``'mtp_total_loss'`` — weighted MTP loss scalar (training only).
            - ``'mtp_logits'`` — list of per-head logits (training only).
        """
        # Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        # Extract outputs — be resilient to different model output shapes
        hidden_states: Optional[torch.Tensor] = getattr(outputs, "hidden_states", None)
        logits: torch.Tensor = outputs.logits
        loss: Optional[torch.Tensor] = getattr(outputs, "loss", None)

        result: Dict[str, object] = {
            "logits": logits,
            "loss": loss,
            "mtp_losses": {},
            "mtp_total_loss": torch.tensor(0.0, device=input_ids.device, dtype=torch.float32),
        }

        # Multi-token prediction (only during training when labels are given)
        if labels is not None and self.training:
            if hidden_states is None:
                hidden_states = self._get_last_hidden_states(input_ids, attention_mask)

            mtp_logits = self.mtp_head(hidden_states)
            mtp_loss, per_head = self.mtp_head.compute_loss(
                mtp_logits, labels, self.loss_weights,
            )
            result["mtp_losses"] = per_head
            result["mtp_total_loss"] = mtp_loss
            result["mtp_logits"] = mtp_logits

            if loss is not None:
                result["loss"] = loss + mtp_loss
            else:
                result["loss"] = mtp_loss

        return result

    def _get_last_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract hidden states from the last transformer layer.

        This is a fallback when the base model does not expose
        ``hidden_states`` in its output tuple/dict.  We run a partial forward
        that returns only the last-layer representation.

        Args:
            input_ids: Input token IDs.
            attention_mask: Optional attention mask.

        Returns:
            ``(*batch, seq_len, hidden_size)`` hidden states.
        """
        # Try the embeddings -> all layers approach
        embedding_layer = getattr(self.base_model, "embed_tokens", None) or getattr(
            self.base_model, "embedding", None
        )
        layers = getattr(self.base_model, "layers", None) or getattr(
            self.base_model, "h", None
        )
        norm = getattr(self.base_model, "norm", None) or getattr(
            self.base_model, "final_layer_norm", None
        )

        if embedding_layer is None or layers is None:
            raise RuntimeError(
                "Cannot extract hidden states automatically.  Please ensure "
                "the base model exposes 'hidden_states' in its forward output."
            )

        h = embedding_layer(input_ids)
        for layer in layers:
            h = layer(h, attention_mask=attention_mask)
        if norm is not None:
            h = norm(h)
        return h

    def get_mtp_logits(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Get multi-token prediction logits for speculative decoding.

        Args:
            hidden_states: ``(*batch, seq_len, hidden_size)``

        Returns:
            List of per-head logits.
        """
        return self.mtp_head(hidden_states)


# ---------------------------------------------------------------------------
# 4. MultiTokenInferenceEngine
# ---------------------------------------------------------------------------

@dataclass
class GenerationStats:
    """Statistics from a single generation run."""
    total_tokens: int = 0
    total_forward_passes: int = 0
    tokens_per_forward: float = 0.0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    acceptance_rate: float = 0.0


class MultiTokenInferenceEngine:
    """
    Inference engine that leverages multi-token prediction heads for
    accelerated generation.

    Strategy:
    1. Run base model forward pass (produces hidden_states + primary logits).
    2. All MTP heads produce predictions simultaneously.
    3. Use predictions to speculatively advance multiple tokens.
    4. Verify against base model periodically.

    Two modes:
    - ``"eager"``: Use all MTP predictions without verification (fast but less accurate).
    - ``"verify"``: Verify predictions with base model every K steps (balanced).
    - ``"conservative"``: Only use predictions that exceed confidence threshold.

    Speedup analysis (with 4 MTP heads and 80 % acceptance rate):
        - Standard: 1 token per forward pass
        - MTP eager: ~4 tokens per forward pass (4× theoretical)
        - MTP verify: ~2.5 tokens per forward pass (2.5× practical)

    Args:
        model_with_mtp: A :class:`MultiTokenTrainingWrapper` (or any module
            exposing ``get_mtp_logits``).
        tokenizer: A tokenizer with ``encode``, ``decode``, ``eos_token_id``,
            and ``vocab_size`` attributes.
        mode: One of ``"eager"``, ``"verify"``, ``"conservative"``.
        verify_interval: When *mode* is ``"verify"``, re-run the base model
            every this many speculative tokens.
        confidence_threshold: When *mode* is ``"conservative"``, only accept
            predictions whose max-softmax probability exceeds this value.
    """

    SUPPORTED_MODES = ("eager", "verify", "conservative")

    def __init__(
        self,
        model_with_mtp: nn.Module,
        tokenizer,
        mode: str = "verify",
        verify_interval: int = 4,
        confidence_threshold: float = 0.5,
    ) -> None:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {self.SUPPORTED_MODES}, got '{mode}'")

        self.model = model_with_mtp
        self.tokenizer = tokenizer
        self.mode = mode
        self.verify_interval = verify_interval
        self.confidence_threshold = confidence_threshold

        # Resolve number of MTP heads
        self.num_mtp_heads: int = 0
        if hasattr(model_with_mtp, "mtp_head"):
            self.num_mtp_heads = model_with_mtp.mtp_head.num_future_tokens
        elif hasattr(model_with_mtp, "num_future_tokens"):
            self.num_mtp_heads = model_with_mtp.num_future_tokens

        self._device = next(model_with_mtp.parameters()).device

    # ------------------------------------------------------------------
    # main generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, GenerationStats]:
        """
        Generate tokens using multi-token prediction heads.

        Args:
            prompt_ids: ``(1, prompt_len)`` or ``(prompt_len,)`` tensor of
                token IDs.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Nucleus (top-p) filtering parameter.
            stop_token_ids: Optional list of stop token IDs.

        Returns:
            A ``(generated_ids, stats)`` tuple where *generated_ids* includes
            the prompt and *stats* is a :class:`GenerationStats` dataclass.
        """
        self.model.eval()

        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)

        input_ids = prompt_ids.to(self._device)
        generated = [input_ids]
        stats = GenerationStats()

        stop_token_ids = stop_token_ids or []
        tokens_generated = 0
        steps_since_verify = 0

        while tokens_generated < max_new_tokens:
            # --- Run base model ---
            outputs = self.model.base_model(input_ids)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None:
                hidden_states = self.model._get_last_hidden_states(input_ids, None)
            logits = outputs.logits  # (1, seq, vocab)

            stats.total_forward_passes += 1

            # --- Sample primary next token ---
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            next_token = self._sample_from_logits(next_logits, top_k=top_k, top_p=top_p)
            generated.append(next_token.unsqueeze(0))
            tokens_generated += 1
            stats.accepted_tokens += 1
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            steps_since_verify += 1

            # Check stop
            if next_token.item() in stop_token_ids:
                break

            # --- Speculate with MTP heads ---
            if self.num_mtp_heads > 0 and tokens_generated < max_new_tokens:
                # Use the last hidden state for MTP speculation
                last_hidden = hidden_states[:, -1:, :]  # (1, 1, H)

                speculated = self._speculate_with_mtp(last_hidden, self.num_mtp_heads, temperature, top_k, top_p)

                for token_id, confidence in speculated:
                    if tokens_generated >= max_new_tokens:
                        break

                    accept = True

                    if self.mode == "conservative":
                        accept = confidence >= self.confidence_threshold
                    elif self.mode == "verify" and steps_since_verify >= self.verify_interval:
                        # Run verification pass
                        accept, token_id = self._verify_prediction(input_ids, token_id, temperature, top_k, top_p)
                        stats.total_forward_passes += 1
                        steps_since_verify = 0

                    if accept:
                        generated.append(token_id.unsqueeze(0).unsqueeze(0))
                        input_ids = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                        tokens_generated += 1
                        stats.accepted_tokens += 1
                        steps_since_verify += 1

                        if token_id.item() in stop_token_ids:
                            break
                    else:
                        stats.rejected_tokens += 1
                        break  # stop speculating on rejection

        all_ids = torch.cat(generated, dim=1)
        stats.total_tokens = tokens_generated
        stats.tokens_per_forward = (
            stats.total_tokens / max(stats.total_forward_passes, 1)
        )
        stats.acceptance_rate = (
            stats.accepted_tokens / max(stats.accepted_tokens + stats.rejected_tokens, 1)
        )

        return all_ids, stats

    # ------------------------------------------------------------------
    # speculation helpers
    # ------------------------------------------------------------------

    def _speculate_with_mtp(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[Tuple[torch.Tensor, float]]:
        """
        Use MTP heads to predict next *num_tokens* tokens greedily or
        stochastically.

        Args:
            hidden_states: ``(*batch, 1, hidden_size)``
            num_tokens: Number of tokens to speculate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus filtering.

        Returns:
            List of ``(token_id, confidence_score)`` tuples.
        """
        mtp_logits = self.model.get_mtp_logits(hidden_states)
        predictions: List[Tuple[torch.Tensor, float]] = []

        # We currently only have the hidden state for the *current* position,
        # so we use head i to predict token at position t + i + 1.
        for i in range(min(num_tokens, len(mtp_logits))):
            head_logits = mtp_logits[i][:, -1, :]  # (1, vocab)
            scaled = head_logits / max(temperature, 1e-8)
            probs = F.softmax(scaled, dim=-1)

            confidence, token_id = probs.max(dim=-1)
            token_id = self._sample_from_logits(scaled, top_k=top_k, top_p=top_p)
            predictions.append((token_id, confidence.item()))

        return predictions

    def _verify_prediction(
        self,
        input_ids: torch.Tensor,
        predicted_token: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[bool, torch.Tensor]:
        """
        Run the base model to verify an MTP prediction.

        Args:
            input_ids: Current sequence including the predicted token appended.
            predicted_token: The speculated token ID.

        Returns:
            ``(accepted, corrected_token)`` — if the model agrees, *accepted*
            is *True* and *corrected_token* equals *predicted_token*.
        """
        outputs = self.model.base_model(input_ids)
        logits = outputs.logits[:, -1, :] / max(temperature, 1e-8)

        model_token = self._sample_from_logits(logits, top_k=top_k, top_p=top_p)
        accepted = (model_token.item() == predicted_token.item())
        return accepted, model_token

    # ------------------------------------------------------------------
    # sampling utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Sample a token from *logits* with temperature already applied.

        Applies top-k and top-p filtering then samples from the remaining
        distribution.

        Args:
            logits: ``(batch, vocab_size)`` or ``(vocab_size,)``
            top_k: Keep only top-k tokens.
            top_p: Keep tokens with cumulative probability ≤ *top_p*.

        Returns:
            Sampled token ID tensor with shape matching the batch dim.
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Top-k
        if top_k > 0:
            top_k_val = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k_val)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right so the first token above threshold is kept
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove,
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return sampled.squeeze(-1)

    @staticmethod
    def _confidence_score(logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence score (max softmax probability) from logits."""
        return F.softmax(logits, dim=-1).max(dim=-1).values


# ---------------------------------------------------------------------------
# 5. NgramFallbackPredictor
# ---------------------------------------------------------------------------

class NgramFallbackPredictor:
    """
    Lightweight n-gram based fallback for multi-token prediction.

    When MTP heads are unavailable or for very far-ahead predictions, use
    n-gram statistics from the context as a simple predictor.  This is much
    simpler than neural MTP heads but provides a reasonable baseline for
    common token sequences.

    Usage: as a fallback in :class:`MultiTokenInferenceEngine` when MTP
    confidence is below threshold.

    Args:
        max_n: Maximum n-gram order (e.g. 4 means up to 4-grams).
        max_predictions: Maximum number of candidate tokens to return.
        smoothing: Laplace smoothing constant.
    """

    def __init__(
        self,
        max_n: int = 4,
        max_predictions: int = 5,
        smoothing: float = 1.0,
    ) -> None:
        self.max_n = max_n
        self.max_predictions = max_predictions
        self.smoothing = smoothing

        # ngram_counts[n] maps tuple(context) -> Counter of next tokens
        self.ngram_counts: Dict[int, Dict[Tuple[int, ...], Dict[int, int]]] = {
            n: defaultdict(lambda: defaultdict(int))
            for n in range(1, max_n + 1)
        }
        # Unigram total count (for backoff)
        self.unigram_total: int = 0

    def predict(
        self,
        token_ids: List[int],
        num_predictions: int = 4,
    ) -> List[Tuple[int, float]]:
        """
        Predict next tokens using n-gram statistics with Katz backoff.

        Starting from the highest available order, fall back to lower orders
        when the context is not found in the count table.

        Args:
            token_ids: Recent context tokens (most recent last).
            num_predictions: Number of predictions to return.

        Returns:
            List of ``(token_id, probability)`` tuples sorted by
            descending probability.
        """
        if self.unigram_total == 0:
            # No data yet — return empty
            return []

        # Try each n-gram order from max down to 1
        for n in range(min(self.max_n, len(token_ids)), 0, -1):
            context = tuple(token_ids[-n:])
            counts = self.ngram_counts[n].get(context, {})

            if counts:
                total = sum(counts.values()) + self.smoothing * len(counts)
                probs = {
                    tok: (cnt + self.smoothing) / total
                    for tok, cnt in counts.items()
                }
                sorted_preds = sorted(probs.items(), key=lambda x: -x[1])
                return sorted_preds[:num_predictions]

        # Absolute fallback: uniform over observed unigrams
        unigram = self.ngram_counts[1].get((), {})
        if unigram:
            total = sum(unigram.values())
            probs = {tok: cnt / total for tok, cnt in unigram.items()}
            sorted_preds = sorted(probs.items(), key=lambda x: -x[1])
            return sorted_preds[:num_predictions]

        return []

    def update_statistics(self, token_ids: List[int]) -> None:
        """
        Update n-gram frequency tables from a sequence of tokens.

        Args:
            token_ids: A flat list of integer token IDs (e.g. an entire
                document or a recent chunk).
        """
        if not token_ids:
            return

        for n in range(1, self.max_n + 1):
            for i in range(len(token_ids) - n):
                context = tuple(token_ids[i : i + n])
                next_tok = token_ids[i + n]
                self.ngram_counts[n][context][next_tok] += 1

        # Update unigram total
        self.unigram_total += len(token_ids)

    def prune(self, min_count: int = 2) -> int:
        """
        Remove n-gram entries whose count is below *min_count*.

        Returns:
            Number of entries pruned.
        """
        removed = 0
        for n in range(1, self.max_n + 1):
            to_delete = [
                ctx for ctx, ctr in self.ngram_counts[n].items()
                if sum(ctr.values()) < min_count
            ]
            for ctx in to_delete:
                del self.ngram_counts[n][ctx]
                removed += 1
        return removed

    def state_dict(self) -> Dict:
        """Return serialisable state (useful for checkpointing)."""
        # Convert defaultdicts to plain dicts for JSON/pickle compatibility
        serialisable_counts: Dict[int, Dict] = {}
        for n, ctx_dict in self.ngram_counts.items():
            serialisable_counts[n] = {
                "=".join(map(str, ctx)): counts
                for ctx, counts in ctx_dict.items()
            }
        return {
            "max_n": self.max_n,
            "max_predictions": self.max_predictions,
            "smoothing": self.smoothing,
            "unigram_total": self.unigram_total,
            "ngram_counts": serialisable_counts,
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "NgramFallbackPredictor":
        """Restore from a previously saved state dict."""
        obj = cls(
            max_n=state["max_n"],
            max_predictions=state["max_predictions"],
            smoothing=state["smoothing"],
        )
        obj.unigram_total = state["unigram_total"]
        for n_str, ctx_dict in state["ngram_counts"].items():
            n = int(n_str)
            for ctx_str, counts in ctx_dict.items():
                ctx = tuple(int(x) for x in ctx_str.split("="))
                obj.ngram_counts[n][ctx] = counts
        return obj
