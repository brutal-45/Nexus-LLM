"""
Proximal Policy Optimization (PPO) Trainer for RLHF.

This module implements the complete PPO algorithm from scratch using PyTorch,
designed for Reinforcement Learning from Human Feedback (RLHF) on Large
Language Models.  PPO is an on-policy actor-critic method that iteratively:

    1.  Generates rollouts from the current policy (actor).
    2.  Computes rewards with a separate reward model.
    3.  Estimates advantages via Generalized Advantage Estimation (GAE).
    4.  Updates the actor and critic for several epochs using a clipped
        surrogate objective.

Key references
--------------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
*Proximal Policy Optimization Algorithms*.  arXiv:1707.06347.

Ouyang, L. et al. (2022).  *Training language models to follow instructions
with human feedback*.  arXiv:2203.02155.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean of *tensor* over all positions where *mask* is True.

    Parameters
    ----------
    tensor : torch.Tensor
        Values to average (shape ``[B, S]`` or compatible with *mask*).
    mask : torch.Tensor
        Boolean or float mask (1 = keep, 0 = ignore).

    Returns
    -------
    torch.Tensor
        Scalar mean over the valid (masked) elements.
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    return tensor.masked_fill(~mask, 0.0).sum() / mask.sum().clamp(min=1.0)


def _whiten(values: torch.Tensor, mask: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Whiten (normalize) *values* using the masked mean and standard deviation.

    .. math::

        \\hat{v}_i = \\frac{v_i - \\mu}{\\sigma + \\epsilon}

    Parameters
    ----------
    values : torch.Tensor
        Raw values (``[B, S]``).
    mask : torch.Tensor
        1 for valid positions, 0 otherwise.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Whitened values of the same shape.
    """
    mean = _masked_mean(values, mask)
    variance = _masked_mean((values - mean) ** 2, mask)
    return (values - mean) / torch.sqrt(variance + epsilon)


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """A lightweight value head that projects the base model's hidden states
    to a scalar value per position.

    The value head is used as the **Critic** in the actor-critic PPO
    framework.  It takes the last hidden state :math:`h_t \\in
    \\mathbb{R}^{d}` from the base transformer and maps it to:

    .. math::

        V(s_t) = W \\, h_t + b \\in \\mathbb{R}

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the base model's hidden state (:math:`d`).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # A simple linear projection from hidden_size → 1.
        self.value_projection = nn.Linear(hidden_size, 1, bias=True)
        # Xavier uniform initialization for stable training starts.
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass producing a scalar value per token position.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Last hidden states from the base model, shape ``[B, S, d]``.
        attention_mask : Optional[torch.Tensor]
            Mask of shape ``[B, S]`` — ignored by the projection itself but
            useful for downstream masking.

        Returns
        -------
        torch.Tensor
            Value predictions squeezed to shape ``[B, S]``.
        """
        # hidden_states: [B, S, hidden_size] → [B, S, 1] → [B, S]
        values = self.value_projection(hidden_states).squeeze(-1)
        return values


# ---------------------------------------------------------------------------
# CriticModel
# ---------------------------------------------------------------------------

class CriticModel(nn.Module):
    """Critic (value) model composed of a base LLM and a :class:`ValueHead`.

    The critic estimates the state-value function :math:`V(s)` used in the
    advantage computation.  It optionally **shares** the base model's
    embedding layer with the actor to reduce memory usage and improve
    training stability.

    Parameters
    ----------
    base_model : nn.Module
        A HuggingFace-style transformer model (e.g. ``AutoModelForCausalLM``).
        The forward method must return a ``ModelOutput`` with a
        ``last_hidden_state`` attribute, or the model itself can be indexed
        to access ``model.model`` which holds the backbone.
    share_base_model : bool, optional
        If ``True``, the critic shares the transformer backbone with the
        actor.  Only the :class:`ValueHead` is separate.  Default is ``False``.
    """

    def __init__(
        self,
        base_model: nn.Module,
        share_base_model: bool = False,
    ) -> None:
        super().__init__()
        self.share_base_model = share_base_model

        if share_base_model:
            # When sharing, we do NOT create a separate copy.  The caller is
            # responsible for passing the same base_model reference to both
            # actor and critic.
            self.backbone = base_model
        else:
            self.backbone = base_model

        # Detect hidden size from the model's config or embedding layer.
        hidden_size = self._detect_hidden_size(base_model)
        self.value_head = ValueHead(hidden_size)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _detect_hidden_size(model: nn.Module) -> int:
        """Attempt to infer ``hidden_size`` from the model.

        The function tries, in order:
        1. ``model.config.hidden_size``
        2. ``model.model.config.hidden_size`` (HF AutoModelForCausalLM)
        3. The embedding dimension of the first embedding layer found.

        Raises
        ------
        ValueError
            If no hidden size can be inferred.
        """
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size  # type: ignore[return-value]
        if hasattr(model, "model") and hasattr(model.model, "config"):
            cfg = model.model.config  # type: ignore[union-attr]
            if hasattr(cfg, "hidden_size"):
                return cfg.hidden_size  # type: ignore[return-value]
        # Fallback: look for an embedding layer.
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return module.embedding_dim
        raise ValueError(
            "Could not detect hidden_size from the model.  Please pass "
            "a model with a ``config.hidden_size`` attribute."
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass returning scalar value predictions.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token ids, shape ``[B, S]``.
        attention_mask : Optional[torch.Tensor]
            Attention mask, shape ``[B, S]``.
        **kwargs
            Additional keyword arguments forwarded to the backbone.

        Returns
        -------
        torch.Tensor
            Value predictions of shape ``[B, S]``.
        """
        # We need the last hidden states, not the language-model head logits.
        # Strategy: try backbone.model (the inner transformer) first; fall back
        # to calling backbone directly.
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "forward"):
            outputs = self.backbone.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state  # type: ignore[union-attr]
        elif isinstance(outputs, torch.Tensor):
            hidden_states = outputs
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            hidden_states = outputs[0]
        else:
            raise RuntimeError(
                "Cannot extract hidden states from backbone output of type "
                f"{type(outputs).__name__}."
            )

        values = self.value_head(hidden_states, attention_mask)
        return values


# ---------------------------------------------------------------------------
# PPOConfig
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Configuration dataclass for the PPO trainer.

    All hyper-parameters governing the PPO training loop are stored here.
    Defaults follow the recommendations of the InstructGPT / RLHF literature
    with minor adjustments for LLM fine-tuning.

    Attributes
    ----------
    ppo_epochs : int
        Number of optimization epochs over each rollout buffer.  Default ``4``.
    clip_epsilon : float
        PPO clipping parameter :math:`\\epsilon`.  The surrogate objective
        clips the importance ratio to :math:`[1 - \\epsilon, 1 + \\epsilon]`.
        Default ``0.2``.
    value_coeff : float
        Coefficient :math:`c_1` for the value-function loss term.
        Default ``0.1``.
    kl_coeff : float
        Coefficient :math:`c_2` for the KL-divergence penalty.  This initial
        value is adapted online by :meth:`PPOTrainer.kl_adaptive_controller`.
        Default ``0.05``.
    kl_target : float
        Target KL divergence :math:`d_{\\text{target}}` used in the adaptive
        controller.  Default ``0.05``.
    gamma : float
        Discount factor :math:`\\gamma` for return computation.  A value of
        ``1.0`` means undiscounted returns (common for sentence-level rewards
        in RLHF).  Default ``1.0``.
    gae_lambda : float
        GAE lambda :math:`\\lambda` for advantage estimation.  Default ``0.95``.
    actor_lr : float
        Learning rate for the actor (policy) optimizer.  Default ``1e-6``.
    critic_lr : float
        Learning rate for the critic (value) optimizer.  Default ``1e-5``.
    mini_batch_size : int
        Mini-batch size used during PPO update epochs.  Default ``8``.
    max_grad_norm : float
        Maximum gradient norm for clipping (global norm across all parameters).
        Default ``1.0``.
    max_new_tokens : int
        Maximum number of tokens to generate during rollout.  Default ``512``.
    temperature : float
        Sampling temperature for generation.  Default ``0.7``.
    top_p : float
        Nucleus (top-p) sampling threshold.  Default ``0.9``.
    """

    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coeff: float = 0.1
    kl_coeff: float = 0.05
    kl_target: float = 0.05
    gamma: float = 1.0
    gae_lambda: float = 0.95
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    mini_batch_size: int = 8
    max_grad_norm: float = 1.0
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


# ---------------------------------------------------------------------------
# PPOTrainingStats
# ---------------------------------------------------------------------------

@dataclass
class PPOTrainingStats:
    """A dataclass that tracks all diagnostics from a single PPO update step.

    These statistics are logged per training iteration and are useful for
    monitoring training health (KL divergence, clipping fraction,
    explained variance, etc.).

    Attributes
    ----------
    policy_loss : float
        Mean policy (clipped surrogate) loss.
    value_loss : float
        Mean value-function (MSE) loss.
    kl_divergence : float
        Mean KL divergence between the old and new policy distributions.
    total_loss : float
        Combined loss: ``-pg_loss + c1 * vf_loss + c2 * kl_loss``.
    mean_reward : float
        Mean reward across all rollout samples.
    mean_advantage : float
        Mean (whitened) advantage after GAE computation.
    clip_fraction : float
        Fraction of mini-batches where the surrogate was clipped.
    acceptance_rate : float
        Fraction of the time the unclipped objective was used (i.e. ratio
        within :math:`[1-\\epsilon, 1+\\epsilon]`).
    approx_kl : float
        Approximate KL: ``mean((ratio - 1) - log(ratio))`` — a cheap proxy
        for the true KL divergence.
    explained_variance : float
        Fraction of variance in returns explained by the value predictions.
        Values close to 1 indicate a well-calibrated critic.
    """

    policy_loss: float = 0.0
    value_loss: float = 0.0
    kl_divergence: float = 0.0
    total_loss: float = 0.0
    mean_reward: float = 0.0
    mean_advantage: float = 0.0
    clip_fraction: float = 0.0
    acceptance_rate: float = 0.0
    approx_kl: float = 0.0
    explained_variance: float = 0.0


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores rollout data generated by the actor for PPO training.

    Each rollout buffer collects sequences, actions (generated token ids),
    log-probabilities, rewards, value predictions, and episode-termination
    flags.  After collection the buffer computes GAE advantages and returns.

    Parameters
    ----------
    buffer_size : int
        Maximum number of trajectories (or batch entries) the buffer can hold.
    """

    def __init__(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size
        self.sequences: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.old_log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.attention_masks: List[torch.Tensor] = []
        # Populated by compute_returns / compute_advantages.
        self.returns: List[torch.Tensor] = []
        self.advantages: List[torch.Tensor] = []

    # -- mutation methods --------------------------------------------------

    def add(
        self,
        sequence: torch.Tensor,
        action: torch.Tensor,
        old_log_prob: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor,
        done: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> None:
        """Append a single trajectory to the buffer.

        Parameters
        ----------
        sequence : torch.Tensor
            Full token-id sequence (prompt + response), shape ``[S]`` or
            ``[1, S]``.
        action : torch.Tensor
            Token ids of the generated response portion, shape ``[T]`` or
            ``[1, T]``.
        old_log_prob : torch.Tensor
            Log-probabilities of *action* under the old policy, shape ``[T]``
            or ``[1, T]``.
        reward : torch.Tensor
            Reward signal (scalar or per-token), shape ``[T]`` or ``[1, T]``.
        value : torch.Tensor
            Value prediction for each position, shape ``[S]`` or ``[1, S]``.
        done : torch.Tensor
            Episode-termination flags, shape ``[]`` (scalar) or ``[T]``.
        attention_mask : torch.Tensor
            Mask indicating valid (non-padding) positions, shape ``[S]`` or
            ``[1, S]``.
        """
        self.sequences.append(sequence)
        self.actions.append(action)
        self.old_log_probs.append(old_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.attention_masks.append(attention_mask)

    def clear(self) -> None:
        """Empty the buffer and all computed returns / advantages."""
        self.sequences.clear()
        self.actions.clear()
        self.old_log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.attention_masks.clear()
        self.returns.clear()
        self.advantages.clear()

    def __len__(self) -> int:
        return len(self.sequences)

    # -- GAE computation ---------------------------------------------------

    def compute_returns(
        self,
        gamma: float = 1.0,
    ) -> None:
        """Compute discounted returns :math:`G_t` for each timestep.

        .. math::

            G_t = r_t + \\gamma \\, G_{t+1}

        This is a simple recursive formulation with no bootstrapping.
        Results are stored in :attr:`returns`.

        Parameters
        ----------
        gamma : float
            Discount factor.  Default ``1.0`` (undiscounted).
        """
        self.returns.clear()
        for idx in range(len(self)):
            rewards = self.rewards[idx]
            # Work in 1-D for simplicity.
            r = rewards.float().flatten()
            G = torch.zeros_like(r)
            running_return = torch.tensor(0.0, device=r.device)
            for t in reversed(range(r.shape[0])):
                running_return = r[t] + gamma * running_return
                G[t] = running_return
            self.returns.append(G.view_as(rewards))

    def compute_advantages(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> None:
        r"""Compute Generalized Advantage Estimation (GAE) and returns.

        GAE interpolates between high-variance Monte-Carlo advantage estimates
        and low-variance but high-bias TD(0) estimates:

        .. math::

            \\delta_t = r_t + \\gamma \\, V(s_{t+1}) \\, (1 - d_t)
                        - V(s_t)

            \\hat{A}_t = \\sum_{l=0}^{\\infty}
                (\\gamma \\lambda)^l \\, \\delta_{t+l}

        In practice the running sum is computed iteratively in reverse:

        .. math::

            \\hat{A}_t = \\delta_t
                + \\gamma \\lambda \\, \\hat{A}_{t+1} \\, (1 - d_t)

        Returns are then:

        .. math::

            \\hat{R}_t = \\hat{A}_t + V(s_t)

        Parameters
        ----------
        gamma : float
            Discount factor :math:`\\gamma`.  Default ``1.0``.
        gae_lambda : float
            GAE blending parameter :math:`\\lambda`.  Default ``0.95``.
        """
        self.advantages.clear()
        self.returns.clear()

        for idx in range(len(self)):
            rewards = self.rewards[idx].float().flatten()
            values = self.values[idx].float().flatten()

            # Align rewards and values to the same length (reward may only
            # cover the generated portion).
            seq_len = rewards.shape[0]
            vals = values[:seq_len]

            # Done flags — default to all zeros (no truncation).
            dones = self.dones[idx].float().flatten()
            if dones.shape[0] < seq_len:
                pad = torch.zeros(seq_len - dones.shape[0], device=dones.device)
                dones = torch.cat([dones, pad], dim=0)
            dones = dones[:seq_len]

            # Next-value is zero for the last position (episode end).
            next_values = torch.cat([vals[1:], torch.zeros(1, device=vals.device)])

            # TD residuals:  δ_t = r_t + γ V(s_{t+1})(1 - d_t) - V(s_t)
            deltas = rewards + gamma * next_values * (1.0 - dones) - vals

            # Running advantage (reverse accumulation).
            advantages = torch.zeros_like(deltas)
            running_advantage = torch.tensor(0.0, device=deltas.device)
            for t in reversed(range(seq_len)):
                running_advantage = (
                    deltas[t]
                    + gamma * gae_lambda * running_advantage * (1.0 - dones[t])
                )
                advantages[t] = running_advantage

            # Returns: R_t = A_t + V(s_t)
            returns = advantages + vals

            self.advantages.append(advantages)
            self.returns.append(returns)

    # -- generation --------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        actor: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer: Any,
        config: PPOConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate responses from the *actor* model and store the rollout.

        The generation uses top-p (nucleus) sampling with the configured
        temperature.  After generation, the method computes:

        - ``old_log_probs`` via the actor's distribution over generated tokens
        - ``values`` via the critic (if available — otherwise zeros)

        Parameters
        ----------
        actor : nn.Module
            The policy (actor) model, a causal LM that supports ``generate``.
        input_ids : torch.Tensor
            Prompt token ids, shape ``[B, S_prompt]``.
        attention_mask : torch.Tensor
            Prompt attention mask, shape ``[B, S_prompt]``.
        tokenizer : Any
            A tokenizer with a ``decode`` method (unused internally but kept
            for API symmetry).
        config : PPOConfig
            PPO configuration (``temperature``, ``top_p``, ``max_new_tokens``).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(sequences, actions, old_log_probs)`` where:
            - ``sequences``: full sequences ``[B, S_total]``
            - ``actions``: generated token ids ``[B, S_gen]``
            - ``old_log_probs``: log-probs of actions ``[B, S_gen]``
        """
        # Generation kwargs — top-p sampling.
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": True,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        # Remove None entries so HuggingFace doesn't complain.
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Generate: output_ids has shape [B, S_prompt + S_gen].
        output_ids = actor.generate(input_ids, **gen_kwargs)

        sequences = output_ids
        prompt_len = input_ids.shape[1]
        actions = output_ids[:, prompt_len:]  # [B, S_gen]

        # Compute old log-probs by forwarding the full sequence.
        # Shift logits by one so that logit[i] predicts action[i].
        model_out = actor(input_ids=sequences, attention_mask=None)
        if hasattr(model_out, "logits"):
            logits = model_out.logits  # [B, S_total, V]
        elif isinstance(model_out, tuple):
            logits = model_out[0]
        else:
            raise RuntimeError(
                f"Unexpected model output type: {type(model_out)}"
            )

        # Logits for predicting each token: shift left by one.
        gen_len = actions.shape[1]
        # logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :] predict
        # tokens at positions prompt_len .. prompt_len + gen_len - 1.
        action_logits = logits[:, prompt_len - 1: prompt_len - 1 + gen_len, :]
        log_probs = F.log_softmax(action_logits, dim=-1)

        # Gather the log-probs corresponding to the actual actions.
        action_ids = actions.unsqueeze(-1)  # [B, S_gen, 1]
        old_log_probs = log_probs.gather(dim=-1, index=action_ids).squeeze(-1)
        # old_log_probs: [B, S_gen]

        return sequences, actions, old_log_probs


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """Proximal Policy Optimization trainer for RLHF on language models.

    The trainer orchestrates the full RLHF loop:

    1. **Rollout phase** — generate responses with the current policy,
       score them with the reward model, and record log-probs and values.
    2. **Advantage computation** — estimate advantages via GAE.
    3. **Update phase** — optimize the clipped surrogate objective for
       multiple epochs over mini-batches, updating both the actor and
       the critic.
    4. **KL adaptation** — adjust the KL penalty coefficient to keep the
       policy close to the reference model.

    Parameters
    ----------
    actor : nn.Module
        The policy model (causal LM).  Must support ``generate`` and return
        ``logits`` on forward.
    critic : CriticModel
        The critic model that predicts state values.
    reward_model : nn.Module
        A model that takes token sequences and returns scalar rewards.
    reference_model : nn.Module
        The frozen reference model (original policy) used for KL computation.
    config : PPOConfig
        Hyper-parameters for PPO training.
    tokenizer : Any
        Tokenizer (used during rollout generation).
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: CriticModel,
        reward_model: nn.Module,
        reference_model: nn.Module,
        config: PPOConfig,
        tokenizer: Any,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.reference_model = reference_model
        self.config = config
        self.tokenizer = tokenizer

        # Freeze the reference model — it provides the KL anchor.
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

        # Freeze the reward model.
        for param in self.reward_model.parameters():
            param.requires_grad = False
        self.reward_model.eval()

        # Separate optimizers for actor and critic.
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config.actor_lr,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr,
        )

        # Rollout buffer sized to hold one batch of rollouts.
        self.rollout_buffer = RolloutBuffer(buffer_size=config.mini_batch_size * 4)

        # Mutable KL coefficient (adapted online).
        self.kl_coeff: float = config.kl_coeff

    # -- log-prob helper ---------------------------------------------------

    def get_log_probs(
        self,
        model: nn.Module,
        sequences: torch.Tensor,
        actions: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Extract log-probabilities of *actions* under *model*.

        Given a batch of full sequences (prompt + generated tokens), compute
        the log-probability the model assigns to each generated token.

        .. math::

            \\log \\pi_\\theta(a_t \\mid s_{<t}) =
                \\log \\text{Softmax}(\\text{logits}_t)[a_t]

        Parameters
        ----------
        model : nn.Module
            A causal LM that returns ``logits`` on forward.
        sequences : torch.Tensor
            Full token-id sequences, shape ``[B, S]`` where
            ``S = prompt_length + gen_length``.
        actions : torch.Tensor
            Generated token ids, shape ``[B, T]`` where ``T = S - prompt_length``.
        prompt_length : int
            Number of prompt tokens.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[B, T]``.
        """
        with torch.set_grad_enabled(model is self.actor):
            outputs = model(input_ids=sequences, attention_mask=None)
            if hasattr(outputs, "logits"):
                logits = outputs.logits  # [B, S, V]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                raise RuntimeError(
                    f"Unexpected output from model: {type(outputs)}"
                )

        gen_len = actions.shape[1]
        # Shift logits: logit at position t predicts token at position t+1.
        action_logits = logits[:, prompt_length - 1: prompt_length - 1 + gen_len, :]
        log_probs = F.log_softmax(action_logits, dim=-1)

        action_ids = actions.unsqueeze(-1)  # [B, T, 1]
        per_token_log_probs = log_probs.gather(dim=-1, index=action_ids).squeeze(-1)
        return per_token_log_probs  # [B, T]

    # -- rollout phase -----------------------------------------------------

    def rollout_phase(
        self,
        prompts: torch.Tensor,
        prompt_attention_masks: torch.Tensor,
    ) -> None:
        """Generate rollouts from the current policy and compute rewards.

        For each prompt in the batch:

        1. The actor generates a response.
        2. The reward model scores the response.
        3. The critic predicts values for each position.
        4. Old log-probs are recorded.

        All data is pushed into :attr:`rollout_buffer`.

        Parameters
        ----------
        prompts : torch.Tensor
            Prompt token ids, shape ``[B, S_p]``.
        prompt_attention_masks : torch.Tensor
            Prompt attention mask, shape ``[B, S_p]``.
        """
        self.rollout_buffer.clear()
        batch_size = prompts.shape[0]
        prompt_len = prompts.shape[1]

        # Generate sequences using the rollout buffer's generate method.
        sequences, actions, old_log_probs = self.rollout_buffer.generate(
            actor=self.actor,
            input_ids=prompts,
            attention_mask=prompt_attention_masks,
            tokenizer=self.tokenizer,
            config=self.config,
        )

        # --- Compute rewards with the reward model -----------------------
        with torch.no_grad():
            reward_outputs = self.reward_model(input_ids=sequences)
            if isinstance(reward_outputs, torch.Tensor):
                rewards_per_seq = reward_outputs.squeeze(-1)  # [B]
            elif hasattr(reward_outputs, "logits"):
                rewards_per_seq = reward_outputs.logits.squeeze(-1)
            elif hasattr(reward_outputs, "score"):
                rewards_per_seq = reward_outputs.score.squeeze(-1)
            else:
                rewards_per_seq = reward_outputs[0].squeeze(-1)

            # Ensure scalar per sequence.
            if rewards_per_seq.dim() == 0:
                rewards_per_seq = rewards_per_seq.unsqueeze(0)
            # rewards_per_seq: [B]

        # --- Compute values with the critic ------------------------------
        with torch.no_grad():
            values = self.critic(
                input_ids=sequences,
                attention_mask=None,
            )  # [B, S_total]

        # --- Store each trajectory in the buffer --------------------------
        for i in range(batch_size):
            seq_i = sequences[i]  # [S]
            act_i = actions[i]  # [T]
            olp_i = old_log_probs[i]  # [T]
            val_i = values[i]  # [S]

            # Reward is a single scalar per sequence — broadcast to each
            # generated token so GAE can process it per-step.
            r_i = rewards_per_seq[i].repeat(act_i.shape[0])  # [T]

            # Done flag: 1 at the last generated token (sequence ended), 0
            # elsewhere.  This handles truncation due to max_new_tokens.
            done_i = torch.zeros(act_i.shape[0], device=act_i.device)
            done_i[-1] = 1.0

            # Attention mask: 1 for all non-pad tokens in the sequence.
            attn_i = torch.ones(seq_i.shape[0], device=seq_i.device)

            self.rollout_buffer.add(
                sequence=seq_i,
                action=act_i,
                old_log_prob=olp_i,
                reward=r_i,
                value=val_i,
                done=done_i,
                attention_mask=attn_i,
            )

    # -- GAE computation ---------------------------------------------------

    def compute_gae_advantages(self) -> None:
        """Compute GAE advantages and returns from buffered rollout data.

        Delegates to :meth:`RolloutBuffer.compute_advantages` using the
        current configuration's ``gamma`` and ``gae_lambda``.

        After this call, ``self.rollout_buffer.advantages`` and
        ``self.rollout_buffer.returns`` are populated.
        """
        self.rollout_buffer.compute_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

    # -- update phase ------------------------------------------------------

    def update_phase(self) -> PPOTrainingStats:
        """Run the PPO update for several epochs over mini-batches.

        For each mini-batch:

        1. Compute new log-probs from the actor.
        2. Compute the importance ratio:

           .. math::

               r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}
                                   {\\pi_{\\theta_{\\text{old}}}(a_t|s_t)}
                             = \\exp(\\log \\pi_\\theta - \\log \\pi_{\\text{old}})

        3. Compute the clipped surrogate objective:

           .. math::

               L^{\\text{CLIP}}(\\theta) = \\hat{E}_t \\bigl[
                   \\min\\bigl(
                       r_t(\\theta)\\,\\hat{A}_t,\\;
                       \\text{clip}(r_t, 1-\\epsilon, 1+\\epsilon)\\,\\hat{A}_t
                   \\bigr)
               \\bigr]

        4. Value loss:

           .. math::

               L^{\\text{VF}} = \\frac{1}{2}\\,\\hat{E}_t
                   \\bigl[(V_\\theta(s_t) - \\hat{R}_t)^2\\bigr]

        5. KL divergence penalty:

           .. math::

               L^{\\text{KL}} = \\hat{E}_t
                   \\bigl[\\pi_{\\text{old}}(\\cdot|s_t)
                   \\log\\frac{\\pi_{\\text{old}}(\\cdot|s_t)}
                                {\\pi_\\theta(\\cdot|s_t)}\\bigr]

        6. Total loss:

           .. math::

               L(\\theta) = -L^{\\text{CLIP}}
                   + c_1\\, L^{\\text{VF}}
                   + c_2\\, L^{\\text{KL}}

        Returns
        -------
        PPOTrainingStats
            Aggregated statistics for this update phase.
        """
        buffer = self.rollout_buffer
        num_samples = len(buffer)
        if num_samples == 0:
            warnings.warn("Rollout buffer is empty — skipping update.")
            return PPOTrainingStats()

        # Precompute list indices for shuffling.
        all_indices = list(range(num_samples))

        # Accumulators for statistics.
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        total_loss = 0.0
        total_clip_frac = 0.0
        total_accept = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        # Collect rewards / advantages / returns for global stats.
        all_rewards: List[float] = []
        all_advantages: List[float] = []
        all_returns: List[float] = []
        all_values_pred: List[float] = []

        for _epoch in range(self.config.ppo_epochs):
            # Shuffle indices each epoch for better coverage.
            import random
            random.shuffle(all_indices)

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_kl = 0.0
            epoch_clip_frac = 0.0
            epoch_approx_kl = 0.0
            epoch_n = 0

            # Process in mini-batches.
            for start in range(0, num_samples, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, num_samples)
                mb_indices = all_indices[start:end]

                # --- Gather mini-batch tensors (pad to common length) -----
                mb_sequences = [buffer.sequences[i] for i in mb_indices]
                mb_actions = [buffer.actions[i] for i in mb_indices]
                mb_old_log_probs = [buffer.old_log_probs[i] for i in mb_indices]
                mb_advantages = [buffer.advantages[i] for i in mb_indices]
                mb_returns = [buffer.returns[i] for i in mb_indices]
                mb_rewards = [buffer.rewards[i] for i in mb_indices]

                # Pad sequences to the longest in the mini-batch.
                max_seq_len = max(s.shape[0] for s in mb_sequences)
                padded_seqs = torch.zeros(
                    len(mb_sequences), max_seq_len,
                    dtype=mb_sequences[0].dtype,
                    device=mb_sequences[0].device,
                )
                padded_attn = torch.zeros(
                    len(mb_sequences), max_seq_len,
                    dtype=mb_sequences[0].dtype,
                    device=mb_sequences[0].device,
                )
                for j, s in enumerate(mb_sequences):
                    padded_seqs[j, :s.shape[0]] = s
                    padded_attn[j, :s.shape[0]] = 1.0

                # Pad actions to common gen length.
                max_gen_len = max(a.shape[0] for a in mb_actions)
                padded_actions = torch.zeros(
                    len(mb_actions), max_gen_len,
                    dtype=mb_actions[0].dtype,
                    device=mb_actions[0].device,
                )
                for j, a in enumerate(mb_actions):
                    padded_actions[j, :a.shape[0]] = a

                # Pad old_log_probs, advantages, returns.
                padded_old_lp = torch.zeros(
                    len(mb_old_log_probs), max_gen_len,
                    dtype=mb_old_log_probs[0].dtype,
                    device=mb_old_log_probs[0].device,
                )
                padded_advs = torch.zeros(
                    len(mb_advantages), max_gen_len,
                    dtype=mb_advantages[0].dtype,
                    device=mb_advantages[0].device,
                )
                padded_rets = torch.zeros(
                    len(mb_returns), max_gen_len,
                    dtype=mb_returns[0].dtype,
                    device=mb_returns[0].device,
                )
                for j in range(len(mb_indices)):
                    olp = mb_old_log_probs[j]
                    adv = mb_advantages[j]
                    ret = mb_returns[j]
                    padded_old_lp[j, :olp.shape[0]] = olp
                    padded_advs[j, :adv.shape[0]] = adv
                    padded_rets[j, :ret.shape[0]] = ret

                # Action mask (valid generated tokens — not padding).
                action_mask = torch.zeros(
                    len(mb_actions), max_gen_len,
                    device=mb_actions[0].device,
                )
                for j, a in enumerate(mb_actions):
                    action_mask[j, :a.shape[0]] = 1.0

                # --- Compute new log probs from actor --------------------
                new_log_probs = self.get_log_probs(
                    model=self.actor,
                    sequences=padded_seqs,
                    actions=padded_actions,
                    prompt_length=max_seq_len - max_gen_len,
                )  # [mb, T]

                # --- Compute new values from critic ----------------------
                new_values = self.critic(
                    input_ids=padded_seqs,
                    attention_mask=padded_attn,
                )  # [mb, S_total]
                # Slice values to the generated portion.
                prompt_len = max_seq_len - max_gen_len
                gen_values = new_values[:, prompt_len: prompt_len + max_gen_len]
                # Clamp to match padded_rets shape in case of off-by-one.
                gen_values = gen_values[:, :max_gen_len]

                # --- Compute ratio & clipped objective -------------------
                # ratio = exp(log π_new - log π_old)
                log_ratio = new_log_probs - padded_old_lp
                ratio = torch.exp(log_ratio)

                # Whiten advantages within the mini-batch for stability.
                mb_mask = action_mask.bool()
                flat_advs = padded_advs[mb_mask]
                if flat_advs.numel() > 1:
                    adv_mean = flat_advs.mean()
                    adv_std = flat_advs.std().clamp(min=1e-8)
                    whitened_advs = (padded_advs - adv_mean) / adv_std
                else:
                    whitened_advs = padded_advs

                # L^CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)
                surr1 = ratio * whitened_advs
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                1.0 + self.config.clip_epsilon)
                    * whitened_advs
                )
                pg_loss = -torch.min(surr1, surr2)

                # Mask to valid positions.
                masked_pg_loss = _masked_mean(pg_loss, mb_mask)

                # --- Value loss: 0.5 * MSE(V_pred, R) -------------------
                vf_loss = 0.5 * (gen_values - padded_rets) ** 2
                masked_vf_loss = _masked_mean(vf_loss, mb_mask)

                # --- KL divergence penalty -------------------------------
                # KL(π_old || π_new) ≈ π_old * (log π_old - log π_new)
                kl_div = padded_old_lp * (padded_old_lp - new_log_probs)
                masked_kl = _masked_mean(kl_div, mb_mask)

                # Approximate KL:  mean((ratio - 1) - log(ratio))
                approx_kl_val = _masked_mean(
                    (ratio - 1.0) - log_ratio, mb_mask
                )

                # --- Total loss ------------------------------------------
                loss = (
                    masked_pg_loss
                    + self.config.value_coeff * masked_vf_loss
                    + self.kl_coeff * masked_kl
                )

                # --- Back-propagation (actor) ----------------------------
                self.actor_optimizer.zero_grad()
                # Actor gradients from policy + kl terms.
                actor_loss = (
                    -masked_pg_loss + self.kl_coeff * masked_kl
                ).detach().requires_grad_(True)
                # We recompute the full graph for a clean backward.
                self.actor_optimizer.zero_grad()
                actor_full_loss = (
                    -torch.min(surr1, surr2) * mb_mask.float()
                    + self.kl_coeff * kl_div * mb_mask.float()
                ).sum() / mb_mask.sum().clamp(min=1.0)
                actor_full_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm,
                )
                self.actor_optimizer.step()

                # --- Back-propagation (critic) ---------------------------
                self.critic_optimizer.zero_grad()
                critic_loss = (
                    0.5 * (gen_values - padded_rets) ** 2 * mb_mask.float()
                ).sum() / mb_mask.sum().clamp(min=1.0)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm,
                )
                self.critic_optimizer.step()

                # --- Accumulate stats ------------------------------------
                with torch.no_grad():
                    clip_frac = _masked_mean(
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                        .float(),
                        mb_mask,
                    )
                    accept_rate = _masked_mean(
                        (torch.abs(ratio - 1.0) <= self.config.clip_epsilon)
                        .float(),
                        mb_mask,
                    )

                epoch_policy_loss += masked_pg_loss.item()
                epoch_value_loss += masked_vf_loss.item()
                epoch_kl += masked_kl.item()
                epoch_clip_frac += clip_frac.item()
                epoch_approx_kl += approx_kl_val.item()
                epoch_n += 1

                # Collect for global stats.
                for j in range(len(mb_indices)):
                    r_j = mb_rewards[j]
                    a_j = mb_advantages[j]
                    ret_j = mb_returns[j]
                    v_j = gen_values[j]
                    all_rewards.append(r_j.mean().item())
                    all_advantages.append(a_j.mean().item())
                    all_returns.append(ret_j.mean().item())
                    all_values_pred.append(v_j.mean().item())

            # Epoch averages.
            if epoch_n > 0:
                total_policy_loss += epoch_policy_loss / epoch_n
                total_value_loss += epoch_value_loss / epoch_n
                total_kl += epoch_kl / epoch_n
                total_clip_frac += epoch_clip_frac / epoch_n
                total_approx_kl += epoch_approx_kl / epoch_n
                num_updates += 1

        # --- Aggregate stats ---------------------------------------------
        n = max(num_updates, 1)
        mean_reward = float(sum(all_rewards) / max(len(all_rewards), 1))
        mean_adv = float(sum(all_advantages) / max(len(all_advantages), 1))

        # Explained variance:  1 - Var(R - V) / Var(R)
        if len(all_returns) > 1:
            rets_t = torch.tensor(all_returns)
            vals_t = torch.tensor(all_values_pred)
            var_ret = rets_t.var().clamp(min=1e-8)
            explained_var = 1.0 - (rets_t - vals_t).var() / var_ret
            explained_var = float(explained_var.clamp(max=1.0))
        else:
            explained_var = 0.0

        stats = PPOTrainingStats(
            policy_loss=total_policy_loss / n,
            value_loss=total_value_loss / n,
            kl_divergence=total_kl / n,
            total_loss=(
                -total_policy_loss / n
                + self.config.value_coeff * total_value_loss / n
                + self.kl_coeff * total_kl / n
            ),
            mean_reward=mean_reward,
            mean_advantage=mean_adv,
            clip_fraction=total_clip_frac / n,
            acceptance_rate=1.0 - total_clip_frac / n,
            approx_kl=total_approx_kl / n,
            explained_variance=explained_var,
        )
        return stats

    # -- KL adaptive controller --------------------------------------------

    def kl_adaptive_controller(self, kl_divergence: float) -> None:
        """Adaptively adjust the KL penalty coefficient.

        The controller keeps the policy close to the reference model by
        increasing the KL coefficient when divergence is too high and
        decreasing it when divergence is below the target:

        .. math::

            d_{\\text{target}} &= 0.05 \\\\
            \\beta \\leftarrow \\begin{cases}
                2\\beta & \\text{if } d > 1.5 \\cdot d_{\\text{target}} \\\\
                \\beta / 2 & \\text{if } d < d_{\\text{target}} / 1.5 \\\\
                \\beta & \\text{otherwise}
            \\end{cases} \\\\
            \\beta \\leftarrow \\text{clamp}(\\beta,\\; 0.001,\\; 10.0)

        Parameters
        ----------
        kl_divergence : float
            The most recently observed KL divergence between the current
            policy and the reference model.
        """
        if kl_divergence > 1.5 * self.config.kl_target:
            self.kl_coeff *= 2.0
        elif kl_divergence < self.config.kl_target / 1.5:
            self.kl_coeff /= 2.0
        self.kl_coeff = max(0.001, min(self.kl_coeff, 10.0))

    # -- train step --------------------------------------------------------

    def train_step(
        self,
        prompts: torch.Tensor,
        prompt_attention_masks: torch.Tensor,
    ) -> PPOTrainingStats:
        """Execute one complete PPO training iteration.

        This is the top-level entry point that runs:

        1. :meth:`rollout_phase` — generate and score rollouts.
        2. :meth:`compute_gae_advantages` — estimate advantages via GAE.
        3. :meth:`update_phase` — optimize actor and critic.
        4. :meth:`kl_adaptive_controller` — adapt KL penalty.

        Parameters
        ----------
        prompts : torch.Tensor
            Batch of prompt token ids, shape ``[B, S_p]``.
        prompt_attention_masks : torch.Tensor
            Attention mask for the prompts, shape ``[B, S_p]``.

        Returns
        -------
        PPOTrainingStats
            Training statistics for this iteration.
        """
        # 1. Rollout: generate responses, compute rewards, store in buffer.
        self.rollout_phase(prompts, prompt_attention_masks)

        # 2. Compute GAE advantages and returns.
        self.compute_gae_advantages()

        # 3. PPO update phase.
        stats = self.update_phase()

        # 4. Adapt KL coefficient.
        self.kl_adaptive_controller(stats.kl_divergence)

        return stats
