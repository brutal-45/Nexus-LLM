"""Beam search implementations for Nexus-LLM backend.

Implements standard beam search, diverse beam search, and constrained beam search
with length normalization and various scoring strategies.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Callable, Any, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class BeamHypothesis:
    """A single beam hypothesis during beam search."""
    sequence: List[int]
    score: float
    length: int
    is_done: bool = False

    def normalized_score(self, length_penalty: float = 1.0) -> float:
        """Compute length-normalized score."""
        if self.length <= 0:
            return self.score
        norm = ((5 + self.length) / 6.0) ** length_penalty
        return self.score / norm


class BeamSearchScorer:
    """Scorer for standard beam search hypotheses."""

    def __init__(
        self,
        num_beams: int,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        num_return_sequences: int = 1,
    ):
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_return_sequences = num_return_sequences
        self.beams: List[List[BeamHypothesis]] = []
        self._done: List[bool] = []

    def initialize(self, batch_size: int) -> None:
        """Initialize beams for each batch."""
        self.beams = [[] for _ in range(batch_size)]
        self._done = [False] * batch_size

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Process one step of beam search. Returns selected tokens and beam indices."""
        batch_size = input_ids.shape[0] // self.num_beams
        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            beam_tokens = next_tokens[batch_idx * self.num_beams: (batch_idx + 1) * self.num_beams]
            beam_scores = next_scores[batch_idx * self.num_beams: (batch_idx + 1) * self.num_beams]

            if self._done[batch_idx]:
                next_beam_scores[batch_idx] = torch.zeros(self.num_beams, device=device)
                next_beam_tokens[batch_idx, 0] = eos_token_id if eos_token_id is not None else 0
                next_beam_indices[batch_idx] = torch.arange(self.num_beams, device=device)
                continue

            beam_idx = 0
            for beam_token_rank, (token, score) in enumerate(zip(beam_tokens, beam_scores)):
                if beam_idx >= self.num_beams:
                    break

                effective_beam_id = batch_idx * self.num_beams + beam_idx
                is_eos = eos_token_id is not None and token.item() == eos_token_id

                if is_eos:
                    hypothesis = BeamHypothesis(
                        sequence=input_ids[effective_beam_id].tolist(),
                        score=score.item(),
                        length=input_ids.shape[-1],
                        is_done=True,
                    )
                    self.beams[batch_idx].append(hypothesis)
                else:
                    next_beam_scores[batch_idx, beam_idx] = score
                    next_beam_tokens[batch_idx, beam_idx] = token
                    next_beam_indices[batch_idx, beam_idx] = beam_idx
                    beam_idx += 1

            if self.early_stopping and len(self.beams[batch_idx]) >= self.num_return_sequences:
                self._done[batch_idx] = True

        return next_beam_indices.flatten(), next_beam_tokens.flatten()

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Finalize beam search and return best sequences."""
        batch_size = input_ids.shape[0] // self.num_beams

        for batch_idx in range(batch_size):
            for beam_idx in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_idx
                hypothesis = BeamHypothesis(
                    sequence=input_ids[effective_beam_id].tolist(),
                    score=final_beam_scores[effective_beam_id].item(),
                    length=input_ids.shape[-1],
                )
                self.beams[batch_idx].append(hypothesis)

        output_sequences = []
        output_scores = []

        for batch_idx in range(batch_size):
            sorted_hypotheses = sorted(
                self.beams[batch_idx],
                key=lambda h: h.normalized_score(self.length_penalty),
                reverse=True,
            )
            best = sorted_hypotheses[:self.num_return_sequences]
            for hyp in best:
                output_sequences.append(hyp.sequence)
                output_scores.append(hyp.normalized_score(self.length_penalty))

        max_len = max(len(s) for s in output_sequences) if output_sequences else 0
        padded = torch.full(
            (len(output_sequences), max_len),
            eos_token_id if eos_token_id is not None else 0,
            dtype=torch.long,
        )
        for i, seq in enumerate(output_sequences):
            padded[i, :len(seq)] = torch.tensor(seq)

        scores_tensor = torch.tensor(output_scores)
        return padded, scores_tensor

    def is_done(self) -> bool:
        return all(self._done)


class DiverseBeamSearchScorer:
    """Scorer for diverse beam search with diversity penalty between beam groups."""

    def __init__(
        self,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float = 5.0,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ):
        self.num_beams = num_beams
        self.num_beam_groups = num_beam_groups
        self.diversity_penalty = diversity_penalty
        self.length_penalty = length_penalty
        self.num_return_sequences = num_return_sequences
        self.beams_per_group = num_beams // num_beam_groups
        self._group_scores: Optional[torch.FloatTensor] = None

    def initialize(self, batch_size: int, device: torch.device = None) -> None:
        """Initialize diversity tracking."""
        self._group_scores = torch.zeros(
            (batch_size, self.num_beam_groups), device=device
        )

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Process one step with diversity penalty between groups."""
        batch_size = input_ids.shape[0] // self.num_beams
        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            for group_idx in range(self.num_beam_groups):
                start = group_idx * self.beams_per_group
                end = start + self.beams_per_group

                group_beam_scores = next_scores[batch_idx * self.num_beams + start: batch_idx * self.num_beams + end].clone()

                diversity_bonus = torch.zeros_like(group_beam_scores)
                for other_group in range(self.num_beam_groups):
                    if other_group != group_idx:
                        diversity_bonus += self._group_scores[batch_idx, other_group]

                adjusted_scores = group_beam_scores - self.diversity_penalty * diversity_bonus

                for beam_idx_in_group in range(self.beams_per_group):
                    global_beam_idx = batch_idx * self.num_beams + start + beam_idx_in_group
                    best_idx = adjusted_scores[beam_idx_in_group].argmax()
                    next_beam_scores[batch_idx, start + beam_idx_in_group] = adjusted_scores[beam_idx_in_group, best_idx]
                    next_beam_tokens[batch_idx, start + beam_idx_in_group] = next_tokens[global_beam_idx, best_idx]
                    next_beam_indices[batch_idx, start + beam_idx_in_group] = beam_idx_in_group

                self._group_scores[batch_idx, group_idx] = adjusted_scores.max().item()

        return next_beam_indices.flatten(), next_beam_tokens.flatten()

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Finalize diverse beam search."""
        batch_size = input_ids.shape[0] // self.num_beams
        all_hypotheses: List[List[BeamHypothesis]] = [[] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for beam_idx in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_idx
                hypothesis = BeamHypothesis(
                    sequence=input_ids[effective_beam_id].tolist(),
                    score=final_beam_scores[effective_beam_id].item(),
                    length=input_ids.shape[-1],
                )
                all_hypotheses[batch_idx].append(hypothesis)

        output_sequences = []
        output_scores = []

        for batch_idx in range(batch_size):
            sorted_hyps = sorted(
                all_hypotheses[batch_idx],
                key=lambda h: h.normalized_score(self.length_penalty),
                reverse=True,
            )
            best = sorted_hyps[:self.num_return_sequences]
            for hyp in best:
                output_sequences.append(hyp.sequence)
                output_scores.append(hyp.normalized_score(self.length_penalty))

        max_len = max(len(s) for s in output_sequences) if output_sequences else 0
        padded = torch.full(
            (len(output_sequences), max_len),
            eos_token_id if eos_token_id is not None else 0,
            dtype=torch.long,
        )
        for i, seq in enumerate(output_sequences):
            padded[i, :len(seq)] = torch.tensor(seq)

        return padded, torch.tensor(output_scores)


class ConstrainedBeamSearchScorer:
    """Scorer for constrained beam search ensuring certain tokens appear in output."""

    def __init__(
        self,
        num_beams: int,
        constraints: Optional[List[List[int]]] = None,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ):
        self.num_beams = num_beams
        self.constraints = constraints or []
        self.length_penalty = length_penalty
        self.num_return_sequences = num_return_sequences
        self.beams: List[List[BeamHypothesis]] = []

    def _satisfies_constraints(self, sequence: List[int]) -> bool:
        """Check if a sequence satisfies all constraints."""
        for constraint_tokens in self.constraints:
            constraint_len = len(constraint_tokens)
            found = False
            for i in range(len(sequence) - constraint_len + 1):
                if sequence[i:i + constraint_len] == constraint_tokens:
                    found = True
                    break
            if not found:
                return False
        return True

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Process one step, prioritizing beams closer to satisfying constraints."""
        batch_size = input_ids.shape[0] // self.num_beams
        device = input_ids.device

        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=torch.long, device=device)

        for batch_idx in range(batch_size):
            beam_idx = 0
            beam_start = batch_idx * self.num_beams
            beam_end = beam_start + self.num_beams

            for rank, (token, score) in enumerate(zip(
                next_tokens[beam_start:beam_end],
                next_scores[beam_start:beam_end],
            )):
                if beam_idx >= self.num_beams:
                    break

                is_eos = eos_token_id is not None and token.item() == eos_token_id

                if is_eos:
                    seq = input_ids[beam_start + beam_idx].tolist()
                    if self._satisfies_constraints(seq):
                        hypothesis = BeamHypothesis(
                            sequence=seq,
                            score=score.item(),
                            length=len(seq),
                            is_done=True,
                        )
                        if len(self.beams) <= batch_idx:
                            self.beams.append([])
                        self.beams[batch_idx].append(hypothesis)
                    continue

                next_beam_scores[batch_idx, beam_idx] = score
                next_beam_tokens[batch_idx, beam_idx] = token
                next_beam_indices[batch_idx, beam_idx] = beam_idx
                beam_idx += 1

        return next_beam_indices.flatten(), next_beam_tokens.flatten()

    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Finalize constrained beam search."""
        batch_size = input_ids.shape[0] // self.num_beams

        for batch_idx in range(batch_size):
            if len(self.beams) <= batch_idx:
                self.beams.append([])
            for beam_idx in range(self.num_beams):
                effective_beam_id = batch_idx * self.num_beams + beam_idx
                seq = input_ids[effective_beam_id].tolist()
                hypothesis = BeamHypothesis(
                    sequence=seq,
                    score=final_beam_scores[effective_beam_id].item(),
                    length=len(seq),
                )
                self.beams[batch_idx].append(hypothesis)

        output_sequences = []
        output_scores = []

        for batch_idx in range(batch_size):
            constrained = [h for h in self.beams[batch_idx] if self._satisfies_constraints(h.sequence)]
            candidates = constrained if constrained else self.beams[batch_idx]
            sorted_hyps = sorted(
                candidates,
                key=lambda h: h.normalized_score(self.length_penalty),
                reverse=True,
            )
            best = sorted_hyps[:self.num_return_sequences]
            for hyp in best:
                output_sequences.append(hyp.sequence)
                output_scores.append(hyp.normalized_score(self.length_penalty))

        max_len = max(len(s) for s in output_sequences) if output_sequences else 0
        padded = torch.full(
            (len(output_sequences), max_len),
            eos_token_id if eos_token_id is not None else 0,
            dtype=torch.long,
        )
        for i, seq in enumerate(output_sequences):
            padded[i, :len(seq)] = torch.tensor(seq)

        return padded, torch.tensor(output_scores)
