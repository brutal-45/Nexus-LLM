"""Preference dataset for alignment training.

Provides :class:`PreferenceDataset` — a container for preference pairs
(chosen vs. rejected responses) used in DPO and RLHF training.
Supports programmatic construction, JSONL serialisation, and batched
sampling.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreferenceSample:
    """A single preference comparison sample.

    Attributes:
        prompt: The shared prompt / context.
        chosen: The preferred (chosen) response.
        rejected: The dis-preferred (rejected) response.
        metadata: Optional metadata (e.g. annotator ID, timestamp).
    """

    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceSample":
        """Deserialise from a plain dictionary."""
        return cls(
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            metadata=data.get("metadata", {}),
        )


class PreferenceDataset:
    """Container and sampler for preference-aligned training data.

    Stores a collection of :class:`PreferenceSample` instances and
    provides methods for adding samples, random batched sampling, and
    JSONL-based persistence.

    Usage::

        ds = PreferenceDataset()
        ds.add_sample("What is AI?", "AI is ...", "AI means ...")
        batch = ds.get_batch(2)
        ds.to_jsonl("prefs.jsonl")

        ds2 = PreferenceDataset.from_jsonl("prefs.jsonl")
        print(ds2.size())
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialise an empty PreferenceDataset.

        Args:
            seed: Optional random seed for reproducible batch sampling.
        """
        self._samples: List[PreferenceSample] = []
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_sample(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a preference sample to the dataset.

        Args:
            prompt: The shared prompt / context.
            chosen: The preferred response.
            rejected: The dis-preferred response.
            metadata: Optional key-value metadata.
        """
        sample = PreferenceSample(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            metadata=metadata or {},
        )
        self._samples.append(sample)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def get_batch(self, n: int) -> List[PreferenceSample]:
        """Randomly sample *n* preference pairs (with replacement).

        If *n* exceeds the dataset size, the entire dataset is
        returned (shuffled).

        Args:
            n: Number of samples to retrieve.

        Returns:
            List of PreferenceSample instances.
        """
        if n >= len(self._samples):
            batch = list(self._samples)
            self._rng.shuffle(batch)
            return batch

        indices = self._rng.choices(range(len(self._samples)), k=n)
        return [self._samples[i] for i in indices]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)

    def __len__(self) -> int:
        return self.size()

    def __getitem__(self, index: int) -> PreferenceSample:
        return self._samples[index]

    def __iter__(self):
        return iter(self._samples)

    @property
    def samples(self) -> List[PreferenceSample]:
        """Read-only view of all samples."""
        return list(self._samples)

    # ------------------------------------------------------------------
    # JSONL persistence
    # ------------------------------------------------------------------

    def to_jsonl(self, path: str) -> None:
        """Write the dataset to a JSONL file.

        Each line is a JSON object with keys ``prompt``, ``chosen``,
        ``rejected``, and optionally ``metadata``.

        Args:
            path: Destination file path.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for sample in self._samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        logger.info("Wrote %d samples to %s", len(self._samples), path)

    @classmethod
    def from_jsonl(cls, path: str, seed: Optional[int] = None) -> "PreferenceDataset":
        """Load a PreferenceDataset from a JSONL file.

        Args:
            path: Source JSONL file path.
            seed: Optional random seed for the new dataset.

        Returns:
            A populated PreferenceDataset.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If a line contains invalid JSON.
        """
        dataset = cls(seed=seed)
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = PreferenceSample.from_dict(data)
                    dataset._samples.append(sample)
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed line %d in %s: %s",
                        line_no,
                        path,
                        exc,
                    )

        logger.info("Loaded %d samples from %s", dataset.size(), path)
        return dataset
