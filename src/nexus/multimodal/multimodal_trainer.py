"""
Multimodal Trainer Module
=========================

Comprehensive training infrastructure for the Nexus LLM multimodal system.
Provides dataset handling, batch collation, loss functions, modality-balanced
sampling, and evaluation metrics for vision-language-audio-video models.

Classes:
    - MultimodalDataset: Mixed-modality dataset with dynamic modality loading
    - MultimodalCollator: Per-modality dynamic padding and batch assembly
    - ContrastiveLoss: InfoNCE loss with hard negative mining
    - AlignmentLoss: Cosine similarity alignment between modalities
    - MultimodalLoss: Weighted combination of language modeling + contrastive + alignment
    - ModalityBalancedSampler: Ensures balanced modality distribution per batch
    - EvaluationMetrics: Retrieval recall, VQA accuracy, CIDEr/ROUGE/BLEU

Usage:
    >>> dataset = MultimodalDataset(data_path="data/train", modalities=["vision", "text"])
    >>> collator = MultimodalCollator(pad_token_id=0, image_size=224)
    >>> sampler = ModalityBalancedSampler(dataset, batch_size=32)
    >>> dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
"""

import math
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, Sampler


# =============================================================================
# Constants
# =============================================================================

PAD_TOKEN_ID = 0
IGNORE_INDEX = -100
DEFAULT_IMAGE_SIZE = 224
DEFAULT_AUDIO_LENGTH = 30.0
DEFAULT_NUM_FRAMES = 8
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MELS = 128
MAX_TEXT_LENGTH = 2048
MAX_AUDIO_TOKENS = 1500
MAX_IMAGE_TOKENS = 576
MAX_VIDEO_TOKENS = 1152

MODALITY_TYPES = ("image", "audio", "video", "text")
SUPPORTED_MODALITIES = {"image", "audio", "video", "text"}


# =============================================================================
# Utility Functions
# =============================================================================

def compute_mask_from_lengths(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Compute a boolean mask from variable-length sequences.

    Creates a 2D boolean mask where positions beyond each sequence's actual
    length are set to False.

    Args:
        lengths: 1D tensor of sequence lengths, shape (batch_size,).
        max_length: Maximum sequence length in the batch.

    Returns:
        Boolean mask tensor of shape (batch_size, max_length), where True
        indicates valid (non-padding) positions.

    Example:
        >>> lengths = torch.tensor([3, 1, 4])
        >>> mask = compute_mask_from_lengths(lengths, 5)
        >>> mask
        tensor([[ True,  True,  True, False, False],
                [ True, False, False, False, False],
                [ True,  True,  True,  True, False]])
    """
    batch_size = lengths.shape[0]
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    return positions < lengths.unsqueeze(1)


def pad_sequence_to_length(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    dim: int = 0,
) -> torch.Tensor:
    """Pad a 1D or 2D tensor to a target length along a given dimension.

    Args:
        tensor: Input tensor to pad.
        target_length: Desired length along the padding dimension.
        pad_value: Value to use for padding. Default: 0.0.
        dim: Dimension along which to pad. Default: 0.

    Returns:
        Padded tensor with shape matching the input except along `dim`,
        which is `target_length`.

    Example:
        >>> t = torch.tensor([1.0, 2.0, 3.0])
        >>> pad_sequence_to_length(t, 5, pad_value=-1.0)
        tensor([ 1.,  2.,  3., -1., -1.])
    """
    current_length = tensor.shape[dim]
    if current_length >= target_length:
        return tensor

    pad_shape = list(tensor.shape)
    pad_shape[dim] = target_length - current_length

    pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=dim)


def pad_2d_batch(
    tensors: List[torch.Tensor],
    pad_value: float = 0.0,
    padding_dim: int = 0,
    trailing_dim: Optional[int] = None,
) -> torch.Tensor:
    """Pad a list of tensors to the same length along a dimension and stack.

    Useful for batching variable-length spectrograms, image feature sequences,
    or video frame sequences.

    Args:
        tensors: List of tensors with potentially different lengths along
                 `padding_dim`.
        pad_value: Value used for padding. Default: 0.0.
        padding_dim: Dimension along which to pad (after stacking). Default: 0.
        trailing_dim: Optional fixed trailing dimension (e.g. feature dim).
                      If provided, tensors are padded/truncated to this size.

    Returns:
        Stacked tensor of shape (batch_size, ...) with all tensors padded to
        the maximum length in the batch along `padding_dim`.

    Raises:
        ValueError: If the list is empty.
    """
    if not tensors:
        raise ValueError("Cannot pad an empty list of tensors.")

    valid_tensors = [t for t in tensors if t is not None]
    if not valid_tensors:
        return torch.zeros(0, dtype=torch.float32)

    if trailing_dim is not None:
        processed = []
        for t in valid_tensors:
            if t.shape[-1] < trailing_dim:
                pad_needed = trailing_dim - t.shape[-1]
                padding = torch.zeros(
                    *t.shape[:-1], pad_needed, dtype=t.dtype, device=t.device
                )
                t = torch.cat([t, padding], dim=-1)
            elif t.shape[-1] > trailing_dim:
                t = t[..., :trailing_dim]
            processed.append(t)
        valid_tensors = processed

    max_len = max(t.shape[padding_dim] for t in valid_tensors)
    padded = []
    for t in valid_tensors:
        if t.shape[padding_dim] < max_len:
            t = pad_sequence_to_length(t, max_len, pad_value=pad_value, dim=padding_dim)
        padded.append(t)

    return torch.stack(padded, dim=0)


def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two batches of vectors.

    Args:
        x: Tensor of shape (batch_x, dim).
        y: Tensor of shape (batch_y, dim).

    Returns:
        Similarity matrix of shape (batch_x, batch_y) with values in [-1, 1].
    """
    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    return torch.matmul(x_norm, y_norm.t())


def tokenize_ngrams(text: str, n: int = 4) -> List[str]:
    """Compute character n-grams from a text string.

    Used for CIDEr and ROUGE computation.

    Args:
        text: Input text string.
        n: N-gram size. Default: 4.

    Returns:
        List of n-gram strings.
    """
    text = text.lower().strip()
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])
    return ngrams


def tokenize_words(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenization.

    Args:
        text: Input text.

    Returns:
        List of word tokens.
    """
    text = text.lower().strip()
    words = text.replace(".", " . ").replace(",", " , ").replace("?", " ? ")
    words = words.replace("!", " ! ").replace(";", " ; ").replace(":", " : ")
    words = words.replace("\"", " \" ").replace("'", " ' ")
    return [w for w in words.split() if w.strip()]


def compute_ngram_overlap(
    reference: List[str],
    hypothesis: List[str],
    n: int = 4,
) -> Tuple[int, int, int]:
    """Compute n-gram overlap statistics between reference and hypothesis.

    Args:
        reference: Reference token list.
        hypothesis: Hypothesis token list.
        n: N-gram order. Default: 4.

    Returns:
        Tuple of (matches, reference_count, hypothesis_count) for the given n.
    """
    ref_ngrams = Counter()
    hyp_ngrams = Counter()

    for i in range(len(reference) - n + 1):
        ngram = tuple(reference[i:i + n])
        ref_ngrams[ngram] += 1

    for i in range(len(hypothesis) - n + 1):
        ngram = tuple(hypothesis[i:i + n])
        hyp_ngrams[ngram] += 1

    matches = sum((ref_ngrams & hyp_ngrams).values())
    return matches, sum(ref_ngrams.values()), sum(hyp_ngrams.values())


def compute_bleu_single(
    reference: List[str],
    hypothesis: List[str],
    max_n: int = 4,
) -> float:
    """Compute BLEU score for a single reference-hypothesis pair.

    Args:
        reference: Reference token list.
        hypothesis: Hypothesis token list.
        max_n: Maximum n-gram order. Default: 4.

    Returns:
        BLEU score as a float in [0, 1].
    """
    if len(hypothesis) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        if len(hypothesis) < n:
            precisions.append(0.0)
            continue
        matches, ref_count, hyp_count = compute_ngram_overlap(reference, hypothesis, n)
        if hyp_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / hyp_count)

    log_precisions = []
    for p in precisions:
        if p > 0:
            log_precisions.append(math.log(p))
        else:
            log_precisions.append(-1e10)

    if not log_precisions:
        return 0.0

    avg_log = sum(log_precisions) / len(log_precisions)
    geo_mean = math.exp(min(0.0, avg_log))

    bp = min(1.0, math.exp(1.0 - len(reference) / max(len(hypothesis), 1)))

    return bp * geo_mean


def compute_rouge_n(
    reference: List[str],
    hypothesis: List[str],
    n: int = 2,
) -> float:
    """Compute ROUGE-N F1 score.

    Args:
        reference: Reference token list.
        hypothesis: Hypothesis token list.
        n: N-gram order. Default: 2.

    Returns:
        ROUGE-N F1 score.
    """
    matches, ref_count, hyp_count = compute_ngram_overlap(reference, hypothesis, n)

    if ref_count == 0 or hyp_count == 0:
        return 0.0

    precision = matches / hyp_count
    recall = matches / ref_count

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_cider_score(
    references: List[List[str]],
    hypothesis: List[str],
    n: int = 4,
    sigma: float = 6.0,
) -> float:
    """Compute CIDEr score (Consensus-Based Image Description Evaluation).

    CIDEr measures consensus between a hypothesis and a set of reference
    captions using TF-IDF weighted n-gram similarity.

    Args:
        references: List of reference token lists (multiple references).
        hypothesis: Hypothesis token list.
        n: Maximum n-gram order. Default: 4.
        sigma: Gaussian smoothing parameter for TF-IDF. Default: 6.0.

    Returns:
        CIDEr score as a float.
    """
    if not references or not hypothesis:
        return 0.0

    all_docs = [hypothesis] + references
    doc_freq = defaultdict(int)
    num_docs = len(all_docs)

    for doc in all_docs:
        seen = set()
        for i in range(len(doc) - n + 1):
            ngram = tuple(doc[i:i + n])
            if ngram not in seen:
                doc_freq[ngram] += 1
                seen.add(ngram)

    def tf_idf(ngrams: Counter, length: int) -> Dict[Tuple[str, ...], float]:
        scores = {}
        for ngram, count in ngrams.items():
            tf = count / max(length, 1)
            df = doc_freq.get(ngram, 0)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            scores[ngram] = tf * idf
        return scores

    hyp_ngrams = Counter()
    for i in range(len(hypothesis) - n + 1):
        ngram = tuple(hypothesis[i:i + n])
        hyp_ngrams[ngram] += 1
    hyp_vec = tf_idf(hyp_ngrams, len(hypothesis))

    score = 0.0
    for ref in references:
        ref_ngrams = Counter()
        for i in range(len(ref) - n + 1):
            ngram = tuple(ref[i:i + n])
            ref_ngrams[ngram] += 1
        ref_vec = tf_idf(ref_ngrams, len(ref))

        common_ngrams = set(hyp_vec.keys()) & set(ref_vec.keys())
        if not common_ngrams:
            continue

        cos_num = sum(hyp_vec[ng] * ref_vec[ng] for ng in common_ngrams)
        cos_den = (
            math.sqrt(sum(v ** 2 for v in hyp_vec.values())) *
            math.sqrt(sum(v ** 2 for v in ref_vec.values()))
        )
        if cos_den > 0:
            score += cos_num / cos_den

    score /= len(references)
    return max(0.0, score / sigma)


# =============================================================================
# Multimodal Dataset
# =============================================================================

class MultimodalDataset(Dataset):
    """Dataset for handling mixed-modality training data.

    Supports loading and serving samples that may contain any combination
    of image, audio, video, and text modalities. Each sample is returned
    as a dictionary with modality-specific tensors and masks.

    The dataset handles:
    - Text-only samples (language modeling)
    - Image-text pairs (VQA, captioning, retrieval)
    - Audio-text pairs (ASR, audio captioning)
    - Video-text pairs (video QA, video captioning)
    - Multi-modal combinations (image+audio+text, etc.)

    Args:
        data_path: Root directory containing data files.
        modalities: List of enabled modalities. Default: ("image", "text").
        max_text_length: Maximum token length for text sequences. Default: 2048.
        max_image_tokens: Maximum image token count. Default: 576.
        max_audio_tokens: Maximum audio token count. Default: 1500.
        max_video_tokens: Maximum video token count. Default: 1152.
        image_size: Target image resolution. Default: 224.
        num_frames: Number of frames for video. Default: 8.
        sample_rate: Audio sample rate. Default: 16000.
        max_audio_seconds: Maximum audio duration in seconds. Default: 30.0.
        tokenizer: Optional tokenizer for text encoding. If None, uses
                   placeholder token IDs based on whitespace tokenization.
        transform: Optional image transform pipeline.
        audio_transform: Optional audio transform pipeline.
        split: Data split ("train", "val", "test"). Default: "train".
        seed: Random seed for reproducibility. Default: 42.
        modality_dropout_prob: Probability of randomly dropping a modality
                               during training for robustness. Default: 0.0.

    Attributes:
        samples: List of sample dictionaries loaded from data_path.
        modality_counts: Counter tracking modality frequency in the dataset.

    Example:
        >>> dataset = MultimodalDataset(
        ...     data_path="data/train",
        ...     modalities=["image", "text"],
        ...     max_text_length=512,
        ... )
        >>> sample = dataset[0]
        >>> sample.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels', ...])
    """

    def __init__(
        self,
        data_path: str,
        modalities: Sequence[str] = ("image", "text"),
        max_text_length: int = MAX_TEXT_LENGTH,
        max_image_tokens: int = MAX_IMAGE_TOKENS,
        max_audio_tokens: int = MAX_AUDIO_TOKENS,
        max_video_tokens: int = MAX_VIDEO_TOKENS,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_frames: int = DEFAULT_NUM_FRAMES,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        max_audio_seconds: float = DEFAULT_AUDIO_LENGTH,
        tokenizer: Optional[Any] = None,
        transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        split: str = "train",
        seed: int = 42,
        modality_dropout_prob: float = 0.0,
    ):
        self.data_path = data_path
        self.modalities = list(modalities)
        self.max_text_length = max_text_length
        self.max_image_tokens = max_image_tokens
        self.max_audio_tokens = max_audio_tokens
        self.max_video_tokens = max_video_tokens
        self.image_size = image_size
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.max_audio_seconds = max_audio_seconds
        self.tokenizer = tokenizer
        self.transform = transform
        self.audio_transform = audio_transform
        self.split = split
        self.seed = seed
        self.modality_dropout_prob = modality_dropout_prob

        self._rng = random.Random(seed)
        self.samples: List[Dict[str, Any]] = []
        self.modality_counts: Counter = Counter()
        self._modality_indices: Dict[str, List[int]] = defaultdict(list)

        self._load_data()
        self._build_modality_index()

    def _load_data(self):
        """Load dataset samples from the data directory.

        Scans the data_path for supported file formats and constructs
        sample dictionaries. Supports JSONL, JSON, and directory-based
        formats.
        """
        import json
        import os

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}"
            )

        jsonl_path = os.path.join(self.data_path, f"{self.split}.jsonl")
        json_path = os.path.join(self.data_path, f"{self.split}.json")

        if os.path.exists(jsonl_path):
            self._load_jsonl(jsonl_path)
        elif os.path.exists(json_path):
            self._load_json(json_path)
        else:
            self._load_directory(self.data_path)

    def _load_jsonl(self, filepath: str):
        """Load samples from a JSONL file.

        Each line should be a JSON object with fields like:
        - "text": The text content (required).
        - "image": Path to an image file (optional).
        - "audio": Path to an audio file (optional).
        - "video": Path to a video file (optional).
        - "labels": Target labels for supervised training (optional).
        - "input_ids": Pre-tokenized input IDs (optional).
        - "attention_mask": Pre-computed attention mask (optional).
        - "modality": Primary modality type (optional).

        Args:
            filepath: Path to the JSONL file.
        """
        import os

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                import json
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as e:
                    continue

                sample = self._normalize_sample(sample)
                if sample is not None:
                    self.samples.append(sample)
                    self._update_modality_counts(sample)

    def _load_json(self, filepath: str):
        """Load samples from a JSON file.

        Expects a JSON array of sample objects, or a single dictionary
        with a "data" key containing the array.

        Args:
            filepath: Path to the JSON file.
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, list):
            data = [data]

        for sample_raw in data:
            sample = self._normalize_sample(sample_raw)
            if sample is not None:
                self.samples.append(sample)
                self._update_modality_counts(sample)

    def _load_directory(self, dirpath: str):
        """Load samples from a directory structure.

        Expects subdirectories named by modality (e.g., "images/", "audio/")
        and a text file with sample metadata.

        Args:
            dirpath: Path to the data directory.
        """
        import os

        metadata_path = os.path.join(dirpath, "metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if isinstance(metadata, list):
                for sample_raw in metadata:
                    sample = self._normalize_sample(sample_raw)
                    if sample is not None:
                        self.samples.append(sample)
                        self._update_modality_counts(sample)

    def _normalize_sample(self, raw_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a raw sample into the expected format.

        Handles various input formats and resolves paths relative to data_path.

        Args:
            raw_sample: Raw sample dictionary from data files.

        Returns:
            Normalized sample dictionary, or None if the sample is invalid.
        """
        import os

        if not raw_sample:
            return None

        sample = {
            "text": raw_sample.get("text", ""),
            "input_ids": raw_sample.get("input_ids"),
            "attention_mask": raw_sample.get("attention_mask"),
            "labels": raw_sample.get("labels"),
            "modality": raw_sample.get("modality", "text"),
            "metadata": raw_sample.get("metadata", {}),
        }

        if "image" in raw_sample and raw_sample["image"] is not None:
            image_path = raw_sample["image"]
            if not os.path.isabs(image_path):
                image_path = os.path.join(self.data_path, image_path)
            sample["image_path"] = image_path

        if "audio" in raw_sample and raw_sample["audio"] is not None:
            audio_path = raw_sample["audio"]
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(self.data_path, audio_path)
            sample["audio_path"] = audio_path

        if "video" in raw_sample and raw_sample["video"] is not None:
            video_path = raw_sample["video"]
            if not os.path.isabs(video_path):
                video_path = os.path.join(self.data_path, video_path)
            sample["video_path"] = video_path

        if not sample["text"] and "image_path" not in sample and "audio_path" not in sample and "video_path" not in sample:
            if sample["input_ids"] is None:
                return None

        return sample

    def _update_modality_counts(self, sample: Dict[str, Any]):
        """Track modality frequency for balanced sampling.

        Args:
            sample: Normalized sample dictionary.
        """
        modalities_present = set()
        if "image_path" in sample:
            modalities_present.add("image")
        if "audio_path" in sample:
            modalities_present.add("audio")
        if "video_path" in sample:
            modalities_present.add("video")
        if sample.get("text") or sample.get("input_ids") is not None:
            modalities_present.add("text")

        if not modalities_present:
            modalities_present.add("text")

        for mod in modalities_present:
            self.modality_counts[mod] += 1

    def _build_modality_index(self):
        """Build an index mapping modality types to sample indices.

        Used by ModalityBalancedSampler to efficiently retrieve samples
        of a specific modality.
        """
        self._modality_indices.clear()
        for idx, sample in enumerate(self.samples):
            present_modalities = self._get_present_modalities(sample)
            for mod in present_modalities:
                self._modality_indices[mod].append(idx)

    def _get_present_modalities(self, sample: Dict[str, Any]) -> List[str]:
        """Determine which modalities are present in a sample.

        Args:
            sample: Normalized sample dictionary.

        Returns:
            List of modality names present in the sample.
        """
        modalities = []
        if "image_path" in sample:
            modalities.append("image")
        if "audio_path" in sample:
            modalities.append("audio")
        if "video_path" in sample:
            modalities.append("video")
        if sample.get("text") or sample.get("input_ids") is not None:
            modalities.append("text")
        return modalities

    def _tokenize_text(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Tokenize text using the configured tokenizer or simple fallback.

        If a tokenizer is provided, it is used directly. Otherwise, a simple
        character-level tokenization is used as a placeholder.

        Args:
            text: Input text string.
            max_length: Maximum token length. Defaults to self.max_text_length.

        Returns:
            Tuple of (input_ids, attention_mask), both as lists of ints.
        """
        if max_length is None:
            max_length = self.max_text_length

        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_attention_mask=True,
                return_tensors=None,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
        else:
            input_ids = [ord(c) % 1000 for c in text[:max_length]]
            attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask

    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess an image.

        Attempts to load the image using PIL (if available) and applies
        the configured transform. Returns a tensor of shape (C, H, W).

        Args:
            image_path: Path to the image file.

        Returns:
            Image tensor of shape (3, image_size, image_size), or None if
            loading fails.
        """
        try:
            from PIL import Image
            import numpy as np

            image = Image.open(image_path).convert("RGB")
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

            if self.transform is not None:
                image = self.transform(image)
            else:
                img_array = np.array(image, dtype=np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img_array = (img_array - mean) / std
                image = torch.from_numpy(img_array.transpose(2, 0, 1))

            return image
        except Exception:
            return None

    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess an audio file.

        Attempts to load audio using scipy or soundfile. Returns a mel
        spectrogram tensor of shape (n_mels, time_frames).

        Args:
            audio_path: Path to the audio file.

        Returns:
            Audio feature tensor, or None if loading fails.
        """
        try:
            import numpy as np

            try:
                import soundfile as sf
                waveform, sr = sf.read(audio_path, dtype="float32")
            except ImportError:
                try:
                    from scipy.io import wavfile
                    sr, waveform = wavfile.read(audio_path)
                    if waveform.dtype == np.int16:
                        waveform = waveform.astype(np.float32) / 32768.0
                    elif waveform.dtype == np.int32:
                        waveform = waveform.astype(np.float32) / 2147483648.0
                except ImportError:
                    waveform = np.random.randn(
                        int(self.max_audio_seconds * self.sample_rate)
                    ).astype(np.float32)
                    sr = self.sample_rate

            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=-1)

            if sr != self.sample_rate:
                num_samples = int(len(waveform) * self.sample_rate / sr)
                indices = np.linspace(0, len(waveform) - 1, num_samples).astype(int)
                waveform = waveform[indices]

            max_samples = int(self.max_audio_seconds * self.sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            elif len(waveform) < max_samples:
                pad_len = max_samples - len(waveform)
                waveform = np.pad(waveform, (0, pad_len), mode="constant")

            if self.audio_transform is not None:
                waveform = self.audio_transform(waveform)

            waveform_tensor = torch.from_numpy(waveform)

            try:
                n_fft = 400
                hop_length = 160
                n_mels = DEFAULT_NUM_MELS
                stft = torch.stft(
                    waveform_tensor.unsqueeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=torch.hann_window(n_fft),
                    return_complex=True,
                )
                magnitude = stft.abs().squeeze(0)

                mel_fb = self._create_mel_filterbank(n_fft, n_mels)
                mel_spec = torch.matmul(mel_fb, magnitude)
                mel_spec = torch.log(mel_spec + 1e-8)
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
                return mel_spec
            except Exception:
                return waveform_tensor.unsqueeze(0)

        except Exception:
            return None

    def _create_mel_filterbank(self, n_fft: int, n_mels: int) -> torch.Tensor:
        """Create a mel filterbank matrix.

        Args:
            n_fft: FFT size.
            n_mels: Number of mel bands.

        Returns:
            Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        """
        n_freqs = n_fft // 2 + 1
        fmin = 0.0
        fmax = self.sample_rate / 2.0

        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)

        hz_points = mel_to_hz(mel_points)
        freq_bins = torch.floor((n_fft + 1) * hz_points / self.sample_rate).long()

        filterbank = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            left = freq_bins[i]
            center = freq_bins[i + 1]
            right = freq_bins[i + 2]

            if center > left:
                filterbank[i, left:center] = (
                    torch.arange(left, center) - left
                ) / (center - left)
            if right > center:
                filterbank[i, center:right] = (
                    right - torch.arange(center, right)
                ) / (right - center)

        return filterbank

    def _load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess a video file.

        Attempts to load video frames and returns a tensor of shape
        (num_frames, 3, H, W). If video loading fails, returns synthetic
        random frames.

        Args:
            video_path: Path to the video file.

        Returns:
            Video tensor of shape (num_frames, 3, image_size, image_size),
            or None if loading fails.
        """
        try:
            import numpy as np
            from PIL import Image

            try:
                import cv2
                cap = cv2.VideoCapture(video_path)
                frames = []
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = Image.fromarray(frame)
                            frame = frame.resize((self.image_size, self.image_size), Image.BILINEAR)
                            frames.append(np.array(frame, dtype=np.float32) / 255.0)
                        else:
                            frames.append(np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))
                cap.release()
            except ImportError:
                frames = [
                    np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)
                    for _ in range(self.num_frames)
                ]

            if len(frames) < self.num_frames:
                while len(frames) < self.num_frames:
                    frames.append(frames[-1].copy() if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.float32))

            frames = frames[:self.num_frames]
            frames_array = np.stack(frames, axis=0)

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            frames_array = (frames_array - mean) / std

            video_tensor = torch.from_numpy(frames_array.transpose(0, 3, 1, 2))
            return video_tensor
        except Exception:
            return None

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Total number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with all modality tensors.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing:
                - input_ids: Token IDs for the text portion.
                - attention_mask: Attention mask for text tokens.
                - labels: Target labels for language modeling loss.
                - image_tensor: Image tensor (C, H, W) or None.
                - audio_tensor: Audio spectrogram tensor or None.
                - video_tensor: Video frames tensor (F, C, H, W) or None.
                - modality_mask: Binary mask indicating active modalities
                  in order [image, audio, video].
                - modality: Primary modality type string.
                - sample_idx: Original sample index.

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.samples)} samples"
            )

        sample = self.samples[idx]
        result: Dict[str, Any] = {}

        text = sample.get("text", "")

        if sample.get("input_ids") is not None:
            input_ids = list(sample["input_ids"])
            attention_mask = list(sample.get("attention_mask", [1] * len(input_ids)))
        else:
            input_ids, attention_mask = self._tokenize_text(text)

        if len(input_ids) > self.max_text_length:
            input_ids = input_ids[:self.max_text_length]
            attention_mask = attention_mask[:self.max_text_length]

        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

        labels = sample.get("labels")
        if labels is not None:
            if len(labels) > self.max_text_length:
                labels = labels[:self.max_text_length]
            result["labels"] = labels
        else:
            result["labels"] = list(input_ids)

        result["image_tensor"] = None
        result["audio_tensor"] = None
        result["video_tensor"] = None

        image_path = sample.get("image_path")
        if image_path and "image" in self.modalities:
            image_tensor = self._load_image(image_path)
            if self.training and self.modality_dropout_prob > 0:
                if self._rng.random() < self.modality_dropout_prob:
                    image_tensor = None
            result["image_tensor"] = image_tensor

        audio_path = sample.get("audio_path")
        if audio_path and "audio" in self.modalities:
            audio_tensor = self._load_audio(audio_path)
            if self.training and self.modality_dropout_prob > 0:
                if self._rng.random() < self.modality_dropout_prob:
                    audio_tensor = None
            result["audio_tensor"] = audio_tensor

        video_path = sample.get("video_path")
        if video_path and "video" in self.modalities:
            video_tensor = self._load_video(video_path)
            if self.training and self.modality_dropout_prob > 0:
                if self._rng.random() < self.modality_dropout_prob:
                    video_tensor = None
            result["video_tensor"] = video_tensor

        modality_mask = [
            1.0 if result["image_tensor"] is not None else 0.0,
            1.0 if result["audio_tensor"] is not None else 0.0,
            1.0 if result["video_tensor"] is not None else 0.0,
        ]
        result["modality_mask"] = modality_mask
        result["modality"] = sample.get("modality", "text")
        result["sample_idx"] = idx

        return result

    def get_modality_indices(self, modality: str) -> List[int]:
        """Get all sample indices that contain a specific modality.

        Args:
            modality: Modality name ("image", "audio", "video", "text").

        Returns:
            List of sample indices that contain the requested modality.

        Raises:
            ValueError: If the modality is not recognized.
        """
        if modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unknown modality '{modality}'. Supported: {SUPPORTED_MODALITIES}"
            )
        return self._modality_indices.get(modality, [])

    def get_modality_distribution(self) -> Dict[str, float]:
        """Get the distribution of modalities across the dataset.

        Returns:
            Dictionary mapping modality names to their proportion in the
            dataset (values sum to >= 1.0 since samples can have multiple
            modalities).
        """
        total = len(self.samples)
        if total == 0:
            return {m: 0.0 for m in self.modalities}

        return {
            modality: count / total
            for modality, count in self.modality_counts.items()
        }

    def subset_by_modality(self, modality: str) -> "MultimodalDataset":
        """Create a subset containing only samples with a specific modality.

        Args:
            modality: Modality to filter by.

        Returns:
            A new MultimodalDataset containing only samples with the
            specified modality.
        """
        indices = self.get_modality_indices(modality)
        subset = MultimodalDataset.__new__(MultimodalDataset)
        subset.data_path = self.data_path
        subset.modalities = self.modalities
        subset.max_text_length = self.max_text_length
        subset.max_image_tokens = self.max_image_tokens
        subset.max_audio_tokens = self.max_audio_tokens
        subset.max_video_tokens = self.max_video_tokens
        subset.image_size = self.image_size
        subset.num_frames = self.num_frames
        subset.sample_rate = self.sample_rate
        subset.max_audio_seconds = self.max_audio_seconds
        subset.tokenizer = self.tokenizer
        subset.transform = self.transform
        subset.audio_transform = self.audio_transform
        subset.split = self.split
        subset.seed = self.seed
        subset.modality_dropout_prob = self.modality_dropout_prob
        subset._rng = self._rng
        subset.samples = [self.samples[i] for i in indices]
        subset.modality_counts = Counter()
        subset._modality_indices = defaultdict(list)
        for s in subset.samples:
            subset._update_modality_counts(s)
        subset._build_modality_index()
        return subset

    def __repr__(self) -> str:
        """String representation of the dataset.

        Returns:
            Summary string with dataset statistics.
        """
        modality_info = ", ".join(
            f"{mod}: {cnt}"
            for mod, cnt in self.modality_counts.most_common()
        )
        return (
            f"MultimodalDataset(num_samples={len(self.samples)}, "
            f"modalities=[{modality_info}], split='{self.split}')"
        )


# =============================================================================
# Multimodal Collator
# =============================================================================

class MultimodalCollator:
    """Batch collation with per-modality dynamic padding.

    Handles batching of heterogeneous multimodal samples where different
    samples may have different modalities present, different text lengths,
    different audio durations, and different video frame counts.

    Features:
    - Dynamic text padding to the longest sequence in the batch.
    - Image tensor stacking (all images must be same shape after transform).
    - Audio spectrogram padding to the longest duration in the batch.
    - Video frame tensor padding to the most frames in the batch.
    - Modality mask creation indicating which modalities are active per sample.
    - Label padding with IGNORE_INDEX for language modeling loss.

    Args:
        pad_token_id: Token ID used for padding. Default: 0.
        image_pad_value: Padding value for image tensors. Default: 0.0.
        audio_pad_value: Padding value for audio tensors. Default: 0.0.
        video_pad_value: Padding value for video tensors. Default: 0.0.
        max_text_length: Maximum text sequence length. Default: 2048.
        pad_to_multiple_of: Pad text length to a multiple of this value.
                             Useful for tensor core efficiency. Default: None.
        return_tensors: Whether to convert all outputs to tensors. Default: True.
        label_pad_token_id: Token ID for padding labels (ignored in loss).
                             Default: -100.
        image_padding_mode: How to pad images if sizes differ.
                            Options: "constant", "replicate", "reflect".
                            Default: "constant".
        audio_padding_dim: Dimension along which to pad audio. Default: 1.
        video_padding_dim: Dimension along which to pad video. Default: 0.

    Example:
        >>> collator = MultimodalCollator(pad_token_id=0, max_text_length=512)
        >>> batch = collator([dataset[i] for i in range(4)])
        >>> batch["input_ids"].shape
        torch.Size([4, 512])
        >>> batch["modality_mask"].shape
        torch.Size([4, 3])
    """

    def __init__(
        self,
        pad_token_id: int = PAD_TOKEN_ID,
        image_pad_value: float = 0.0,
        audio_pad_value: float = 0.0,
        video_pad_value: float = 0.0,
        max_text_length: int = MAX_TEXT_LENGTH,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: bool = True,
        label_pad_token_id: int = IGNORE_INDEX,
        image_padding_mode: str = "constant",
        audio_padding_dim: int = 1,
        video_padding_dim: int = 0,
    ):
        self.pad_token_id = pad_token_id
        self.image_pad_value = image_pad_value
        self.audio_pad_value = audio_pad_value
        self.video_pad_value = video_pad_value
        self.max_text_length = max_text_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.label_pad_token_id = label_pad_token_id
        self.image_padding_mode = image_padding_mode
        self.audio_padding_dim = audio_padding_dim
        self.video_padding_dim = video_padding_dim

    def _pad_text_sequences(
        self,
        input_ids_list: List[List[int]],
        attention_mask_list: List[List[int]],
        labels_list: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad text sequences to the longest length in the batch.

        Optionally pads to a multiple of pad_to_multiple_of for efficiency.

        Args:
            input_ids_list: List of input ID sequences.
            attention_mask_list: List of attention mask sequences.
            labels_list: List of label sequences.

        Returns:
            Tuple of padded (input_ids, attention_mask, labels) tensors.
        """
        max_len = max(len(ids) for ids in input_ids_list) if input_ids_list else 1
        max_len = min(max_len, self.max_text_length)

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len += self.pad_to_multiple_of - remainder

        padded_ids = []
        padded_masks = []
        padded_labels = []

        for ids, mask, labels in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_masks.append(mask + [0] * pad_len)

            if labels is not None:
                padded_labels.append(
                    list(labels) + [self.label_pad_token_id] * pad_len
                )
            else:
                padded_labels.append([self.label_pad_token_id] * max_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long)
        attention_mask = torch.tensor(padded_masks, dtype=torch.long)
        labels = torch.tensor(padded_labels, dtype=torch.long)

        return input_ids, attention_mask, labels

    def _collate_images(
        self,
        image_tensors: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Stack image tensors into a batch.

        All images should have the same shape after preprocessing. If any
        image is None (modality not present), it is replaced with a zero
        tensor of the correct shape.

        Args:
            image_tensors: List of image tensors or None values.

        Returns:
            Batched image tensor of shape (B, C, H, W), or None if no
            images are present in the batch.
        """
        valid_images = [img for img in image_tensors if img is not None]

        if not valid_images:
            return None

        target_shape = valid_images[0].shape
        device = valid_images[0].device

        processed = []
        for img in image_tensors:
            if img is None:
                processed.append(
                    torch.zeros(target_shape, dtype=torch.float32, device=device)
                )
            else:
                if img.shape != target_shape:
                    img = self._resize_tensor(img, target_shape)
                processed.append(img)

        return torch.stack(processed, dim=0)

    def _resize_tensor(
        self,
        tensor: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """Resize a tensor to match a target shape using interpolation.

        Args:
            tensor: Input tensor.
            target_shape: Desired output shape.

        Returns:
            Resized tensor.
        """
        if len(tensor.shape) == 3:
            return F.interpolate(
                tensor.unsqueeze(0),
                size=target_shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tensor

    def _collate_audio(
        self,
        audio_tensors: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Pad and stack audio spectrogram tensors.

        Audio tensors may have different temporal lengths. They are padded
        to the longest length in the batch.

        Args:
            audio_tensors: List of audio tensors or None values.

        Returns:
            Batched audio tensor of shape (B, n_mels, max_time), or None
            if no audio is present.
        """
        valid_audio = [a for a in audio_tensors if a is not None]

        if not valid_audio:
            return None

        if len(valid_audio) == 1:
            target_shape = valid_audio[0].shape
            device = valid_audio[0].device
            processed = []
            for a in audio_tensors:
                if a is None:
                    processed.append(
                        torch.zeros(target_shape, dtype=torch.float32, device=device)
                    )
                else:
                    processed.append(a)
            return torch.stack(processed, dim=0)

        return pad_2d_batch(
            valid_audio,
            pad_value=self.audio_pad_value,
            padding_dim=self.audio_padding_dim,
        )

    def _collate_video(
        self,
        video_tensors: List[Optional[torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Pad and stack video frame tensors.

        Video tensors may have different numbers of frames. They are padded
        to the maximum frame count in the batch.

        Args:
            video_tensors: List of video tensors or None values.

        Returns:
            Batched video tensor of shape (B, F, C, H, W), or None if no
            video is present.
        """
        valid_videos = [v for v in video_tensors if v is not None]

        if not valid_videos:
            return None

        max_frames = max(v.shape[0] for v in valid_videos)
        target_shape = valid_videos[0].shape
        device = valid_videos[0].device

        processed = []
        for v in video_tensors:
            if v is None:
                padded = torch.zeros(
                    (max_frames,) + target_shape[1:],
                    dtype=torch.float32,
                    device=device,
                )
            else:
                if v.shape[0] < max_frames:
                    pad_size = max_frames - v.shape[0]
                    padding = torch.zeros(
                        (pad_size,) + v.shape[1:],
                        dtype=v.dtype,
                        device=v.device,
                    )
                    v = torch.cat([v, padding], dim=self.video_padding_dim)
                elif v.shape[0] > max_frames:
                    v = v[:max_frames]
                if v.shape != (max_frames,) + target_shape[1:]:
                    pad_needed = (max_frames,) + target_shape[1:]
                    if v.shape[1:] != target_shape[1:]:
                        v = v.view(v.shape[0], -1)
                        target_flat = max_frames * math.prod(target_shape[1:])
                        if v.shape[1] < target_flat:
                            pad_w = target_flat - v.shape[1]
                            v = torch.cat([
                                v,
                                torch.zeros(v.shape[0], pad_w, device=v.device)
                            ], dim=-1)
                        v = v.view(pad_needed)
                processed.append(v)

        return torch.stack(processed, dim=0)

    def _create_modality_mask(
        self,
        modality_masks: List[List[float]],
    ) -> torch.Tensor:
        """Create a batched modality mask tensor.

        Args:
            modality_masks: List of per-sample modality masks, each of
                            length 3 (image, audio, video).

        Returns:
            Modality mask tensor of shape (B, 3).
        """
        return torch.tensor(modality_masks, dtype=torch.float32)

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collate a list of samples into a batch.

        Args:
            features: List of sample dictionaries from MultimodalDataset.

        Returns:
            Dictionary containing batched tensors:
                - input_ids: (B, max_text_len) padded token IDs.
                - attention_mask: (B, max_text_len) attention mask.
                - labels: (B, max_text_len) target labels.
                - image_tensor: (B, C, H, W) or None.
                - audio_tensor: (B, n_mels, max_time) or None.
                - video_tensor: (B, max_frames, C, H, W) or None.
                - modality_mask: (B, 3) modality presence mask.
                - modality_types: List of primary modality type strings.
                - sample_indices: (B,) original sample indices.

        Example:
            >>> collator = MultimodalCollator()
            >>> batch = collator([dataset[0], dataset[1], dataset[2]])
        """
        if not features:
            return {}

        input_ids_list = [f["input_ids"] for f in features]
        attention_mask_list = [f["attention_mask"] for f in features]
        labels_list = [f.get("labels", f["input_ids"]) for f in features]

        input_ids, attention_mask, labels = self._pad_text_sequences(
            input_ids_list, attention_mask_list, labels_list
        )

        image_tensors = [f.get("image_tensor") for f in features]
        audio_tensors = [f.get("audio_tensor") for f in features]
        video_tensors = [f.get("video_tensor") for f in features]
        modality_masks = [f.get("modality_mask", [0.0, 0.0, 0.0]) for f in features]

        image_batch = self._collate_images(image_tensors)
        audio_batch = self._collate_audio(audio_tensors)
        video_batch = self._collate_video(video_tensors)
        modality_mask = self._create_modality_mask(modality_masks)

        modality_types = [f.get("modality", "text") for f in features]
        sample_indices = torch.tensor(
            [f.get("sample_idx", i) for i, f in enumerate(features)],
            dtype=torch.long,
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_tensor": image_batch,
            "audio_tensor": audio_batch,
            "video_tensor": video_batch,
            "modality_mask": modality_mask,
            "modality_types": modality_types,
            "sample_indices": sample_indices,
        }

        if not self.return_tensors:
            batch = {
                k: v if not isinstance(v, torch.Tensor) else v.tolist()
                for k, v in batch.items()
            }

        return batch


# =============================================================================
# Contrastive Loss
# =============================================================================

class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with hard negative mining.

    Implements the InfoNCE (Information Noise Contrastive Estimation) loss
    used in contrastive learning frameworks like CLIP. The loss encourages
    positive (matching) pairs to have higher similarity than negative
    (non-matching) pairs.

    With hard negative mining enabled, the loss focuses on the most
    difficult negatives (those with similarity closest to the positive pair)
    to improve discriminative power.

    The loss is computed as:
        L = -log(exp(sim(q, p) / tau) / sum(exp(sim(q, n_i) / tau)))

    where q is the query, p is the positive, n_i are negatives, and tau
    is the temperature parameter.

    Args:
        temperature: Temperature parameter for scaling logits. Higher
                     values produce softer probability distributions.
                     Default: 0.07.
        use_hard_negative_mining: Whether to focus on hard negatives.
                                  Default: True.
        hard_negative_ratio: Fraction of negatives to consider as hard.
                             Default: 0.5.
        hard_negative_threshold: Minimum similarity for a negative to be
                                 considered "hard". Default: 0.3.
        max_hard_negatives: Maximum number of hard negatives per query.
                            Default: 64.
        reduction: Loss reduction method ("mean", "sum", "none").
                   Default: "mean".
        label_smoothing: Label smoothing factor in [0, 1]. Default: 0.0.
        augmentation_temperature: Temperature for augmented view
                                  consistency. Default: None.

    Shape:
        - query: (B, D) or (B, N, D)
        - positive: (B, D) or (B, N, D)
        - negatives: (B, K, D) or (B, K, N, D) where K is number of negatives

    Example:
        >>> loss_fn = ContrastiveLoss(temperature=0.07)
        >>> query = torch.randn(32, 768)
        >>> positive = torch.randn(32, 768)
        >>> negatives = torch.randn(32, 64, 768)
        >>> loss = loss_fn(query, positive, negatives)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_hard_negative_mining: bool = True,
        hard_negative_ratio: float = 0.5,
        hard_negative_threshold: float = 0.3,
        max_hard_negatives: int = 64,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        augmentation_temperature: Optional[float] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_hard_negative_mining = use_hard_negative_mining
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_threshold = hard_negative_threshold
        self.max_hard_negatives = max_hard_negatives
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.augmentation_temperature = augmentation_temperature

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not (0.0 <= self.hard_negative_ratio <= 1.0):
            raise ValueError(
                f"hard_negative_ratio must be in [0, 1], "
                f"got {self.hard_negative_ratio}"
            )
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )

    def _compute_similarity(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between query and key vectors.

        L2-normalizes both query and key vectors before computing
        the dot product, yielding cosine similarity scores.

        Args:
            query: Query vectors of shape (B, D) or (B, N, D).
            keys: Key vectors of shape (B, K, D) or (B, K, N, D).

        Returns:
            Similarity scores of shape (B, K) or (B, K, N).
        """
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)

        if query.dim() == 2 and keys.dim() == 3:
            similarity = torch.matmul(query_norm, keys_norm.transpose(-2, -1))
            return similarity
        elif query.dim() == 3 and keys.dim() == 3:
            similarity = torch.matmul(query_norm, keys_norm.transpose(-2, -1))
            return similarity
        elif query.dim() == 3 and keys.dim() == 4:
            similarity = torch.matmul(query_norm, keys_norm.transpose(-2, -1))
            return similarity
        else:
            return torch.matmul(query_norm, keys_norm.t())

    def _select_hard_negatives(
        self,
        positive_similarity: torch.Tensor,
        negative_similarity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select hard negatives based on similarity scores.

        Hard negatives are those with similarity scores closest to the
        positive similarity. A threshold is applied to filter out
        trivially easy negatives.

        Args:
            positive_similarity: Shape (B,) similarity with positives.
            negative_similarity: Shape (B, K) similarity with all negatives.

        Returns:
            Tuple of (selected_indices, selected_similarities) where
            selected_indices has shape (B, K') and K' <= K.
        """
        pos_sim_expanded = positive_similarity.unsqueeze(-1)

        difficulty = (negative_similarity - pos_sim_expanded).abs()
        difficulty = difficulty.clamp(min=0.0)

        above_threshold = negative_similarity > self.hard_negative_threshold
        difficulty = difficulty * above_threshold.float()

        k = min(
            int(self.hard_negative_ratio * negative_similarity.shape[-1]),
            self.max_hard_negatives,
        )
        k = max(k, 1)

        _, hard_indices = difficulty.topk(k, dim=-1, largest=True)
        hard_indices_expanded = hard_indices.unsqueeze(-1).expand(
            -1, -1, negative_similarity.shape[-1]
        )

        selected_negatives = torch.gather(negative_similarity, 1, hard_indices)

        return hard_indices, selected_negatives

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
        negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the InfoNCE contrastive loss.

        Args:
            query: Query embeddings of shape (B, D).
            positive: Positive embeddings of shape (B, D).
            negatives: Negative embeddings of shape (B, K, D). If None,
                       other batch items are used as negatives (in-batch).
            negative_mask: Optional mask of shape (B, K) where 1 indicates
                           a valid negative and 0 indicates padding/invalid.
                           Default: None (all negatives are valid).

        Returns:
            Scalar loss value (or per-sample losses if reduction="none").
        """
        if query.dim() != 2:
            raise ValueError(
                f"Expected query to be 2D (B, D), got {query.dim()}D"
            )
        if positive.dim() != 2:
            raise ValueError(
                f"Expected positive to be 2D (B, D), got {positive.dim()}D"
            )

        batch_size = query.shape[0]
        embed_dim = query.shape[1]

        positive_similarity = self._compute_similarity(query, positive)
        positive_similarity = positive_similarity.diag() if positive_similarity.dim() == 2 else positive_similarity.squeeze(-1)

        if negatives is None:
            all_embeddings = torch.cat([query, positive], dim=0)
            similarity_matrix = self._compute_similarity(all_embeddings, all_embeddings)

            self_mask = torch.eye(batch_size, device=query.device, dtype=torch.bool)
            positive_mask = torch.zeros(batch_size, 2 * batch_size, device=query.device, dtype=torch.bool)
            for i in range(batch_size):
                positive_mask[i, batch_size + i] = True
                positive_mask[i, i] = True

            negative_mask_full = ~(self_mask | positive_mask)
            similarity_matrix = similarity_matrix.masked_fill(~negative_mask_full, float("-inf"))

            positive_logits = positive_similarity / self.temperature
            all_logits = similarity_matrix / self.temperature

            labels = torch.full(
                (batch_size, 2 * batch_size),
                self.label_smoothing / (2 * batch_size - 1),
                device=query.device,
            )
            for i in range(batch_size):
                labels[i, batch_size + i] = 1.0 - self.label_smoothing

            loss = F.cross_entropy(all_logits, labels, reduction=self.reduction)
            return loss

        if negatives.dim() != 3:
            raise ValueError(
                f"Expected negatives to be 3D (B, K, D), got {negatives.dim()}D"
            )

        negative_similarity = self._compute_similarity(query, negatives)

        if self.use_hard_negative_mining:
            _, negative_similarity = self._select_hard_negatives(
                positive_similarity, negative_similarity
            )

        if negative_mask is not None:
            if negative_mask.shape != negative_similarity.shape:
                min_size = min(negative_mask.shape[1], negative_similarity.shape[1])
                negative_mask = negative_mask[:, :min_size]
                negative_similarity = negative_similarity[:, :min_size]
            negative_similarity = negative_similarity.masked_fill(
                ~negative_mask.bool(), float("-inf")
            )

        logits = torch.cat(
            [positive_similarity.unsqueeze(-1), negative_similarity],
            dim=-1,
        )
        logits = logits / self.temperature

        num_negatives = logits.shape[-1] - 1
        if self.label_smoothing > 0:
            labels = torch.full(
                logits.shape,
                self.label_smoothing / num_negatives,
                device=logits.device,
                dtype=logits.dtype,
            )
            labels[:, 0] = 1.0 - self.label_smoothing
            loss = -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
            loss = F.cross_entropy(logits, labels, reduction="none")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def compute_accuracy(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """Compute retrieval accuracy metrics.

        Args:
            query: Query embeddings (B, D).
            positive: Positive embeddings (B, D).
            negatives: Optional negative embeddings (B, K, D).

        Returns:
            Tuple of (recall@1, recall@5) as floats.
        """
        with torch.no_grad():
            if negatives is None:
                all_embeds = torch.cat([query, positive], dim=0)
                sim = compute_cosine_similarity(all_embeds, all_embeds)
                batch_size = query.shape[0]

                positive_idx = torch.arange(batch_size, 2 * batch_size, device=query.device)

                sorted_indices = sim.argsort(dim=-1, descending=True)
                ranks = (sorted_indices == positive_idx.unsqueeze(-1)).nonzero(as_tuple=True)[1]

                recall_at_1 = (ranks < 1).float().mean().item()
                recall_at_5 = (ranks < 5).float().mean().item()
            else:
                pos_sim = self._compute_similarity(query, positive).diag()
                neg_sim = self._compute_similarity(query, negatives)

                all_sims = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
                top5_indices = all_sims.topk(5, dim=-1).indices
                is_top1 = (top5_indices[:, 0] == 0).float()
                is_top5 = (top5_indices == 0).any(dim=-1).float()

                recall_at_1 = is_top1.mean().item()
                recall_at_5 = is_top5.mean().item()

            return recall_at_1, recall_at_5


# =============================================================================
# Alignment Loss
# =============================================================================

class AlignmentLoss(nn.Module):
    """Cosine similarity alignment loss between modality embeddings.

    Encourages embeddings from different modalities to align in a shared
    representation space. Supports both symmetric alignment (both modalities
    pulled toward each other equally) and asymmetric alignment (one modality
    acts as the anchor while the other is aligned to it).

    The loss is computed as:
        L_symmetric = 1 - cosine_sim(embedding_a, embedding_b)
        L_asymmetric = 1 - cosine_sim(anchor, target)
        L_weighted = weights * (1 - cosine_sim(embedding_a, embedding_b))

    A margin can be specified so that pairs with similarity above the
    margin incur no loss, enabling the model to focus on harder pairs.

    Args:
        alignment_type: Type of alignment ("symmetric" or "asymmetric").
                        Default: "symmetric".
        margin: Margin for margin-based alignment loss. Pairs with cosine
                similarity above this margin are not penalized. Default: 0.0
                (no margin).
        temperature: Temperature for scaling cosine similarity. Default: 1.0.
        reduction: Loss reduction method ("mean", "sum", "none").
                   Default: "mean".
        modality_weights: Optional per-modality weights for weighted
                          alignment. Default: None (uniform weights).
        normalize_embeddings: Whether to L2-normalize embeddings before
                              computing similarity. Default: True.
        use_projection_head: Apply a learned projection head before
                             computing alignment. Default: False.
        projection_dim: Dimension of the projection head output.
                        Default: 256.
        dropout: Dropout probability for the projection head. Default: 0.1.
        epsilon: Small constant for numerical stability. Default: 1e-8.

    Shape:
        - embedding_a: (B, D) or (B, N, D)
        - embedding_b: (B, D) or (B, N, D)

    Example:
        >>> loss_fn = AlignmentLoss(alignment_type="symmetric", margin=0.5)
        >>> vision_embeds = torch.randn(32, 768)
        >>> text_embeds = torch.randn(32, 768)
        >>> loss = loss_fn(vision_embeds, text_embeds)
    """

    def __init__(
        self,
        alignment_type: str = "symmetric",
        margin: float = 0.0,
        temperature: float = 1.0,
        reduction: str = "mean",
        modality_weights: Optional[Dict[str, float]] = None,
        normalize_embeddings: bool = True,
        use_projection_head: bool = False,
        projection_dim: int = 256,
        dropout: float = 0.1,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.alignment_type = alignment_type
        self.margin = margin
        self.temperature = temperature
        self.reduction = reduction
        self.modality_weights = modality_weights
        self.normalize_embeddings = normalize_embeddings
        self.epsilon = epsilon

        if self.alignment_type not in ("symmetric", "asymmetric"):
            raise ValueError(
                f"alignment_type must be 'symmetric' or 'asymmetric', "
                f"got '{self.alignment_type}'"
            )

        self.projection_head = None
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim, projection_dim),
            )

    def _normalize(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """L2-normalize embeddings along the last dimension.

        Args:
            embeddings: Input embeddings.

        Returns:
            Normalized embeddings.
        """
        if not self.normalize_embeddings:
            return embeddings

        return F.normalize(embeddings, p=2, dim=-1, eps=self.epsilon)

    def _cosine_similarity(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute element-wise cosine similarity between paired embeddings.

        Args:
            x: First embedding tensor, shape (B, D) or (B, N, D).
            y: Second embedding tensor, shape (B, D) or (B, N, D).

        Returns:
            Cosine similarity tensor with shape matching the input after
            removing the last dimension.
        """
        x_norm = self._normalize(x)
        y_norm = self._normalize(y)

        if x.dim() == 2:
            similarity = (x_norm * y_norm).sum(dim=-1)
        elif x.dim() == 3:
            similarity = (x_norm * y_norm).sum(dim=-1)
        else:
            similarity = (x_norm * y_norm).sum(dim=-1)

        return similarity / self.temperature

    def _symmetric_loss(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric alignment loss.

        Both modalities are equally penalized for misalignment.

        Args:
            similarity: Pairwise cosine similarities, shape (B,) or (B, N).

        Returns:
            Symmetric alignment loss per sample.
        """
        if self.margin > 0:
            loss = F.relu(self.margin - similarity)
        else:
            loss = 1.0 - similarity

        return loss

    def _asymmetric_loss(
        self,
        anchor: torch.Tensor,
        target: torch.Tensor,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric alignment loss.

        The anchor modality serves as the reference; the target is
        aligned toward the anchor. This is useful when one modality
        is more reliable (e.g., text anchors for image-text alignment).

        The loss also penalizes deviation of the anchor norm from 1,
        encouraging well-conditioned embeddings.

        Args:
            anchor: Anchor embeddings (B, D).
            target: Target embeddings (B, D).
            similarity: Pairwise cosine similarities (B,).

        Returns:
            Asymmetric alignment loss per sample.
        """
        alignment_component = 1.0 - similarity

        anchor_norm = anchor.norm(p=2, dim=-1)
        norm_regularization = (anchor_norm - 1.0).pow(2).mean()

        total_loss = alignment_component + 0.01 * norm_regularization

        if self.margin > 0:
            total_loss = F.relu(self.margin - similarity)

        return total_loss

    def forward(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute alignment loss between two modality embeddings.

        Args:
            embedding_a: First modality embeddings, shape (B, D) or (B, N, D).
            embedding_b: Second modality embeddings, shape (B, D) or (B, N, D).
            weights: Optional per-sample weights, shape (B,). Used to weight
                     the contribution of each sample to the total loss.
            mask: Optional mask, shape (B,) or (B, N), where 1 indicates
                  valid pairs and 0 indicates padding. Default: None.

        Returns:
            Scalar alignment loss (or per-sample losses if reduction="none").

        Raises:
            ValueError: If embedding shapes are incompatible.
        """
        if embedding_a.shape[0] != embedding_b.shape[0]:
            raise ValueError(
                f"Batch size mismatch: embedding_a has {embedding_a.shape[0]} "
                f"samples, embedding_b has {embedding_b.shape[0]}"
            )

        if self.projection_head is not None:
            embedding_a = self.projection_head(embedding_a)
            embedding_b = self.projection_head(embedding_b)

        similarity = self._cosine_similarity(embedding_a, embedding_b)

        if self.alignment_type == "symmetric":
            per_sample_loss = self._symmetric_loss(similarity)
        else:
            per_sample_loss = self._asymmetric_loss(
                embedding_a, embedding_b, similarity
            )

        if mask is not None:
            mask = mask.float()
            if mask.dim() < per_sample_loss.dim():
                while mask.dim() < per_sample_loss.dim():
                    mask = mask.unsqueeze(-1)
            per_sample_loss = per_sample_loss * mask
            num_valid = mask.sum().clamp(min=1.0)
            per_sample_loss = per_sample_loss.sum() / num_valid
            if self.reduction == "none":
                return (per_sample_loss * mask.squeeze()).squeeze()
            return per_sample_loss

        if weights is not None:
            if weights.dim() == 1 and per_sample_loss.dim() > 1:
                weights = weights.unsqueeze(-1)
            per_sample_loss = per_sample_loss * weights

        if self.reduction == "mean":
            return per_sample_loss.mean()
        elif self.reduction == "sum":
            return per_sample_loss.sum()
        else:
            return per_sample_loss

    def compute_alignment_score(
        self,
        embedding_a: torch.Tensor,
        embedding_b: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute alignment statistics between two embedding sets.

        Args:
            embedding_a: First modality embeddings (B, D).
            embedding_b: Second modality embeddings (B, D).

        Returns:
            Dictionary with alignment metrics:
                - mean_cosine_similarity: Average cosine similarity.
                - median_cosine_similarity: Median cosine similarity.
                - std_cosine_similarity: Standard deviation of similarities.
                - alignment_ratio: Fraction of pairs with similarity > 0.5.
        """
        with torch.no_grad():
            similarity = self._cosine_similarity(embedding_a, embedding_b)
            sim_list = similarity.flatten().cpu().tolist()

            mean_sim = sum(sim_list) / max(len(sim_list), 1)
            sorted_sims = sorted(sim_list)
            n = len(sorted_sims)
            if n % 2 == 0:
                median_sim = (sorted_sims[n // 2 - 1] + sorted_sims[n // 2]) / 2
            else:
                median_sim = sorted_sims[n // 2]
            variance = sum((s - mean_sim) ** 2 for s in sim_list) / max(len(sim_list), 1)
            std_sim = math.sqrt(variance)
            alignment_ratio = sum(1.0 for s in sim_list if s > 0.5) / max(len(sim_list), 1)

            return {
                "mean_cosine_similarity": mean_sim,
                "median_cosine_similarity": median_sim,
                "std_cosine_similarity": std_sim,
                "alignment_ratio": alignment_ratio,
            }


# =============================================================================
# Multimodal Loss
# =============================================================================

class MultimodalLoss(nn.Module):
    """Combined multimodal loss function.

    Aggregates multiple loss components for end-to-end multimodal training:
    - Language modeling loss (cross-entropy on text generation)
    - Contrastive loss (InfoNCE between modality embeddings)
    - Alignment loss (cosine similarity alignment between modalities)

    Each loss component can be independently enabled/disabled and weighted,
    allowing flexible training schedules and multi-stage pretraining.

    The total loss is:
        L = w_lm * L_lm + w_con * L_con + w_align * L_align

    where w_lm, w_con, w_align are configurable weights.

    Args:
        lm_weight: Weight for language modeling loss. Default: 1.0.
        contrastive_weight: Weight for contrastive loss. Default: 1.0.
        alignment_weight: Weight for alignment loss. Default: 0.5.
        enable_lm: Enable language modeling loss. Default: True.
        enable_contrastive: Enable contrastive loss. Default: True.
        enable_alignment: Enable alignment loss. Default: True.
        contrastive_temperature: Temperature for contrastive loss.
                                 Default: 0.07.
        alignment_margin: Margin for alignment loss. Default: 0.0.
        alignment_type: Type of alignment ("symmetric" or "asymmetric").
                        Default: "symmetric".
        label_smoothing: Label smoothing for language modeling loss.
                         Default: 0.0.
        ignore_index: Index to ignore in language modeling loss.
                      Default: -100.
        kl_divergence_weight: Weight for optional KL divergence
                              regularization. Default: 0.0.
        diversity_weight: Weight for modality embedding diversity
                          regularization. Default: 0.0.
        reconstruction_weight: Weight for modality reconstruction loss.
                               Default: 0.0.
        gradient_clipping: Maximum gradient norm for loss scaling.
                           Default: None (no clipping).

    Example:
        >>> loss_fn = MultimodalLoss(
        ...     lm_weight=1.0,
        ...     contrastive_weight=0.5,
        ...     alignment_weight=0.2,
        ... )
        >>> logits = torch.randn(4, 128, 32000)
        >>> labels = torch.randint(0, 32000, (4, 128))
        >>> vision_embeds = torch.randn(4, 768)
        >>> text_embeds = torch.randn(4, 768)
        >>> loss = loss_fn(
        ...     lm_logits=logits,
        ...     labels=labels,
        ...     vision_embeddings=vision_embeds,
        ...     text_embeddings=text_embeds,
        ... )
    """

    def __init__(
        self,
        lm_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        alignment_weight: float = 0.5,
        enable_lm: bool = True,
        enable_contrastive: bool = True,
        enable_alignment: bool = True,
        contrastive_temperature: float = 0.07,
        alignment_margin: float = 0.0,
        alignment_type: str = "symmetric",
        label_smoothing: float = 0.0,
        ignore_index: int = IGNORE_INDEX,
        kl_divergence_weight: float = 0.0,
        diversity_weight: float = 0.0,
        reconstruction_weight: float = 0.0,
        gradient_clipping: Optional[float] = None,
    ):
        super().__init__()
        self.lm_weight = lm_weight
        self.contrastive_weight = contrastive_weight
        self.alignment_weight = alignment_weight
        self.enable_lm = enable_lm
        self.enable_contrastive = enable_contrastive
        self.enable_alignment = enable_alignment
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.kl_divergence_weight = kl_divergence_weight
        self.diversity_weight = diversity_weight
        self.reconstruction_weight = reconstruction_weight
        self.gradient_clipping = gradient_clipping

        if self.enable_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss(
                temperature=contrastive_temperature,
            )
        else:
            self.contrastive_loss_fn = None

        if self.enable_alignment:
            self.alignment_loss_fn = AlignmentLoss(
                alignment_type=alignment_type,
                margin=alignment_margin,
            )
        else:
            self.alignment_loss_fn = None

    def _compute_lm_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the language modeling cross-entropy loss.

        Shifts logits and labels for next-token prediction, applies
        attention masking, and computes cross-entropy with optional
        label smoothing.

        Args:
            logits: Model output logits, shape (B, seq_len, vocab_size).
            labels: Target token IDs, shape (B, seq_len).
            attention_mask: Optional attention mask, shape (B, seq_len).

        Returns:
            Scalar language modeling loss.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
            shift_labels = shift_labels.masked_fill(shift_mask == 0, self.ignore_index)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )

        return loss

    def _compute_kl_divergence(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """Compute KL divergence between teacher and student distributions.

        Used for knowledge distillation between modality encoders.

        Args:
            teacher_logits: Teacher model logits, shape (B, D).
            student_logits: Student model logits, shape (B, D).
            temperature: Temperature for softening distributions. Default: 2.0.

        Returns:
            KL divergence loss.
        """
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        return kl_loss * (temperature ** 2)

    def _compute_diversity_loss(
        self,
        modality_embeddings: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute modality embedding diversity loss.

        Encourages embeddings from different modalities to occupy
        different regions of the shared representation space.

        Args:
            modality_embeddings: Dictionary mapping modality names to
                                embedding tensors of shape (B, D).

        Returns:
            Diversity regularization loss.
        """
        if len(modality_embeddings) < 2:
            return torch.tensor(0.0, device=next(iter(modality_embeddings.values())).device)

        embeddings = list(modality_embeddings.values())
        total_loss = torch.tensor(0.0, device=embeddings[0].device)

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                mean_i = embeddings[i].mean(dim=0)
                mean_j = embeddings[j].mean(dim=0)

                distance = 1.0 - F.cosine_similarity(
                    mean_i.unsqueeze(0), mean_j.unsqueeze(0)
                )

                total_loss = total_loss - distance.mean()

        return total_loss / max(len(embeddings), 1)

    def _compute_reconstruction_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute modality reconstruction loss (MSE).

        Used when a modality is auto-encoded or reconstructed from
        cross-modal features.

        Args:
            original: Original modality features.
            reconstructed: Reconstructed modality features.
            mask: Optional mask for valid positions.

        Returns:
            Reconstruction MSE loss.
        """
        loss = F.mse_loss(reconstructed, original, reduction="none")

        if mask is not None:
            if mask.dim() < loss.dim():
                while mask.dim() < loss.dim():
                    mask = mask.unsqueeze(-1)
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)

        return loss.mean()

    def forward(
        self,
        lm_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeddings: Optional[torch.Tensor] = None,
        audio_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        video_embeddings: Optional[torch.Tensor] = None,
        negative_embeddings: Optional[torch.Tensor] = None,
        teacher_embeddings: Optional[torch.Tensor] = None,
        student_embeddings: Optional[torch.Tensor] = None,
        reconstructed_features: Optional[torch.Tensor] = None,
        original_features: Optional[torch.Tensor] = None,
        reconstruction_mask: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined multimodal loss.

        Only computes loss components for which inputs are provided and
        the corresponding loss is enabled.

        Args:
            lm_logits: Language model logits, shape (B, seq_len, vocab_size).
            labels: Target labels, shape (B, seq_len).
            attention_mask: Attention mask, shape (B, seq_len).
            vision_embeddings: Vision encoder embeddings (B, D).
            audio_embeddings: Audio encoder embeddings (B, D).
            text_embeddings: Text embeddings (B, D).
            video_embeddings: Video encoder embeddings (B, D).
            negative_embeddings: Hard negatives for contrastive loss (B, K, D).
            teacher_embeddings: Teacher embeddings for KL divergence (B, D).
            student_embeddings: Student embeddings for KL divergence (B, D).
            reconstructed_features: Reconstructed modality features.
            original_features: Original modality features for reconstruction.
            reconstruction_mask: Mask for reconstruction loss.
            modality_mask: Binary mask indicating active modalities (B, num_modalities).

        Returns:
            Dictionary containing:
                - total_loss: Weighted sum of all enabled loss components.
                - lm_loss: Language modeling loss (if enabled).
                - contrastive_loss: Contrastive loss (if enabled).
                - alignment_loss: Alignment loss (if enabled).
                - kl_divergence_loss: KL divergence loss (if applicable).
                - diversity_loss: Diversity regularization loss.
                - reconstruction_loss: Reconstruction loss (if applicable).
                - loss_components: Dictionary of individual loss values.
        """
        losses = {}
        loss_components = {}

        if self.enable_lm and lm_logits is not None and labels is not None:
            lm_loss = self._compute_lm_loss(lm_logits, labels, attention_mask)
            loss_components["lm_loss"] = lm_loss * self.lm_weight
            losses["lm_loss"] = lm_loss

        if self.enable_contrastive and self.contrastive_loss_fn is not None:
            contrastive_loss = torch.tensor(0.0, device=labels.device if labels is not None else torch.device("cpu"))

            if vision_embeddings is not None and text_embeddings is not None:
                if modality_mask is not None and modality_mask[:, 0].any():
                    vision_mask = modality_mask[:, 0].bool()
                    v_emb = vision_embeddings[vision_mask]
                    t_emb = text_embeddings[vision_mask]
                    if v_emb.shape[0] > 1:
                        contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                            v_emb, t_emb, negative_embeddings
                        )
                else:
                    contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                        vision_embeddings, text_embeddings, negative_embeddings
                    )

            if audio_embeddings is not None and text_embeddings is not None:
                if modality_mask is not None and modality_mask[:, 1].any():
                    audio_mask = modality_mask[:, 1].bool()
                    a_emb = audio_embeddings[audio_mask]
                    t_emb = text_embeddings[audio_mask]
                    if a_emb.shape[0] > 1:
                        contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                            a_emb, t_emb, negative_embeddings
                        )
                else:
                    contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                        audio_embeddings, text_embeddings, negative_embeddings
                    )

            if video_embeddings is not None and text_embeddings is not None:
                if modality_mask is not None and modality_mask[:, 2].any():
                    video_mask = modality_mask[:, 2].bool()
                    vid_emb = video_embeddings[video_mask]
                    t_emb = text_embeddings[video_mask]
                    if vid_emb.shape[0] > 1:
                        contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                            vid_emb, t_emb, negative_embeddings
                        )
                else:
                    contrastive_loss = contrastive_loss + self.contrastive_loss_fn(
                        video_embeddings, text_embeddings, negative_embeddings
                    )

            loss_components["contrastive_loss"] = contrastive_loss * self.contrastive_weight
            losses["contrastive_loss"] = contrastive_loss

        if self.enable_alignment and self.alignment_loss_fn is not None:
            alignment_loss = torch.tensor(0.0, device=labels.device if labels is not None else torch.device("cpu"))

            if vision_embeddings is not None and text_embeddings is not None:
                alignment_loss = alignment_loss + self.alignment_loss_fn(
                    vision_embeddings, text_embeddings, mask=modality_mask[:, 0] if modality_mask is not None else None
                )

            if audio_embeddings is not None and text_embeddings is not None:
                alignment_loss = alignment_loss + self.alignment_loss_fn(
                    audio_embeddings, text_embeddings, mask=modality_mask[:, 1] if modality_mask is not None else None
                )

            if video_embeddings is not None and text_embeddings is not None:
                alignment_loss = alignment_loss + self.alignment_loss_fn(
                    video_embeddings, text_embeddings, mask=modality_mask[:, 2] if modality_mask is not None else None
                )

            loss_components["alignment_loss"] = alignment_loss * self.alignment_weight
            losses["alignment_loss"] = alignment_loss

        if self.kl_divergence_weight > 0 and teacher_embeddings is not None and student_embeddings is not None:
            kl_loss = self._compute_kl_divergence(
                teacher_embeddings, student_embeddings
            )
            loss_components["kl_divergence_loss"] = kl_loss * self.kl_divergence_weight
            losses["kl_divergence_loss"] = kl_loss

        if self.diversity_weight > 0:
            modality_embs = {}
            if vision_embeddings is not None:
                modality_embs["vision"] = vision_embeddings
            if audio_embeddings is not None:
                modality_embs["audio"] = audio_embeddings
            if text_embeddings is not None:
                modality_embs["text"] = text_embeddings
            if video_embeddings is not None:
                modality_embs["video"] = video_embeddings

            if len(modality_embs) >= 2:
                diversity_loss = self._compute_diversity_loss(modality_embs)
                loss_components["diversity_loss"] = diversity_loss * self.diversity_weight
                losses["diversity_loss"] = diversity_loss

        if self.reconstruction_weight > 0 and reconstructed_features is not None and original_features is not None:
            recon_loss = self._compute_reconstruction_loss(
                original_features, reconstructed_features, reconstruction_mask
            )
            loss_components["reconstruction_loss"] = recon_loss * self.reconstruction_weight
            losses["reconstruction_loss"] = recon_loss

        total_loss = sum(loss_components.values()) if loss_components else torch.tensor(0.0)
        losses["total_loss"] = total_loss
        losses["loss_components"] = loss_components

        return losses

    def get_loss_weights(self) -> Dict[str, float]:
        """Get the current loss component weights.

        Returns:
            Dictionary mapping loss component names to their weights.
        """
        weights = {}
        if self.enable_lm:
            weights["lm_loss"] = self.lm_weight
        if self.enable_contrastive:
            weights["contrastive_loss"] = self.contrastive_weight
        if self.enable_alignment:
            weights["alignment_loss"] = self.alignment_weight
        if self.kl_divergence_weight > 0:
            weights["kl_divergence_loss"] = self.kl_divergence_weight
        if self.diversity_weight > 0:
            weights["diversity_loss"] = self.diversity_weight
        if self.reconstruction_weight > 0:
            weights["reconstruction_loss"] = self.reconstruction_weight
        return weights

    def update_weights(
        self,
        lm_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
        alignment_weight: Optional[float] = None,
        kl_divergence_weight: Optional[float] = None,
        diversity_weight: Optional[float] = None,
        reconstruction_weight: Optional[float] = None,
    ):
        """Update loss component weights for curriculum learning.

        Args:
            lm_weight: New language modeling weight.
            contrastive_weight: New contrastive weight.
            alignment_weight: New alignment weight.
            kl_divergence_weight: New KL divergence weight.
            diversity_weight: New diversity weight.
            reconstruction_weight: New reconstruction weight.
        """
        if lm_weight is not None:
            self.lm_weight = lm_weight
        if contrastive_weight is not None:
            self.contrastive_weight = contrastive_weight
        if alignment_weight is not None:
            self.alignment_weight = alignment_weight
        if kl_divergence_weight is not None:
            self.kl_divergence_weight = kl_divergence_weight
        if diversity_weight is not None:
            self.diversity_weight = diversity_weight
        if reconstruction_weight is not None:
            self.reconstruction_weight = reconstruction_weight


# =============================================================================
# Modality Balanced Sampler
# =============================================================================

class ModalityBalancedSampler(Sampler):
    """Sampler that ensures balanced modality distribution per batch.

    Addresses the common problem of modality imbalance in multimodal
    datasets where text-only samples vastly outnumber multimodal ones.
    This sampler constructs batches such that each batch contains a
    roughly equal number of samples from each enabled modality.

    Strategy:
    1. Group dataset indices by their primary modality.
    2. For each batch, sample approximately batch_size/num_modalities
       indices from each modality group.
    3. Shuffle within each epoch and across modality groups.
    4. If a modality has fewer samples than needed, oversample with
       replacement to maintain balance.

    Args:
        dataset: The MultimodalDataset instance to sample from.
        batch_size: Number of samples per batch. Default: 32.
        num_modalities: Number of modality groups to balance.
                        Default: 3 (image, audio, video).
        shuffle: Whether to shuffle within epochs. Default: True.
        seed: Random seed for reproducibility. Default: 42.
        drop_last: Whether to drop the last incomplete batch.
                   Default: True.
        replacement: Allow sampling with replacement when a modality
                     has too few samples. Default: True.
        max_samples_per_modality: Cap on samples per modality per epoch.
                                  None means use all available. Default: None.

    Attributes:
        num_samples: Total number of samples generated per epoch.
        modality_groups: Mapping of modality to list of sample indices.
        batch_size_per_modality: Approximate samples per modality per batch.

    Example:
        >>> sampler = ModalityBalancedSampler(
        ...     dataset, batch_size=32, num_modalities=3
        ... )
        >>> dataloader = DataLoader(
        ...     dataset, batch_sampler=sampler, collate_fn=collator
        ... )
        >>> for batch in dataloader:
        ...     print(batch["modality_mask"].sum(dim=0))
    """

    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size: int = 32,
        num_modalities: int = 3,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
        replacement: bool = True,
        max_samples_per_modality: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_modalities = num_modalities
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.replacement = replacement
        self.max_samples_per_modality = max_samples_per_modality

        self._rng = random.Random(seed)
        self._epoch = 0

        self.modality_groups: Dict[str, List[int]] = {}
        self._build_modality_groups()

        self.batch_size_per_modality = max(1, self.batch_size // self.num_modalities)

        self.num_samples = self._compute_num_samples()

    def _build_modality_groups(self):
        """Build groups of sample indices by modality.

        Uses the dataset's modality index to efficiently partition
        samples into modality groups. Falls back to iterating over
        all samples if the index is not available.
        """
        self.modality_groups.clear()

        target_modalities = list(SUPPORTED_MODALITIES)[:self.num_modalities]

        for modality in target_modalities:
            try:
                indices = self.dataset.get_modality_indices(modality)
            except (AttributeError, ValueError):
                indices = []
            if not indices:
                for idx in range(len(self.dataset)):
                    sample_modalities = []
                    if hasattr(self.dataset, "_get_present_modalities"):
                        sample = self.dataset.samples[idx] if hasattr(self.dataset, "samples") and idx < len(self.dataset.samples) else {}
                        sample_modalities = self.dataset._get_present_modalities(sample)
                    if modality in sample_modalities:
                        indices.append(idx)
            self.modality_groups[modality] = indices

        if not self.modality_groups:
            self.modality_groups["text"] = list(range(len(self.dataset)))

    def _compute_num_samples(self) -> int:
        """Compute the total number of samples per epoch.

        The number of samples is determined by the largest modality group
        multiplied by the batch size, ensuring all groups contribute equally.

        Returns:
            Total number of sample indices per epoch.
        """
        if not self.modality_groups:
            return 0

        max_group_size = max(len(indices) for indices in self.modality_groups.values())

        if self.max_samples_per_modality is not None:
            max_group_size = min(max_group_size, self.max_samples_per_modality)

        num_batches = max_group_size // self.batch_size_per_modality
        if not self.drop_last and max_group_size % self.batch_size_per_modality != 0:
            num_batches += 1

        return num_batches * self.batch_size

    def _sample_modality_indices(
        self,
        modality: str,
        num_samples: int,
    ) -> List[int]:
        """Sample indices from a specific modality group.

        If the group has fewer indices than requested and replacement
        is enabled, oversampling is performed.

        Args:
            modality: Modality name to sample from.
            num_samples: Number of indices to sample.

        Returns:
            List of sampled indices.
        """
        indices = self.modality_groups.get(modality, [])

        if not indices:
            return []

        if self.max_samples_per_modality is not None:
            indices = indices[:self.max_samples_per_modality]

        if len(indices) >= num_samples:
            sampled = self._rng.sample(indices, num_samples)
        elif self.replacement:
            sampled = self._rng.choices(indices, k=num_samples)
        else:
            sampled = list(indices)
            remaining = num_samples - len(indices)
            sampled.extend([indices[-1]] * remaining)

        return sampled

    def __iter__(self):
        """Generate batched indices with balanced modality distribution.

        Yields:
            Lists of sample indices, each of length approximately
            batch_size, with balanced modality representation.
        """
        self._rng.seed(self.seed + self._epoch)

        num_batches = self.num_samples // self.batch_size
        if not self.drop_last and self.num_samples % self.batch_size != 0:
            num_batches += 1

        for _ in range(num_batches):
            batch_indices = []

            modalities_list = list(self.modality_groups.keys())

            samples_per_mod = self.batch_size_per_modality
            remainder = self.batch_size - samples_per_mod * len(modalities_list)

            for i, modality in enumerate(modalities_list):
                count = samples_per_mod
                if i < remainder:
                    count += 1
                mod_indices = self._sample_modality_indices(modality, count)
                batch_indices.extend(mod_indices)

            if self.shuffle:
                self._rng.shuffle(batch_indices)

            yield batch_indices

    def __len__(self) -> int:
        """Return the number of batches per epoch.

        Returns:
            Total number of batches in one epoch.
        """
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch: int):
        """Set the current epoch for shuffling.

        Must be called at the beginning of each epoch when using
        distributed training to ensure different shuffling per epoch.

        Args:
            epoch: Current epoch number.
        """
        self._epoch = epoch

    def get_modality_counts_per_batch(self) -> Dict[str, int]:
        """Get the expected modality distribution per batch.

        Returns:
            Dictionary mapping modality names to the number of samples
            from that modality per batch.
        """
        counts = {}
        modalities_list = list(self.modality_groups.keys())
        samples_per_mod = self.batch_size_per_modality
        remainder = self.batch_size - samples_per_mod * len(modalities_list)

        for i, modality in enumerate(modalities_list):
            counts[modality] = samples_per_mod + (1 if i < remainder else 0)

        return counts

    def get_effective_dataset_size(self) -> int:
        """Get the effective dataset size considering oversampling.

        Returns:
            Effective number of unique samples used per epoch.
        """
        total = sum(len(indices) for indices in self.modality_groups.values())
        return min(total, self.num_samples)

    def __repr__(self) -> str:
        """String representation of the sampler.

        Returns:
            Summary of sampler configuration.
        """
        group_info = ", ".join(
            f"{mod}: {len(indices)}"
            for mod, indices in self.modality_groups.items()
        )
        return (
            f"ModalityBalancedSampler("
            f"batch_size={self.batch_size}, "
            f"num_batches={len(self)}, "
            f"num_samples={self.num_samples}, "
            f"groups=[{group_info}])"
        )


# =============================================================================
# Evaluation Metrics
# =============================================================================

class EvaluationMetrics:
    """Comprehensive evaluation metrics for multimodal models.

    Computes standard metrics for evaluating multimodal understanding tasks:
    - Image-text retrieval: Recall@1, Recall@5, Recall@10, Mean Reciprocal Rank
    - Visual Question Answering (VQA): Accuracy
    - Image Captioning: CIDEr, ROUGE-L, BLEU-1/2/3/4, METEOR, SPICE-like

    The class accumulates predictions across batches and computes aggregate
    metrics. It supports both single-reference and multi-reference evaluation.

    Args:
        num_recall_k: List of K values for recall computation.
                      Default: [1, 5, 10].
        compute_cider: Whether to compute CIDEr score. Default: True.
        compute_rouge: Whether to compute ROUGE scores. Default: True.
        compute_bleu: Whether to compute BLEU scores. Default: True.
        compute_meteor: Whether to compute METEOR score. Default: True.
        max_ngrams: Maximum n-gram order for CIDEr. Default: 4.
        cider_sigma: Gaussian smoothing for CIDEr. Default: 6.0.

    Example:
        >>> metrics = EvaluationMetrics()
        >>> for batch in dataloader:
        ...     predictions = model(batch)
        ...     metrics.add_batch(
        ...         queries=vision_embeds,
        ...         keys=text_embeds,
        ...         query_ids=batch["image_ids"],
        ...         key_ids=batch["text_ids"],
        ...     )
        >>> metrics.add_caption_batch(
        ...     predictions=["a cat on a mat"],
        ...     references=[["a cat sitting on a mat"]],
        ... )
        >>> results = metrics.compute()
        >>> print(results["retrieval"]["recall@1"])
    """

    def __init__(
        self,
        num_recall_k: Optional[List[int]] = None,
        compute_cider: bool = True,
        compute_rouge: bool = True,
        compute_bleu: bool = True,
        compute_meteor: bool = True,
        max_ngrams: int = 4,
        cider_sigma: float = 6.0,
    ):
        if num_recall_k is None:
            num_recall_k = [1, 5, 10]

        self.num_recall_k = sorted(num_recall_k)
        self.compute_cider = compute_cider
        self.compute_rouge = compute_rouge
        self.compute_bleu = compute_bleu
        self.compute_meteor = compute_meteor
        self.max_ngrams = max_ngrams
        self.cider_sigma = cider_sigma

        self._retrieval_queries: List[torch.Tensor] = []
        self._retrieval_keys: List[torch.Tensor] = []
        self._retrieval_query_ids: List[Any] = []
        self._retrieval_key_ids: List[Any] = []
        self._retrieval_positive_pairs: List[Tuple[Any, Any]] = []

        self._caption_predictions: List[str] = []
        self._caption_references: List[List[str]] = []

        self._vqa_predictions: List[str] = []
        self._vqa_answers: List[Union[str, List[str]]] = []

        self._bleu_scores: List[List[float]] = []
        self._rouge_scores: List[List[float]] = []
        self._cider_scores: List[float] = []

    def reset(self):
        """Reset all accumulated predictions and metrics.

        Clears all stored retrieval embeddings, caption predictions,
        VQA answers, and intermediate scores.
        """
        self._retrieval_queries.clear()
        self._retrieval_keys.clear()
        self._retrieval_query_ids.clear()
        self._retrieval_key_ids.clear()
        self._retrieval_positive_pairs.clear()
        self._caption_predictions.clear()
        self._caption_references.clear()
        self._vqa_predictions.clear()
        self._vqa_answers.clear()
        self._bleu_scores.clear()
        self._rouge_scores.clear()
        self._cider_scores.clear()

    def add_batch(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_ids: Optional[Sequence[Any]] = None,
        key_ids: Optional[Sequence[Any]] = None,
        positive_pairs: Optional[Sequence[Tuple[Any, Any]]] = None,
    ):
        """Add a batch of embeddings for retrieval evaluation.

        Args:
            queries: Query embeddings, shape (B, D).
            keys: Key (candidate) embeddings, shape (B, D) or (B, K, D).
            query_ids: Optional unique identifiers for queries.
            key_ids: Optional unique identifiers for keys.
            positive_pairs: Optional list of (query_id, key_id) positive pairs.
        """
        with torch.no_grad():
            if keys.dim() == 3:
                keys_flat = keys.view(-1, keys.shape[-1])
                if key_ids is not None:
                    key_ids_expanded = []
                    kid = 0
                    for kid_val in key_ids:
                        for _ in range(keys.shape[1]):
                            key_ids_expanded.append(f"{kid_val}_{kid}")
                            kid += 1
                    key_ids = key_ids_expanded
            else:
                keys_flat = keys

            self._retrieval_queries.append(queries.cpu())
            self._retrieval_keys.append(keys_flat.cpu())

            if query_ids is not None:
                self._retrieval_query_ids.extend(list(query_ids))
            if key_ids is not None:
                self._retrieval_key_ids.extend(list(key_ids))
            if positive_pairs is not None:
                self._retrieval_positive_pairs.extend(list(positive_pairs))

    def add_caption_batch(
        self,
        predictions: List[str],
        references: List[List[str]],
    ):
        """Add a batch of caption predictions and references.

        Computes incremental metrics for each prediction.

        Args:
            predictions: List of predicted caption strings.
            references: List of reference caption lists (multiple references
                        per prediction are supported).

        Raises:
            ValueError: If predictions and references lengths differ.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"predictions length ({len(predictions)}) must match "
                f"references length ({len(references)})"
            )

        for pred, refs in zip(predictions, references):
            self._caption_predictions.append(pred)
            self._caption_references.append(refs)

            pred_tokens = tokenize_words(pred)

            if self.compute_bleu:
                bleu_scores = []
                for n in range(1, 5):
                    ref_bleus = []
                    for ref in refs:
                        ref_tokens = tokenize_words(ref)
                        score = compute_bleu_single(ref_tokens, pred_tokens, max_n=n)
                        ref_bleus.append(score)
                    bleu_scores.append(max(ref_bleus) if ref_bleus else 0.0)
                self._bleu_scores.append(bleu_scores)

            if self.compute_rouge:
                rouge_scores = []
                for n in [1, 2, 3]:
                    ref_rouges = []
                    for ref in refs:
                        ref_tokens = tokenize_words(ref)
                        score = compute_rouge_n(ref_tokens, pred_tokens, n=n)
                        ref_rouges.append(score)
                    rouge_scores.append(max(ref_rouges) if ref_rouges else 0.0)
                self._rouge_scores.append(rouge_scores)

            if self.compute_cider:
                ref_token_lists = [tokenize_words(ref) for ref in refs]
                cider = compute_cider_score(
                    ref_token_lists, pred_tokens,
                    n=self.max_ngrams, sigma=self.cider_sigma,
                )
                self._cider_scores.append(cider)

    def add_vqa_batch(
        self,
        predictions: List[str],
        answers: List[Union[str, List[str]]],
    ):
        """Add a batch of VQA predictions and ground truth answers.

        Args:
            predictions: List of predicted answer strings.
            answers: List of ground truth answers (string or list of strings
                     for multiple acceptable answers).
        """
        if len(predictions) != len(answers):
            raise ValueError(
                f"predictions length ({len(predictions)}) must match "
                f"answers length ({len(answers)})"
            )

        self._vqa_predictions.extend(predictions)
        self._vqa_answers.extend(answers)

    def _compute_retrieval_metrics(self) -> Dict[str, Any]:
        """Compute image-text retrieval metrics from accumulated embeddings.

        Computes Recall@K for each configured K, Mean Reciprocal Rank (MRR),
        and Median Rank.

        Returns:
            Dictionary with retrieval metrics:
                - recall@{k}: Recall at rank k.
                - mrr: Mean Reciprocal Rank.
                - median_rank: Median rank of the correct item.
                - mean_rank: Mean rank of the correct item.
        """
        if not self._retrieval_queries or not self._retrieval_keys:
            return {}

        all_queries = torch.cat(self._retrieval_queries, dim=0)
        all_keys = torch.cat(self._retrieval_keys, dim=0)

        similarity = compute_cosine_similarity(all_queries, all_keys)

        results = {}

        for k in self.num_recall_k:
            topk_indices = similarity.topk(min(k, similarity.shape[1]), dim=-1).indices
            recall_values = []

            if self._retrieval_positive_pairs:
                query_id_to_idx = {}
                for idx, qid in enumerate(self._retrieval_query_ids):
                    query_id_to_idx[qid] = idx

                key_id_to_idx = {}
                for idx, kid in enumerate(self._retrieval_key_ids):
                    key_id_to_idx[kid] = idx

                for qid, kid in self._retrieval_positive_pairs:
                    if qid in query_id_to_idx and kid in key_id_to_idx:
                        q_idx = query_id_to_idx[qid]
                        k_idx = key_id_to_idx[kid]
                        if k_idx in topk_indices[q_idx]:
                            recall_values.append(1.0)
                        else:
                            recall_values.append(0.0)
            else:
                diag_indices = torch.arange(
                    min(similarity.shape[0], similarity.shape[1]),
                    device=similarity.device,
                )
                for i in range(min(similarity.shape[0], similarity.shape[1])):
                    if diag_indices[i] in topk_indices[i]:
                        recall_values.append(1.0)
                    else:
                        recall_values.append(0.0)

            if recall_values:
                results[f"recall@{k}"] = sum(recall_values) / len(recall_values)
            else:
                results[f"recall@{k}"] = 0.0

        if self._retrieval_positive_pairs:
            mrr_values = []
            rank_values = []

            for qid, kid in self._retrieval_positive_pairs:
                if qid in query_id_to_idx and kid in key_id_to_idx:
                    q_idx = query_id_to_idx[qid]
                    k_idx = key_id_to_idx[kid]
                    sorted_indices = similarity[q_idx].argsort(dim=-1, descending=True)
                    rank = (sorted_indices == k_idx).nonzero(as_tuple=True)[0]

                    if len(rank) > 0:
                        rank_val = rank[0].item() + 1
                        mrr_values.append(1.0 / rank_val)
                        rank_values.append(rank_val)

            if mrr_values:
                results["mrr"] = sum(mrr_values) / len(mrr_values)
                results["mean_rank"] = sum(rank_values) / len(rank_values)
                sorted_ranks = sorted(rank_values)
                n = len(sorted_ranks)
                if n % 2 == 0:
                    results["median_rank"] = (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2]) / 2
                else:
                    results["median_rank"] = sorted_ranks[n // 2]
        else:
            num_pairs = min(similarity.shape[0], similarity.shape[1])
            diag_indices = torch.arange(num_pairs, device=similarity.device)

            mrr_values = []
            rank_values = []
            for i in range(num_pairs):
                sorted_indices = similarity[i].argsort(dim=-1, descending=True)
                rank = (sorted_indices == diag_indices[i]).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    rank_val = rank[0].item() + 1
                    mrr_values.append(1.0 / rank_val)
                    rank_values.append(rank_val)

            if mrr_values:
                results["mrr"] = sum(mrr_values) / len(mrr_values)
                results["mean_rank"] = sum(rank_values) / len(rank_values)
                sorted_ranks = sorted(rank_values)
                n = len(sorted_ranks)
                if n % 2 == 0:
                    results["median_rank"] = (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2]) / 2
                else:
                    results["median_rank"] = sorted_ranks[n // 2]

        return results

    def _compute_vqa_metrics(self) -> Dict[str, float]:
        """Compute VQA accuracy metrics.

        VQA accuracy is computed as:
        accuracy = min(1, #humans_that_gave_answer / 3)

        For single-reference answers, accuracy is 1.0 if the prediction
        exactly matches and 0.0 otherwise (with optional soft matching).

        Returns:
            Dictionary with VQA metrics:
                - vqa_accuracy: Average VQA accuracy.
                - vqa_exact_match: Exact match accuracy.
                - vqa_soft_match: Soft match accuracy (with stemming).
        """
        if not self._vqa_predictions:
            return {}

        total = len(self._vqa_predictions)
        accuracy_scores = []
        exact_matches = 0
        soft_matches = 0

        for pred, answer in zip(self._vqa_predictions, self._vqa_answers):
            pred_normalized = pred.lower().strip()

            if isinstance(answer, list):
                answer_normalized = [a.lower().strip() for a in answer]
                matching_count = sum(1 for a in answer_normalized if a == pred_normalized)
                accuracy = min(1.0, matching_count / 3.0)
                accuracy_scores.append(accuracy)

                if pred_normalized in answer_normalized:
                    exact_matches += 1

                pred_words = set(pred_normalized.split())
                for ans in answer_normalized:
                    ans_words = set(ans.split())
                    if pred_words == ans_words:
                        soft_matches += 1
                        break
            else:
                answer_normalized = answer.lower().strip()

                if pred_normalized == answer_normalized:
                    accuracy_scores.append(1.0)
                    exact_matches += 1
                    soft_matches += 1
                else:
                    pred_words = set(pred_normalized.split())
                    ans_words = set(answer_normalized.split())
                    if pred_words == ans_words:
                        accuracy_scores.append(0.5)
                        soft_matches += 1
                    else:
                        accuracy_scores.append(0.0)

        results = {
            "vqa_accuracy": sum(accuracy_scores) / max(total, 1),
            "vqa_exact_match": exact_matches / max(total, 1),
            "vqa_soft_match": soft_matches / max(total, 1),
        }

        return results

    def _compute_caption_metrics(self) -> Dict[str, float]:
        """Compute captioning evaluation metrics.

        Aggregates BLEU, ROUGE, CIDEr, and METEOR scores across
        all accumulated predictions.

        Returns:
            Dictionary with captioning metrics:
                - bleu_1, bleu_2, bleu_3, bleu_4: BLEU scores.
                - rouge_1, rouge_2, rouge_3: ROUGE F1 scores.
                - cider: CIDEr score.
                - meteor: METEOR score.
        """
        if not self._caption_predictions:
            return {}

        results = {}

        if self.compute_bleu and self._bleu_scores:
            for n in range(4):
                key = f"bleu_{n + 1}"
                scores = [s[n] for s in self._bleu_scores]
                results[key] = sum(scores) / len(scores)

        if self.compute_rouge and self._rouge_scores:
            for n, name in enumerate(["rouge_1", "rouge_2", "rouge_3"]):
                if n < len(self._rouge_scores[0]) if self._rouge_scores else False:
                    scores = [s[n] for s in self._rouge_scores]
                    results[name] = sum(scores) / len(scores)

        if self.compute_cider and self._cider_scores:
            results["cider"] = sum(self._cider_scores) / len(self._cider_scores)

        if self.compute_meteor and self._caption_predictions:
            meteor_scores = []
            for pred, refs in zip(self._caption_predictions, self._caption_references):
                pred_tokens = tokenize_words(pred)
                best_meteor = 0.0
                for ref in refs:
                    ref_tokens = tokenize_words(ref)
                    meteor = self._compute_meteor_single(pred_tokens, ref_tokens)
                    best_meteor = max(best_meteor, meteor)
                meteor_scores.append(best_meteor)
            results["meteor"] = sum(meteor_scores) / max(len(meteor_scores), 1)

        return results

    def _compute_meteor_single(
        self,
        hypothesis: List[str],
        reference: List[str],
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
    ) -> float:
        """Compute METEOR score for a single hypothesis-reference pair.

        METEOR (Metric for Evaluation of Translation with Explicit ORdering)
        is based on the harmonic mean of unigram precision and recall, with
        a penalty for fragmentation.

        Args:
            hypothesis: Hypothesis token list.
            reference: Reference token list.
            alpha: Weight for precision in F-measure. Default: 0.9.
            beta: Fragmentation penalty weight. Default: 3.0.
            gamma: Fragmentation penalty threshold. Default: 0.5.

        Returns:
            METEOR score as a float.
        """
        if not hypothesis and not reference:
            return 1.0
        if not hypothesis or not reference:
            return 0.0

        hyp_counts = Counter(hypothesis)
        ref_counts = Counter(reference)

        matches = sum((hyp_counts & ref_counts).values())
        precision = matches / len(hypothesis) if hypothesis else 0.0
        recall = matches / len(reference) if reference else 0.0

        if precision + recall == 0:
            return 0.0

        f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)

        chunks = 0
        i = 0
        j = 0
        ref_remaining = list(reference)

        for h_token in hypothesis:
            if h_token in ref_remaining:
                idx = ref_remaining.index(h_token)
                if idx != 0 or (idx == 0 and i == 0):
                    chunks += 1
                ref_remaining.pop(idx)
            i += 1

        if chunks == 0:
            return f_mean

        penalty = gamma * (chunks / matches) ** beta
        meteor = f_mean * (1 - penalty)

        return max(0.0, meteor)

    def compute(self) -> Dict[str, Any]:
        """Compute all evaluation metrics.

        Aggregates all accumulated predictions and computes final metrics
        for retrieval, VQA, and captioning tasks.

        Returns:
            Dictionary with metric categories:
                - retrieval: Dict of retrieval metrics (recall@k, mrr, ranks).
                - vqa: Dict of VQA metrics (accuracy, exact_match, soft_match).
                - captioning: Dict of captioning metrics (bleu, rouge, cider).
                - summary: Dict of high-level summary statistics.
        """
        results = {}

        retrieval_metrics = self._compute_retrieval_metrics()
        if retrieval_metrics:
            results["retrieval"] = retrieval_metrics

        vqa_metrics = self._compute_vqa_metrics()
        if vqa_metrics:
            results["vqa"] = vqa_metrics

        caption_metrics = self._compute_caption_metrics()
        if caption_metrics:
            results["captioning"] = caption_metrics

        summary = {
            "num_retrieval_queries": len(self._retrieval_queries),
            "num_caption_samples": len(self._caption_predictions),
            "num_vqa_samples": len(self._vqa_predictions),
        }
        results["summary"] = summary

        return results

    def compute_single_sample(
        self,
        prediction: str,
        references: List[str],
    ) -> Dict[str, float]:
        """Compute all captioning metrics for a single sample.

        Convenience method for evaluating a single prediction without
        accumulating state.

        Args:
            prediction: Predicted caption string.
            references: List of reference caption strings.

        Returns:
            Dictionary of metric scores for this sample.
        """
        pred_tokens = tokenize_words(prediction)
        ref_token_lists = [tokenize_words(ref) for ref in references]

        metrics = {}

        if self.compute_bleu:
            for n in range(1, 5):
                best_bleu = 0.0
                for ref_tokens in ref_token_lists:
                    bleu = compute_bleu_single(ref_tokens, pred_tokens, max_n=n)
                    best_bleu = max(best_bleu, bleu)
                metrics[f"bleu_{n}"] = best_bleu

        if self.compute_rouge:
            for n in [1, 2, 3]:
                best_rouge = 0.0
                for ref_tokens in ref_token_lists:
                    rouge = compute_rouge_n(ref_tokens, pred_tokens, n=n)
                    best_rouge = max(best_rouge, rouge)
                metrics[f"rouge_{n}"] = best_rouge

        if self.compute_cider:
            metrics["cider"] = compute_cider_score(
                ref_token_lists, pred_tokens,
                n=self.max_ngrams, sigma=self.cider_sigma,
            )

        if self.compute_meteor:
            best_meteor = 0.0
            for ref_tokens in ref_token_lists:
                meteor = self._compute_meteor_single(pred_tokens, ref_tokens)
                best_meteor = max(best_meteor, meteor)
            metrics["meteor"] = best_meteor

        return metrics

    def format_results(
        self,
        results: Dict[str, Any],
        precision: int = 4,
    ) -> str:
        """Format evaluation results as a human-readable string.

        Args:
            results: Dictionary from compute().
            precision: Number of decimal places. Default: 4.

        Returns:
            Formatted string with all metrics.
        """
        lines = ["=" * 60, "Evaluation Results", "=" * 60]

        if "retrieval" in results:
            lines.append("\n--- Image-Text Retrieval ---")
            for key, value in results["retrieval"].items():
                lines.append(f"  {key}: {value:.{precision}f}")

        if "vqa" in results:
            lines.append("\n--- Visual Question Answering ---")
            for key, value in results["vqa"].items():
                lines.append(f"  {key}: {value:.{precision}f}")

        if "captioning" in results:
            lines.append("\n--- Image Captioning ---")
            for key, value in results["captioning"].items():
                lines.append(f"  {key}: {value:.{precision}f}")

        if "summary" in results:
            lines.append("\n--- Summary ---")
            for key, value in results["summary"].items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_metric_names(self) -> List[str]:
        """Get the list of all metric names that will be computed.

        Returns:
            List of metric name strings.
        """
        names = []
        for k in self.num_recall_k:
            names.append(f"recall@{k}")
        names.extend(["mrr", "mean_rank", "median_rank"])
        names.extend(["vqa_accuracy", "vqa_exact_match", "vqa_soft_match"])
        if self.compute_bleu:
            names.extend([f"bleu_{n}" for n in range(1, 5)])
        if self.compute_rouge:
            names.extend(["rouge_1", "rouge_2", "rouge_3"])
        if self.compute_cider:
            names.append("cider")
        if self.compute_meteor:
            names.append("meteor")
        return names

    def __repr__(self) -> str:
        """String representation of the evaluation metrics.

        Returns:
            Summary of configured metrics.
        """
        return (
            f"EvaluationMetrics("
            f"recall_k={self.num_recall_k}, "
            f"cider={self.compute_cider}, "
            f"rouge={self.compute_rouge}, "
            f"bleu={self.compute_bleu}, "
            f"meteor={self.compute_meteor})"
        )
