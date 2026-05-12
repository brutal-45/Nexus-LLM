"""
Audio Encoder Module
====================

Production-grade audio encoder implementations for the Nexus LLM multimodal framework.
Provides multiple audio encoding architectures including Whisper-style, HuBERT-style,
audio tokenization, and supporting components like mel spectrogram computation,
feature extraction, and voice activity detection.

All components implement audio processing from raw waveforms using pure PyTorch
operations — no external audio processing library dependencies.

References:
    - Whisper: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
    - HuBERT: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction" (Hsu et al., 2021)
    - SpecAugment: "SpecAugment: A Simple Data Augmentation Method" (Park et al., 2019)
    - VAD: Various voice activity detection algorithms
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.multimodal.multimodal_config import AudioConfig


# =============================================================================
# Output Data Structures
# =============================================================================

@dataclass
class AudioEncoderOutput:
    """Output from an audio encoder.

    Attributes:
        last_hidden_state: Final hidden state (batch, seq_len, encoder_dim).
        hidden_states: Optional list of all hidden states from each layer.
        attentions: Optional list of attention weights from each layer.
        audio_features: Raw audio features / spectrogram (batch, freq, time).
        attention_mask: Attention mask (batch, seq_len).
        pooler_output: Pooled representation (batch, encoder_dim).
        frame_level_features: Frame-level features before temporal encoding.
    """
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    audio_features: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None
    frame_level_features: Optional[torch.Tensor] = None


@dataclass
class AudioTokenizerOutput:
    """Output from an audio tokenizer.

    Attributes:
        token_ids: Discrete token indices (batch, num_tokens).
        token_embeddings: Token embedding vectors (batch, num_tokens, dim).
        num_tokens: Number of tokens per sample.
        reconstruction: Optional reconstructed audio.
        codebook_indices: Codebook usage indices.
    """
    token_ids: torch.Tensor
    token_embeddings: torch.Tensor
    num_tokens: int
    reconstruction: Optional[torch.Tensor] = None
    codebook_indices: Optional[torch.Tensor] = None


# =============================================================================
# Mel Spectrogram (from scratch using torch)
# =============================================================================

class MelSpectrogram(nn.Module):
    """Custom mel spectrogram computation from raw audio waveforms.

    Implements the complete mel spectrogram pipeline using pure PyTorch operations:
    1. Short-Time Fourier Transform (STFT)
    2. Mel filterbank application
    3. Log power compression

    No external dependencies (librosa, torchaudio) required.

    Args:
        config: Audio configuration object.
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: Optional[int] = None,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        log: str = "log",
        log_offset: float = 1e-6,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        mel_norm: str = "slaney",
    ):
        """Initialize mel spectrogram computation.

        Args:
            config: Audio configuration. Overrides other args if provided.
            sample_rate: Audio sample rate in Hz.
            n_fft: Size of FFT window.
            hop_length: Hop length between frames.
            win_length: Window length. Default: n_fft.
            n_mels: Number of mel filter banks.
            fmin: Minimum frequency.
            fmax: Maximum frequency. Default: sample_rate // 2.
            power: Power for power spectrogram (1.0 = magnitude, 2.0 = power).
            log: Log compression type: 'log', 'log10', 'db', or 'none'.
            log_offset: Offset added before log to avoid log(0).
            normalized: Whether to normalize the STFT.
            center: Pad the waveform so frames are centered.
            pad_mode: Padding mode for centering.
            mel_norm: Mel filterbank normalization ('slaney' or 'none').
        """
        super().__init__()
        if config is not None:
            sample_rate = config.sample_rate
            n_fft = config.n_fft
            hop_length = config.hop_length
            win_length = config.win_length
            n_mels = config.n_mels
            fmin = config.fmin
            fmax = config.fmax

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2.0
        self.power = power
        self.log = log
        self.log_offset = log_offset
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        self.mel_norm = mel_norm

        # Build window function (Hann window)
        self.register_buffer(
            "window",
            self._create_hann_window(self.win_length),
            persistent=False,
        )

        # Build mel filterbank
        mel_fb = self._create_mel_filterbank()
        self.register_buffer("mel_filterbank", mel_fb, persistent=False)

        # Precompute FFT frequency bins
        fft_freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        self.register_buffer("fft_freqs", fft_freqs, persistent=False)

    def _create_hann_window(self, length: int) -> torch.Tensor:
        """Create a Hann window function.

        Args:
            length: Window length in samples.

        Returns:
            Hann window tensor of shape (length,).
        """
        n = torch.arange(length, dtype=torch.float32)
        window = 0.5 * (1.0 - torch.cos(2.0 * math.pi * n / (length - 1)))
        return window

    def _hz_to_mel(self, hz: torch.Tensor) -> torch.Tensor:
        """Convert frequency in Hz to mel scale.

        Args:
            hz: Frequency values in Hz.

        Returns:
            Frequency values in mel scale.
        """
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert frequency in mel scale to Hz.

        Args:
            mel: Frequency values in mel scale.

        Returns:
            Frequency values in Hz.
        """
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel-spaced filterbank matrix.

        Constructs triangular filters spaced on the mel scale, mapping
        from linear frequency bins to mel frequency bands.

        Returns:
            Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
        """
        n_freqs = self.n_fft // 2 + 1

        mel_min = self._hz_to_mel(torch.tensor(self.fmin))
        mel_max = self._hz_to_mel(torch.tensor(self.fmax))
        mel_points = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        fft_freqs = torch.linspace(0, self.sample_rate / 2, n_freqs)

        filterbank = torch.zeros(self.n_mels, n_freqs)

        for i in range(self.n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]

            # Rising slope
            rising = (fft_freqs - lower) / max(center - lower, 1e-10)
            rising = torch.clamp(rising, min=0.0, max=1.0)

            # Falling slope
            falling = (upper - fft_freqs) / max(upper - center, 1e-10)
            falling = torch.clamp(falling, min=0.0, max=1.0)

            filterbank[i] = rising * falling

        # Slaney normalization: normalize by bandwidth
        if self.mel_norm == "slaney":
            enorm = 2.0 / (hz_points[2:] - hz_points[:-2])
            filterbank *= enorm.unsqueeze(1)

        return filterbank

    def _stft(
        self,
        waveform: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Short-Time Fourier Transform.

        Args:
            waveform: Audio waveform of shape (batch, num_samples).

        Returns:
            Tuple of (magnitude, phase) each of shape (batch, n_fft//2+1, num_frames).
        """
        if self.center:
            pad_length = self.n_fft // 2
            waveform = F.pad(
                waveform, (pad_length, pad_length), mode=self.pad_mode
            )

        # Extract overlapping windows
        batch_size = waveform.shape[0]
        num_samples = waveform.shape[1]
        num_frames = 1 + (num_samples - self.n_fft) // self.hop_length

        # Create frame indices
        frame_starts = torch.arange(
            0, num_frames, device=waveform.device
        ) * self.hop_length
        frame_starts = frame_starts.unsqueeze(0).unsqueeze(2)
        window_offsets = torch.arange(
            self.win_length, device=waveform.device
        ).unsqueeze(0).unsqueeze(0)

        indices = frame_starts + window_offsets
        frames = waveform[:, indices.squeeze(0)]

        # Apply window
        windowed = frames * self.window.unsqueeze(0).unsqueeze(1)

        # Apply zero-padding if win_length < n_fft
        if self.win_length < self.n_fft:
            pad_size = self.n_fft - self.win_length
            windowed = F.pad(windowed, (0, pad_size))

        # Compute FFT
        fft_result = torch.fft.rfft(windowed, dim=-1)

        magnitude = torch.abs(fft_result)
        phase = torch.angle(fft_result)

        if self.normalized:
            magnitude = magnitude / torch.sqrt(
                torch.tensor(self.n_fft, dtype=magnitude.dtype, device=magnitude.device)
            )

        return magnitude, phase

    def _apply_log_compression(
        self,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """Apply logarithmic compression to spectrogram.

        Args:
            spectrogram: Power spectrogram (batch, freq, time).

        Returns:
            Log-compressed spectrogram.
        """
        spectrogram = spectrogram + self.log_offset

        if self.log == "log":
            return torch.log(spectrogram)
        elif self.log == "log10":
            return torch.log10(spectrogram)
        elif self.log == "db":
            return 10.0 * torch.log10(spectrogram)
        else:
            return spectrogram

    def forward(
        self,
        waveform: torch.Tensor,
        return_phase: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute mel spectrogram from waveform.

        Args:
            waveform: Audio waveform of shape (batch, num_samples).
                     Should be single-channel (mono).
            return_phase: Whether to also return phase information.

        Returns:
            Mel spectrogram of shape (batch, n_mels, num_frames).
            If return_phase, also returns phase tensor.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # Compute STFT
        magnitude, phase = self._stft(waveform)

        # Compute power spectrogram
        if self.power == 1.0:
            power_spec = magnitude
        elif self.power == 2.0:
            power_spec = magnitude ** 2
        else:
            power_spec = magnitude ** self.power

        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_filterbank, power_spec)

        # Apply log compression
        if self.log != "none":
            mel_spec = self._apply_log_compression(mel_spec)

        if return_phase:
            return mel_spec, phase
        return mel_spec

    def inverse(self, mel_spec: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
        """Approximate inverse mel spectrogram (griffin-lim).

        Args:
            mel_spec: Mel spectrogram (batch, n_mels, num_frames).
            n_iter: Number of Griffin-Lim iterations.

        Returns:
            Reconstructed waveform of shape (batch, num_samples).
        """
        with torch.no_grad():
            # Approximate linear spectrogram from mel spectrogram
            mel_power = torch.exp(mel_spec) - self.log_offset
            mel_power = torch.clamp(mel_power, min=0.0)

            # Pseudo-inverse of mel filterbank
            mel_fb_t = self.mel_filterbank.t()
            mel_fb_pinv = torch.linalg.pinv(self.mel_filterbank)
            linear_spec = torch.matmul(mel_fb_pinv, mel_power)
            linear_spec = torch.clamp(linear_spec, min=0.0)

            # Estimate magnitude from power
            magnitude = torch.sqrt(linear_spec + self.log_offset)

            # Random phase initialization
            num_frames = mel_spec.shape[-1]
            num_samples = self.hop_length * (num_frames - 1) + self.n_fft
            phase = 2.0 * math.pi * torch.rand(
                magnitude.shape[0], 1, num_frames,
                device=magnitude.device, dtype=magnitude.dtype
            )

            for _ in range(n_iter):
                # Construct complex spectrogram
                complex_spec = magnitude * torch.exp(1j * phase)

                # Inverse FFT
                frames = torch.fft.irfft(complex_spec, n=self.n_fft, dim=-1)

                # Overlap-add
                waveform = self._overlap_add(frames, num_samples)

                # Re-compute STFT to get consistent phase
                magnitude_new, phase = self._stft(waveform)

            # Final reconstruction
            complex_spec = magnitude * torch.exp(1j * phase)
            frames = torch.fft.irfft(complex_spec, n=self.n_fft, dim=-1)
            waveform = self._overlap_add(frames, num_samples)

        return waveform

    def _overlap_add(
        self,
        frames: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Overlap-add frames to reconstruct waveform.

        Args:
            frames: STFT frames (batch, n_fft//2+1, num_frames).
            num_samples: Target number of output samples.

        Returns:
            Reconstructed waveform (batch, num_samples).
        """
        batch_size, _, num_frames = frames.shape

        # Apply inverse window
        if self.win_length < self.n_fft:
            pad_size = self.n_fft - self.win_length
            window = F.pad(self.window, (0, pad_size))
        else:
            window = self.window[:self.n_fft]

        window_sum = torch.zeros(
            batch_size, num_samples + self.n_fft,
            device=frames.device, dtype=frames.dtype,
        )
        waveform = torch.zeros_like(window_sum)

        for t in range(num_frames):
            start = t * self.hop_length
            end = start + self.n_fft
            if end <= num_samples + self.n_fft:
                waveform[:, start:end] += frames[:, :, t] * window
                window_sum[:, start:end] += window ** 2

        window_sum = torch.clamp(window_sum, min=1e-10)
        waveform = waveform[:, :num_samples] / window_sum[:, :num_samples]
        return waveform

    def get_num_frames(self, num_samples: int) -> int:
        """Compute number of frames for given number of samples.

        Args:
            num_samples: Number of audio samples.

        Returns:
            Number of spectrogram frames.
        """
        if self.center:
            num_samples += self.n_fft
        return max(1, (num_samples - self.n_fft) // self.hop_length + 1)


# =============================================================================
# Audio Feature Extractor
# =============================================================================

class AudioFeatureExtractor(nn.Module):
    """Frame-level audio feature extraction.

    Extracts various features from audio waveforms or spectrograms including:
    - Delta features (first and second derivatives)
    - Cepstral Mean and Variance Normalization (CMVN)
    - Feature stacking and subsampling
    - Energy-based features
    """

    def __init__(
        self,
        feature_dim: int = 80,
        num_delta_features: int = 2,
        use_cmvn: bool = True,
        stack_frames: int = 1,
        subsample: int = 1,
        specaugment: bool = False,
        specaugment_time_mask: int = 100,
        specaugment_freq_mask: int = 80,
        specaugment_num_time_masks: int = 2,
        specaugment_num_freq_masks: int = 2,
    ):
        """Initialize audio feature extractor.

        Args:
            feature_dim: Input feature dimension.
            num_delta_features: Number of delta feature orders (0=none, 1=delta, 2=delta+delta-delta).
            use_cmvn: Apply cepstral mean/variance normalization.
            stack_frames: Stack this many consecutive frames.
            subsample: Subsample rate after stacking.
            specaugment: Apply SpecAugment.
            specaugment_time_mask: SpecAugment time mask parameter.
            specaugment_freq_mask: SpecAugment freq mask parameter.
            specaugment_num_time_masks: Number of time masks.
            specaugment_num_freq_masks: Number of frequency masks.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_delta_features = num_delta_features
        self.use_cmvn = use_cmvn
        self.stack_frames = stack_frames
        self.subsample = subsample
        self.specaugment = specaugment
        self.specaugment_time_mask = specaugment_time_mask
        self.specaugment_freq_mask = specaugment_freq_mask
        self.specaugment_num_time_masks = specaugment_num_time_masks
        self.specaugment_num_freq_masks = specaugment_num_freq_masks

        self.output_dim = feature_dim * (num_delta_features + 1) * stack_frames

    def compute_delta(
        self,
        features: torch.Tensor,
        order: int = 1,
    ) -> torch.Tensor:
        """Compute delta (derivative) features.

        Uses the regression formula:
            d[t] = sum_{n=1}^{N} n * (c[t+n] - c[t-n]) / (2 * sum_{n=1}^{N} n^2)

        Args:
            features: Input features (batch, dim, time).
            order: Delta order (1 = first derivative, 2 = acceleration).

        Returns:
            Delta features of same shape.
        """
        N = 2
        denominator = 2 * sum(n ** 2 for n in range(1, N + 1))

        # Pad features
        padded = F.pad(features, (N, N), mode="replicate")

        delta = torch.zeros_like(features)
        for n in range(1, N + 1):
            delta += n * (padded[:, :, 2 * N - n:-n] - padded[:, :, n:2 * N + n])

        delta = delta / denominator

        return delta

    def apply_cmvn(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Cepstral Mean and Variance Normalization.

        Args:
            features: Features (batch, dim, time).
            attention_mask: Optional mask (batch, time).

        Returns:
            Normalized features.
        """
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()
            num_frames = mask_expanded.sum(dim=-1, keepdim=True)
            num_frames = torch.clamp(num_frames, min=1.0)

            mean = (features * mask_expanded).sum(dim=-1, keepdim=True) / num_frames
            variance = ((features - mean) ** 2 * mask_expanded).sum(
                dim=-1, keepdim=True
            ) / num_frames
        else:
            mean = features.mean(dim=-1, keepdim=True)
            variance = features.var(dim=-1, unbiased=False, keepdim=True)

        std = torch.sqrt(variance + 1e-6)
        features = (features - mean) / std

        return features

    def stack_and_subsample(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Stack consecutive frames and optionally subsample.

        Args:
            features: Features (batch, dim, time).

        Returns:
            Stacked features (batch, dim * stack_frames, time // subsample).
        """
        batch_size, dim, time_steps = features.shape

        if self.stack_frames > 1:
            # Pad to make divisible by stack_frames
            pad_size = (
                self.stack_frames - (time_steps % self.stack_frames)
            ) % self.stack_frames
            if pad_size > 0:
                features = F.pad(features, (0, pad_size), mode="replicate")

            time_steps = features.shape[2]
            features = features.reshape(
                batch_size, dim, time_steps // self.stack_frames, self.stack_frames
            )
            features = features.permute(0, 1, 3, 2).reshape(
                batch_size, dim * self.stack_frames, time_steps // self.stack_frames
            )

        if self.subsample > 1:
            features = features[:, :, ::self.subsample]

        return features

    def apply_specaugment(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpecAugment data augmentation.

        Args:
            features: Features (batch, dim, time).

        Returns:
            Augmented features.
        """
        if not self.training or not self.specaugment:
            return features

        batch_size, dim, time_steps = features.shape
        augmented = features.clone()

        # Time masking
        for _ in range(self.specaugment_num_time_masks):
            t = torch.randint(
                0, self.specaugment_time_mask, (batch_size,)
            ).to(features.device)
            t0 = torch.randint(
                0, max(time_steps - t.min().item(), 1), (batch_size,)
            ).to(features.device)

            for b in range(batch_size):
                t_end = min(t0[b] + t[b], time_steps)
                if t[b] > 0 and t0[b] < time_steps:
                    augmented[b, :, t0[b]:t_end] = 0.0

        # Frequency masking
        for _ in range(self.specaugment_num_freq_masks):
            f = torch.randint(
                0, min(self.specaugment_freq_mask, dim), (batch_size,)
            ).to(features.device)
            f0 = torch.randint(
                0, max(dim - f.min().item(), 1), (batch_size,)
            ).to(features.device)

            for b in range(batch_size):
                f_end = min(f0[b] + f[b], dim)
                if f[b] > 0 and f0[b] < dim:
                    augmented[b, f0[b]:f_end, :] = 0.0

        return augmented

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract and process audio features.

        Args:
            features: Input features (batch, dim, time).
            attention_mask: Optional attention mask (batch, time).

        Returns:
            Tuple of (processed features, updated attention mask).
        """
        # Apply SpecAugment
        features = self.apply_specaugment(features)

        # Compute delta features
        all_features = [features]
        for order in range(1, self.num_delta_features + 1):
            delta = self.compute_delta(features, order=order)
            all_features.append(delta)

        features = torch.cat(all_features, dim=1)

        # Apply CMVN
        if self.use_cmvn:
            features = self.apply_cmvn(features, attention_mask)

        # Stack and subsample
        if self.stack_frames > 1 or self.subsample > 1:
            features = self.stack_and_subsample(features)
            if attention_mask is not None:
                attention_mask = attention_mask[:, ::self.subsample]

        return features, attention_mask


# =============================================================================
# Whisper-style Audio Encoder
# =============================================================================

class WhisperConvStem(nn.Module):
    """Convolutional stem for Whisper encoder.

    Processes the mel spectrogram with two 1D convolutional layers to
    reduce the temporal resolution and project to encoder dimension.
    """

    def __init__(
        self,
        in_channels: int = 128,
        encoder_dim: int = 768,
        num_conv_layers: int = 2,
    ):
        """Initialize convolutional stem.

        Args:
            in_channels: Number of mel frequency bins.
            encoder_dim: Output encoder dimension.
            num_conv_layers: Number of convolution layers.
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()

        in_ch = in_channels
        out_ch = encoder_dim
        stride = 2

        for i in range(num_conv_layers):
            kernel_size = 3
            padding = kernel_size // 2
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.GELU(),
                nn.LayerNorm(out_ch),
            ))
            in_ch = out_ch
            if i == 0:
                stride = 2
                out_ch = out_ch

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """Apply convolutional stem.

        Args:
            mel_spectrogram: Mel spectrogram (batch, n_mels, time).

        Returns:
            Convolved features (batch, encoder_dim, time // 4).
        """
        hidden_states = mel_spectrogram
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class WhisperBlock(nn.Module):
    """Whisper transformer encoder block.

    Implements the attention and FFN layers used in the Whisper audio encoder.
    Uses pre-normalization with GELU activation.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize Whisper block.

        Args:
            encoder_dim: Encoder hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: FFN expansion ratio.
            dropout: General dropout.
            attention_dropout: Attention dropout.
            layer_norm_eps: Layer norm epsilon.
        """
        super().__init__()
        self.embed_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = encoder_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(encoder_dim, eps=layer_norm_eps)
        self.q_proj = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.k_proj = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.v_proj = nn.Linear(encoder_dim, encoder_dim, bias=False)
        self.out_proj = nn.Linear(encoder_dim, encoder_dim, bias=False)

        self.norm2 = nn.LayerNorm(encoder_dim, eps=layer_norm_eps)
        self.fc1 = nn.Linear(encoder_dim, int(encoder_dim * mlp_ratio), bias=True)
        self.fc2 = nn.Linear(int(encoder_dim * mlp_ratio), encoder_dim, bias=True)
        self.act = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attention_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Whisper block.

        Args:
            hidden_states: Input tensor (batch, seq_len, encoder_dim).
            attention_mask: Optional attention mask.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of (output, attention_weights).
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_weights = attn_weights + attention_mask[:, None, None, :]
            elif attention_mask.dim() == 4:
                attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(hidden_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)
        hidden_states = residual + self.dropout1(attn_output)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.act(self.fc1(hidden_states))
        ffn_output = self.fc2(ffn_output)
        hidden_states = residual + self.dropout2(ffn_output)

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states, None


class WhisperEncoder(nn.Module):
    """Whisper-style audio encoder.

    Implements the encoder from "Robust Speech Recognition via Large-Scale
    Weak Supervision" (Radford et al., 2022).

    Architecture:
    1. Mel spectrogram computation
    2. Convolutional stem (2 conv layers with stride 2)
    3. Sinusoidal position embeddings
    4. N transformer encoder blocks
    5. Final layer normalization

    Args:
        config: Audio configuration object.
    """

    def __init__(self, config: AudioConfig):
        """Initialize Whisper encoder.

        Args:
            config: Audio configuration.
        """
        super().__init__()
        self.config = config

        # Mel spectrogram
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )

        # Convolutional stem
        if config.use_conv_stem:
            self.conv_stem = WhisperConvStem(
                in_channels=config.n_mels,
                encoder_dim=config.encoder_dim,
                num_conv_layers=config.num_conv_layers,
            )
        else:
            self.conv_stem = None
            self.input_proj = nn.Linear(config.n_mels, config.encoder_dim)

        # Sinusoidal position embeddings
        self.max_position_embeddings = 1500
        self.position_embedding = self._build_sinusoidal_positions(
            config.encoder_dim,
            self.max_position_embeddings,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            WhisperBlock(
                encoder_dim=config.encoder_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

        self.gradient_checkpointing = config.use_gradient_checkpointing

    def _build_sinusoidal_positions(
        self,
        dim: int,
        max_len: int,
    ) -> torch.Tensor:
        """Build sinusoidal position embedding table.

        Args:
            dim: Embedding dimension.
            max_len: Maximum sequence length.

        Returns:
            Position embedding tensor (max_len, dim).
        """
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        waveform: torch.Tensor,
        mel_spectrogram: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> AudioEncoderOutput:
        """Forward pass through Whisper encoder.

        Args:
            waveform: Raw audio waveform (batch, num_samples).
            mel_spectrogram: Pre-computed mel spectrogram (optional).
            attention_mask: Optional attention mask.
            output_hidden_states: Return all hidden states.
            output_attentions: Return attention weights.
            return_dict: Return structured output.

        Returns:
            AudioEncoderOutput with encoded features.
        """
        if mel_spectrogram is None:
            mel_spectrogram = self.mel_spectrogram(waveform)

        # Store original features
        frame_level_features = mel_spectrogram

        # Apply convolutional stem or linear projection
        if self.conv_stem is not None:
            hidden_states = self.conv_stem(mel_spectrogram)
        else:
            hidden_states = mel_spectrogram.transpose(1, 2)
            hidden_states = self.input_proj(hidden_states)

        # Add position embeddings
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype

        if seq_len > self.max_position_embeddings:
            pos_embed = self._build_sinusoidal_positions(
                self.config.encoder_dim, seq_len
            ).to(device=device, dtype=dtype)
        else:
            pos_embed = self.position_embedding[:seq_len].to(
                device=device, dtype=dtype
            )

        hidden_states = hidden_states + pos_embed.unsqueeze(0)
        hidden_states = self.dropout(hidden_states)

        # Update attention mask for downsampled sequence
        if attention_mask is not None and self.conv_stem is not None:
            attention_mask = attention_mask[:, ::4]
            attention_mask = attention_mask[:, :seq_len]

        # Create default attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[0], seq_len,
                device=device, dtype=torch.float32,
            )

        # Expand attention mask for multi-head attention
        expanded_mask = self._expand_attention_mask(attention_mask, seq_len, device)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Pass through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, expanded_mask, output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights = block(
                    hidden_states,
                    attention_mask=expanded_mask,
                    output_attentions=output_attentions,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.final_norm(hidden_states)

        # Pool over time for global representation
        mask_expanded = attention_mask.unsqueeze(-1)
        pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)

        if not return_dict:
            return (
                hidden_states, all_hidden_states, all_attentions,
                frame_level_features, attention_mask, pooled,
            )

        return AudioEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            audio_features=frame_level_features,
            attention_mask=attention_mask,
            pooler_output=pooled,
            frame_level_features=frame_level_features,
        )

    def _expand_attention_mask(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand 2D attention mask to 4D for multi-head attention.

        Args:
            attention_mask: 2D mask (batch, seq_len).
            seq_len: Target sequence length.
            device: Target device.

        Returns:
            4D attention mask (batch, 1, 1, seq_len).
        """
        if attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :]
            expanded_mask = (1.0 - expanded_mask.float()) * torch.finfo(
                torch.float32
            ).min
            if expanded_mask.shape[-1] < seq_len:
                pad_size = seq_len - expanded_mask.shape[-1]
                expanded_mask = F.pad(expanded_mask, (0, pad_size), value=0.0)
            elif expanded_mask.shape[-1] > seq_len:
                expanded_mask = expanded_mask[:, :, :, :seq_len]
            return expanded_mask
        return attention_mask


# =============================================================================
# HuBERT-style Audio Encoder
# =============================================================================

class HuBERTFeatureExtractor(nn.Module):
    """CNN feature extractor for HuBERT.

    Extracts frame-level features from raw audio using stacked
    1D convolutions with GELU activation, similar to wav2vec 2.0.
    """

    def __init__(
        self,
        num_channels: int = 1,
        encoder_dim: int = 768,
        num_conv_layers: int = 7,
        conv_dim: int = 512,
        conv_kernel_sizes: Optional[List[int]] = None,
        conv_strides: Optional[List[int]] = None,
    ):
        """Initialize HuBERT feature extractor.

        Args:
            num_channels: Number of input audio channels.
            encoder_dim: Final output dimension.
            num_conv_layers: Number of convolutional layers.
            conv_dim: Hidden dimension for conv layers.
            conv_kernel_sizes: Kernel sizes per layer. Default: [10,3,3,3,3,2,2].
            conv_strides: Strides per layer. Default: [5,2,2,2,2,2,2].
        """
        super().__init__()
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [10, 3, 3, 3, 3, 2, 2]
        if conv_strides is None:
            conv_strides = [5, 2, 2, 2, 2, 2, 2]

        self.conv_layers = nn.ModuleList()
        in_dim = num_channels

        for i in range(num_conv_layers):
            out_dim = conv_dim if i < num_conv_layers - 1 else encoder_dim
            kernel_size = conv_kernel_sizes[i] if i < len(conv_kernel_sizes) else 3
            stride = conv_strides[i] if i < len(conv_strides) else 2

            layer = nn.Sequential(
                nn.Conv1d(
                    in_dim, out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.Dropout(0.1),
                nn.GroupNorm(out_dim, out_dim, affine=True),
                nn.GELU(),
            )
            self.conv_layers.append(layer)
            in_dim = out_dim

        self.num_layers = num_conv_layers

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from waveform.

        Args:
            waveform: Raw audio (batch, num_samples) or (batch, 1, num_samples).
            attention_mask: Optional attention mask (batch, num_samples).

        Returns:
            Tuple of (features (batch, encoder_dim, time), updated attention_mask).
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden_states = waveform
        conv_feature_masks = None

        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        # Compute attention mask for output
        if attention_mask is not None:
            conv_feature_masks = self._compute_conv_attention_mask(
                attention_mask, waveform.shape[-1], hidden_states.shape[-1],
                hidden_states.device,
            )

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states, conv_feature_masks

    def _compute_conv_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_length: int,
        output_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute attention mask after convolutional downsampling.

        Args:
            attention_mask: Input attention mask (batch, input_length).
            input_length: Original input length.
            output_length: Output sequence length after convs.
            device: Target device.

        Returns:
            Output attention mask (batch, output_length).
        """
        # Interpolate mask to match output length
        mask = attention_mask.float().unsqueeze(1)
        mask = F.interpolate(
            mask, size=output_length, mode="nearest"
        ).squeeze(1)

        # Binary threshold: if any input position in the receptive field is valid
        mask = (mask > 0.5).float()

        return mask


class HuBERTBlock(nn.Module):
    """HuBERT transformer encoder block.

    Follows the pre-norm transformer architecture with relative
    position bias support.
    """

    def __init__(
        self,
        encoder_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1500,
    ):
        """Initialize HuBERT block.

        Args:
            encoder_dim: Encoder dimension.
            num_heads: Number of attention heads.
            mlp_ratio: FFN expansion ratio.
            dropout: General dropout.
            attention_dropout: Attention dropout.
            layer_norm_eps: Layer norm epsilon.
            max_position_embeddings: Max sequence length.
        """
        super().__init__()
        self.embed_dim = encoder_dim
        self.num_heads = num_heads
        self.head_dim = encoder_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(encoder_dim, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            encoder_dim, num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(encoder_dim, eps=layer_norm_eps)
        intermediate_size = int(encoder_dim * mlp_ratio)
        self.fc1 = nn.Linear(encoder_dim, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, encoder_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through HuBERT block.

        Args:
            hidden_states: Input tensor (batch, seq_len, encoder_dim).
            attention_mask: Optional attention mask.
            output_attentions: Return attention weights.

        Returns:
            Tuple of (output, attention_weights).
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        # PyTorch MultiheadAttention expects mask in (batch*num_heads, seq, seq)
        attn_output, attn_weights = self.self_attn(
            hidden_states,
            hidden_states,
            hidden_states,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None,
            need_weights=output_attentions,
            average_attn_weights=output_attentions,
        )

        hidden_states = residual + self.dropout1(attn_output)

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        ffn_output = self.act(self.fc1(hidden_states))
        ffn_output = self.dropout2(ffn_output)
        ffn_output = self.fc2(ffn_output)
        hidden_states = residual + self.dropout3(ffn_output)

        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states, None


class HuBERTEncoder(nn.Module):
    """HuBERT-style audio encoder.

    Implements the encoder from "HuBERT: Self-Supervised Speech Representation
    Learning by Masked Prediction of Hidden Units" (Hsu et al., 2021).

    Features:
    - CNN feature extraction from raw audio
    - Positional embeddings
    - Transformer encoder with pre-norm
    - Optional masked prediction for self-supervised learning

    Args:
        config: Audio configuration object.
    """

    def __init__(self, config: AudioConfig):
        """Initialize HuBERT encoder.

        Args:
            config: Audio configuration.
        """
        super().__init__()
        self.config = config

        self.feature_extractor = HuBERTFeatureExtractor(
            num_channels=1,
            encoder_dim=config.encoder_dim,
            num_conv_layers=config.num_conv_layers,
        )

        self.position_embedding = nn.Embedding(
            config.max_audio_length * 100,
            config.encoder_dim,
        )

        self.blocks = nn.ModuleList([
            HuBERTBlock(
                encoder_dim=config.encoder_dim,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                layer_norm_eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.encoder_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_dropout = nn.Dropout(0.1)

        self.gradient_checkpointing = config.use_gradient_checkpointing

        # Masked prediction head (optional)
        self.masked_pred_head = nn.Linear(config.encoder_dim, config.encoder_dim)
        self.label_embedding = nn.Embedding(config.num_audio_tokens, config.encoder_dim)

    def _compute_mask(
        self,
        shape: Tuple[int, int],
        mask_prob: float = 0.65,
        mask_length: int = 10,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Compute random span mask for self-supervised learning.

        Args:
            shape: (batch_size, sequence_length).
            mask_prob: Probability of masking.
            mask_length: Length of each mask span.
            device: Target device.

        Returns:
            Boolean mask tensor.
        """
        batch_size, seq_len = shape
        mask = torch.zeros(shape, dtype=torch.bool, device=device)

        num_masked = int(mask_prob * seq_len)
        num_spans = max(1, num_masked // mask_length)

        for b in range(batch_size):
            for _ in range(num_spans):
                start = torch.randint(0, max(seq_len - mask_length, 1), (1,)).item()
                mask[b, start:start + mask_length] = True

        return mask

    def forward(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> AudioEncoderOutput:
        """Forward pass through HuBERT encoder.

        Args:
            waveform: Raw audio waveform (batch, num_samples).
            attention_mask: Optional attention mask.
            mask: Optional pre-computed mask for self-supervised learning.
            output_hidden_states: Return all hidden states.
            output_attentions: Return attention weights.
            return_dict: Return structured output.

        Returns:
            AudioEncoderOutput with encoded features.
        """
        batch_size = waveform.shape[0]

        # Feature extraction
        hidden_states, conv_attention_mask = self.feature_extractor(
            waveform, attention_mask
        )

        frame_level_features = hidden_states

        # Add position embeddings
        position_ids = torch.arange(
            hidden_states.shape[1],
            device=hidden_states.device,
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = hidden_states + self.position_embedding(position_ids)
        hidden_states = self.dropout(hidden_states)

        # Create attention mask
        attn_mask = conv_attention_mask if conv_attention_mask is not None else attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Pass through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states, attn_weights = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, attn_mask, output_attentions,
                    use_reentrant=False,
                )
            else:
                hidden_states, attn_weights = block(
                    hidden_states,
                    attention_mask=attn_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = self.layer_dropout(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

        hidden_states = self.final_norm(hidden_states)

        # Pool over time
        if attn_mask is not None:
            mask_expanded = attn_mask.float().unsqueeze(-1)
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden_states.mean(dim=1)

        if not return_dict:
            return (
                hidden_states, all_hidden_states, all_attentions,
                frame_level_features, attn_mask, pooled,
            )

        return AudioEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            audio_features=frame_level_features,
            attention_mask=attn_mask,
            pooler_output=pooled,
            frame_level_features=frame_level_features,
        )


# =============================================================================
# Audio Tokenizer
# =============================================================================

class AudioCodebook(nn.Module):
    """Vector quantization codebook for discrete audio tokens.

    Uses learnable codebook vectors and straight-through estimator
    for gradient computation.
    """

    def __init__(
        self,
        num_tokens: int = 8192,
        token_dim: int = 1024,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """Initialize audio codebook.

        Args:
            num_tokens: Number of codebook entries.
            token_dim: Dimension of each codebook vector.
            commitment_cost: Weight for commitment loss.
            decay: EMA decay for codebook updates.
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.commitment_cost = commitment_cost

        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(num_tokens),
        )
        self.register_buffer(
            "ema_w",
            torch.randn(token_dim, num_tokens),
        )
        self.ema_w.data.uniform_(-1.0 / num_tokens, 1.0 / num_tokens)

        self.codebook = nn.Parameter(
            torch.randn(num_tokens, token_dim) * 0.02
        )
        self.decay = decay
        self.epsilon = epsilon

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize continuous features to discrete tokens.

        Args:
            z: Continuous features (batch, seq_len, token_dim).

        Returns:
            Tuple of (quantized, token_ids, loss).
        """
        # Flatten for distance computation
        z_flat = z.reshape(-1, self.token_dim)

        # Compute distances to codebook entries
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(z_flat, self.codebook.t())
            + torch.sum(self.codebook ** 2, dim=1, keepdim=True).t()
        )

        # Find nearest codebook entry
        token_ids = torch.argmin(distances, dim=-1)
        z_q = self.codebook[token_ids]

        # Straight-through estimator
        z_q_flat = z_flat + (z_q - z_flat).detach()

        # Reshape
        z_q = z_q_flat.view(z.shape)
        token_ids = token_ids.view(z.shape[0], z.shape[1])

        # EMA update
        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay)
                self.ema_w.mul_(self.decay)

                one_hot = F.one_hot(
                    token_ids.view(-1), self.num_tokens
                ).float()

                cluster_size = one_hot.sum(0)
                self.ema_cluster_size.add_(
                    cluster_size * (1 - self.decay)
                )

                dw = torch.matmul(
                    z_flat.t(), one_hot
                )
                self.ema_w.add_(
                    dw * (1 - self.decay)
                )

                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_tokens * self.epsilon)
                    * n
                )
                self.codebook.data.copy_(
                    self.ema_w.t() / cluster_size.unsqueeze(1)
                )

        # Compute loss
        e_latent_loss = F.mse_loss(z_q_flat, z_flat.detach())
        loss = self.commitment_cost * e_latent_loss

        return z_q, token_ids, loss

    def get_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up codebook embeddings for given token IDs.

        Args:
            token_ids: Token indices (batch, seq_len).

        Returns:
            Embeddings (batch, seq_len, token_dim).
        """
        return self.codebook[token_ids]


class KMeansQuantizer(nn.Module):
    """Online k-means quantizer for audio feature discretization.

    Implements a differentiable k-means clustering that can be trained
    end-to-end with the audio encoder.
    """

    def __init__(
        self,
        num_clusters: int = 8192,
        feature_dim: int = 1024,
        num_iterations: int = 10,
    ):
        """Initialize k-means quantizer.

        Args:
            num_clusters: Number of clusters (vocabulary size).
            feature_dim: Dimension of input features.
            num_iterations: Number of k-means iterations.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations

        # Cluster centers (learnable)
        self.centers = nn.Parameter(
            torch.randn(num_clusters, feature_dim) * 0.1
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize features using k-means clustering.

        Args:
            features: Input features (batch, seq_len, feature_dim).

        Returns:
            Tuple of (quantized_features, cluster_ids, distances).
        """
        batch_size, seq_len, dim = features.shape
        flat_features = features.reshape(-1, dim)

        cluster_ids = None
        distances = None

        for _ in range(self.num_iterations):
            # Compute distances to all centers
            dist = (
                torch.sum(flat_features ** 2, dim=1, keepdim=True)
                - 2 * torch.matmul(flat_features, self.centers.t())
                + torch.sum(self.centers ** 2, dim=1).unsqueeze(0)
            )
            distances = dist

            # Assign to nearest cluster
            cluster_ids = torch.argmin(dist, dim=-1)

        # Get quantized features
        quantized = self.centers[cluster_ids]
        quantized = quantized.view(batch_size, seq_len, dim)
        cluster_ids = cluster_ids.view(batch_size, seq_len)

        return quantized, cluster_ids, distances


class AudioTokenizer(nn.Module):
    """Discrete audio tokenization pipeline.

    Combines feature extraction and quantization to convert raw audio
    waveforms into discrete token sequences for LLM processing.
    """

    def __init__(
        self,
        config: AudioConfig,
        feature_dim: int = 1024,
        codebook_size: int = 8192,
        num_codebooks: int = 1,
    ):
        """Initialize audio tokenizer.

        Args:
            config: Audio configuration.
            feature_dim: Feature dimension for quantization.
            codebook_size: Number of tokens in codebook.
            num_codebooks: Number of parallel codebooks (RVQ).
        """
        super().__init__()
        self.config = config
        self.num_codebooks = num_codebooks

        self.mel_spectrogram = MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            fmin=config.fmin,
            fmax=config.fmax,
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(config.n_mels, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        self.codebooks = nn.ModuleList([
            AudioCodebook(
                num_tokens=codebook_size,
                token_dim=feature_dim,
            )
            for _ in range(num_codebooks)
        ])

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Conv1d(512, config.n_mels, kernel_size=3, stride=1, padding=1),
        )

    def encode(
        self,
        waveform: torch.Tensor,
    ) -> AudioTokenizerOutput:
        """Encode waveform to discrete tokens.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            AudioTokenizerOutput with token IDs and embeddings.
        """
        mel_spec = self.mel_spectrogram(waveform)

        # Encode to continuous features
        hidden = self.encoder(mel_spec)
        hidden = hidden.transpose(1, 2)

        # Quantize
        total_loss = 0.0
        quantized = torch.zeros_like(hidden)
        all_token_ids = []

        for i, codebook in enumerate(self.codebooks):
            residual = hidden - quantized
            z_q, token_ids, loss = codebook(residual)
            quantized = quantized + z_q
            total_loss = total_loss + loss
            all_token_ids.append(token_ids)

        token_ids = torch.stack(all_token_ids, dim=1)

        return AudioTokenizerOutput(
            token_ids=token_ids,
            token_embeddings=quantized,
            num_tokens=token_ids.shape[-1],
            codebook_indices=token_ids,
        )

    def decode(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Decode discrete tokens back to mel spectrogram.

        Args:
            token_ids: Token indices (batch, num_codebooks, num_tokens).

        Returns:
            Reconstructed mel spectrogram.
        """
        quantized = torch.zeros(
            token_ids.shape[0], token_ids.shape[2],
            self.codebooks[0].token_dim,
            device=token_ids.device, dtype=token_ids.dtype,
        )

        for i, codebook in enumerate(self.codebooks):
            ids = token_ids[:, i, :]
            z_q = codebook.get_embeddings(ids)
            quantized = quantized + z_q

        mel_recon = self.decoder(quantized.transpose(1, 2))
        return mel_recon

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> AudioTokenizerOutput:
        """Full encode-decode pass.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            AudioTokenizerOutput with tokens and reconstruction.
        """
        output = self.encode(waveform)
        reconstruction = self.decode(output.token_ids)
        output.reconstruction = reconstruction
        return output


# =============================================================================
# Audio Augmentation
# =============================================================================

class AudioAugmentation(nn.Module):
    """Audio data augmentation for training.

    Implements various augmentation strategies:
    - Speed perturbation (time stretching via resampling)
    - Noise injection (Gaussian, colored)
    - Time masking (zero-out time spans)
    - Frequency masking (zero-out frequency bands)
    - SpecAugment combined
    - Random gain changes
    - Random polarity inversion
    """

    def __init__(
        self,
        speed_perturbation_range: Tuple[float, float] = (0.9, 1.1),
        noise_snr_range: Tuple[float, float] = (10.0, 40.0),
        time_mask_param: int = 100,
        freq_mask_param: int = 80,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        gain_range: Tuple[float, float] = (0.8, 1.2),
        p_speed: float = 0.5,
        p_noise: float = 0.3,
        p_time_mask: float = 0.5,
        p_freq_mask: float = 0.5,
        p_gain: float = 0.3,
        p_polarity: float = 0.1,
    ):
        """Initialize audio augmentation.

        Args:
            speed_perturbation_range: Range for speed perturbation factor.
            noise_snr_range: Range for noise SNR in dB.
            time_mask_param: Maximum time mask width.
            freq_mask_param: Maximum frequency mask width.
            num_time_masks: Number of time masks per sample.
            num_freq_masks: Number of frequency masks per sample.
            gain_range: Range for random gain.
            p_speed: Probability of applying speed perturbation.
            p_noise: Probability of applying noise injection.
            p_time_mask: Probability of applying time masking.
            p_freq_mask: Probability of applying frequency masking.
            p_gain: Probability of applying gain change.
            p_polarity: Probability of polarity inversion.
        """
        super().__init__()
        self.speed_perturbation_range = speed_perturbation_range
        self.noise_snr_range = noise_snr_range
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.gain_range = gain_range
        self.p_speed = p_speed
        self.p_noise = p_noise
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.p_gain = p_gain
        self.p_polarity = p_polarity

    def speed_perturbation(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Apply speed perturbation via resampling.

        Args:
            waveform: Audio waveform (batch, num_samples).
            sample_rate: Sample rate.

        Returns:
            Perturbed waveform.
        """
        factor = torch.empty(1, device=waveform.device).uniform_(*self.speed_perturbation_range).item()

        new_length = int(waveform.shape[-1] / factor)
        indices = torch.linspace(0, waveform.shape[-1] - 1, new_length, device=waveform.device)

        index_floor = torch.floor(indices).long()
        index_ceil = torch.ceil(indices).long()
        index_ceil = torch.clamp(index_ceil, max=waveform.shape[-1] - 1)

        alpha = (indices - index_floor.float()).unsqueeze(-1)

        # If waveform is (batch, samples), handle batched
        if waveform.dim() == 2:
            interp = (
                waveform[:, index_floor] * (1 - alpha) +
                waveform[:, index_ceil] * alpha
            )
        else:
            interp = (
                waveform[..., index_floor] * (1 - alpha) +
                waveform[..., index_ceil] * alpha
            )

        # Pad or trim to original length
        if interp.shape[-1] < waveform.shape[-1]:
            pad_size = waveform.shape[-1] - interp.shape[-1]
            interp = F.pad(interp, (0, pad_size))
        elif interp.shape[-1] > waveform.shape[-1]:
            interp = interp[..., :waveform.shape[-1]]

        return interp

    def inject_noise(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Inject additive noise at random SNR.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            Noisy waveform.
        """
        snr_db = torch.empty(1, device=waveform.device).uniform_(*self.noise_snr_range).item()
        snr = 10.0 ** (snr_db / 10.0)

        signal_power = torch.mean(waveform ** 2, dim=-1, keepdim=True)
        noise_power = signal_power / snr
        noise_std = torch.sqrt(noise_power)

        noise = torch.randn_like(waveform) * noise_std
        return waveform + noise

    def time_mask(
        self,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """Apply time masking to spectrogram.

        Args:
            spectrogram: Spectrogram (batch, freq, time).

        Returns:
            Masked spectrogram.
        """
        batch_size, _, time_steps = spectrogram.shape
        augmented = spectrogram.clone()

        for _ in range(self.num_time_masks):
            mask_width = torch.randint(
                1, min(self.time_mask_param, time_steps),
                (1,)
            ).item()
            mask_start = torch.randint(
                0, max(time_steps - mask_width, 1),
                (1,)
            ).item()

            augmented[:, :, mask_start:mask_start + mask_width] = 0.0

        return augmented

    def freq_mask(
        self,
        spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        """Apply frequency masking to spectrogram.

        Args:
            spectrogram: Spectrogram (batch, freq, time).

        Returns:
            Masked spectrogram.
        """
        batch_size, num_freq, _ = spectrogram.shape
        augmented = spectrogram.clone()

        for _ in range(self.num_freq_masks):
            mask_width = torch.randint(
                1, min(self.freq_mask_param, num_freq),
                (1,)
            ).item()
            mask_start = torch.randint(
                0, max(num_freq - mask_width, 1),
                (1,)
            ).item()

            augmented[:, mask_start:mask_start + mask_width, :] = 0.0

        return augmented

    def random_gain(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random gain to waveform.

        Args:
            waveform: Audio waveform.

        Returns:
            Gain-adjusted waveform.
        """
        gain = torch.empty(1, device=waveform.device).uniform_(*self.gain_range)
        return waveform * gain

    def polarity_inversion(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Randomly invert polarity of waveform.

        Args:
            waveform: Audio waveform.

        Returns:
            Potentially inverted waveform.
        """
        return -waveform

    def forward(
        self,
        waveform: Optional[torch.Tensor] = None,
        spectrogram: Optional[torch.Tensor] = None,
        sample_rate: int = 16000,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply random augmentations.

        Args:
            waveform: Audio waveform for waveform-level augmentations.
            spectrogram: Spectrogram for spectral-level augmentations.
            sample_rate: Sample rate for speed perturbation.

        Returns:
            Tuple of (augmented_waveform, augmented_spectrogram).
        """
        aug_waveform = waveform
        aug_spectrogram = spectrogram

        if not self.training:
            return aug_waveform, aug_spectrogram

        # Waveform-level augmentations
        if waveform is not None:
            if torch.rand(1).item() < self.p_speed:
                aug_waveform = self.speed_perturbation(waveform, sample_rate)

            if torch.rand(1).item() < self.p_noise:
                aug_waveform = self.inject_noise(aug_waveform)

            if torch.rand(1).item() < self.p_gain:
                aug_waveform = self.random_gain(aug_waveform)

            if torch.rand(1).item() < self.p_polarity:
                aug_waveform = self.polarity_inversion(aug_waveform)

        # Spectrogram-level augmentations
        if spectrogram is not None:
            if torch.rand(1).item() < self.p_time_mask:
                aug_spectrogram = self.time_mask(spectrogram)

            if torch.rand(1).item() < self.p_freq_mask:
                aug_spectrogram = self.freq_mask(aug_spectrogram)

        return aug_waveform, aug_spectrogram


# =============================================================================
# Voice Activity Detection
# =============================================================================

class EnergyBasedVAD(nn.Module):
    """Energy-based voice activity detection.

    Determines which segments of audio contain speech based on
    short-time energy thresholds.
    """

    def __init__(
        self,
        frame_length: int = 512,
        hop_length: int = 256,
        energy_threshold: float = 0.01,
        silence_duration: float = 0.3,
        min_speech_duration: float = 0.1,
        sample_rate: int = 16000,
    ):
        """Initialize energy-based VAD.

        Args:
            frame_length: Frame length in samples.
            hop_length: Hop length in samples.
            energy_threshold: Energy threshold for speech detection.
            silence_duration: Duration of silence to mark speech end.
            min_speech_duration: Minimum speech segment duration.
            sample_rate: Audio sample rate.
        """
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.silence_frames = int(silence_duration * sample_rate / hop_length)
        self.min_speech_frames = int(min_speech_duration * sample_rate / hop_length)
        self.sample_rate = sample_rate

    def compute_energy(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Compute short-time energy of waveform.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            Energy per frame (batch, num_frames).
        """
        num_frames = max(1, (waveform.shape[-1] - self.frame_length) // self.hop_length + 1)
        energy = torch.zeros(waveform.shape[0], num_frames, device=waveform.device)

        for t in range(num_frames):
            start = t * self.hop_length
            end = start + self.frame_length
            if end <= waveform.shape[-1]:
                frame = waveform[:, start:end]
            else:
                frame = waveform[:, start:]
            energy[:, t] = torch.mean(frame ** 2, dim=-1)

        return energy

    def detect(
        self,
        waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Detect voice activity in waveform.

        Args:
            waveform: Audio waveform (batch, num_samples).
            attention_mask: Optional mask for valid samples.

        Returns:
            Tuple of (vad_mask (batch, num_frames), speech_segments).
        """
        energy = self.compute_energy(waveform)
        batch_size = waveform.shape[0]

        # Adaptive thresholding: use percentile of energy
        all_speech_masks = []

        for b in range(batch_size):
            frame_energy = energy[b]

            # Dynamic threshold based on energy distribution
            sorted_energy = torch.sort(frame_energy)[0]
            non_zero = sorted_energy[sorted_energy > 0]
            if len(non_zero) > 0:
                threshold = torch.quantile(non_zero, 0.2).item()
            else:
                threshold = self.energy_threshold

            speech_mask = frame_energy > threshold

            # Apply minimum speech duration constraint
            segments = self._find_speech_segments(speech_mask)

            # Filter short segments
            filtered_segments = []
            for start, end in segments:
                if end - start >= self.min_speech_frames:
                    filtered_segments.append((start, end))

            # Reconstruct mask from segments
            final_mask = torch.zeros_like(speech_mask)
            for start, end in filtered_segments:
                final_mask[start:end] = True

            all_speech_masks.append(final_mask)

        vad_mask = torch.stack(all_speech_masks, dim=0)
        return vad_mask, filtered_segments

    def _find_speech_segments(
        self,
        mask: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """Find contiguous speech segments from binary mask.

        Args:
            mask: Boolean mask of speech frames.

        Returns:
            List of (start, end) frame indices.
        """
        segments = []
        in_speech = False
        start = 0
        silence_count = 0

        for t in range(len(mask)):
            if mask[t].item():
                if not in_speech:
                    start = t
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= self.silence_frames:
                        segments.append((start, t - silence_count))
                        in_speech = False

        if in_speech:
            segments.append((start, len(mask)))

        return segments

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAD mask.

        Args:
            waveform: Audio waveform.

        Returns:
            VAD mask (batch, num_frames).
        """
        vad_mask, _ = self.detect(waveform)
        return vad_mask


class SpectralBasedVAD(nn.Module):
    """Spectral-based voice activity detection.

    Uses spectral flatness and other spectral features for more
    robust voice activity detection compared to pure energy-based methods.
    """

    def __init__(
        self,
        frame_length: int = 512,
        hop_length: int = 256,
        n_fft: int = 512,
        flatness_threshold: float = 0.5,
        sample_rate: int = 16000,
    ):
        """Initialize spectral-based VAD.

        Args:
            frame_length: Frame length.
            hop_length: Hop length.
            n_fft: FFT size.
            flatness_threshold: Threshold for spectral flatness.
            sample_rate: Sample rate.
        """
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.flatness_threshold = flatness_threshold
        self.sample_rate = sample_rate

    def compute_spectral_flatness(
        self,
        spectrum: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spectral flatness (Wiener entropy).

        Spectral flatness is the geometric mean divided by the arithmetic
        mean of the power spectrum. Values close to 1 indicate noise-like
        signals, values close to 0 indicate tonal (speech-like) signals.

        Args:
            spectrum: Power spectrum (batch, freq_bins, num_frames).

        Returns:
            Spectral flatness per frame (batch, num_frames).
        """
        epsilon = 1e-10
        spectrum = spectrum + epsilon

        log_spectrum = torch.log(spectrum)
        geometric_mean = torch.exp(
            torch.mean(log_spectrum, dim=1, keepdim=False)
        )
        arithmetic_mean = torch.mean(spectrum, dim=1, keepdim=False)

        flatness = geometric_mean / arithmetic_mean.clamp(min=epsilon)
        return flatness

    def detect(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Detect voice activity using spectral features.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            VAD mask (batch, num_frames).
        """
        mel = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=80,
        )
        mel_spec = mel(waveform)
        power_spec = mel_spec ** 2

        flatness = self.compute_spectral_flatness(power_spec)

        # Speech has lower flatness than noise
        speech_mask = flatness < self.flatness_threshold

        # Also check overall energy
        energy = torch.mean(power_spec, dim=1)
        energy_threshold = torch.quantile(
            energy[energy > 0], 0.1
        ) if (energy > 0).any() else torch.tensor(0.01)
        energy_mask = energy > energy_threshold

        combined_mask = speech_mask & energy_mask
        return combined_mask.float()

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Compute VAD mask.

        Args:
            waveform: Audio waveform.

        Returns:
            VAD mask (batch, num_frames).
        """
        return self.detect(waveform)


class VAD(nn.Module):
    """Unified voice activity detection module.

    Combines energy-based and spectral-based VAD methods with configurable
    fusion strategy.
    """

    def __init__(
        self,
        method: str = "energy",
        sample_rate: int = 16000,
        frame_length: int = 512,
        hop_length: int = 256,
        **kwargs,
    ):
        """Initialize VAD module.

        Args:
            method: VAD method ('energy', 'spectral', 'combined').
            sample_rate: Sample rate.
            frame_length: Frame length.
            hop_length: Hop length.
            **kwargs: Additional arguments for specific methods.
        """
        super().__init__()
        self.method = method

        if method in ("energy", "combined"):
            self.energy_vad = EnergyBasedVAD(
                frame_length=frame_length,
                hop_length=hop_length,
                sample_rate=sample_rate,
            )
        else:
            self.energy_vad = None

        if method in ("spectral", "combined"):
            self.spectral_vad = SpectralBasedVAD(
                frame_length=frame_length,
                hop_length=hop_length,
                sample_rate=sample_rate,
            )
        else:
            self.spectral_vad = None

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """Compute voice activity detection mask.

        Args:
            waveform: Audio waveform (batch, num_samples).

        Returns:
            VAD mask (batch, num_frames).
        """
        if self.method == "energy" and self.energy_vad is not None:
            return self.energy_vad(waveform)
        elif self.method == "spectral" and self.spectral_vad is not None:
            return self.spectral_vad(waveform)
        elif self.method == "combined":
            energy_mask = self.energy_vad(waveform)
            spectral_mask = self.spectral_vad(waveform)

            # Interpolate to common length
            if energy_mask.shape[-1] != spectral_mask.shape[-1]:
                min_len = min(energy_mask.shape[-1], spectral_mask.shape[-1])
                energy_mask = energy_mask[:, :min_len]
                spectral_mask = spectral_mask[:, :min_len]

            # Union: speech detected by either method
            combined = (energy_mask + spectral_mask).clamp(0, 1)
            return combined
        else:
            raise ValueError(f"Unknown VAD method: {self.method}")
