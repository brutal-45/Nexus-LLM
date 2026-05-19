"""Audio Processor for Nexus-LLM.

Provides audio loading, resampling, feature extraction, format
conversion, and basic analysis for WAV and MP3 files.
"""

from __future__ import annotations

import io
import os
import struct
import wave
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    RAW = "raw"


class FeatureType(str, Enum):
    """Audio feature extraction types."""
    MFCC = "mfcc"
    SPECTROGRAM = "spectrogram"
    MEL_SPECTROGRAM = "mel_spectrogram"
    RMS_ENERGY = "rms_energy"
    ZERO_CROSSING_RATE = "zero_crossing_rate"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AudioInfo:
    """Metadata about an audio file."""
    sample_rate: int = 0
    channels: int = 0
    duration_seconds: float = 0.0
    frames: int = 0
    sample_width: int = 0  # bytes per sample
    format: Optional[str] = None
    file_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": round(self.duration_seconds, 4),
            "frames": self.frames,
            "sample_width": self.sample_width,
            "format": self.format,
            "file_size": self.file_size,
        }


@dataclass
class AudioData:
    """Container for audio sample data."""
    samples: Any  # numpy array or list of floats
    sample_rate: int = 16000
    channels: int = 1

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if HAS_NUMPY and isinstance(self.samples, np.ndarray):
            return len(self.samples) / max(self.sample_rate, 1)
        return len(self.samples) / max(self.sample_rate, 1)

    @property
    def num_samples(self) -> int:
        """Number of sample frames."""
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "num_samples": self.num_samples,
            "duration_seconds": round(self.duration, 4),
        }


# ---------------------------------------------------------------------------
# Audio Processor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """Audio processing utilities for multimodal workflows.

    Provides WAV/MP3 loading, resampling, feature extraction, format
    conversion, and basic audio analysis.  Uses the standard library
    ``wave`` module for WAV I/O and optionally numpy for numerical
    operations.

    Example::

        proc = AudioProcessor()
        audio = proc.load("speech.wav")
        resampled = proc.resample(audio, target_rate=16000)
        features = proc.extract_features(resampled, [FeatureType.RMS_ENERGY])
        info = proc.get_info("speech.wav")
    """

    def __init__(self, default_sample_rate: int = 16000) -> None:
        self._default_sample_rate = default_sample_rate

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self, path: str) -> AudioData:
        """Load an audio file (WAV supported natively; MP3 if pydub available).

        Args:
            path: Path to the audio file.

        Returns:
            AudioData with sample array, sample rate, and channel info.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = os.path.splitext(path)[1].lower().lstrip(".")

        if ext == "wav":
            return self._load_wav(path)
        elif ext == "mp3":
            return self._load_mp3(path)
        else:
            # Attempt WAV as fallback
            try:
                return self._load_wav(path)
            except Exception:
                raise ValueError(f"Unsupported audio format: .{ext}")

    def _load_wav(self, path: str) -> AudioData:
        """Load a WAV file using the standard library."""
        with wave.open(path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        samples = self._decode_pcm(raw, sample_width, channels)
        return AudioData(samples=samples, sample_rate=sample_rate, channels=channels)

    def _load_mp3(self, path: str) -> AudioData:
        """Load an MP3 file. Requires pydub and ffmpeg."""
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(path)
            samples = self._pydub_to_samples(seg)
            return AudioData(
                samples=samples,
                sample_rate=seg.frame_rate,
                channels=seg.channels,
            )
        except ImportError:
            raise ImportError(
                "MP3 loading requires pydub and ffmpeg: "
                "pip install pydub && install ffmpeg"
            )

    def save_wav(
        self,
        audio: AudioData,
        path: str,
        sample_width: int = 2,
    ) -> str:
        """Save audio data as a WAV file.

        Args:
            audio: AudioData to save.
            path: Destination file path.
            sample_width: Bytes per sample (1, 2, 3, or 4).

        Returns:
            The file path written.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        raw = self._encode_pcm(audio.samples, sample_width)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(audio.channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(audio.sample_rate)
            wf.writeframes(raw)

        return path

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def resample(
        self,
        audio: AudioData,
        target_rate: int = 16000,
    ) -> AudioData:
        """Resample audio to a target sample rate using linear interpolation.

        Args:
            audio: Input AudioData.
            target_rate: Desired sample rate in Hz.

        Returns:
            New AudioData at the target sample rate.
        """
        if audio.sample_rate == target_rate:
            return AudioData(
                samples=audio.samples.copy() if HAS_NUMPY and isinstance(audio.samples, np.ndarray) else list(audio.samples),
                sample_rate=target_rate,
                channels=audio.channels,
            )

        if HAS_NUMPY and isinstance(audio.samples, np.ndarray):
            ratio = target_rate / audio.sample_rate
            old_len = len(audio.samples)
            new_len = int(old_len * ratio)
            old_indices = np.arange(new_len) / ratio
            idx_floor = np.floor(old_indices).astype(int)
            idx_ceil = np.minimum(idx_floor + 1, old_len - 1)
            frac = old_indices - idx_floor
            new_samples = audio.samples[idx_floor] * (1 - frac) + audio.samples[idx_ceil] * frac
        else:
            ratio = target_rate / audio.sample_rate
            old = list(audio.samples)
            new_len = int(len(old) * ratio)
            new_samples = []
            for i in range(new_len):
                src_idx = i / ratio
                lo = int(src_idx)
                hi = min(lo + 1, len(old) - 1)
                frac = src_idx - lo
                new_samples.append(old[lo] * (1 - frac) + old[hi] * frac)

        return AudioData(samples=new_samples, sample_rate=target_rate, channels=audio.channels)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        audio: AudioData,
        feature_types: Optional[List[FeatureType]] = None,
    ) -> Dict[str, Any]:
        """Extract audio features from the sample data.

        Args:
            audio: Input AudioData.
            feature_types: List of features to extract. Extracts all if None.

        Returns:
            Dictionary mapping feature names to computed values.
        """
        if feature_types is None:
            feature_types = list(FeatureType)

        features: Dict[str, Any] = {}

        for ft in feature_types:
            if ft == FeatureType.RMS_ENERGY:
                features["rms_energy"] = self._compute_rms(audio)
            elif ft == FeatureType.ZERO_CROSSING_RATE:
                features["zero_crossing_rate"] = self._compute_zcr(audio)
            elif ft == FeatureType.SPECTROGRAM:
                features["spectrogram"] = self._compute_spectrogram(audio)
            elif ft == FeatureType.MEL_SPECTROGRAM:
                features["mel_spectrogram"] = self._compute_mel_spectrogram(audio)
            elif ft == FeatureType.MFCC:
                features["mfcc"] = self._compute_mfcc(audio)

        return features

    def _compute_rms(self, audio: AudioData) -> float:
        """Compute root mean square energy."""
        if HAS_NUMPY and isinstance(audio.samples, np.ndarray):
            s = audio.samples.astype(float)
        else:
            s = np.array(list(audio.samples), dtype=float) if HAS_NUMPY else list(map(float, audio.samples))

        if HAS_NUMPY:
            return float(np.sqrt(np.mean(s ** 2)))
        else:
            s_list = list(map(float, audio.samples))
            return (sum(x ** 2 for x in s_list) / max(len(s_list), 1)) ** 0.5

    def _compute_zcr(self, audio: AudioData) -> float:
        """Compute zero crossing rate."""
        if HAS_NUMPY and isinstance(audio.samples, np.ndarray):
            s = audio.samples.astype(float)
            crossings = np.sum(np.abs(np.diff(np.sign(s))) > 0)
            return float(crossings / max(len(s) - 1, 1))
        else:
            s_list = list(map(float, audio.samples))
            crossings = sum(1 for i in range(1, len(s_list)) if (s_list[i] >= 0) != (s_list[i - 1] >= 0))
            return crossings / max(len(s_list) - 1, 1)

    def _compute_spectrogram(self, audio: AudioData) -> Any:
        """Compute a basic magnitude spectrogram via DFT on windowed frames."""
        if not HAS_NUMPY:
            return None
        s = np.array(audio.samples, dtype=float)
        frame_len = 512
        hop = 256
        num_frames = max(1, (len(s) - frame_len) // hop + 1)
        spec = []
        for i in range(num_frames):
            start = i * hop
            frame = s[start:start + frame_len]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            windowed = frame * np.hanning(frame_len)
            spectrum = np.abs(np.fft.rfft(windowed))
            spec.append(spectrum.tolist())
        return spec

    def _compute_mel_spectrogram(self, audio: AudioData) -> Any:
        """Compute a mel spectrogram (simplified)."""
        spec = self._compute_spectrogram(audio)
        if spec is None or not HAS_NUMPY:
            return None
        # Simplified mel filterbank approximation
        spec_arr = np.array(spec)
        n_mels = 40
        n_fft_bins = spec_arr.shape[1]
        mel_weights = self._mel_filterbank(n_mels, n_fft_bins, audio.sample_rate)
        mel_spec = np.dot(spec_arr, mel_weights.T)
        return mel_spec.tolist()

    def _compute_mfcc(self, audio: AudioData) -> Any:
        """Compute MFCCs (simplified: DCT of mel spectrogram)."""
        mel_spec = self._compute_mel_spectrogram(audio)
        if mel_spec is None or not HAS_NUMPY:
            return None
        mel_arr = np.array(mel_spec)
        log_mel = np.log(mel_arr + 1e-10)
        n_mfcc = min(13, log_mel.shape[1])
        mfccs = []
        for frame in log_mel:
            dct = np.fft.dct(frame, type=2, norm="ortho")[:n_mfcc]
            mfccs.append(dct.tolist())
        return mfccs

    @staticmethod
    def _mel_filterbank(n_mels: int, n_fft_bins: int, sample_rate: int) -> "np.ndarray":
        """Create a mel filterbank matrix."""
        if not HAS_NUMPY:
            raise ImportError("numpy is required for mel filterbank computation")
        low = 0.0
        high = sample_rate / 2.0
        mel_low = 1127.0 * np.log1p(low / 700.0)
        mel_high = 1127.0 * np.log1p(high / 700.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (np.expm1(mel_points / 1127.0))
        bin_points = np.floor((n_fft_bins + 1) * hz_points / sample_rate).astype(int)
        fbank = np.zeros((n_mels, n_fft_bins))
        for m in range(n_mels):
            f_left = bin_points[m]
            f_center = bin_points[m + 1]
            f_right = bin_points[m + 2]
            for k in range(f_left, f_center):
                if k < n_fft_bins and f_center != f_left:
                    fbank[m, k] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if k < n_fft_bins and f_right != f_center:
                    fbank[m, k] = (f_right - k) / (f_right - f_center)
        return fbank

    # ------------------------------------------------------------------
    # Info / Utilities
    # ------------------------------------------------------------------

    def get_info(self, path: str) -> AudioInfo:
        """Extract metadata from a WAV audio file.

        Args:
            path: Path to the WAV file.

        Returns:
            AudioInfo with format details.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        ext = os.path.splitext(path)[1].lower().lstrip(".")
        file_size = os.path.getsize(path)

        if ext == "wav":
            with wave.open(path, "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                nf = wf.getnframes()
                dur = nf / max(sr, 1)
            return AudioInfo(
                sample_rate=sr, channels=ch, duration_seconds=dur,
                frames=nf, sample_width=sw, format="wav", file_size=file_size,
            )

        return AudioInfo(format=ext, file_size=file_size)

    def split_channels(self, audio: AudioData) -> List[AudioData]:
        """Split multi-channel audio into separate mono channels.

        Args:
            audio: Multi-channel AudioData.

        Returns:
            List of mono AudioData, one per channel.
        """
        if audio.channels <= 1:
            return [audio]

        if HAS_NUMPY and isinstance(audio.samples, np.ndarray):
            reshaped = audio.samples.reshape(-1, audio.channels)
            return [
                AudioData(samples=reshaped[:, i].copy(), sample_rate=audio.sample_rate, channels=1)
                for i in range(audio.channels)
            ]
        else:
            samples = list(audio.samples)
            n = len(samples) // audio.channels
            channels = []
            for ch in range(audio.channels):
                ch_samples = [samples[i * audio.channels + ch] for i in range(n)]
                channels.append(AudioData(samples=ch_samples, sample_rate=audio.sample_rate, channels=1))
            return channels

    # ------------------------------------------------------------------
    # PCM helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_pcm(raw: bytes, sample_width: int, channels: int) -> Any:
        """Decode raw PCM bytes into a sample array."""
        fmt = {1: "b", 2: "h", 3: None, 4: "i"}.get(sample_width)
        if fmt is None:
            # 24-bit: unpack manually
            if HAS_NUMPY:
                raw_arr = np.frombuffer(raw, dtype=np.uint8)
                n_samples = len(raw_arr) // 3
                reshaped = raw_arr[:n_samples * 3].reshape(-1, 3)
                samples = (
                    reshaped[:, 0].astype(np.int32)
                    | (reshaped[:, 1].astype(np.int32) << 8)
                    | (reshaped[:, 2].astype(np.int32) << 16)
                )
                # Sign-extend
                samples = np.where(samples > 0x7FFFFF, samples - 0x1000000, samples)
                return samples.astype(np.float32) / 0x7FFFFF
            else:
                samples = []
                for i in range(0, len(raw) - 2, 3):
                    val = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
                    if val > 0x7FFFFF:
                        val -= 0x1000000
                    samples.append(val / 0x7FFFFF)
                return samples

        max_val = {1: 127, 2: 32767, 4: 2147483647}[sample_width]

        if HAS_NUMPY:
            arr = np.frombuffer(raw, dtype=np.dtype(f"<{fmt}")).astype(np.float32)
            arr /= max_val
            return arr
        else:
            n = len(raw) // sample_width
            samples = []
            for i in range(n):
                val = int.from_bytes(raw[i * sample_width:(i + 1) * sample_width], "little", signed=True)
                samples.append(val / max_val)
            return samples

    @staticmethod
    def _encode_pcm(samples: Any, sample_width: int) -> bytes:
        """Encode a sample array into raw PCM bytes."""
        max_val = {1: 127, 2: 32767, 4: 2147483647}[sample_width]

        if HAS_NUMPY and isinstance(samples, np.ndarray):
            int_samples = (samples * max_val).astype(np.int16 if sample_width == 2 else np.int32)
            return int_samples.tobytes()
        else:
            buf = io.BytesIO()
            for s in samples:
                val = int(s * max_val)
                val = max(-max_val - 1, min(max_val, val))
                buf.write(val.to_bytes(sample_width, "little", signed=True))
            return buf.getvalue()

    @staticmethod
    def _pydub_to_samples(segment: Any) -> Any:
        """Convert a pydub AudioSegment to a numpy/sample array."""
        if HAS_NUMPY:
            raw = segment.raw_data
            sw = segment.sample_width
            if sw == 2:
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
            elif sw == 4:
                arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483647.0
            else:
                arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 127.0 - 1.0
            if segment.channels > 1:
                arr = arr.reshape(-1, segment.channels).mean(axis=1)
            return arr
        else:
            sw = segment.sample_width
            raw = segment.raw_data
            n = len(raw) // sw
            max_val = {1: 127, 2: 32767, 4: 2147483647}.get(sw, 32767)
            samples = []
            for i in range(n):
                val = int.from_bytes(raw[i * sw:(i + 1) * sw], "little", signed=True)
                samples.append(val / max_val)
            return samples
