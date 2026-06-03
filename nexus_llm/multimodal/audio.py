"""Audio processing for Nexus-LLM multimodal support."""

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AudioSegment:
    """A segment of transcribed audio."""
    start: float  # seconds
    end: float    # seconds
    text: str


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str
    segments: List[AudioSegment]
    language: Optional[str]
    duration: float


class AudioProcessor:
    """Handles loading, transcribing, and analyzing audio data.

    Uses mock/local transcription by default. Can be extended with
    Whisper or other ASR backends.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._transcriber: Optional[Any] = None

    # -- Loading --------------------------------------------------------------

    def load_audio(self, path: str) -> bytes:
        """Load audio data from a file.

        Args:
            path: Filesystem path to the audio file.

        Returns:
            Raw audio bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return file_path.read_bytes()

    # -- Transcription --------------------------------------------------------

    def transcribe(self, audio: Any) -> TranscriptionResult:
        """Transcribe audio data to text.

        Attempts to use OpenAI Whisper if available; otherwise falls back
        to a mock transcription.

        Args:
            audio: Audio data (file path string, bytes, or numpy array).

        Returns:
            ``TranscriptionResult`` with text and segment details.
        """
        # Try Whisper
        try:
            return self._transcribe_whisper(audio)
        except (ImportError, RuntimeError):
            pass

        # Fallback: mock transcription
        return self._mock_transcribe(audio)

    # -- Info -----------------------------------------------------------------

    def get_audio_info(self, path: str) -> Dict[str, Any]:
        """Extract metadata from an audio file.

        Args:
            path: Filesystem path to the audio file.

        Returns:
            Dict with keys: ``file_size``, ``sample_rate``,
            ``channels``, ``duration``, ``format``.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        info: Dict[str, Any] = {
            "file_size": file_path.stat().st_size,
            "format": file_path.suffix.lstrip("."),
            "sample_rate": None,
            "channels": None,
            "duration": None,
        }

        suffix = file_path.suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(str(file_path), "rb") as wf:
                    info["channels"] = wf.getnchannels()
                    info["sample_rate"] = wf.getframerate()
                    frames = wf.getnframes()
                    info["duration"] = round(frames / wf.getframerate(), 2)
            except (wave.Error, EOFError):
                pass

        return info

    # -- Utilities ------------------------------------------------------------

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Format seconds as ``HH:MM:SS.mmm`` string.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    # -- Private helpers ------------------------------------------------------

    def _transcribe_whisper(self, audio: Any) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper (requires the `whisper` package)."""
        import whisper  # type: ignore[import-untyped]

        if self._transcriber is None:
            self._transcriber = whisper.load_model("base")

        # Whisper expects a file path or numpy array
        if isinstance(audio, (str, Path)):
            result = self._transcriber.transcribe(str(audio))
        elif isinstance(audio, bytes):
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio)
                tmp_path = tmp.name
            try:
                result = self._transcriber.transcribe(tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            # Assume numpy array
            result = self._transcriber.transcribe(audio)

        segments = [
            AudioSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result.get("segments", [])
        ]

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=segments,
            language=result.get("language"),
            duration=result.get("segments", [{}])[-1].get("end", 0.0) if segments else 0.0,
        )

    def _mock_transcribe(self, audio: Any) -> TranscriptionResult:
        """Produce a mock transcription result for testing."""
        duration = 1.0
        if isinstance(audio, (str, Path)):
            try:
                info = self.get_audio_info(str(audio))
                duration = info.get("duration") or 1.0
            except (FileNotFoundError, Exception):
                pass
        elif isinstance(audio, bytes):
            # Rough estimate: assume 16kHz 16-bit mono
            duration = len(audio) / (self._sample_rate * 2)

        return TranscriptionResult(
            text="[Mock transcription] Audio content could not be transcribed locally.",
            segments=[
                AudioSegment(
                    start=0.0,
                    end=duration,
                    text="[Mock transcription] Audio content could not be transcribed locally.",
                )
            ],
            language=None,
            duration=duration,
        )
