"""Multimodal processing pipeline for Nexus-LLM."""

from typing import Any, Dict, List, Optional

from nexus_llm.multimodal.image import ImageProcessor
from nexus_llm.multimodal.audio import AudioProcessor
from nexus_llm.multimodal.vision import VisionEngine
from nexus_llm.multimodal.processor import BaseProcessor


class TextProcessor(BaseProcessor):
    """Simple text processor that acts as a pass-through with optional cleanup."""

    def process(self, data: Any) -> str:
        if not self.validate_input(data):
            raise ValueError("Invalid text input: expected a non-empty string")
        return str(data).strip()

    def validate_input(self, data: Any) -> bool:
        return isinstance(data, str) and len(data.strip()) > 0

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "text_processor",
            "modality": "text",
            "description": "Pass-through text processor with whitespace cleanup",
            "version": "1.0.0",
        }


class MultimodalPipeline:
    """Orchestrates processing of multimodal inputs.

    Routes input data to the appropriate processor based on modality
    and supports chaining multiple processors.
    """

    SUPPORTED_MODALITIES = ("text", "image", "audio")

    def __init__(
        self,
        image_processor: Optional[ImageProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        vision_engine: Optional[VisionEngine] = None,
    ) -> None:
        self._image_processor = image_processor or ImageProcessor()
        self._audio_processor = audio_processor or AudioProcessor()
        self._vision_engine = vision_engine or VisionEngine()
        self._text_processor = TextProcessor()

        self._processors: Dict[str, BaseProcessor] = {
            "text": self._text_processor,
            "image": self._image_processor,  # type: ignore[assignment]
            "audio": self._audio_processor,  # type: ignore[assignment]
        }

        self._chain: List[Dict[str, Any]] = []

    # -- Core API -------------------------------------------------------------

    def process(self, input_data: Any, modality: str) -> Dict[str, Any]:
        """Process input data of a given modality.

        Args:
            input_data: The data to process (text string, image, audio, etc.).
            modality: One of ``"text"``, ``"image"``, ``"audio"``.

        Returns:
            Dict with keys: ``modality``, ``result``, ``success``,
            ``error``.

        Raises:
            ValueError: If the modality is not supported.
        """
        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Must be one of: {self.SUPPORTED_MODALITIES}"
            )

        try:
            processor = self._processors.get(modality)
            if processor is None:
                raise ValueError(f"No processor registered for modality '{modality}'")

            result = processor.process(input_data)

            # Run chained processors if any
            for step in self._chain:
                if step.get("modality") == modality:
                    step_processor = step["processor"]
                    if step_processor.validate_input(result):
                        result = step_processor.process(result)

            return {
                "modality": modality,
                "result": result,
                "success": True,
                "error": None,
            }
        except Exception as exc:
            return {
                "modality": modality,
                "result": None,
                "success": False,
                "error": str(exc),
            }

    # -- Processor registration ------------------------------------------------

    def register_processor(self, modality: str, processor: BaseProcessor) -> None:
        """Register or replace a processor for a modality.

        Args:
            modality: Modality name.
            processor: ``BaseProcessor`` implementation.
        """
        self._processors[modality] = processor

    def add_chain_step(self, modality: str, processor: BaseProcessor) -> None:
        """Add a chained processing step for a modality.

        Chained steps run in order after the primary processor.

        Args:
            modality: The modality this step applies to.
            processor: ``BaseProcessor`` to run in the chain.
        """
        self._chain.append({"modality": modality, "processor": processor})

    # -- Convenience methods --------------------------------------------------

    def process_image(self, path_or_url: str) -> Dict[str, Any]:
        """Convenience: load and process an image.

        Args:
            path_or_url: Image source.

        Returns:
            Processing result dict.
        """
        image = self._image_processor.load_image(path_or_url)
        description = self._vision_engine.describe_image(image)
        info = self._image_processor.get_image_info(image)
        return {
            "modality": "image",
            "description": description,
            "image_info": info,
            "success": True,
            "error": None,
        }

    def process_audio(self, path: str) -> Dict[str, Any]:
        """Convenience: load and transcribe audio.

        Args:
            path: Audio file path.

        Returns:
            Processing result dict.
        """
        audio_data = self._audio_processor.load_audio(path)
        info = self._audio_processor.get_audio_info(path)
        transcription = self._audio_processor.transcribe(audio_data)
        return {
            "modality": "audio",
            "transcription": transcription.text,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcription.segments
            ],
            "audio_info": info,
            "success": True,
            "error": None,
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Convenience: process text input.

        Args:
            text: Input text string.

        Returns:
            Processing result dict.
        """
        return self.process(text, "text")

    # -- Info -----------------------------------------------------------------

    def list_modalities(self) -> List[str]:
        """Return supported modality names."""
        return list(self.SUPPORTED_MODALITIES)

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Return information about the pipeline configuration."""
        return {
            "supported_modalities": list(self.SUPPORTED_MODALITIES),
            "registered_processors": {
                mod: proc.get_info() if isinstance(proc, BaseProcessor) else {"name": str(proc)}
                for mod, proc in self._processors.items()
            },
            "chain_steps": len(self._chain),
        }
