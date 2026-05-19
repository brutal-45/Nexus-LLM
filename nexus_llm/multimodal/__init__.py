"""Nexus-LLM Multimodal Module.

Provides multimodal processing capabilities including image, audio, and
document processing, OCR text extraction, and vision-language model
integration for multimodal AI workflows.
"""

from nexus_llm.multimodal.image_processor import (
    ImageFormat,
    ImageMetadata,
    ImageProcessor,
    NormalizeConfig,
    ResizeMode,
)
from nexus_llm.multimodal.audio_processor import (
    AudioData,
    AudioFormat,
    AudioInfo,
    AudioProcessor,
    FeatureType,
)
from nexus_llm.multimodal.document_processor import (
    DocumentChunk,
    DocumentContent,
    DocumentFormat,
    DocumentProcessor,
    detect_format,
)
from nexus_llm.multimodal.vision_model import (
    MultimodalInput,
    VLMBackend,
    VLMConfig,
    VLMResponse,
    VisionLanguageModel,
)
from nexus_llm.multimodal.ocr import (
    BoundingBox,
    OCREngine,
    OCRLine,
    OCRResult,
    OCRWord,
    OCRBackend,
)

__all__ = [
    # Image Processor
    "ImageFormat",
    "ImageMetadata",
    "ImageProcessor",
    "NormalizeConfig",
    "ResizeMode",
    # Audio Processor
    "AudioData",
    "AudioFormat",
    "AudioInfo",
    "AudioProcessor",
    "FeatureType",
    # Document Processor
    "DocumentChunk",
    "DocumentContent",
    "DocumentFormat",
    "DocumentProcessor",
    "detect_format",
    # Vision Model
    "MultimodalInput",
    "VLMBackend",
    "VLMConfig",
    "VLMResponse",
    "VisionLanguageModel",
    # OCR
    "BoundingBox",
    "OCREngine",
    "OCRLine",
    "OCRResult",
    "OCRWord",
    "OCRBackend",
]
