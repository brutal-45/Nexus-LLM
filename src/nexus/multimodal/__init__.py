"""
Nexus Multimodal Module
=======================
Multimodal capabilities for the Nexus LLM including vision, audio, video
encoders, cross-modal fusion, projectors, and multimodal training infrastructure.

Architecture Overview:
    - Vision: ViT, SigLIP, ConvNeXt with configurable patch embeddings
    - Audio: Whisper-style encoder, Mel spectrogram from raw waveforms
    - Video: TimeSformer, VideoSwin, ViViT with tubelet embeddings
    - Fusion: Cross-attention, co-attention, gated, adaptive, compact bilinear
    - Projectors: MLP (LLaVA), Q-Former (BLIP-2), Resampler (Perceiver)
    - Training: Contrastive loss, alignment loss, modality-balanced sampling

All components support mixed precision (BF16), gradient checkpointing,
and are designed for integration with the core Nexus transformer.
"""

from nexus.multimodal.vision_encoder import (
    PatchEmbedding2D,
    ViTEncoder,
    SigLIPEncoder,
    ConvNeXtBlock,
    ConvNeXtEncoder,
    EfficientViTBlock,
    ImageAugmentation,
    ResolutionAdaptor,
)

from nexus.multimodal.audio_encoder import (
    MelSpectrogram,
    AudioFeatureExtractor,
    WhisperEncoder,
    AudioAugmentation,
    AudioTokenizer,
    VoiceActivityDetector,
)

from nexus.multimodal.video_encoder import (
    VideoPatchEmbedding,
    TimeSformerEncoder,
    VideoSwinBlock,
    ViViTEncoder,
    TemporalAggregator,
    FrameSampler,
    VideoAugmentation,
)

from nexus.multimodal.cross_modal_fusion import (
    ModalityEmbedding,
    CrossAttentionFusion,
    CoAttentionFusion,
    ConcatenationFusion,
    GatedFusion,
    CompactBilinearFusion,
    AdaptiveFusion,
    MultiModalTransformerLayer,
    MultiModalTransformer,
    FusionGate,
    ModalityDropout,
)

from nexus.multimodal.multimodal_projector import (
    LinearProjector,
    MLPProjector,
    QFormerProjector,
    ResamplerProjector,
    CAbstractor,
    ModalityAdapter,
    ProjectorFactory,
)

from nexus.multimodal.multimodal_trainer import (
    MultimodalDataset,
    MultimodalCollator,
    MultimodalTrainer,
    MultimodalLoss,
    ContrastiveLoss,
    AlignmentLoss,
    ModalityBalancedSampler,
    EvaluationMetrics,
)

from nexus.multimodal.multimodal_config import (
    VisionConfig,
    AudioConfig,
    VideoConfig,
    CrossModalConfig,
    MultimodalConfig,
    ProcessorConfig,
)

__all__ = [
    # Vision
    "PatchEmbedding2D", "ViTEncoder", "SigLIPEncoder",
    "ConvNeXtBlock", "ConvNeXtEncoder", "EfficientViTBlock",
    "ImageAugmentation", "ResolutionAdaptor",
    # Audio
    "MelSpectrogram", "AudioFeatureExtractor", "WhisperEncoder",
    "AudioAugmentation", "AudioTokenizer", "VoiceActivityDetector",
    # Video
    "VideoPatchEmbedding", "TimeSformerEncoder", "VideoSwinBlock",
    "ViViTEncoder", "TemporalAggregator", "FrameSampler", "VideoAugmentation",
    # Fusion
    "ModalityEmbedding", "CrossAttentionFusion", "CoAttentionFusion",
    "ConcatenationFusion", "GatedFusion", "CompactBilinearFusion",
    "AdaptiveFusion", "MultiModalTransformerLayer", "MultiModalTransformer",
    "FusionGate", "ModalityDropout",
    # Projectors
    "LinearProjector", "MLPProjector", "QFormerProjector",
    "ResamplerProjector", "CAbstractor", "ModalityAdapter", "ProjectorFactory",
    # Training
    "MultimodalDataset", "MultimodalCollator", "MultimodalTrainer",
    "MultimodalLoss", "ContrastiveLoss", "AlignmentLoss",
    "ModalityBalancedSampler", "EvaluationMetrics",
    # Config
    "VisionConfig", "AudioConfig", "VideoConfig",
    "CrossModalConfig", "MultimodalConfig", "ProcessorConfig",
]
