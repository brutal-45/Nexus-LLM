"""Model exporter for Nexus-LLM.

Exports model weights, configuration, and tokenizers.  Supports
the HuggingFace Transformers format natively; ONNX and GGUF are
mock exports that produce placeholder files.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

ModelLike = Any
TokenizerLike = Any


class ModelExporter:
    """Export models, configs, and tokenizers to various formats.

    Example::

        exporter = ModelExporter()
        exporter.export_model(model, "output_dir", format="transformers")
        exporter.export_config(config, "output_dir/config.json")
    """

    # ------------------------------------------------------------------
    # Model export
    # ------------------------------------------------------------------

    def export_model(
        self,
        model: ModelLike,
        path: Union[str, Path],
        format: str = "transformers",
    ) -> str:
        """Export a model to the specified format.

        Args:
            model: A model-like object.
            path: Destination directory or file path.
            format: One of ``"transformers"``, ``"onnx"``, ``"gguf"``.

        Returns:
            The path the model was exported to.

        Raises:
            ValueError: If the format is unsupported.
        """
        format = format.lower()
        path = str(path)
        os.makedirs(path, exist_ok=True)

        if format == "transformers":
            return self._export_transformers(model, path)
        elif format == "onnx":
            return self._export_onnx_mock(model, path)
        elif format == "gguf":
            return self._export_gguf_mock(model, path)
        else:
            raise ValueError(
                f"Unsupported model export format {format!r}. "
                f"Supported: transformers, onnx, gguf"
            )

    # ------------------------------------------------------------------
    # Config export
    # ------------------------------------------------------------------

    def export_config(
        self,
        model_config: Dict[str, Any],
        path: Union[str, Path],
    ) -> str:
        """Export a model configuration dict to a JSON file.

        Args:
            model_config: Configuration dictionary.
            path: Destination file path.

        Returns:
            The path the config was written to.
        """
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2, default=str, ensure_ascii=False)

        logger.info("Exported model config to %s", path)
        return path

    # ------------------------------------------------------------------
    # Tokenizer export
    # ------------------------------------------------------------------

    def export_tokenizer(
        self,
        tokenizer: TokenizerLike,
        path: Union[str, Path],
    ) -> str:
        """Export a tokenizer to the specified directory.

        If the tokenizer has a ``save_pretrained`` method (HuggingFace)
        it is used.  Otherwise a minimal ``tokenizer_config.json`` is
        written.

        Args:
            tokenizer: A tokenizer-like object.
            path: Destination directory.

        Returns:
            The path the tokenizer was exported to.
        """
        path = str(path)
        os.makedirs(path, exist_ok=True)

        save_fn = getattr(tokenizer, "save_pretrained", None)
        if save_fn is not None and callable(save_fn):
            save_fn(path)
            logger.info("Exported tokenizer via save_pretrained to %s", path)
        else:
            # Write a minimal config
            config = {
                "tokenizer_class": getattr(tokenizer, "__class__.__name__", "UnknownTokenizer"),
                "model_max_length": getattr(tokenizer, "model_max_length", 512),
                "vocab_size": getattr(tokenizer, "vocab_size", len(getattr(tokenizer, "get_vocab", lambda: {})())),
            }
            config_path = os.path.join(path, "tokenizer_config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(config, fh, indent=2, ensure_ascii=False)
            logger.info("Exported tokenizer config to %s", config_path)

        return path

    # ------------------------------------------------------------------
    # Internal exporters
    # ------------------------------------------------------------------

    @staticmethod
    def _export_transformers(model: ModelLike, path: str) -> str:
        """Export using HuggingFace Transformers ``save_pretrained``."""
        save_fn = getattr(model, "save_pretrained", None)
        if save_fn is not None and callable(save_fn):
            save_fn(path)
            logger.info("Exported model via save_pretrained to %s", path)
        else:
            # Write a placeholder model card
            model_card = {
                "model_type": getattr(model, "config", {})
                if isinstance(getattr(model, "config", None), dict)
                else "unknown",
                "export_format": "transformers",
                "note": "Placeholder export — no save_pretrained method available",
            }
            card_path = os.path.join(path, "config.json")
            with open(card_path, "w", encoding="utf-8") as fh:
                json.dump(model_card, fh, indent=2, default=str, ensure_ascii=False)
            logger.info("Exported placeholder model config to %s", card_path)

        return path

    @staticmethod
    def _export_onnx_mock(model: ModelLike, path: str) -> str:
        """Mock ONNX export — writes a placeholder file.

        In production, ``torch.onnx.export`` or ``optimum.onnxruntime``
        would be used.
        """
        onnx_path = os.path.join(path, "model.onnx")
        placeholder = {
            "format": "onnx",
            "note": "Mock ONNX export — use optimum or torch.onnx.export in production",
            "model_class": getattr(model, "__class__.__name__", "UnknownModel"),
        }
        with open(onnx_path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(placeholder, fh, indent=2, default=str)

        # Write a minimal ONNX file header (just the bytes marker)
        with open(onnx_path, "wb") as fh:
            fh.write(b"MOCK_ONNX_PLACEHOLDER")

        logger.info("Exported mock ONNX model to %s", onnx_path)
        return path

    @staticmethod
    def _export_gguf_mock(model: ModelLike, path: str) -> str:
        """Mock GGUF export — writes a placeholder file.

        In production, ``llama.cpp``'s ``convert.py`` or the
        ``gguf`` Python package would be used.
        """
        gguf_path = os.path.join(path, "model.gguf")
        placeholder = {
            "format": "gguf",
            "note": "Mock GGUF export — use llama.cpp convert in production",
            "model_class": getattr(model, "__class__.__name__", "UnknownModel"),
        }
        with open(gguf_path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(placeholder, fh, indent=2, default=str)

        with open(gguf_path, "wb") as fh:
            fh.write(b"MOCK_GGUF_PLACEHOLDER")

        logger.info("Exported mock GGUF model to %s", gguf_path)
        return path
