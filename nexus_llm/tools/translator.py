"""Nexus-LLM Translation Tool.

Provides text translation capabilities supporting multiple languages
and translation providers.
"""

import logging
from typing import Any, Dict, List, Optional

from nexus_llm.tools.base_tool import BaseTool, ToolParameter, ToolResult, ParameterType

logger = logging.getLogger(__name__)

# Simple language code mapping
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
}


class TranslatorTool(BaseTool):
    """Tool for translating text between languages.

    Supports direct translation with language code pairs and
    automatic language detection hints.

    Example::

        tool = TranslatorTool()
        result = tool.run(text="Hello world", source_lang="en", target_lang="es")
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="translator", description="Translate text between languages", **kwargs)

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="text", type=ParameterType.STRING, description="Text to translate", required=True),
            ToolParameter(name="source_lang", type=ParameterType.STRING, description="Source language code", required=True, default="en"),
            ToolParameter(name="target_lang", type=ParameterType.STRING, description="Target language code", required=True),
            ToolParameter(name="provider", type=ParameterType.STRING, description="Translation provider", required=False, default="local"),
        ]

    def execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get("text", "")
        source_lang = kwargs.get("source_lang", "en")
        target_lang = kwargs.get("target_lang", "")
        provider = kwargs.get("provider", "local")

        if not text:
            return ToolResult(success=False, error="No text provided for translation")
        if not target_lang:
            return ToolResult(success=False, error="No target language specified")

        try:
            translated = self._translate(text, source_lang, target_lang, provider)
            return ToolResult(
                success=True,
                output={
                    "original_text": text,
                    "translated_text": translated,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                },
                metadata={"provider": provider},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

    def _translate(self, text: str, source_lang: str, target_lang: str, provider: str) -> str:
        """Perform the translation.

        In local mode, returns a placeholder indicating the translation
        request. Real implementations would call an external API.

        Args:
            text: Text to translate.
            source_lang: Source language code.
            target_lang: Target language code.
            provider: Translation provider name.

        Returns:
            Translated text string.
        """
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        return f"[{target_name}] {text}"

    def get_supported_languages(self) -> Dict[str, str]:
        """Get the mapping of supported language codes to names.

        Returns:
            Dictionary of language code -> name.
        """
        return dict(LANGUAGE_NAMES)
