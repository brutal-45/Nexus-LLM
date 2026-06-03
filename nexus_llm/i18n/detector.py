"""Heuristic-based language detection for short texts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class _ScriptRange:
    """Unicode character-range specification for a script."""

    name: str
    ranges: List[Tuple[int, int]]  # (start, end) inclusive
    locales: List[str]  # locale codes that use this script


# Define script ranges used for heuristic detection.
_SCRIPTS: List[_ScriptRange] = [
    _ScriptRange(
        name="cjk_unified",
        ranges=[(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
        locales=["zh"],
    ),
    _ScriptRange(
        name="hiragana_katakana",
        ranges=[(0x3040, 0x309F), (0x30A0, 0x30FF)],
        locales=["ja"],
    ),
    _ScriptRange(
        name="hangul",
        ranges=[(0xAC00, 0xD7AF), (0x1100, 0x11FF)],
        locales=["ko"],
    ),
    _ScriptRange(
        name="arabic",
        ranges=[(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
        locales=["ar"],
    ),
    _ScriptRange(
        name="devanagari",
        ranges=[(0x0900, 0x097F)],
        locales=["hi"],
    ),
    _ScriptRange(
        name="cyrillic",
        ranges=[(0x0400, 0x04FF), (0x0500, 0x052F)],
        locales=["ru"],
    ),
]

# Latin-script locales that we try to disambiguate with diacritics.
_LATIN_LOCALE_MARKERS: Dict[str, str] = {
    "de": "äöüßÄÖÜ",
    "fr": "àâçéèêëîïôùûüÿŒœÀÂÇÉÈÊËÎÏÔÙÛÜŸ",
    "es": "áéíóúñ¿¡ÁÉÍÓÚÑ",
    "pt": "àáâãçéêíóôõúÃÀÁÂÇÉÊÍÓÔÕÚ",
    "it": "àèéìòùÀÈÉÌÒÙ",
}


class LanguageDetector:
    """Simple heuristic language detector based on Unicode character ranges.

    This is **not** a statistical NLP detector; it relies on the
    distribution of Unicode scripts and diacritical markers in the
    input text.  It works well for texts written in a single language
    that uses a distinctive script (CJK, Arabic, Cyrillic, Devanagari,
    Hangul) and provides best-effort guesses for Latin-script languages
    via diacritic heuristics.

    Example::

        det = LanguageDetector()
        det.detect("こんにちは世界")          # "ja"
        det.detect("Hello world")            # "en"
        det.detect_confidence("Привет")      # ("ru", 0.95)
    """

    def __init__(self, default_locale: str = "en") -> None:
        self._default = default_locale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> str:
        """Detect the most likely locale for *text*.

        Args:
            text: Input string.

        Returns:
            A locale code (e.g. ``"en"``, ``"ja"``).
        """
        locale, _ = self.detect_confidence(text)
        return locale

    def detect_confidence(self, text: str) -> Tuple[str, float]:
        """Detect locale and return a confidence score.

        Args:
            text: Input string.

        Returns:
            A ``(locale_code, confidence)`` tuple.  *confidence* is
            between 0.0 and 1.0.
        """
        if not text or not text.strip():
            return (self._default, 0.0)

        cleaned = self._clean(text)
        if not cleaned:
            return (self._default, 0.0)

        # Count characters per script
        script_counts = self._count_scripts(cleaned)
        total_chars = sum(script_counts.values())

        if total_chars == 0:
            return (self._default, 0.0)

        # If a non-Latin script dominates, return its locale
        for script_name, count in script_counts.most_common():
            # Skip Latin — we handle it separately
            if script_name == "latin":
                continue
            ratio = count / total_chars
            if ratio >= 0.15:
                # Find the associated locale
                for script_def in _SCRIPTS:
                    if script_def.name == script_name:
                        return (script_def.locales[0], min(ratio + 0.2, 1.0))

        # Latin-script disambiguation via diacritics
        latin_locale, latin_conf = self._disambiguate_latin(cleaned)
        if latin_conf > 0.0:
            return (latin_locale, latin_conf)

        # Pure ASCII — assume English with moderate confidence
        return (self._default, 0.5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Remove digits, punctuation, and whitespace for analysis."""
        import string
        # Build a set of characters to remove
        remove = set(string.digits + string.whitespace + string.punctuation)
        return "".join(ch for ch in text if ch not in remove)

    @staticmethod
    def _count_scripts(text: str) -> Dict[str, int]:
        """Count characters by Unicode script category."""
        from collections import Counter

        counts: Counter = Counter()

        for ch in text:
            cp = ord(ch)

            # Check defined scripts
            matched = False
            for script_def in _SCRIPTS:
                for start, end in script_def.ranges:
                    if start <= cp <= end:
                        counts[script_def.name] += 1
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                # Classify as Latin or other
                try:
                    name = unicodedata.name(ch, "")
                    if name.startswith("LATIN"):
                        counts["latin"] += 1
                    else:
                        counts["other"] += 1
                except ValueError:
                    counts["other"] += 1

        return counts

    @staticmethod
    def _disambiguate_latin(text: str) -> Tuple[str, float]:
        """Try to distinguish Latin-script languages via diacritics."""
        best_locale = ""
        best_score = 0.0
        text_len = max(len(text), 1)

        for locale, markers in _LATIN_LOCALE_MARKERS.items():
            count = sum(1 for ch in text if ch in markers)
            score = count / text_len
            # Boost score: even a few markers can be decisive
            if count > 0:
                score = min(score * 5, 1.0)
            if score > best_score:
                best_score = score
                best_locale = locale

        if best_score > 0.0:
            return (best_locale, best_score)

        return ("", 0.0)
