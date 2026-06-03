"""Locale data and formatting utilities."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional


# ------------------------------------------------------------------
# Number formatting constants (module-level to avoid dataclass issues)
# ------------------------------------------------------------------

_NUM_FORMATS: Dict[str, Dict[str, str]] = {
    "en": {"thousands": ",", "decimal": "."},
    "es": {"thousands": ".", "decimal": ","},
    "fr": {"thousands": "\u202f", "decimal": ","},
    "de": {"thousands": ".", "decimal": ","},
    "zh": {"thousands": ",", "decimal": "."},
    "ja": {"thousands": ",", "decimal": "."},
    "ko": {"thousands": ",", "decimal": "."},
    "pt": {"thousands": ".", "decimal": ","},
    "ru": {"thousands": "\u202f", "decimal": ","},
    "ar": {"thousands": ",", "decimal": "."},
    "hi": {"thousands": ",", "decimal": "."},
    "it": {"thousands": ".", "decimal": ","},
}

_DATE_FORMATS: Dict[str, str] = {
    "en": "%B %d, %Y",
    "es": "%d de %B de %Y",
    "fr": "%d %B %Y",
    "de": "%d. %B %Y",
    "zh": "%Y年%m月%d日",
    "ja": "%Y年%m月%d日",
    "ko": "%Y년 %m월 %d일",
    "pt": "%d de %B de %Y",
    "ru": "%d %B %Y г.",
    "ar": "%d %B %Y",
    "hi": "%d %B %Y",
    "it": "%d %B %Y",
}


@dataclass(frozen=True)
class Locale:
    """Represents a single locale with metadata and formatting helpers.

    Attributes:
        code: ISO 639-1 language code (e.g. ``"en"``, ``"zh"``).
        name: English name of the language.
        native_name: Name of the language in its own script.
        direction: Text direction — ``"ltr"`` or ``"rtl"``.
    """

    code: str
    name: str
    native_name: str
    direction: str = "ltr"

    def __post_init__(self) -> None:
        if self.direction not in ("ltr", "rtl"):
            raise ValueError(
                f"Invalid direction '{self.direction}'. Must be 'ltr' or 'rtl'."
            )

    # ------------------------------------------------------------------
    # Number formatting
    # ------------------------------------------------------------------

    def format_number(self, n: float, locale: Optional[str] = None) -> str:
        """Format a number according to locale conventions.

        Uses the appropriate thousands separator and decimal point for
        the locale.

        Args:
            n: The number to format.
            locale: Override locale code; defaults to ``self.code``.

        Returns:
            The formatted number string.
        """
        loc = locale or self.code
        fmt = _NUM_FORMATS.get(loc, {"thousands": ",", "decimal": "."})

        is_negative = n < 0
        # Use string representation to avoid floating-point subtraction errors
        raw = f"{abs(n):.10f}".rstrip("0").rstrip(".")
        if "." in raw:
            int_str, frac_str = raw.split(".", 1)
        else:
            int_str = raw
            frac_str = ""

        # Format integer part with thousands separators
        groups: list[str] = []
        while len(int_str) > 3:
            groups.append(int_str[-3:])
            int_str = int_str[:-3]
        groups.append(int_str)
        groups.reverse()

        result = fmt["thousands"].join(groups)

        # Add fractional part if present
        if frac_str:
            result += fmt["decimal"] + frac_str

        if is_negative:
            result = "-" + result

        return result

    # ------------------------------------------------------------------
    # Date formatting
    # ------------------------------------------------------------------

    def format_date(
        self,
        date: datetime.date,
        locale: Optional[str] = None,
    ) -> str:
        """Format a date according to locale conventions.

        Args:
            date: The date to format.
            locale: Override locale code; defaults to ``self.code``.

        Returns:
            The locale-formatted date string.
        """
        loc = locale or self.code
        fmt = _DATE_FORMATS.get(loc, "%Y-%m-%d")
        try:
            return date.strftime(fmt)
        except (ValueError, TypeError):
            return str(date)


# ------------------------------------------------------------------
# Built-in locale definitions
# ------------------------------------------------------------------

BUILTIN_LOCALES: List[Locale] = [
    Locale(code="en", name="English", native_name="English", direction="ltr"),
    Locale(code="es", name="Spanish", native_name="Español", direction="ltr"),
    Locale(code="fr", name="French", native_name="Français", direction="ltr"),
    Locale(code="de", name="German", native_name="Deutsch", direction="ltr"),
    Locale(code="zh", name="Chinese", native_name="中文", direction="ltr"),
    Locale(code="ja", name="Japanese", native_name="日本語", direction="ltr"),
    Locale(code="ko", name="Korean", native_name="한국어", direction="ltr"),
    Locale(code="pt", name="Portuguese", native_name="Português", direction="ltr"),
    Locale(code="ru", name="Russian", native_name="Русский", direction="ltr"),
    Locale(code="ar", name="Arabic", native_name="العربية", direction="rtl"),
    Locale(code="hi", name="Hindi", native_name="हिन्दी", direction="ltr"),
    Locale(code="it", name="Italian", native_name="Italiano", direction="ltr"),
]
