"""Languages N-ATLAS supports, and the official speech model for each."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    asr_model: str


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "NCAIR1/NigerianAccentedEnglish"),
    "ha": Language("ha", "Hausa", "NCAIR1/Hausa-ASR"),
    "ig": Language("ig", "Igbo", "NCAIR1/Igbo-ASR"),
    "yo": Language("yo", "Yoruba", "NCAIR1/Yoruba-ASR"),
}

_ALIASES = {lang.name.lower(): code for code, lang in LANGUAGES.items()}


def get_language(code_or_name: str) -> Language:
    """Look up a language by ISO 639-1 code ("yo") or English name ("Yoruba")."""
    key = code_or_name.strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return LANGUAGES[key]
    except KeyError:
        supported = ", ".join(f"{c} ({l.name})" for c, l in LANGUAGES.items())
        raise ValueError(f"Unsupported language {code_or_name!r}. Supported: {supported}") from None
