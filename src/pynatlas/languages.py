"""Languages N-ATLaS supports, the official speech model for each, and detection."""

from __future__ import annotations

import functools
import importlib.resources
import json
import unicodedata
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


def trigrams(text: str) -> list[str]:
    padded = " " + " ".join(unicodedata.normalize("NFC", text.lower()).split()) + " "
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


@functools.cache
def _profile() -> dict[str, dict[str, float]]:
    raw = importlib.resources.files(__package__).joinpath("language_profile.json")
    profile: dict[str, dict[str, float]] = json.loads(raw.read_text(encoding="utf-8"))
    return profile


def detect_language(text: str) -> str | None:
    """The code ("en", "ha", "ig" or "yo") of the language `text` is written in.

    Runs offline from character patterns, with no model call. It assumes the text
    is in one of the four languages, so text in any other language still gets one
    of these codes. Returns None when there is nothing to go on. Accuracy is
    measured by scripts/evaluate_language_detection.py.
    """
    grams = trigrams(text)
    if not any(ch.isalpha() for ch in text):
        return None
    best, best_score = None, float("-inf")
    for code, weights in _profile().items():
        unseen = min(weights.values()) - 2.0
        score = sum(weights.get(g, unseen) for g in grams)
        if score > best_score:
            best, best_score = code, score
    return best
