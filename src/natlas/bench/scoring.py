"""Turning free-text model replies into answers that can be scored."""

from __future__ import annotations

import re

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_choice(reply: str) -> str | None:
    """First standalone capital A-D, so "B", "B.", "(B)" and "Answer: B" all count.

    Case matters: "a" is a common word in Hausa and Yoruba and must not read as option A.
    """
    match = re.search(r"\b([ABCD])\b", reply.strip())
    return match.group(1) if match else None


def parse_nli(reply: str) -> str | None:
    text = reply.lower()
    hits = [(text.find(label), label) for label in ("entailment", "neutral", "contradiction")]
    hits = [h for h in hits if h[0] >= 0]
    return min(hits)[1] if hits else None


def parse_number(reply: str) -> str | None:
    """Prefer the number after the last "Answer:", else the last number in the reply."""
    tail = reply.rsplit("Answer:", 1)[-1] if "Answer:" in reply else reply
    numbers = _NUMBER.findall(tail) or _NUMBER.findall(reply)
    if not numbers:
        return None
    value = numbers[-1].replace(",", "").rstrip(".")
    return value.removesuffix(".0")


def numbers_equal(a: str | None, b: str) -> bool:
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def chrf(hypotheses: list[str], references: list[str]) -> float:
    """Corpus chrF++ (character n-grams plus word bigrams), the MAFAND paper's metric."""
    from sacrebleu.metrics import CHRF

    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def normalise_transcript(text: str) -> str:
    """Lowercase and drop punctuation, keeping tone marks: in Yoruba and Igbo they
    change the word, so a missing diacritic is a real recognition error."""
    import unicodedata

    text = unicodedata.normalize("NFC", text.lower())
    text = "".join(ch if ch.isalnum() or ch.isspace() or ch == "'" or
                   unicodedata.category(ch) == "Mn" else " " for ch in text)
    return " ".join(text.split())


def strip_tone_marks(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def wer(hypotheses: list[str], references: list[str], ignore_tones: bool = False) -> float:
    """Corpus word error rate as a fraction, after `normalise_transcript`.

    `ignore_tones` also strips diacritics from both sides. Use it alongside the strict
    score when references are inconsistently toned, as FLEURS Yoruba is.
    """
    import jiwer

    def prep(text: str) -> str:
        text = normalise_transcript(text)
        return strip_tone_marks(text) if ignore_tones else text

    return jiwer.wer([prep(r) for r in references], [prep(h) for h in hypotheses])
