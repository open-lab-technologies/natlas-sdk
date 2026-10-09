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
