"""Download and parse benchmark test sets. Files are cached, never vendored:
MAFAND is GPL-3.0, so redistributing it inside an Apache-2.0 package is not an option.
"""

from __future__ import annotations

import ast
import csv
import io
import random
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..models import cache_dir

IROKO = "https://huggingface.co/datasets/masakhane/{task}/resolve/main/data/{lang}/test.tsv"
MAFAND = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/text_files/en-{lang}/test.{ext}"

# Dataset language codes are ISO 639-3; the SDK uses 639-1.
ISO3 = {"en": "eng", "ha": "hau", "ig": "ibo", "yo": "yor"}


@dataclass
class Item:
    id: str
    prompt: str
    gold: str
    meta: dict = field(default_factory=dict)


def _fetch(url: str, name: str) -> str:
    path: Path = cache_dir() / "bench" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        path.write_text(resp.text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


def _tsv(task: str, lang: str) -> list[dict[str, str]]:
    text = _fetch(IROKO.format(task=task, lang=ISO3[lang]), f"{task}-{ISO3[lang]}.tsv")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"{task}/{lang}: malformed rows; the upstream file may have changed")
    return rows


def sample(items: list[Item], n: int | None, seed: int) -> list[Item]:
    if n is None or n >= len(items):
        return items
    return sorted(random.Random(seed).sample(items, n), key=lambda i: int(i.id.rsplit("-", 1)[1]))


def afrimmlu(lang: str) -> list[Item]:
    items = []
    for i, row in enumerate(_tsv("afrimmlu", lang)):
        choices = ast.literal_eval(row["choices"])
        options = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", choices))
        prompt = (
            "Answer the following multiple choice question. "
            "Reply with the letter of the correct answer only (A, B, C or D).\n\n"
            f"Question: {row['question']}\n{options}\nAnswer:"
        )
        items.append(Item(f"afrimmlu-{lang}-{i}", prompt, row["answer"].strip().upper(),
                          {"subject": row.get("subject")}))
    return items


XNLI_LABELS = {"0": "entailment", "1": "neutral", "2": "contradiction"}


def afrixnli(lang: str) -> list[Item]:
    items = []
    for i, row in enumerate(_tsv("afrixnli", lang)):
        prompt = (
            "Read the premise and the hypothesis. Decide whether the premise entails the "
            "hypothesis, contradicts it, or is neutral. Reply with one word: "
            "entailment, neutral or contradiction.\n\n"
            f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}\nAnswer:"
        )
        items.append(Item(f"afrixnli-{lang}-{i}", prompt, XNLI_LABELS[row["label"].strip()]))
    return items


def afrimgsm(lang: str) -> list[Item]:
    items = []
    for i, row in enumerate(_tsv("afrimgsm", lang)):
        prompt = (
            "Solve the following maths problem step by step. "
            "End your reply with a final line of the form 'Answer: <number>'.\n\n"
            f"Problem: {row['question']}"
        )
        gold = row["answer_number"].strip().replace(",", "")
        items.append(Item(f"afrimgsm-{lang}-{i}", prompt, gold))
    return items


def mafand(lang: str, direction: str) -> list[Item]:
    """direction is "en-xx" (English into `lang`) or "xx-en"."""
    from ..languages import get_language

    code3 = ISO3[lang]
    en = _fetch(MAFAND.format(lang=code3, ext="en"), f"mafand-{code3}.en").splitlines()
    xx = _fetch(MAFAND.format(lang=code3, ext=code3), f"mafand-{code3}.{code3}").splitlines()
    name = get_language(lang).name
    src_name, dst_name = ("English", name) if direction == "en-xx" else (name, "English")
    items = []
    for i, (e, x) in enumerate(zip(en, xx)):
        if not e.strip() or not x.strip():
            continue
        src, ref = (e, x) if direction == "en-xx" else (x, e)
        prompt = (
            f"Translate the following {src_name} text into {dst_name}. "
            f"Reply with the {dst_name} translation only.\n\n{src}"
        )
        items.append(Item(f"mafand-{direction.replace('xx', lang)}-{i}", prompt, ref))
    return items
