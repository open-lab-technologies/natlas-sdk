"""Download and parse benchmark test sets. Files are cached, never vendored:
MAFAND is GPL-3.0, so redistributing it inside an Apache-2.0 package is not an option.
"""

from __future__ import annotations

import ast
import csv
import io
import json
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


FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{code}/{name}"
# FLEURS has no Nigerian-accented English, so English is scored on US speakers.
FLEURS_CODES = {"en": "en_us", "ha": "ha_ng", "ig": "ig_ng", "yo": "yo_ng"}


def fleurs(lang: str, n: int) -> list[Item]:
    """The first `n` clips of the FLEURS dev split, with their reference transcripts.

    The audio archives are hundreds of MB, so the archive is streamed and
    reading stops once `n` clips are on disk.
    """
    import tarfile

    code = FLEURS_CODES[lang]
    rows = _fetch(FLEURS.format(code=code, name="dev.tsv"), f"fleurs-{code}-dev.tsv")
    transcripts = {}
    for line in rows.splitlines():
        cols = line.split("\t")
        if len(cols) >= 3:
            transcripts[cols[1]] = cols[2]

    audio_dir = cache_dir() / "bench" / f"fleurs-{code}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(p.name for p in audio_dir.glob("*.wav"))
    if len(have) < n:
        url = FLEURS.format(code=code, name="audio/dev.tar.gz")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with tarfile.open(fileobj=_StreamReader(resp.iter_bytes()), mode="r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    if not member.isfile() or name not in transcripts:
                        continue
                    target = audio_dir / name
                    if not target.exists():
                        target.write_bytes(tar.extractfile(member).read())
                        have.append(name)
                    if len(set(have)) >= n:
                        break
        have = sorted(p.name for p in audio_dir.glob("*.wav"))

    return [
        Item(f"fleurs-{lang}-{i}", "", transcripts[name], {"audio": str(audio_dir / name)})
        for i, name in enumerate(have[:n])
    ]


class _StreamReader(io.RawIOBase):
    """File-like view over an iterator of byte chunks, for streaming tarfile reads."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
        size = min(len(b), len(self._buffer))
        b[:size], self._buffer = self._buffer[:size], self._buffer[size:]
        return size


def tool_calling(lang: str) -> list[Item]:
    """Requests that do and do not need one of four tools.

    Written for this package, with tools and questions that played no part in
    designing the tool-calling prompt, so the scores are out of sample. The
    Hausa, Igbo and Yoruba questions have not been reviewed by native speakers.
    """
    import importlib.resources

    spec = json.loads(importlib.resources.files(__package__).joinpath("tool_cases.json")
                      .read_text(encoding="utf-8"))
    return [
        Item(f"tools-{lang}-{i}", prompt, name or "",
             {"tools": spec["tools"], "arguments": arguments})
        for i, (prompt, name, arguments) in enumerate(spec["cases"][lang])
    ]
