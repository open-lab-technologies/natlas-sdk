"""Measure `detect_language` on text it was not built from.

    python scripts/evaluate_language_detection.py

Writes bench-results/language-detection/summary.json. AfriXNLI test is held out
from the profile, which uses the dev split only; FLEURS is a different domain.
"""

import csv
import io
import json
from pathlib import Path

from pynatlas.bench.data import FLEURS, FLEURS_CODES, ISO3, _fetch
from pynatlas.bench.scoring import strip_tone_marks
from pynatlas.languages import detect_language

URL = "https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{lang}/test.tsv"
OUT = Path(__file__).parents[1] / "bench-results" / "language-detection" / "summary.json"


def afrixnli_test(lang: str) -> list[str]:
    text = _fetch(URL.format(lang=ISO3[lang]), f"afrixnli-{ISO3[lang]}-test.tsv")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    return list(dict.fromkeys([r["premise"] for r in rows] + [r["hypothesis"] for r in rows]))


def fleurs_transcripts(lang: str) -> list[str]:
    code = FLEURS_CODES[lang]
    rows = _fetch(FLEURS.format(code=code, name="dev.tsv"), f"fleurs-{code}-dev.tsv")
    return list(dict.fromkeys(r.split("\t")[2] for r in rows.splitlines() if r.count("\t") >= 2))


def accuracy(texts: list[str], lang: str) -> dict[str, float | int]:
    return {"n": len(texts),
            "accuracy": round(100 * sum(detect_language(t) == lang for t in texts) / len(texts), 1)}


results = {}
for lang in ISO3:
    test = afrixnli_test(lang)
    short = [t for t in test if len(t.split()) <= 5]
    results[lang] = {
        "afrixnli_test": accuracy(test, lang),
        "afrixnli_test_without_tone_marks": accuracy([strip_tone_marks(t) for t in test], lang),
        "afrixnli_test_five_words_or_fewer": accuracy(short, lang),
        "fleurs_transcripts": accuracy(fleurs_transcripts(lang), lang),
    }
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results, indent=2) + "\n")
for lang, sets in results.items():
    print(lang, {name: r["accuracy"] for name, r in sets.items()})
