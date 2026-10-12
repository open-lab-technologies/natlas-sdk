"""Rebuild src/pynatlas/language_profile.json from the AfriXNLI dev split.

    python scripts/build_language_profile.py

Only the dev split is used; the test split stays held out for measuring
`detect_language`. Tone-stripped copies of every sentence are included because
much real Yoruba and Igbo is typed without tone marks.
"""

import collections
import csv
import io
import json
import math
from pathlib import Path

from pynatlas.bench.data import ISO3, _fetch
from pynatlas.bench.scoring import strip_tone_marks
from pynatlas.languages import trigrams

TOP = 400
URL = "https://huggingface.co/datasets/masakhane/afrixnli/resolve/main/data/{lang}/dev.tsv"
OUT = Path(__file__).parents[1] / "src" / "pynatlas" / "language_profile.json"


def sentences(lang: str) -> list[str]:
    text = _fetch(URL.format(lang=ISO3[lang]), f"afrixnli-{ISO3[lang]}-dev.tsv")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    return list(dict.fromkeys([r["premise"] for r in rows] + [r["hypothesis"] for r in rows]))


profile = {}
for lang in ISO3:
    texts = sentences(lang)
    texts += [strip_tone_marks(t) for t in texts]
    counts = collections.Counter(g for t in texts for g in trigrams(t))
    total = sum(counts.values())
    profile[lang] = {g: round(math.log(n / total), 3) for g, n in counts.most_common(TOP)}

OUT.write_text(json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
