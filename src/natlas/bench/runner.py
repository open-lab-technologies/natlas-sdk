"""Run benchmark tasks against any N-ATLAS endpoint and write per-item results."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..client import NAtlas
from ..errors import NAtlasError
from . import data, scoring

log = logging.getLogger("natlas.bench")


@dataclass(frozen=True)
class Task:
    name: str
    languages: tuple[str, ...]
    load: Callable[[str], list[data.Item]]
    max_tokens: int
    metric: str  # "accuracy", "chrf" or "wer"
    parse: Callable[[str], str | None] | None = None
    chance: float | None = None  # score from guessing, to read results against
    audio: bool = False  # items carry an audio file to transcribe instead of a prompt


TASKS: dict[str, Task] = {
    "afrimmlu": Task("afrimmlu", ("en", "ha", "ig", "yo"), data.afrimmlu, 8, "accuracy",
                     scoring.parse_choice, 0.25),
    "afrixnli": Task("afrixnli", ("en", "ha", "ig", "yo"), data.afrixnli, 8, "accuracy",
                     scoring.parse_nli, 1 / 3),
    "afrimgsm": Task("afrimgsm", ("en", "ha", "ig", "yo"), data.afrimgsm, 512, "accuracy",
                     scoring.parse_number),
    "mt-en-xx": Task("mt-en-xx", ("ha", "ig", "yo"), lambda l: data.mafand(l, "en-xx"), 256,
                     "chrf"),
    "mt-xx-en": Task("mt-xx-en", ("ha", "ig", "yo"), lambda l: data.mafand(l, "xx-en"), 256,
                     "chrf"),
    "asr-fleurs": Task("asr-fleurs", ("en", "ha", "ig", "yo"), data.fleurs, 0, "wer",
                       audio=True),
}


def _correct(task: Task, parsed: str | None, gold: str) -> bool:
    if task.name == "afrimgsm":
        return scoring.numbers_equal(parsed, gold)
    return parsed == gold


def run(
    client: NAtlas,
    tasks: list[str],
    languages: list[str] | None,
    n: int | None,
    out_dir: Path,
    seed: int = 0,
    label: str = "n-atlas",
    progress: Callable[[str, str, int, int], None] | None = None,
) -> dict:
    """Run `tasks` and write results to `out_dir`.

    `progress` is called with (task, language, items done, items total).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "label": label,
        "endpoint": client.base_url,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_per_language": n,
        "seed": seed,
        "results": [],
    }
    for name in tasks:
        task = TASKS[name]
        for lang in task.languages:
            if languages and lang not in languages:
                continue
            if task.audio:
                # FLEURS clips are streamed from the start of the archive, so take the
                # first n rather than a random sample of the whole split.
                items = task.load(lang, n or 50)
            else:
                items = data.sample(task.load(lang), n, seed)
            rows, hyps, refs = [], [], []
            failures = 0
            t0 = time.time()
            for k, item in enumerate(items, 1):
                try:
                    if task.audio:
                        reply = client.transcribe(item.meta["audio"], language=lang)
                    else:
                        reply = client.chat(item.prompt, temperature=0.0,
                                            max_tokens=task.max_tokens).text
                except NAtlasError as e:
                    reply, failures = "", failures + 1
                    log.warning("%s failed: %s", item.id, e)
                row = {"id": item.id, "reply": reply}
                if task.metric == "accuracy":
                    row["gold"] = item.gold
                    row["parsed"] = task.parse(reply)
                    row["correct"] = _correct(task, row["parsed"], item.gold)
                elif task.metric == "wer":
                    row["gold"] = item.gold
                    row["wer"] = round(100 * scoring.wer([reply], [item.gold]), 1)
                    hyps.append(reply)
                    refs.append(item.gold)
                else:
                    # MAFAND references are GPL-3.0, so they are not written out; the id
                    # is the line number in the upstream test file.
                    hyps.append(reply.strip().splitlines()[0] if reply.strip() else "")
                    refs.append(item.gold)
                rows.append(row)
                if progress:
                    progress(name, lang, k, len(items))

            result = {"task": name, "language": lang, "n": len(items), "metric": task.metric,
                      "errors": failures, "seconds": round(time.time() - t0, 1)}
            if task.metric == "accuracy":
                result["score"] = round(100 * sum(r["correct"] for r in rows) / len(rows), 1)
                result["unparsed"] = sum(r["parsed"] is None for r in rows)
                if task.chance:
                    result["chance"] = round(100 * task.chance, 1)
            elif task.metric == "wer":
                result["score"] = round(100 * scoring.wer(hyps, refs), 1)
                result["score_ignoring_tones"] = round(
                    100 * scoring.wer(hyps, refs, ignore_tones=True), 1)
            else:
                result["score"] = round(scoring.chrf(hyps, refs), 1)
            summary["results"].append(result)

            with (out_dir / f"{name}-{lang}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(summary))
    return summary


def to_markdown(summary: dict) -> str:
    lines = [
        f"# {summary['label']}",
        "",
        (f"Endpoint `{summary['endpoint']}`, {summary['n_per_language'] or 'all'} items per "
         f"language (seed {summary['seed']}), run {summary['started']}."),
        "",
        "| Task | Language | Metric | Score | Chance | n | Unparsed | Errors |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| {r['task']} | {r['language']} | {r['metric']} | {r['score']} | "
            f"{r.get('chance', '')} | {r['n']} | {r.get('unparsed', '')} | {r['errors']} |"
        )
    return "\n".join(lines) + "\n"
