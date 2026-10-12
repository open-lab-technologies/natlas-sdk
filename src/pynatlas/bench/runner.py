"""Run benchmark tasks against any N-ATLAS endpoint and write per-item results."""

from __future__ import annotations

import json
import logging
import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..client import NAtlas
from ..errors import NAtlasError
from ..types import ToolCall
from . import data, scoring

log = logging.getLogger("pynatlas.bench")


Metric = Literal["accuracy", "chrf", "wer", "tool-calls"]
OUTCOMES = ("false_call", "missed_call", "wrong_tool", "wrong_arguments")


@dataclass(frozen=True)
class Task:
    name: str
    languages: tuple[str, ...]
    load: Callable[..., list[data.Item]]
    max_tokens: int
    metric: Metric
    parse: Callable[[str], str | None] | None = None
    matches: Callable[[str | None, str], bool] = operator.eq
    chance: float | None = None


ALL = ("en", "ha", "ig", "yo")
NIGERIAN = ("ha", "ig", "yo")

TASKS: dict[str, Task] = {
    "afrimmlu": Task("afrimmlu", ALL, data.afrimmlu, 8, "accuracy",
                     parse=scoring.parse_choice, chance=0.25),
    "afrixnli": Task("afrixnli", ALL, data.afrixnli, 8, "accuracy",
                     parse=scoring.parse_nli, chance=1 / 3),
    "afrimgsm": Task("afrimgsm", ALL, data.afrimgsm, 512, "accuracy",
                     parse=scoring.parse_number, matches=scoring.numbers_equal),
    "mt-en-xx": Task("mt-en-xx", NIGERIAN, lambda l: data.mafand(l, "en-xx"), 256, "chrf"),
    "mt-xx-en": Task("mt-xx-en", NIGERIAN, lambda l: data.mafand(l, "xx-en"), 256, "chrf"),
    "asr-fleurs": Task("asr-fleurs", ALL, data.fleurs, 0, "wer"),
    "tool-calling": Task("tool-calling", ALL, data.tool_calling, 200, "tool-calls"),
}


def _items(task: Task, lang: str, n: int | None, seed: int) -> list[data.Item]:
    if task.metric == "wer":
        # FLEURS clips are streamed from the start of the archive, so take the
        # first n rather than a random sample of the whole split.
        return task.load(lang, n or 50)
    return data.sample(task.load(lang), n, seed)


def _ask(client: NAtlas, task: Task, item: data.Item, lang: str) -> tuple[str, ToolCall | None]:
    if task.metric == "wer":
        return client.transcribe(item.meta["audio"], language=lang), None
    if task.metric == "tool-calls":
        reply = client.chat(item.prompt, tools=item.meta["tools"], temperature=0.0,
                            max_tokens=task.max_tokens)
        return reply.text, reply.tool_calls[0] if reply.tool_calls else None
    return client.chat(item.prompt, temperature=0.0, max_tokens=task.max_tokens).text, None


def _row(task: Task, item: data.Item, reply: str, call: ToolCall | None) -> dict[str, Any]:
    row: dict[str, Any] = {"id": item.id, "reply": reply}
    if task.metric == "accuracy":
        parsed = task.parse(reply) if task.parse else reply
        row.update(gold=item.gold, parsed=parsed, correct=task.matches(parsed, item.gold))
    elif task.metric == "wer":
        row.update(gold=item.gold, wer=round(100 * scoring.wer([reply], [item.gold]), 1))
    elif task.metric == "tool-calls":
        called, arguments = (call.name, call.arguments) if call else (None, None)
        row.update(expected=item.gold or None, called=called, arguments=arguments,
                   outcome=scoring.tool_call_outcome(called, arguments or {}, item.gold,
                                                     item.meta["arguments"]))
    # chrf rows carry no reference: MAFAND is GPL-3.0, and the id is the line
    # number in the upstream test file.
    return row


def _score(task: Task, rows: list[dict[str, Any]], replies: list[str],
           refs: list[str]) -> dict[str, Any]:
    if task.metric == "accuracy":
        result = {"score": round(100 * sum(r["correct"] for r in rows) / len(rows), 1),
                  "unparsed": sum(r["parsed"] is None for r in rows)}
        if task.chance:
            result["chance"] = round(100 * task.chance, 1)
        return result
    if task.metric == "wer":
        strict = scoring.wer(replies, refs)
        lenient = scoring.wer(replies, refs, ignore_tones=True)
        return {"score": round(100 * strict, 1), "score_ignoring_tones": round(100 * lenient, 1)}
    if task.metric == "tool-calls":
        return {"score": round(100 * sum(r["outcome"] == "correct" for r in rows) / len(rows), 1),
                **{o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}}
    first_lines = [r.strip().splitlines()[0] if r.strip() else "" for r in replies]
    return {"score": round(scoring.chrf(first_lines, refs), 1)}


def run(
    client: NAtlas,
    tasks: list[str],
    languages: list[str] | None,
    n: int | None,
    out_dir: Path,
    seed: int = 0,
    label: str = "n-atlas",
    progress: Callable[[str, str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run `tasks` and write results to `out_dir`.

    `progress` is called with (task, language, items done, items total).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
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
            items = _items(task, lang, n, seed)
            rows, replies, failures, started = [], [], 0, time.time()
            for done, item in enumerate(items, 1):
                try:
                    reply, call = _ask(client, task, item, lang)
                except NAtlasError as e:
                    reply, call, failures = "", None, failures + 1
                    log.warning("%s failed: %s", item.id, e)
                rows.append(_row(task, item, reply, call))
                replies.append(reply)
                if progress:
                    progress(name, lang, done, len(items))

            summary["results"].append({
                "task": name, "language": lang, "n": len(items), "metric": task.metric,
                "errors": failures, "seconds": round(time.time() - started, 1),
                **_score(task, rows, replies, [i.gold for i in items]),
            })
            with (out_dir / f"{name}-{lang}.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    summary["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(summary))
    return summary


def to_markdown(summary: dict[str, Any]) -> str:
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
