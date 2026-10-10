"""`natlas` command line: download, serve, chat, translate, transcribe."""

from __future__ import annotations

import argparse
import os
import sys

from .client import NAtlas, NAtlasError
from .models import DEFAULT_QUANT, QUANTS, DownloadError, default_model_path, download


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .engine import LlamaCppEngine
    from .server import create_app

    path = args.model or default_model_path(args.quant)
    engine = LlamaCppEngine(path, n_ctx=args.ctx)
    asr = None
    if args.asr:
        from .asr import SpeechRecogniser

        asr = SpeechRecogniser()
    api_key = args.api_key or os.getenv("NATLAS_API_KEY")
    if not api_key and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: serving on {args.host} with no API key; anyone who can reach this "
              "machine can use the model. Set --api-key or NATLAS_API_KEY.", file=sys.stderr)
    print(f"Serving {engine.model_path.name} on http://{args.host}:{args.port}/v1", file=sys.stderr)
    uvicorn.run(create_app(engine, asr, api_key=api_key), host=args.host, port=args.port,
                log_level="warning")


def _chat(args: argparse.Namespace) -> None:
    with NAtlas(base_url=args.base_url) as client:
        if args.prompt:
            for token in client.stream(" ".join(args.prompt)):
                print(token, end="", flush=True)
            print()
            return
        history: list[dict[str, str]] = []
        print("Chatting with N-ATLAS. Ctrl-D to quit.", file=sys.stderr)
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                print()
                return
            history.append({"role": "user", "content": prompt})
            reply = []
            for token in client.stream(history):
                print(token, end="", flush=True)
                reply.append(token)
            print()
            history.append({"role": "assistant", "content": "".join(reply)})


def _bench(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .bench import TASKS, run

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s) {unknown}. Choose from {', '.join(TASKS)}")
    languages = args.languages.split(",") if args.languages else None
    with NAtlas(base_url=args.base_url, timeout=600) as client:
        summary = run(client, tasks, languages, args.n or None, Path(args.out), args.seed,
                      args.label)
    from .bench.runner import to_markdown

    print(to_markdown(summary))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="natlas", description="Tools for building with N-ATLAS")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $NATLAS_BASE_URL)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download quantised N-ATLAS weights")
    p.add_argument("--quant", default=DEFAULT_QUANT, choices=QUANTS,
                   help="Q4_K_M or larger recommended; Q2_K and Q3_K_M lose noticeable quality")

    p = sub.add_parser("serve", help="Run an OpenAI-compatible server on a local model")
    p.add_argument("--model", help="Path to a GGUF file (default: downloaded --quant)")
    p.add_argument("--quant", default=DEFAULT_QUANT, choices=QUANTS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ctx", type=int, default=8192, help="Context window in tokens")
    p.add_argument("--asr", action="store_true", help="Enable /v1/audio/transcriptions")
    p.add_argument("--api-key", help="Require this bearer token (default: $NATLAS_API_KEY)")

    p = sub.add_parser("chat", help="Chat in the terminal, or answer one prompt")
    p.add_argument("prompt", nargs="*")

    p = sub.add_parser("translate", help="Translate between en, ha, ig and yo")
    p.add_argument("text")
    p.add_argument("--to", required=True)
    p.add_argument("--from", dest="source", default="en")

    p = sub.add_parser("bench", help="Score an endpoint on public Nigerian-language benchmarks")
    p.add_argument("--tasks", default="afrimmlu,afrixnli,afrimgsm,mt-en-xx,mt-xx-en")
    p.add_argument("--languages", help="Comma-separated subset of en,ha,ig,yo")
    p.add_argument("-n", type=int, default=100, help="Items per language; 0 for the full set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="n-atlas", help="Name for this run in the report")
    p.add_argument("--out", default="bench-results", help="Directory for results")

    p = sub.add_parser("transcribe", help="Speech to text with the N-ATLAS speech models")
    p.add_argument("audio")
    p.add_argument("--language", "-l", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            print(download(args.quant))
        elif args.command == "serve":
            _serve(args)
        elif args.command == "chat":
            _chat(args)
        elif args.command == "translate":
            with NAtlas(base_url=args.base_url) as client:
                print(client.translate(args.text, to=args.to, source=args.source))
        elif args.command == "bench":
            _bench(args)
        elif args.command == "transcribe":
            with NAtlas(base_url=args.base_url) as client:
                print(client.transcribe(args.audio, language=args.language))
    except (NAtlasError, DownloadError, ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
