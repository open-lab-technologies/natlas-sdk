"""`natlas` command line: download, serve, chat, translate, transcribe."""

from __future__ import annotations

import argparse
import sys

from .client import NAtlas, NAtlasError
from .models import DEFAULT_QUANT, QUANTS, default_model_path, download


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
    print(f"Serving {engine.model_path.name} on http://{args.host}:{args.port}/v1", file=sys.stderr)
    uvicorn.run(create_app(engine, asr), host=args.host, port=args.port, log_level="warning")


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="natlas", description="Tools for building with N-ATLAS")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $NATLAS_BASE_URL)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download quantised N-ATLAS weights")
    p.add_argument("--quant", default=DEFAULT_QUANT, choices=QUANTS)

    p = sub.add_parser("serve", help="Run an OpenAI-compatible server on a local model")
    p.add_argument("--model", help="Path to a GGUF file (default: downloaded --quant)")
    p.add_argument("--quant", default=DEFAULT_QUANT, choices=QUANTS)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ctx", type=int, default=8192, help="Context window in tokens")
    p.add_argument("--asr", action="store_true", help="Enable /v1/audio/transcriptions")

    p = sub.add_parser("chat", help="Chat in the terminal, or answer one prompt")
    p.add_argument("prompt", nargs="*")

    p = sub.add_parser("translate", help="Translate between en, ha, ig and yo")
    p.add_argument("text")
    p.add_argument("--to", required=True)
    p.add_argument("--from", dest="source", default="en")

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
        elif args.command == "transcribe":
            with NAtlas(base_url=args.base_url) as client:
                print(client.transcribe(args.audio, language=args.language))
    except (NAtlasError, ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
