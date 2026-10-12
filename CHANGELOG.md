# Changelog

All notable changes to pynatlas are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may
include breaking changes, each called out under its release.

## [0.1.0] - 2026-10-12

First release.

### Client

- `NAtlas` and `AsyncNAtlas` for any OpenAI-compatible endpoint serving
  N-ATLaS, sending standard OpenAI fields only. Provider-specific options pass
  through as keyword arguments.
- Chat, streaming, translation and speech to text.
- Tool calling from plain Python functions, with `run_tools()` to run the
  call-and-answer loop.
- Structured output from a Pydantic model or JSON schema, with recovery of
  JSON from code fences and one corrective retry where the endpoint does not
  enforce schemas.
- `summarize()`, `classify()` (enforced label set) and `voice_chat()`.
- `detect_language()`, offline, above 99% accurate on held-out text.
- Retries with backoff for connection errors, 408, 409, 429 and 5xx, and typed
  exceptions under `NAtlasError`.
- Settings validated once from `NATLAS_*` environment variables.

### Server

- `pynatlas serve`: N-ATLaS GGUF weights behind the OpenAI API through
  llama.cpp, with tool calling, schema-enforced JSON output and speech
  recognition through the official NCAIR1 models.
- API keys, OpenAI error bodies, and recovery when a client disconnects
  mid-stream.
- Replies independent of earlier requests, and `--date` to make benchmark runs
  reproducible.
- `Engine` and `Transcriber` protocols for other backends.
- A container image and documented GPU routes.

### Tooling

- `pynatlas download`: pinned, SHA-256 verified, resumable weight downloads.
- `pynatlas bench`: AfriMMLU, AfriXNLI, AfriMGSM, MAFAND, FLEURS and a held-out
  tool-calling set, with every reply saved for audit.
- A Streamlit playground and runnable examples.

[0.1.0]: https://github.com/Nazehs/natlas-sdk/releases/tag/v0.1.0
