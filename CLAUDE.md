# pynatlas

Python SDK for N-ATLAS. Layout: `src/pynatlas/` (client, types, tools, errors,
settings, defaults, server, _toolcalls, engine, asr, models, cli, bench/),
`playground/` (Streamlit, not packaged), `examples/`, `bench-results/`
(published runs).

## Commands

```bash
uv venv && CMAKE_ARGS="-DGGML_METAL=on" uv pip install -e ".[server,asr,bench]" --group dev
pytest                                  # unit tests, no model needed
ruff check src tests playground examples
mypy                                    # strict, must stay clean
pynatlas serve --asr --api-key test-key   # then:
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

Run the live tests before calling a change to the server, engine or speech path
done. Unit tests use fakes and cannot catch model or llama.cpp behaviour.

## Invariants

- The core install is httpx only, and `import pynatlas` must not load anything
  else. Heavy dependencies belong in an extra and are imported through
  `_extras.require` at the point of use, so a missing extra names its install
  command.
- Contributor and playground tools go in `[dependency-groups]`, never in
  `[project.optional-dependencies]`; extras are for features users run.
- The client sends standard OpenAI fields only. Provider-specific options such
  as `repetition_penalty` pass through `**extra` when the caller asks, because
  strict OpenAI-compatible APIs reject unknown fields.
- `NAtlas` and `AsyncNAtlas` share request building and parsing in `_Base`.
  Change behaviour there, not in one client.
- Configuration goes through `Settings.from_env()`, which validates once and
  names the variable in `ConfigError`. Do not read `NATLAS_*` with `os.getenv`
  elsewhere, and keep the API key out of reprs and messages.
- Pydantic is never imported by the client. A Pydantic model passed as
  `response_format` or a tool parameter is used through its own methods.
- Every tool decision is generated under a JSON schema listing only the real
  tools, plus `{"name": "none"}` in auto mode, with tools in the system message.
  Free prompting called a tool for every question that needed none. Change
  this only with the held-out `pynatlas bench --tasks tool-calling` numbers, and
  never tune on `bench/tool_cases.json`, or its scores stop meaning anything.
- The engine switches the repetition penalty off under a grammar: JSON repeats
  its punctuation by design and the penalty fights it.
- Library code never prints. Progress goes through callbacks the CLI supplies,
  and diagnostics through the `pynatlas` loggers.
- The server depends on the `Engine` and `Transcriber` protocols, not on
  llama.cpp or transformers.
- Endpoints consume engine output through `_Generation` and call `stop()` in a
  `finally`. Engines hold a lock while generating; iterating their generator
  directly lets a client that hangs up mid-stream lock the server for good.
- Render the chat template in `LlamaCppEngine`, not with llama.cpp's formatter:
  the template stamps today's date, which llama.cpp leaves at a placeholder.
  Because the date is in every prompt, benchmark runs are only comparable with
  the same `pynatlas serve --date`; published runs use "10 Oct 2026".
- Clear the llama.cpp cache before every generation. Prefix reuse makes a
  temperature-0 reply depend on the previous request, and benchmark scores on
  question order.
- Keep the model card defaults (system prompt, temperature 0.1, repetition
  penalty 1.12) in `defaults.py`. Without the system prompt the model drifts
  towards generic Llama-3 answers.
- Never commit MAFAND reference text. It is GPL-3.0 and this repo is
  Apache-2.0; translation results store item ids only.
- The speech loop guard runs only on a retry. Banning repeated 4-grams fixes
  the Igbo model's loops but costs Hausa and Yoruba accuracy on every request.
- Decode m4a and webm through a temporary file: m4a keeps its index at the end,
  which ffmpeg cannot reach from a pipe. Test audio decoding with clips of ten
  seconds or more, since short clips hide this.
- Model downloads are pinned and hash-checked. Bump `GGUF_REVISION` in
  `models.py` deliberately, never to a branch name.
- Report benchmark caveats with the numbers: sample size, 4-bit weights,
  possible training contamination, and untoned Yoruba FLEURS references.

## Gotchas

- The speech models are gated per model on Hugging Face; a valid login still
  gets 403 until each model's terms are accepted.
- llama.cpp needs GPU access. Where Metal is unavailable, it fails with "Failed
  to create llama_context" unless `n_gpu_layers=0`.
