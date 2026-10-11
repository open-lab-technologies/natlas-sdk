# pynatlas

Python SDK for [N-ATLAS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's open
language model for English, Hausa, Igbo and Yoruba, and its speech recognition
models. It works with any OpenAI-compatible endpoint serving N-ATLAS, and can
run one locally.

## Install

```bash
pip install "pynatlas @ git+https://github.com/Nazehs/natlas-sdk"
```

| Install | Adds | Size |
|---|---|---|
| `pynatlas` | Sync and async clients | 2.5 MB |
| `pynatlas[server]` | `pynatlas serve`: N-ATLAS on your machine behind the OpenAI API | 109 MB |
| `pynatlas[asr]` | Speech recognition for `pynatlas serve --asr` | 690 MB |
| `pynatlas[bench]` | `pynatlas bench`: benchmarks for any endpoint | 51 MB |

Python 3.10 or newer. `import pynatlas` loads only httpx.

## Quick start

```python
from pynatlas import NAtlas

client = NAtlas(base_url="http://localhost:8000/v1")

client.chat("Kí ni olú ìlú Nàìjíríà?").text        # 'Olú ìlú Nàìjíríà ni Abuja.'
client.translate("Wash your hands before eating.", to="ig")
client.transcribe("voice-note.ogg", language="ha")

for piece in client.stream("Ka bayyana mini menene noma a takaice"):
    print(piece, end="")
```

To run N-ATLAS locally:

```bash
pip install "pynatlas[server] @ git+https://github.com/Nazehs/natlas-sdk"
pynatlas download      # 4.9 GB, 4-bit weights
pynatlas serve         # http://127.0.0.1:8000/v1
```

On Apple Silicon install with `CMAKE_ARGS="-DGGML_METAL=on"`, on NVIDIA with
`CMAKE_ARGS="-DGGML_CUDA=on"`, so the model runs on the GPU.

## Client

### Providers

| Provider | `base_url` | `model` |
|---|---|---|
| `pynatlas serve` | `http://localhost:8000/v1` | `n-atlas` (default) |
| vLLM with the official weights | `http://host:8000/v1` | `NCAIR1/N-ATLaS` |
| llama.cpp server | `http://host:8080/v1` | its `--model_alias` |
| Ollama | `http://localhost:11434/v1` | your model tag |

Requests carry standard OpenAI fields only. Provider-specific options pass
through as keyword arguments, for example `client.chat(..., repetition_penalty=1.12)`
on vLLM. Tested end to end against `pynatlas serve` and llama.cpp's server.

### Tool calling

Pass plain functions; type hints and the docstring become the tool definition.

```python
def get_market_price(commodity: str, market: str) -> dict:
    """Today's price of a food commodity at a named market."""
    return {"naira_per_kg": 1450}

reply = client.run_tools("Nawa ne farashin shinkafa a kasuwar Kano?", [get_market_price])
reply.text  # answer written from the tool's result
```

`run_tools` loops until the model answers in text, and returns a tool's
exception to the model rather than raising it. For control over each step, use
`chat(..., tools=[...])` and read `reply.tool_calls`; `call.result(value)`
builds the message that returns a result. `tool_choice` takes `"auto"`,
`"none"`, `"required"` or a function name.

### Structured output

```python
from typing import Literal
from pydantic import BaseModel

class Feedback(BaseModel):
    language: Literal["en", "ha", "ig", "yo"]
    sentiment: Literal["positive", "neutral", "negative"]

client.chat("Classify: Kayan sun iso a makare.", response_format=Feedback).parsed
```

`response_format` also takes a JSON schema dict. `pynatlas serve`, vLLM and
llama.cpp enforce the schema while generating, so the reply is always valid;
elsewhere a mismatch raises `StructuredOutputError`. Pydantic is only needed if
you pass a Pydantic model.

### Async

`AsyncNAtlas` has the same methods as coroutines, and `run_tools` accepts
async functions.

```python
async with AsyncNAtlas() as client:
    reply = await client.chat("Sannu!")
```

### Errors and retries

Connection failures and 408, 409, 429 and 5xx responses are retried with
backoff, honouring `Retry-After`. Everything raised derives from `NAtlasError`:

| Exception | When |
|---|---|
| `APIConnectionError`, `APITimeoutError` | Endpoint unreachable or too slow |
| `BadRequestError` | 400 or 422, including a prompt longer than the context window |
| `AuthenticationError` | 401 or 403 |
| `NotFoundError` | 404, or the endpoint lacks the feature |
| `RateLimitError`, `InternalServerError` | 429 or 5xx after retries |
| `StructuredOutputError` | Reply does not match `response_format` |
| `ToolLoopError` | `run_tools` hit `max_rounds` |
| `ConfigError` | A setting is malformed |

### Configuration

| Variable | Default |
|---|---|
| `NATLAS_BASE_URL` | `http://localhost:8000/v1` |
| `NATLAS_API_KEY` | none |
| `NATLAS_MODEL` | `n-atlas` |
| `NATLAS_TIMEOUT` | `120` seconds |
| `NATLAS_MAX_RETRIES` | `2` |
| `NATLAS_MODEL_PATH` | the downloaded weights, for `pynatlas serve` |
| `NATLAS_CACHE` | `~/.cache/pynatlas` |

Settings are validated when a client is created; a malformed value raises
`ConfigError` naming the variable. Arguments passed to the client win over the
environment. `pynatlas config` prints what is in effect, with the key masked.

## Server

`pynatlas serve` puts a GGUF build behind the OpenAI API, so the official
`openai` package works unchanged, including tool calls, streaming and
`chat.completions.parse()`.

```bash
pynatlas serve --asr                                               # speech to text too
pynatlas serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"  # to share it
```

It handles one request at a time; for concurrent users serve the official
weights with vLLM. Other backends plug in through the `Engine` protocol in
`pynatlas.engine`, and speech backends through `pynatlas.asr.Transcriber`.

The speech models (`NCAIR1/Hausa-ASR`, `Igbo-ASR`, `Yoruba-ASR`,
`NigerianAccentedEnglish`) are gated: accept the terms on each model page on
Hugging Face, then run `hf auth login`. WhatsApp voice notes, wav, mp3 and flac
decode directly; m4a and webm need ffmpeg.

## Command line

| Command | Does |
|---|---|
| `pynatlas download [--quant Q4_K_M]` | Fetch weights, resumable and hash-checked |
| `pynatlas serve [--asr] [--api-key KEY]` | Run the local server |
| `pynatlas chat ["prompt"]` | Chat, or answer one prompt |
| `pynatlas translate "text" --to yo` | Translate between en, ha, ig and yo |
| `pynatlas transcribe file --language ha` | Speech to text |
| `pynatlas bench [--tasks ...] [-n 100]` | Run benchmarks |
| `pynatlas config` | Show settings in effect |

## Benchmarks

`pynatlas bench` scores any endpoint and saves every reply next to its expected
answer. Results for the 4-bit model on Apple Silicon, temperature 0:

| | English | Hausa | Igbo | Yoruba |
|---|---:|---:|---:|---:|
| AfriMMLU knowledge, % (chance 25) | 53 | 35 | 41 | 37 |
| AfriXNLI reasoning, % (chance 33) | 69 | 58 | 57 | 53 |
| AfriMGSM maths, % | 69 | 43 | 36 | 40 |
| MAFAND translation from English, chrF++ | | 33.7 | 41.0 | 28.0 |
| MAFAND translation into English, chrF++ | | 37.4 | 40.7 | 41.7 |
| FLEURS speech, WER % (lower is better) | 9.0 | 31.3 | 43.8 | 55.7 |
| FLEURS speech ignoring tone marks, WER % | 9.0 | 31.3 | 37.3 | 28.8 |
| Tool calling, held-out set, % correct | 100 | 90 | 80 | 80 |

- Text tasks use 100 items per language, so accuracy is roughly ±10 points.
  Speech uses 50 clips, tool calling 10 questions per language.
- Many FLEURS Yoruba references lack tone marks that the model writes, which
  inflates the strict Yoruba WER.
- Tool calling was scored on 40 questions and 4 tools not used while designing
  it. Plain prompting scores 21 of 40 on the same set and calls a tool for every
  question that needs none; this SDK scores 34 of 40 with one such false call.
  The Hausa, Igbo and Yoruba questions are not reviewed by native speakers.
- N-ATLAS was trained partly on open datasets, so some test items may have been
  seen in training.
- The model's chat template writes today's date into every prompt, so replies
  drift from one day to the next. These runs used `pynatlas serve --date "10 Oct 2026"`;
  with the same date, a rerun reproduces them exactly.

Datasets: [IrokoBench](https://huggingface.co/collections/masakhane/irokobench)
(Apache-2.0), [MAFAND-MT](https://github.com/masakhane-io/lafand-mt) (GPL-3.0)
and [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0), all
downloaded at run time. Full replies are in [bench-results](bench-results/).

## Known limitations

- Yoruba output is the weakest: free-form answers are often incoherent and drop
  tone marks, and Yoruba words inside tool arguments can come out garbled.
- Structured output guarantees the JSON's shape, not its content. Field values
  can be wrong, such as a Hausa message labelled English.
- The model makes at most one tool call per turn.
- Q2_K and Q3_K_M weights lose real quality; use Q4_K_M or larger.
- The GGUF weights are a community conversion (QuantFactory), pinned and
  hash-checked, not published by NCAIR.

## Development

```bash
uv venv && uv pip install -e ".[server,asr,bench]" --group dev
pytest                                   # unit tests, no model needed
ruff check src tests playground examples
```

End-to-end tests run against a live server:

```bash
pynatlas serve --asr --api-key test-key
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

`examples/` has runnable scripts, and `streamlit run playground/app.py` (after
`uv pip install --group playground`) opens a browser playground.

## Licence

Apache-2.0. The N-ATLAS weights and speech models are released by NCAIR under
their own terms.
