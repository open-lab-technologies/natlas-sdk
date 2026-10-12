# pynatlas

Python SDK for [N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's open
language model for English, Hausa, Igbo and Yoruba, and its speech recognition
models. It works with any OpenAI-compatible endpoint serving N-ATLaS, can run
one locally, and publishes measured results for every language it supports.

## Install

```bash
pip install pynatlas
```

| Install            | Adds                                                            | Size   |
| ------------------ | --------------------------------------------------------------- | ------ |
| `pynatlas`         | Sync and async clients                                          | 2.5 MB |
| `pynatlas[server]` | `pynatlas serve`: N-ATLaS on your machine behind the OpenAI API | 109 MB |
| `pynatlas[asr]`    | Speech recognition for `pynatlas serve --asr`                   | 690 MB |
| `pynatlas[bench]`  | `pynatlas bench`: benchmarks for any endpoint                   | 51 MB  |

Python 3.10 or newer. `import pynatlas` loads only httpx. Release notes are in
the [changelog](https://github.com/Nazehs/natlas-sdk/blob/main/CHANGELOG.md).

## Quick start

```python
from pynatlas import NAtlas

client = NAtlas(base_url="http://localhost:8000/v1")

client.chat("Kí ni olú ìlú Nàìjíríà?").text        # 'Olú ìlú Nàìjíríà ni Abuja.'
client.translate("Wash your hands before eating.", to="ig")
client.transcribe("voice-note.ogg", language="ha")  # needs a server with speech

for piece in client.stream("Ka bayyana mini menene noma a takaice"):
    print(piece, end="")
```

To run N-ATLaS locally:

```bash
pip install "pynatlas[server]"
pynatlas download      # 4.9 GB, 4-bit weights
pynatlas serve         # http://127.0.0.1:8000/v1
```

To run the model on the GPU, install the server extra with
`CMAKE_ARGS="-DGGML_METAL=on"` on Apple Silicon or `CMAKE_ARGS="-DGGML_CUDA=on"`
on NVIDIA.

## Client

### Providers

| Provider                       | `base_url`                  | `model`             |
| ------------------------------ | --------------------------- | ------------------- |
| `pynatlas serve`               | `http://localhost:8000/v1`  | `n-atlas` (default) |
| vLLM with the official weights | `http://host:8000/v1`       | `NCAIR1/N-ATLaS`    |
| llama.cpp server               | `http://host:8080/v1`       | its `--model_alias` |
| Ollama                         | `http://localhost:11434/v1` | your model tag      |

Requests carry standard OpenAI fields only. Provider-specific options pass
through as keyword arguments, such as `repetition_penalty=1.12` on vLLM. Tested
end to end against `pynatlas serve` and llama.cpp's server.

### Helpers

```python
from pynatlas import detect_language

detect_language("Bawo ni o se wa?")                       # 'yo', offline, no model call
client.summarize(article)                                 # in the article's own language
client.classify(message, ["complaint", "praise", "question"])
client.voice_chat("voice-note.ogg", language="ha")        # transcript and reply
```

`detect_language` needs no model and is over 99% accurate on held-out text.
`classify` enforces the label set, so the answer is always one of your labels.

### Tool calling

Pass plain functions; type hints and the docstring become the tool definition.

```python
def get_market_price(commodity: str, market: str) -> dict:
    """Today's price of a food commodity at a named market."""
    return {"naira_per_kg": 1450}

reply = client.run_tools("Nawa ne farashin shinkafa a kasuwar Kano?", [get_market_price])
reply.text  # answer written from the tool's result
```

`run_tools` loops until the model answers in text; a tool's exception goes back
to the model instead of being raised. For step-by-step control, pass
`tools=[...]` to `chat()` and read `reply.tool_calls`. `tool_choice` takes
`"auto"`, `"none"`, `"required"` or a function name.

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
llama.cpp enforce the schema while generating, so the reply is always valid.
Elsewhere the client reads JSON out of code fences, and if the reply still does
not match, it shows the model the error and asks again (`format_retries`)
before raising `StructuredOutputError`. Pydantic is needed only if you pass a
Pydantic model.

### Async

`AsyncNAtlas` has the same methods as coroutines, and `run_tools` accepts
async functions.

```python
async with AsyncNAtlas() as client:
    reply = await client.chat("Sannu!")
```

### Errors and retries

Connection failures and 408, 409, 429 and 5xx responses are retried with
backoff, honouring `Retry-After`. A retried chat request can run the model
twice; pass `max_retries=0` where that matters. Everything raised derives from
`NAtlasError`:

| Exception                               | When                                                          |
| --------------------------------------- | ------------------------------------------------------------- |
| `APIConnectionError`, `APITimeoutError` | Endpoint unreachable or too slow                              |
| `BadRequestError`                       | 400 or 422, including a prompt longer than the context window |
| `AuthenticationError`                   | 401 or 403                                                    |
| `NotFoundError`                         | 404, or the endpoint lacks the feature                        |
| `RateLimitError`, `InternalServerError` | 429 or 5xx after retries                                      |
| `StructuredOutputError`                 | Reply does not match `response_format`                        |
| `ToolLoopError`                         | `run_tools` hit `max_rounds`                                  |
| `ConfigError`                           | A setting is malformed                                        |

### Configuration

| Variable             | Default                                      |
| -------------------- | -------------------------------------------- |
| `NATLAS_BASE_URL`    | `http://localhost:8000/v1`                   |
| `NATLAS_API_KEY`     | none                                         |
| `NATLAS_MODEL`       | `n-atlas`                                    |
| `NATLAS_TIMEOUT`     | `120` seconds                                |
| `NATLAS_MAX_RETRIES` | `2`                                          |
| `NATLAS_MODEL_PATH`  | the downloaded weights, for `pynatlas serve` |
| `NATLAS_CACHE`       | `~/.cache/pynatlas`                          |

Client arguments override these. A malformed value raises `ConfigError` naming
the variable, and `pynatlas config` shows what is in effect.

## Server

`pynatlas serve` puts a GGUF build behind the OpenAI API, so the official
`openai` package works unchanged, including tool calls, streaming and
`chat.completions.parse()`.

```bash
pynatlas serve --asr                                               # speech to text too
pynatlas serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"  # to share it
```

It handles one request at a time; for concurrent users, serve the official
weights with vLLM (see Deployment). Other backends plug in through the `Engine`
protocol in `pynatlas.engine`, and speech backends through
`pynatlas.asr.Transcriber`.

The NCAIR1 speech models are gated: accept the terms on each model page on
Hugging Face, then run `hf auth login`. m4a and webm audio need ffmpeg.

## Deployment

### Docker

```bash
docker build -t pynatlas .
docker run -p 8000:8000 -v pynatlas-data:/data -e NATLAS_API_KEY=change-me pynatlas
```

Weights download to the `/data` volume on first start. For speech, build with
`--build-arg EXTRAS=server,asr`, pass `HF_TOKEN` and add `--asr` to the run.

### On a GPU

Point `base_url` at the GPU host; the client does not change.

- vLLM, for many users: `vllm serve NCAIR1/N-ATLaS` with
  `model="NCAIR1/N-ATLaS"`. Tool-call safeguards are `pynatlas serve` features,
  so run `pynatlas bench --tasks tool-calling` against vLLM before relying on
  tool calls.
- `pynatlas serve`, for demos: install with `CMAKE_ARGS="-DGGML_CUDA=on"`.

## Command line

| Command                                                | Does                                      |
| ------------------------------------------------------ | ----------------------------------------- |
| `pynatlas download [--quant Q4_K_M]`                   | Fetch weights, resumable and hash-checked |
| `pynatlas serve [--asr] [--api-key KEY] [--date DATE]` | Run the local server                      |
| `pynatlas chat ["prompt"]`                             | Chat, or answer one prompt                |
| `pynatlas translate "text" --to yo`                    | Translate between en, ha, ig and yo       |
| `pynatlas transcribe file --language ha`               | Speech to text                            |
| `pynatlas bench [--tasks ...] [-n 100]`                | Run benchmarks                            |
| `pynatlas config`                                      | Show settings in effect                   |

`--base-url` and `--model` point every command except `serve` and `download` at
any endpoint.

## Benchmarks

`pynatlas bench` scores any endpoint and saves every reply next to its expected
answer. Results for the 4-bit model on Apple Silicon, temperature 0:

|                                          | English | Hausa | Igbo | Yoruba |
| ---------------------------------------- | ------: | ----: | ---: | -----: |
| AfriMMLU knowledge, % (chance 25)        |      53 |    35 |   41 |     37 |
| AfriXNLI reasoning, % (chance 33)        |      69 |    58 |   57 |     53 |
| AfriMGSM maths, %                        |      69 |    43 |   36 |     40 |
| MAFAND translation from English, chrF++  |         |  33.7 | 41.0 |   28.0 |
| MAFAND translation into English, chrF++  |         |  37.4 | 40.7 |   41.7 |
| FLEURS speech, WER % (lower is better)   |     9.0 |  31.3 | 43.8 |   55.7 |
| FLEURS speech ignoring tone marks, WER % |     9.0 |  31.3 | 37.3 |   28.8 |
| Tool calling, held-out set, % correct    |     100 |    90 |   80 |     80 |
| Language detection, held-out text, %     |    99.8 |  99.9 | 99.6 |   99.6 |

- 100 items per language for text tasks (about ±10 points), 50 clips for
  speech, 10 questions for tool calling.
- Many FLEURS Yoruba references omit tone marks the model writes, which inflates
  the strict Yoruba WER.
- Tool calling uses questions and tools kept out of its design. Plain prompting
  scores 21 of 40 and calls a tool for every question that needs none; this SDK
  scores 34 of 40. Non-English questions are not reviewed by native speakers.
- Language detection is trained on AfriXNLI dev and scored on its test split.
- Some test items may have been in N-ATLaS's training data.
- Runs used `pynatlas serve --date "10 Oct 2026"`; the template dates every
  prompt, so only the same date reproduces them exactly.

Datasets: [IrokoBench](https://huggingface.co/collections/masakhane/irokobench)
(Apache-2.0), [MAFAND-MT](https://github.com/masakhane-io/lafand-mt) (GPL-3.0)
and [FLEURS](https://huggingface.co/datasets/google/fleurs) (CC-BY-4.0), all
downloaded at run time. Full replies are in
[bench-results](https://github.com/Nazehs/natlas-sdk/tree/main/bench-results/).

## Known limitations

- Yoruba output is the weakest: free-form answers are often incoherent and drop
  tone marks, and Yoruba words inside tool arguments can come out garbled.
- Structured output guarantees the JSON's shape, not its content. Field values
  can be wrong, such as a Hausa message labelled English.
- The model makes at most one tool call per turn.
- `detect_language` knows only English, Hausa, Igbo and Yoruba; text in any
  other language is still labelled as one of them.
- Q2_K and Q3_K_M weights lose real quality; use Q4_K_M or larger.
- The GGUF weights are a community conversion (QuantFactory), pinned and
  hash-checked, not published by NCAIR.

## Development

```bash
uv venv && uv pip install -e ".[server,asr,bench]" --group dev
pytest                                   # unit tests, no model needed
ruff check src tests playground examples
mypy                                     # strict type check
```

End-to-end tests run against a live server:

```bash
pynatlas serve --asr --api-key test-key
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

`examples/` has runnable scripts, and `streamlit run playground/app.py` (after
`uv pip install --group playground`) opens a browser playground.

## Licence

The code is Apache-2.0, maintained by Open Lab Technologies. The N-ATLaS and
speech models it runs are under the N-ATLaS Open-Source Research and Innovation
License, which requires attribution in all public use and caps deployments at
1,000 active end-users a month without a commercial licence. Read the
[NOTICE](https://github.com/Nazehs/natlas-sdk/blob/main/NOTICE) before
deploying.

N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation
and Digital Economy, and powered by Awarri Technologies.
