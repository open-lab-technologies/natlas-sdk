# natlas

Python SDK for [N-ATLAS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's open
multilingual language model (Llama-3 8B fine-tuned for English, Hausa, Igbo and
Yoruba), with its official speech recognition models.

The client works with any OpenAI-compatible endpoint serving N-ATLAS: the
bundled `natlas serve`, vLLM, llama.cpp's server, Ollama, or a hosted API. It
sends standard OpenAI requests, so switching provider means changing a URL.

## Install

```bash
pip install "natlas @ git+https://github.com/Nazehs/natlas-sdk"
```

The base install is the HTTP client and nothing else. Add extras only for the
features you run yourself:

| Install | Adds | Installed size |
|---|---|---|
| `natlas` | Sync and async clients | 2.5 MB (httpx) |
| `natlas[server]` | `natlas serve`: run N-ATLAS locally behind the OpenAI API | 109 MB |
| `natlas[asr]` | Local speech recognition for `natlas serve --asr` | 690 MB, mostly PyTorch |
| `natlas[bench]` | `natlas bench`: score any endpoint on public benchmarks | 51 MB |

Python 3.10 or newer. `import natlas` loads only httpx, and a missing extra
fails with the `pip install` command that fixes it.

## Quick start

```python
from natlas import NAtlas

client = NAtlas(base_url="http://localhost:8000/v1")  # or set NATLAS_BASE_URL

print(client.chat("Kí ni olú ìlú Nàìjíríà?"))
# Olú ìlú Nàìjíríà ni Abuja.

for piece in client.stream("Ka bayyana mini menene noma a takaice"):
    print(piece, end="", flush=True)

client.translate("Wash your hands before eating.", to="ig")
client.transcribe("voice-note.ogg", language="ha")
```

No endpoint yet? Run one on your own machine:

```bash
pip install "natlas[server] @ git+https://github.com/Nazehs/natlas-sdk"
natlas download          # 4.9 GB, 4-bit weights, hash-checked
natlas serve             # http://127.0.0.1:8000/v1
```

On Apple Silicon, install with `CMAKE_ARGS="-DGGML_METAL=on"` so the model runs
on the GPU; with an NVIDIA GPU use `CMAKE_ARGS="-DGGML_CUDA=on"`.

## Using the client

### Async

`AsyncNAtlas` has the same methods as coroutines:

```python
import asyncio
from natlas import AsyncNAtlas

async def main():
    async with AsyncNAtlas() as client:
        replies = await asyncio.gather(
            client.chat("Capital of Nigeria?"),
            client.chat("Babban birnin Najeriya?"),
        )
        async for piece in client.stream("Sannu!"):
            print(piece, end="")

asyncio.run(main())
```

### Choosing a provider

| Provider | `base_url` | `model` |
|---|---|---|
| `natlas serve` | `http://localhost:8000/v1` | `n-atlas` (default) |
| vLLM serving the official weights | `http://host:8000/v1` | `NCAIR1/N-ATLaS` |
| llama.cpp server | `http://host:8080/v1` | the `--model_alias` you set |
| Ollama with an imported GGUF | `http://localhost:11434/v1` | the tag you created |

Set them per client (`NAtlas(base_url=..., model=...)`) or with
`NATLAS_BASE_URL` and `NATLAS_MODEL`. The client is tested end to end against
`natlas serve` and llama.cpp's server; the others follow the same OpenAI API.

The client applies the N-ATLAS model card's system prompt and temperature 0.1.
Pass `system_prompt=None` to the client, or your own `system=` per call, to
change that. Keyword arguments `chat()` does not know go into the request body
unchanged, for provider-specific options:

```python
client.chat("Sannu", repetition_penalty=1.12)  # vLLM; llama.cpp calls it repeat_penalty
```

`natlas serve` applies the model card's repetition penalty (1.12) itself, so
there is nothing to pass there.

### Errors and retries

Connection failures and 408, 409, 429 and 5xx responses are retried with
backoff (`max_retries=2` by default), honouring `Retry-After`. What still fails
raises a subclass of `NAtlasError`:

| Exception | When |
|---|---|
| `APIConnectionError`, `APITimeoutError` | The endpoint could not be reached, or took longer than `timeout` |
| `BadRequestError` | 400 or 422, including a prompt longer than the context window |
| `AuthenticationError` | 401 or 403 |
| `NotFoundError` | 404: wrong URL, or the endpoint lacks the feature (speech, for instance) |
| `RateLimitError`, `InternalServerError` | 429 or 5xx after the retries |

Every status error carries `status_code` and the parsed error `body`. A stream
that fails part-way raises rather than retrying, so no text is duplicated.

### Speech to text

`transcribe()` takes a file path or the file's bytes and needs an endpoint with
`/v1/audio/transcriptions`, such as `natlas serve --asr`:

```python
heard = client.transcribe("voice-note.ogg", language="ha")
answer = client.chat(heard)
```

wav, flac, mp3, ogg and opus (WhatsApp voice notes) decode directly. m4a
(iPhone recordings) and webm (browser recorders) need ffmpeg on the server.

The speech models are the official N-ATLAS ones (`NCAIR1/Yoruba-ASR`,
`Igbo-ASR`, `Hausa-ASR`, `NigerianAccentedEnglish`). They are gated: logging in
is not enough, you also have to click "Agree and access repository" on each
model page on Hugging Face, then run `hf auth login` where the server runs.

## Running N-ATLAS locally

`natlas serve` puts a GGUF build behind the OpenAI API, so existing OpenAI code
works unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
client.chat.completions.create(model="n-atlas",
                               messages=[{"role": "user", "content": "Sannu!"}])
```

Streaming, list-style message content and the `developer` role work, as
current OpenAI clients send them. The OpenAI client's
`audio.transcriptions.create` works too: pass `model="NCAIR1/Hausa-ASR"` (or
another speech model) and the language follows from it.

It listens on localhost only. To share it, bind to a reachable address and
require a key, which clients send as a bearer token:

```bash
natlas serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"
```

`/health` stays open for monitoring. The server handles one request at a time;
for concurrent users, serve the official weights with vLLM instead.

To put the server in front of a different inference backend, implement the
`Engine` protocol in `natlas.engine` (three methods) and pass it to
`natlas.server.create_app`. Speech backends implement `natlas.asr.Transcriber`.

## Examples

| Script | Shows |
|---|---|
| [examples/openai_drop_in.py](examples/openai_drop_in.py) | Unchanged OpenAI code answering in all four languages |
| [examples/voice_assistant.py](examples/voice_assistant.py) | A voice note transcribed and answered in the same language |

## Playground

A browser playground for trying prompts, translation and voice input, and for
reading benchmark results reply by reply. It is a development tool, not part of
the package, so run it from a clone:

```bash
git clone https://github.com/Nazehs/natlas-sdk && cd natlas-sdk
uv venv && uv pip install -e . --group playground
streamlit run playground/app.py
```

## Command line

| Command | What it does |
|---|---|
| `natlas download [--quant Q4_K_M]` | Download quantised weights (Q2_K to Q8_0) |
| `natlas serve [--asr] [--api-key KEY] [--port 8000]` | OpenAI-compatible server on a local model |
| `natlas chat ["prompt"]` | Interactive chat, or answer one prompt |
| `natlas translate "text" --to yo [--from en]` | Translate between en, ha, ig, yo |
| `natlas transcribe file --language ha` | Speech to text |
| `natlas bench [-n 100] [--tasks ...]` | Run the benchmark suite |

`--base-url` and `--model` point every command except `serve` and `download` at
any endpoint.

## Configuration

| Variable | Default | Used for |
|---|---|---|
| `NATLAS_BASE_URL` | `http://localhost:8000/v1` | Endpoint the client talks to |
| `NATLAS_MODEL` | `n-atlas` | Model name sent with each request |
| `NATLAS_API_KEY` | none | Bearer token the client sends, and the key `natlas serve` requires |
| `NATLAS_MODEL_PATH` | downloaded `--quant` | GGUF file for `natlas serve` |
| `NATLAS_CACHE` | `~/.cache/natlas` | Downloaded weights and benchmark data |

The client logs retries to the `natlas` logger and benchmark item failures to
`natlas.bench`.

## Benchmarks

`natlas bench` scores an endpoint on:

| Task | Source | Measures | Languages |
|---|---|---|---|
| `afrimmlu` | IrokoBench AfriMMLU | Multiple-choice knowledge | en, ha, ig, yo |
| `afrixnli` | IrokoBench AfriXNLI | Reasoning about sentence meaning | en, ha, ig, yo |
| `afrimgsm` | IrokoBench AfriMGSM | Grade-school maths | en, ha, ig, yo |
| `mt-en-xx`, `mt-xx-en` | MAFAND-MT news | Translation (chrF++) | ha, ig, yo |
| `asr-fleurs` | FLEURS read speech | Speech to text (word error rate) | en, ha, ig, yo |

Items are a seeded random sample, so runs with the same `-n` and `--seed`
are comparable. Results go to `--out`: a `summary.md` table, `summary.json`,
and one JSONL file per task and language with every prompt's reply, parsed
answer and gold answer.

Datasets: [IrokoBench](https://huggingface.co/collections/masakhane/irokobench)
(Masakhane, Apache-2.0), [MAFAND-MT](https://github.com/masakhane-io/lafand-mt)
(Masakhane, GPL-3.0) and [FLEURS](https://huggingface.co/datasets/google/fleurs)
(Google, CC-BY-4.0). They are downloaded at run time rather than shipped with
the package, because MAFAND is GPL-3.0. N-ATLAS was trained partly on open
Hugging Face datasets, so some test items may have been seen in training; treat scores as
an upper bound until that is ruled out.

### Results: N-ATLAS 4-bit (Q4_K_M), llama.cpp on Apple Silicon

100 items per language per task, seed 0, temperature 0. Full replies in
[bench-results/n-atlas-q4_k_m](bench-results/n-atlas-q4_k_m/). Runs are
reproducible: repeating a run, or asking the same items in a different order,
gives identical replies.

| Task | Metric | English | Hausa | Igbo | Yoruba | Chance |
|---|---|---:|---:|---:|---:|---:|
| AfriMMLU | accuracy % | 53 | 35 | 41 | 37 | 25 |
| AfriXNLI | accuracy % | 69 | 58 | 57 | 53 | 33 |
| AfriMGSM | accuracy % | 69 | 43 | 36 | 40 | |
| MAFAND English to language | chrF++ | | 33.7 | 41.0 | 28.0 | |
| MAFAND language to English | chrF++ | | 37.4 | 40.7 | 41.7 | |

How to read this:

- Every Nigerian-language score is above chance, but well below English. The
  gap is largest on knowledge (AfriMMLU) and maths (AfriMGSM).
- Translating into Yoruba is the weakest direction (28.0 chrF++), which matches
  the low Yoruba fluency in the model card's human evaluation.
- With 100 items, an accuracy score is roughly plus or minus 10 points at 95%
  confidence. Differences smaller than that between languages are not meaningful.
- These are 4-bit numbers. The full-precision model may score higher; run
  `natlas bench` against a vLLM endpoint serving the official weights to compare.

### Results: N-ATLAS speech models on FLEURS

First 50 clips of the FLEURS dev split per language, through `natlas serve --asr`.
Word error rate, so lower is better.

| | English | Hausa | Igbo | Yoruba |
|---|---:|---:|---:|---:|
| WER % | 9.0 | 31.3 | 43.8 | 55.7 |
| WER % ignoring tone marks | 9.0 | 31.3 | 37.3 | 28.8 |

- Many FLEURS Yoruba references are written without tone marks ("Awon iroyin")
  while the model writes them ("àwọn ìròyìn"), so the strict Yoruba score
  overstates the error. The second row strips diacritics from both sides.
- FLEURS has no Nigerian-accented English; English is scored on US speakers.
- The Igbo model sometimes loops on one word ("onye onye onye ..."). The server
  detects that and re-decodes with repeated phrases banned; without that retry,
  Igbo WER on this set is 65.7%.

Reproduce with:

```bash
natlas bench -n 100 --seed 0 --out bench-results/my-run
```

## Known limitations

What testing found that is the model's behaviour or outside this package:

- Yoruba output is the weakest. Translation into Yoruba scores lowest, and
  free-form Yoruba answers are often incoherent and drop tone marks, in line
  with the model card's own human evaluation.
- Nigerian-language accuracy trails English on every benchmark, most on
  knowledge and maths.
- Speech error rates are high outside English: about 31% WER for Hausa, 37% for
  Igbo and 29% for Yoruba, ignoring tone marks.
- Possible test contamination: N-ATLAS was trained partly on open datasets, so
  some benchmark items may have been seen in training.
- Small quantisations break. Q2_K failed to translate in testing; use Q4_K_M or
  larger.
- The GGUF weights are a community conversion, not published by NCAIR. They are
  pinned to a revision and hash-checked, but provenance rests on QuantFactory.

## Model notes

`natlas download` fetches a community GGUF conversion
([QuantFactory/N-ATLaS-GGUF](https://huggingface.co/QuantFactory/N-ATLaS-GGUF))
of the official weights, pinned to a fixed revision and checked against the
SHA-256 Hugging Face publishes. Interrupted downloads resume. The official
repository ships only full-precision safetensors, which need about 16 GB of
GPU memory; serve those with vLLM and point the client at it.

## Development

```bash
uv venv && CMAKE_ARGS="-DGGML_METAL=on" uv pip install -e ".[server,asr,bench]" --group dev
pytest
ruff check src tests playground examples
```

`pytest` runs the unit tests, which use fakes and need no model; they are what
CI runs. `tests/test_live.py` exercises a running server end to end in all four
languages, through this SDK and the official OpenAI client, including speech in
phone and browser formats:

```bash
natlas serve --asr --api-key test-key
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

## Licence

Code: Apache-2.0. N-ATLAS weights and speech models are released by NCAIR under
their own terms on Hugging Face.
