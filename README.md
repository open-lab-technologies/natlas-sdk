# natlas

Python SDK, OpenAI-compatible server and benchmark kit for
[N-ATLAS](https://huggingface.co/NCAIR1/N-ATLaS), Nigeria's open multilingual
language model (Llama-3 8B fine-tuned for English, Hausa, Igbo and Yoruba).

- **`NAtlas` client**: chat, streaming, translation and speech to text against any
  endpoint serving N-ATLAS.
- **`natlas serve`**: runs N-ATLAS on your own machine behind the OpenAI API, so
  existing OpenAI-based code switches to N-ATLAS by changing one URL.
- **`natlas bench`**: scores any N-ATLAS endpoint on public Nigerian-language
  benchmarks (IrokoBench and MAFAND), with every model reply saved for inspection.

## Install

```bash
pip install "natlas[server] @ git+https://github.com/Nazehs/natlas-sdk"
```

Extras: `server` (local inference with llama.cpp), `asr` (speech to text),
`bench` (benchmarks), `dev` (tests and lint). Python 3.10 or newer.

On Apple Silicon, build llama.cpp with Metal so the model runs on the GPU:

```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install "natlas[server] @ git+https://github.com/Nazehs/natlas-sdk"
```

With an NVIDIA GPU use `CMAKE_ARGS="-DGGML_CUDA=on"` instead.

## Quick start

```bash
natlas download          # 4.9 GB, 4-bit quantised weights
natlas serve             # http://127.0.0.1:8000/v1
```

```python
from natlas import NAtlas

client = NAtlas()  # or NAtlas(base_url=..., api_key=...)

print(client.chat("Kí ni olú ìlú Nàìjíríà?"))
# Olú ìlú Nàìjíríà ni Abuja.

for token in client.stream("Ka bayyana mini menene noma a takaice"):
    print(token, end="", flush=True)

client.translate("Wash your hands before eating.", to="ig")
```

The client applies the system prompt and sampling settings from the N-ATLAS
model card (temperature 0.1, repetition penalty 1.12) unless you pass your own.
Without that system prompt the model's answers drift towards generic Llama-3.

### Use it from existing OpenAI code

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="n-atlas",
    messages=[{"role": "user", "content": "What is the capital of Nigeria?"}],
)
```

Streaming, list-style message content and the `developer` role all work, so
current OpenAI and LangChain clients need no changes. The server adds the
N-ATLAS system prompt when a request has none.

### Sharing a server

`natlas serve` listens on localhost only. To let others use it, bind to a
reachable address and require a key:

```bash
natlas serve --host 0.0.0.0 --api-key "$(openssl rand -hex 16)"
```

Clients send the key as a bearer token (`NATLAS_API_KEY` for this SDK, `api_key=`
for the OpenAI client). `/health` stays open for monitoring.

### Speech to text

```bash
pip install "natlas[server,asr] @ git+https://github.com/Nazehs/natlas-sdk"
natlas serve --asr
natlas transcribe voice-note.ogg --language yo
```

```python
heard = client.transcribe("voice-note.ogg", language="ha")
answer = client.chat(heard)
```

On an Apple Silicon laptop a WhatsApp voice note takes about a second to
transcribe once the model is loaded, and about a second more for the answer.

This uses the official N-ATLAS speech models (`NCAIR1/Yoruba-ASR`, `Igbo-ASR`,
`Hausa-ASR`, `NigerianAccentedEnglish`). They are gated: logging in is not
enough, you also have to click "Agree and access repository" on each model page
on Hugging Face, then run `hf auth login`.

wav, mp3, ogg and opus (WhatsApp voice notes) decode directly. m4a (iPhone
recordings) and webm (browser recorders) need ffmpeg installed.

The OpenAI client's `audio.transcriptions.create` works too: pass
`model="NCAIR1/Hausa-ASR"` (or another speech model) and the language follows
from it.

## Examples

Run these against `natlas serve` (add `--asr` for the voice example):

| Script | Shows |
|---|---|
| [examples/openai_drop_in.py](examples/openai_drop_in.py) | Unchanged OpenAI code answering in all four languages |
| [examples/voice_assistant.py](examples/voice_assistant.py) | A voice note transcribed and answered in the same language |

## Playground

A browser playground for trying prompts, translation and voice input, and for
reading benchmark results reply by reply. It lives in the repository rather
than the package, so run it from a clone:

```bash
git clone https://github.com/Nazehs/natlas-sdk && cd natlas-sdk
pip install -e ".[playground]"
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

Every command except `serve` and `download` talks to `--base-url`, so it works
against a remote or hosted N-ATLAS endpoint as well as a local one.

## Configuration

| Variable | Default | Used for |
|---|---|---|
| `NATLAS_BASE_URL` | `http://localhost:8000/v1` | Endpoint the client talks to |
| `NATLAS_API_KEY` | none | Bearer token the client sends, and the key `natlas serve` requires |
| `NATLAS_MODEL_PATH` | downloaded `--quant` | GGUF file for `natlas serve` |
| `NATLAS_CACHE` | `~/.cache/natlas` | Downloaded weights and benchmark data |

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

The datasets are downloaded at run time rather than shipped with the package,
because MAFAND is GPL-3.0. N-ATLAS was trained partly on open Hugging Face
datasets, so some test items may have been seen in training; treat scores as
an upper bound until that is ruled out.

### Results: N-ATLAS 4-bit (Q4_K_M), llama.cpp on Apple Silicon

100 items per language per task, seed 0, temperature 0. Full replies in
[bench-results/n-atlas-q4_k_m](bench-results/n-atlas-q4_k_m/).

| Task | Metric | English | Hausa | Igbo | Yoruba | Chance |
|---|---|---:|---:|---:|---:|---:|
| AfriMMLU | accuracy % | 53 | 35 | 41 | 37 | 25 |
| AfriXNLI | accuracy % | 70 | 58 | 57 | 53 | 33 |
| AfriMGSM | accuracy % | 70 | 42 | 34 | 40 | |
| MAFAND English to language | chrF++ | | 33.7 | 41.1 | 28.0 | |
| MAFAND language to English | chrF++ | | 37.5 | 40.7 | 41.8 | |

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
- The Igbo model sometimes loops on one word ("onye onye onye ..."). The SDK
  detects that and re-decodes with repeated phrases banned, which took Igbo WER
  from 65.7% to 43.8% without changing the other languages.

Reproduce with:

```bash
natlas bench -n 100 --seed 0 --out bench-results/my-run
```

## Model notes

`natlas download` fetches a community GGUF conversion
([QuantFactory/N-ATLaS-GGUF](https://huggingface.co/QuantFactory/N-ATLaS-GGUF))
of the official weights, pinned to a fixed revision and checked against the
SHA-256 Hugging Face publishes. Interrupted downloads resume. The official repository ships only full-precision
safetensors, which need about 16 GB of GPU memory. To serve those, run vLLM or
another OpenAI-compatible server and point the client at it.

Use Q4_K_M (the default) or larger. Smaller builds lose real quality: in
testing, Q2_K returned the English input unchanged when asked to translate it
into Hausa, where Q4_K_M translated it correctly.

The chat template stamps today's date into every prompt. llama.cpp's own chat
formatting leaves that date at the template default, so `natlas serve` renders
the template itself.

## Known limitations

What testing found that is not fixed, because it is the model's behaviour or
outside this package:

- **Yoruba output is the weakest.** Translation into Yoruba scores lowest, and
  free-form Yoruba answers are often incoherent and drop tone marks, in line
  with the model card's own human evaluation.
- **Nigerian-language accuracy trails English** on every benchmark, most on
  knowledge and maths.
- **Speech error rates are high outside English**: about 31% WER for Hausa and
  37% for Igbo, 29% for Yoruba ignoring tone marks.
- **Possible test contamination.** N-ATLAS was trained partly on open datasets,
  so some benchmark items may have been seen in training.
- **One request at a time.** `natlas serve` runs one llama.cpp model and
  serialises requests; for concurrent users serve the official weights with
  vLLM and point the client at it.
- **Small quantisations break.** Q2_K failed to translate in testing; use
  Q4_K_M or larger.
- **Weights are a community conversion.** The GGUF files are not published by
  NCAIR. They are pinned and hash-checked, but provenance rests on QuantFactory.

## Development

```bash
uv venv && CMAKE_ARGS="-DGGML_METAL=on" uv pip install -e ".[server,bench,asr,dev]"
pytest
ruff check src tests playground
```

`pytest` runs the unit tests, which need no model and are what CI runs. The
end-to-end tests in `tests/test_live.py` exercise a running server in all four
languages, through this SDK and the official OpenAI client, including speech in
phone and browser formats:

```bash
natlas serve --asr --api-key test-key
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

## Licence

Code: Apache-2.0. N-ATLAS weights and speech models are released by NCAIR under
their own terms on Hugging Face.
