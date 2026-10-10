# natlas

Python SDK, OpenAI-compatible server and benchmark kit for N-ATLAS. Layout:
`src/natlas/` (client, server, engine, asr, models, cli, bench/), `playground/`
(Streamlit, not packaged), `examples/`, `bench-results/` (published runs).

## Commands

```bash
uv venv && CMAKE_ARGS="-DGGML_METAL=on" uv pip install -e ".[server,bench,asr,dev,playground]"
pytest                                  # unit tests, no model needed
ruff check src tests playground
natlas serve --asr --api-key test-key   # then:
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

Run the live tests before calling a change to the server, engine or speech path
done. Unit tests use fakes and cannot catch model or llama.cpp behaviour.

## Invariants

- **Render the chat template in `engine.py`, not with llama.cpp's formatter.**
  The template stamps today's date; llama.cpp leaves it at the template default.
- **Clear the llama.cpp cache before every generation.** Prefix reuse made the
  same temperature-0 request answer differently depending on the previous one,
  so benchmark scores depended on question order.
- **Keep the model card defaults** (its system prompt, temperature 0.1,
  repetition penalty 1.12) in the client and server. Without the system prompt
  the model drifts towards generic Llama-3 answers.
- **Never commit MAFAND reference text.** It is GPL-3.0 and this repo is
  Apache-2.0. Translation results store item ids only; datasets are fetched at
  run time.
- **The speech loop guard runs only on a retry.** Banning repeated 4-grams fixes
  the Igbo model's "onye onye onye" loops but costs Hausa and Yoruba accuracy
  when applied to every request.
- **Decode m4a and webm through a temporary file.** m4a keeps its index at the
  end, which ffmpeg cannot reach from a pipe; short test clips hide this.
- **Model downloads are pinned and hash-checked.** Bump `GGUF_REVISION` in
  `models.py` deliberately, never to a branch name.
- **Report benchmark caveats with the numbers**: sample size, 4-bit weights,
  possible training contamination, and untoned Yoruba FLEURS references.

## Local quirks

- `natlas serve` needs GPU access; under a sandbox that blocks Metal, llama.cpp
  fails with "Failed to create llama_context".
- The speech models are gated per model on Hugging Face; a valid login still
  gets 403 until each model's terms are accepted.
