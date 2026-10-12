# Contributing to pynatlas

Contributions are welcome: bug reports, fixes, documentation, new benchmark
tasks, and native-speaker review of the Hausa, Igbo and Yoruba test sets. The
project is maintained by Open Lab Technologies.

## Reporting a problem

Open an issue with what you ran, what you expected and what happened. For a
wrong or poor answer from the model, include the prompt, the language and the
reply; `pynatlas config` shows the settings to paste alongside.

## Setting up

```bash
git clone https://github.com/open-lab-technologies/natlas-sdk && cd natlas-sdk
uv venv && uv pip install -e ".[server,asr,bench]" --group dev
```

Set `CMAKE_ARGS` for your GPU when installing the server extra (see the README).

## Before opening a pull request

```bash
ruff check src tests playground examples
mypy
pytest
```

All three must pass; CI runs them on Python 3.10 and 3.13. Changes to the
server, engine or speech path also need the end-to-end tests against a running
model:

```bash
pynatlas serve --asr --api-key test-key
NATLAS_E2E_URL=http://127.0.0.1:8000/v1 NATLAS_E2E_API_KEY=test-key pytest tests/test_live.py
```

## Conventions

- One change per pull request, with tests for new behaviour.
- Commit subjects follow Conventional Commits (`fix(server): ...`,
  `feat(client): ...`); the body says why the change is needed.
- Comments explain why the code is as it is, not what it does.
- The core install stays httpx only: anything heavier goes in an extra and is
  imported where it is used.
- Benchmark claims need a run committed under `bench-results/`. Never tune on
  the held-out sets (`bench/tool_cases.json`, the AfriXNLI test split), or
  their scores stop meaning anything.

## Licence

By contributing, you agree that your contribution is licensed under
Apache-2.0, as the rest of the code is. The N-ATLaS and NCAIR1 models have
their own terms; see NOTICE.
