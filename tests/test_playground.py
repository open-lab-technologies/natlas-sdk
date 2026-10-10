import json
from pathlib import Path

import httpx
import pytest
import respx

pytest.importorskip("streamlit")
pytest.importorskip("pandas")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "playground" / "app.py")
BASE = "http://playground-test/v1"


def completion(text):
    return {"model": "n-atlas", "choices": [
        {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


def sse(*tokens):
    frames = [{"choices": [{"delta": {"content": t}}]} for t in tokens]
    return "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("NATLAS_BASE_URL", BASE)
    monkeypatch.chdir(tmp_path)  # no bench-results here unless a test creates it
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_renders_all_tabs(app):
    assert [t.label for t in app.tabs] == ["Chat", "Translate", "Voice", "Benchmarks"]
    assert app.sidebar.text_input[0].value == BASE


@respx.mock
def test_chat_streams_reply_into_history(app):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, text=sse("Ab", "uja"))
    )
    app.chat_input[0].set_value("Capital of Nigeria?").run()
    assert not app.exception
    assert app.session_state.history[-1] == {"role": "assistant", "content": "Abuja"}
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"][-1] == {"role": "user", "content": "Capital of Nigeria?"}


@respx.mock
def test_translate_shows_result(app):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=completion("Barka da safiya"))
    )
    app.tabs[1].selectbox[1].set_value("ha")
    app.tabs[1].button[0].click().run()
    assert app.tabs[1].success[0].value == "Barka da safiya"


@respx.mock
def test_server_error_is_shown_not_raised(app):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Missing or invalid API key"}})
    )
    app.tabs[1].button[0].click().run()
    assert not app.exception
    assert "invalid API key" in app.tabs[1].error[0].value


def test_benchmarks_tab_reads_results(monkeypatch, tmp_path):
    run = tmp_path / "bench-results" / "demo"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "label": "demo", "endpoint": BASE, "started": "2026-10-10T00:00:00+00:00",
        "results": [{"task": "afrixnli", "language": "yo", "score": 53.0}],
    }))
    (run / "afrixnli-yo.jsonl").write_text(
        json.dumps({"id": "x", "reply": "neutral", "gold": "neutral", "correct": True}) + "\n"
    )
    monkeypatch.setenv("NATLAS_BASE_URL", BASE)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs[3].dataframe) == 2  # score table and reply table
