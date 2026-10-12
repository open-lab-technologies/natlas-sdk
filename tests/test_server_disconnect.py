"""A client that hangs up mid-stream must not leave the engine locked."""

import socket
import threading
import time

import httpx
import pytest

uvicorn = pytest.importorskip("uvicorn")
from pynatlas.engine import Chunk
from pynatlas.server import create_app


class LockingEngine:
    """Holds a lock while generating, as LlamaCppEngine does."""

    name = "locking"

    def __init__(self):
        self.lock = threading.Lock()

    def generate(self, messages, **params):
        with self.lock:
            for i in range(200):
                time.sleep(0.01)
                yield Chunk(f"t{i} ")
            yield Chunk("", "stop")

    def count_prompt_tokens(self, messages):
        return 1

    def count_tokens(self, text):
        return len(text.split())


@pytest.fixture
def server_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = LockingEngine()
    config = uvicorn.Config(create_app(engine), host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


def test_engine_is_released_when_client_disconnects_mid_stream(server_url):
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    with httpx.Client(timeout=10) as client, \
            client.stream("POST", f"{server_url}/chat/completions", json=body) as resp:
        for _ in resp.iter_lines():
            break  # read one line, then hang up

    body["stream"] = False
    resp = httpx.post(f"{server_url}/chat/completions", json=body, timeout=10)
    assert resp.status_code == 200
