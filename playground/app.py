"""Browser playground for N-ATLAS: chat, translate, speak, and inspect benchmark runs.

    streamlit run playground/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from natlas import LANGUAGES, NAtlas, NAtlasError
from natlas.client import DEFAULT_SYSTEM_PROMPT

st.set_page_config(page_title="N-ATLAS Playground", layout="wide")

with st.sidebar:
    st.header("Endpoint")
    base_url = st.text_input(
        "Base URL", os.getenv("NATLAS_BASE_URL", "http://localhost:8000/v1")
    )
    api_key = st.text_input("API key (optional)", type="password")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)
    max_tokens = st.slider("Max tokens", 32, 2048, 512, 32)
    system = st.text_area("System prompt", DEFAULT_SYSTEM_PROMPT, height=120)


@st.cache_resource
def get_client(url: str, key: str) -> NAtlas:
    return NAtlas(base_url=url, api_key=key or None)


client = get_client(base_url, api_key)
names = {code: lang.name for code, lang in LANGUAGES.items()}

chat_tab, translate_tab, voice_tab, bench_tab = st.tabs(
    ["Chat", "Translate", "Voice", "Benchmarks"]
)

with chat_tab:
    if "history" not in st.session_state:
        st.session_state.history = []
    if st.button("Clear conversation"):
        st.session_state.history = []
    for msg in st.session_state.history:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Ask in English, Hausa, Igbo or Yoruba"):
        st.session_state.history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            reply = st.chat_message("assistant").write_stream(
                client.stream(st.session_state.history, system=system,
                              temperature=temperature, max_tokens=max_tokens)
            )
            st.session_state.history.append({"role": "assistant", "content": reply})
        except NAtlasError as e:
            st.error(str(e))

with translate_tab:
    left, right = st.columns(2)
    source = left.selectbox("From", list(names), format_func=names.get, index=0)
    target = right.selectbox("To", list(names), format_func=names.get, index=3)
    text = st.text_area("Text", "Wash your hands with soap before you eat.")
    if st.button("Translate", type="primary"):
        if source == target:
            st.warning("Pick two different languages.")
        else:
            try:
                st.success(client.translate(text, to=target, source=source))
            except NAtlasError as e:
                st.error(str(e))

with voice_tab:
    st.caption("Needs a server started with `natlas serve --asr`.")
    lang = st.selectbox("Spoken language", list(names), format_func=names.get, key="asr")
    audio = st.audio_input("Record a question")
    upload = st.file_uploader("or upload a voice note", type=["wav", "mp3", "ogg", "opus", "m4a"])
    clip = audio or upload
    if clip and st.button("Transcribe and answer", type="primary"):
        try:
            with st.spinner("Transcribing"):
                heard = client.transcribe(clip.getvalue(), language=lang)
            st.markdown(f"**Heard:** {heard}")
            st.chat_message("assistant").write_stream(
                client.stream(heard, system=system, temperature=temperature,
                              max_tokens=max_tokens)
            )
        except NAtlasError as e:
            st.error(str(e))

with bench_tab:
    root = Path(st.text_input("Results directory", "bench-results"))
    runs = sorted(p.parent for p in root.glob("*/summary.json"))
    if not runs:
        st.info("No runs yet. Try `natlas bench -n 20`.")
    else:
        run = st.selectbox("Run", runs, format_func=lambda p: p.name)
        summary = json.loads((run / "summary.json").read_text())
        st.caption(f"{summary['label']} · {summary['endpoint']} · started {summary['started']}")
        df = pd.DataFrame(summary["results"])
        df["language"] = df["language"].map(names)
        pivot = df.pivot_table(index="task", columns="language", values="score")
        st.dataframe(pivot.style.format("{:.1f}"), width="stretch")

        st.subheader("Inspect replies")
        files = sorted(run.glob("*.jsonl"))
        chosen = st.selectbox("Task and language", files, format_func=lambda p: p.stem)
        rows = pd.read_json(chosen, lines=True)
        if "correct" in rows:
            only_wrong = st.toggle("Only wrong answers")
            if only_wrong:
                rows = rows[~rows["correct"]]
        st.dataframe(rows, width="stretch", hide_index=True)
