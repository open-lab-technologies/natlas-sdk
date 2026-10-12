"""Existing OpenAI code running on N-ATLaS: only the base URL changes.

    pynatlas serve
    python examples/openai_drop_in.py
"""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("NATLAS_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("NATLAS_API_KEY", "unused"),
)

questions = {
    "English": "Give one tip for keeping drinking water safe.",
    "Hausa": "Ka ba ni shawara guda daya don kiyaye tsaftar ruwan sha.",
    "Igbo": "Nye m otu ndụmọdụ maka idebe mmiri ọṅụṅụ ka ọ dị ọcha.",
    "Yoruba": "Fún mi ní ìmọ̀ràn kan láti jẹ́ kí omi mímu mọ́.",
}

for language, question in questions.items():
    reply = client.chat.completions.create(
        model="n-atlas",
        messages=[{"role": "user", "content": question}],
        max_tokens=120,
    )
    print(f"{language}\n  Q: {question}\n  A: {reply.choices[0].message.content.strip()}\n")
