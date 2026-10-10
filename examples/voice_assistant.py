"""Answer a voice note in the language it was spoken in.

Works with WhatsApp voice notes (.ogg/.opus), iPhone recordings (.m4a), mp3 and wav.

    natlas serve --asr
    python examples/voice_assistant.py voice-note.ogg --language ha
"""

import argparse

from natlas import NAtlas, get_language

parser = argparse.ArgumentParser()
parser.add_argument("audio")
parser.add_argument("--language", "-l", required=True, help="en, ha, ig or yo")
args = parser.parse_args()

language = get_language(args.language)
with NAtlas() as client:
    heard = client.transcribe(args.audio, language=language.code)
    print(f"Heard ({language.name}): {heard}\n")
    print("Answer: ", end="", flush=True)
    for token in client.stream(
        heard,
        system=f"You are a helpful assistant. Reply briefly in {language.name}, "
               "in plain words a listener can follow if the reply is read aloud.",
        max_tokens=200,
    ):
        print(token, end="", flush=True)
    print()
