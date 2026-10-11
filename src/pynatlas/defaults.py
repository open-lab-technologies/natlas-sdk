"""Settings from the N-ATLAS model card.

The model was tuned with this system prompt; without it, answers drift towards
generic Llama-3 behaviour.
"""

SYSTEM_PROMPT = (
    "you are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)
TEMPERATURE = 0.1
MAX_TOKENS = 512
# Not part of the OpenAI API, so the client never sends it unasked. `pynatlas serve`
# applies it by default; with vLLM, pass repetition_penalty=1.12 to chat() yourself.
REPETITION_PENALTY = 1.12
