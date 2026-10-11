"""Tool calling and structured output.

    natlas serve
    python examples/tools_and_structured_output.py
"""

from typing import Literal

from pydantic import BaseModel, Field

from natlas import NAtlas


def get_market_price(commodity: str, market: str) -> dict:
    """Today's price of a food commodity at a named market.

    Args:
        commodity: For example rice, beans or tomatoes.
        market: Market name, for example Dawanau.
    """
    prices = {"rice": 1450, "shinkafa": 1450, "beans": 1200}
    return {"commodity": commodity, "market": market,
            "naira_per_kg": prices.get(commodity.lower(), 900)}


class Feedback(BaseModel):
    language: Literal["en", "ha", "ig", "yo"] = Field(
        description="Language the message is written in: en English, ha Hausa, ig Igbo, yo Yoruba")
    sentiment: Literal["positive", "neutral", "negative"]
    english_summary: str


with NAtlas() as client:
    reply = client.run_tools("Nawa ne farashin shinkafa a kasuwar Dawanau yau?", [get_market_price])
    print("Answer:", reply.text)

    feedback = client.chat(
        "Classify this customer message: Kayan sun iso a makare kuma sun lalace.",
        response_format=Feedback,
    ).parsed
    print("Parsed:", feedback)
