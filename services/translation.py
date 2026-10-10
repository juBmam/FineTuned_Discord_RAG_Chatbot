import os

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-5-mini").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def translate_to_chinese(text: str) -> str:
    """Translate a message into concise, natural Chinese using the existing OpenAI setup."""
    response = await client.responses.create(
        model=TRANSLATION_MODEL,
        instructions=(
            "Translate the user's message into natural Traditional Chinese. "
            "Return only the translation with no explanation or commentary."
        ),
        input=text,
        max_output_tokens=300,
    )

    return response.output_text.strip()
