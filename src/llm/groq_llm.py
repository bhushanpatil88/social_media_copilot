"""ChatGroq LLM factory."""
from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from config.settings import settings


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.7) -> ChatGroq:
    """Return a cached ChatGroq instance for a given temperature."""
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    return ChatGroq(
        model=settings.groq_model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        max_retries=2,
        timeout=60,
    )
