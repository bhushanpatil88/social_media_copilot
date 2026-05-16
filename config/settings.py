"""Central configuration for the Social Media Copilot."""
from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # API keys
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

    # Models
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector store
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "social_posts")

    # RAG
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    # Generation
    max_refinement_iterations: int = 2
    quality_threshold: float = 0.75


settings = Settings()
