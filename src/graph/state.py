"""Shared state object that flows through the LangGraph workflow."""
from __future__ import annotations

from typing import List, Optional, TypedDict, Dict, Any


class CopilotState(TypedDict, total=False):
    # ---- inputs ----
    topic: str
    platform: str         # Instagram | Twitter | LinkedIn | Facebook
    tone: str             # Professional | Casual | Witty | Inspirational | Educational
    humor_level: int      # 0..10
    length: str           # short | medium | long
    audience: str         # free-text description of target audience
    cta: str              # call-to-action description (or 'none')

    # ---- routing / retrieval ----
    route: str            # 'web' | 'rag' | 'both'
    rag_docs: List[Dict[str, Any]]   # serialised Documents
    web_results: List[Dict[str, str]]
    context: str          # combined context string fed to the post generator

    # ---- generation ----
    post: str
    hashtags: List[str]
    image_prompt: str

    # ---- quality loop ----
    quality_score: float
    quality_feedback: str
    iterations: int

    # ---- diagnostics ----
    error: Optional[str]
    sources: List[str]    # urls / metadata used (for UI)
