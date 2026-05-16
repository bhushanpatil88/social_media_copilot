"""RAG retriever with optional platform-aware filtering."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document

from config.settings import settings
from src.rag.vector_store import get_vector_store


def retrieve(topic: str, platform: Optional[str] = None,
             k: Optional[int] = None) -> List[Document]:
    """Retrieve top-k similar posts. If a platform is given, prefer same-platform
    posts but fall back to cross-platform when too few hits."""
    store = get_vector_store()
    k = k or settings.rag_top_k

    if platform:
        try:
            same = store.similarity_search(
                topic, k=k, filter={"platform": platform}
            )
        except Exception:
            same = []
        if len(same) >= k:
            return same
        # back-fill with cross-platform hits
        extra_needed = k - len(same)
        cross = store.similarity_search(topic, k=k + extra_needed)
        # de-dup by page_content
        seen = {d.page_content for d in same}
        for d in cross:
            if d.page_content not in seen and len(same) < k:
                same.append(d)
                seen.add(d.page_content)
        return same

    return store.similarity_search(topic, k=k)


def format_context(docs: List[Document]) -> str:
    """Format retrieved docs into a single context string for the LLM."""
    if not docs:
        return "(no similar past posts found)"
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        header = (
            f"[Example {i} | platform={meta.get('platform','?')} "
            f"| topic={meta.get('topic','?')}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)
