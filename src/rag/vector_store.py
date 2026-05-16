"""Vector store wrapper around Chroma + HuggingFace embeddings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def add_posts(posts: Iterable[Dict[str, Any]]) -> int:
    """Add posts to the vector DB. Each post is a dict with at least
    'text' and metadata fields like 'platform', 'topic', 'author'."""
    store = get_vector_store()
    docs: List[Document] = []
    for p in posts:
        text = p.get("text", "").strip()
        if not text:
            continue
        metadata = {k: v for k, v in p.items() if k != "text"}
        docs.append(Document(page_content=text, metadata=metadata))
    if docs:
        store.add_documents(docs)
    return len(docs)


def collection_size() -> int:
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0
