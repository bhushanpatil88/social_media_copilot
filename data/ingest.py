"""Run once to load sample posts into the Chroma vector store.

Usage:
    python -m data.ingest
or:
    python data/ingest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make project root importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.vector_store import add_posts, collection_size


def main() -> None:
    data_path = ROOT / "data" / "sample_posts.json"
    with open(data_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    before = collection_size()
    added = add_posts(posts)
    after = collection_size()
    print(f"Ingested {added} posts. Collection size: {before} → {after}")


if __name__ == "__main__":
    main()
