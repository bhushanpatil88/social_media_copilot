"""Web search tool — Tavily wrapper that degrades gracefully if the key
is missing."""
from __future__ import annotations

from typing import List, Dict

from config.settings import settings


def web_search(query: str, max_results: int = 4) -> List[Dict[str, str]]:
    """Return a list of {title, url, content} dicts for the query."""
    if not settings.tavily_api_key:
        return []

    # Imported lazily so the rest of the app works without tavily installed
    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
        )
        results = resp.get("results", []) if isinstance(resp, dict) else []
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]
    except Exception as exc:
        print(f"[web_search] Tavily error: {exc}")
        return []


def format_web_results(results: List[Dict[str, str]]) -> str:
    if not results:
        return "(no web results available)"
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[Web {i}] {r['title']}\n{r['content']}\nSource: {r['url']}"
        )
    return "\n\n".join(parts)
