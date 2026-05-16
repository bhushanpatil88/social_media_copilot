"""Small utilities used across the project."""
from __future__ import annotations

import re
from typing import List


def parse_hashtags(raw: str, max_count: int) -> List[str]:
    """Pull out hashtags from a string, dedupe, cap at max_count."""
    found = re.findall(r"#\w+", raw)
    seen, out = set(), []
    for tag in found:
        # normalise: remove repeated # and trailing punctuation
        tag = "#" + tag.lstrip("#").strip(".,;:!?")
        if tag.lower() not in seen and len(tag) > 1:
            seen.add(tag.lower())
            out.append(tag)
        if len(out) >= max_count:
            break
    return out


def parse_quality_response(text: str) -> tuple[float, str]:
    """Parse 'SCORE: 0.82\nFEEDBACK: ...' format."""
    score = 0.5
    feedback = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score = float(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
    return max(0.0, min(1.0, score)), feedback


def truncate_to_limit(text: str, limit: int) -> str:
    """If text is over the limit, truncate cleanly on a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0]
    return cut + "…"
