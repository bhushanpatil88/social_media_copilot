"""Platform-specific constraints, character limits, and best practices."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    max_chars: int
    ideal_chars: int
    hashtag_count: int
    max_hashtags: int
    supports_emoji: bool
    style_notes: str
    length_targets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # NEW: optional override of style_notes for a given length
    style_overrides: Dict[str, str] = field(default_factory=dict)

    def target_range(self, length: str) -> Tuple[int, int]:
        """Return (min_chars, max_chars) target for a given length preference."""
        return self.length_targets.get(
            length.lower(), self.length_targets.get("medium", (0, self.max_chars))
        )

    def style_for(self, length: str) -> str:
        """Style notes, possibly adjusted for the length preference."""
        return self.style_overrides.get(length.lower(), self.style_notes)


PLATFORMS: Dict[str, PlatformSpec] = {
    "Instagram": PlatformSpec(
        name="Instagram",
        max_chars=2200,
        ideal_chars=150,
        hashtag_count=10,
        max_hashtags=30,
        supports_emoji=True,
        style_notes=(
            "Visual-first, conversational, friendly. Hook in the first line. "
            "Use line breaks for readability. Emojis are welcome. "
            "End with a clear call-to-action or question."
        ),
        length_targets={
            "short":  (80, 200),
            "medium": (300, 700),
            "long":   (900, 1600),
        },
    ),
    "Twitter": PlatformSpec(
        name="Twitter",
        max_chars=280,
        ideal_chars=240,
        hashtag_count=2,
        max_hashtags=3,
        supports_emoji=True,
        style_notes=(
            "Concise, punchy, witty. One core idea only. "
            "Strong hook. Minimal hashtags (1-3 max)."
        ),
        # IMPORTANT: 'long' on Twitter still means a single tweet, just one
        # that uses the full character allowance. The style override removes
        # the 'one core idea only' framing because that fights against length.
        style_overrides={
            "long": (
                "Punchy hook on the first line. Use the full Twitter "
                "character allowance — multiple short lines, a mini-arc, or "
                "a setup-payoff structure. Witty, voice-driven. "
                "Hashtags 1-2 max."
            ),
        },
        length_targets={
            "short":  (60, 140),
            "medium": (140, 220),
            "long":   (180, 280),   # was (220, 280) — gave the model no room
        },
    ),
    "LinkedIn": PlatformSpec(
        name="LinkedIn",
        max_chars=3000,
        ideal_chars=1300,
        hashtag_count=4,
        max_hashtags=5,
        supports_emoji=True,
        style_notes=(
            "Professional, insightful, story-driven. "
            "Strong opening line that earns the click on 'see more'. "
            "Short paragraphs, white space, and a takeaway. "
            "Light emoji use only. Hashtags at the end."
        ),
        length_targets={
            "short":  (300, 700),
            "medium": (800, 1500),
            "long":   (1700, 2800),
        },
    ),
    "Facebook": PlatformSpec(
        name="Facebook",
        max_chars=63206,
        ideal_chars=400,
        hashtag_count=2,
        max_hashtags=3,
        supports_emoji=True,
        style_notes=(
            "Conversational and community-oriented. "
            "Storytelling works well. Encourage comments and shares. "
            "Hashtags are optional and used sparingly."
        ),
        length_targets={
            "short":  (120, 300),
            "medium": (400, 900),
            "long":   (1100, 2200),
        },
    ),
}


def get_platform(name: str) -> PlatformSpec:
    if name not in PLATFORMS:
        raise ValueError(f"Unsupported platform: {name}")
    return PLATFORMS[name]