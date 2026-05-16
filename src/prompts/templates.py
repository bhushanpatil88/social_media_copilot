"""Prompt templates for every node in the graph."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Length guidance — what "short / medium / long" actually mean structurally
# ---------------------------------------------------------------------------
LENGTH_GUIDANCE = {
    "short": (
        "ONE tight idea. No buildup. Get to the point in the first line and "
        "stop. No story, no setup."
    ),
    "medium": (
        "Hook (1 line) → one core insight or mini-anecdote (2–3 sentences) → "
        "closing thought or CTA (1 line)."
    ),
    "long": (
        "Open with a hook, then expand with a story, example, or layered "
        "argument that justifies the length. Use line breaks for rhythm. "
        "End with a clear takeaway or CTA. Use the full target character "
        "range — do NOT stop early."
    ),
}


# ---------------------------------------------------------------------------
# 1. Router: decide if web search is needed
# ---------------------------------------------------------------------------
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routing assistant. Decide whether a topic for a social-media "
            "post requires up-to-date information from the web.\n"
            "Reply with EXACTLY one word: 'web' or 'rag'.\n\n"
            "Choose 'web' when:\n"
            "- The topic mentions a recent event, news, launch, or release\n"
            "- The topic mentions a specific year >= 2024 or 'today/this week/recent'\n"
            "- The topic is about a niche product, person, or company likely missing from training data\n\n"
            "Choose 'rag' when:\n"
            "- The topic is evergreen (productivity tips, motivation, generic advice)\n"
            "- The topic is a well-known concept that does not change over time",
        ),
        ("human", "Topic: {topic}\n\nDecision:"),
    ]
)

# ---------------------------------------------------------------------------
# 2. Post generation
# ---------------------------------------------------------------------------
POST_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an elite social-media copywriter. Write a single post for "
            "{platform}. Follow the platform style guide strictly.\n\n"
            "PLATFORM STYLE GUIDE:\n{style_notes}\n\n"
            "LENGTH — THIS OVERRIDES STYLE WHEN THEY CONFLICT:\n"
            "- Target range: {target_min}–{target_max} characters. "
            "Your output MUST land inside this range.\n"
            "- A post under {target_min} characters will be REJECTED. "
            "A post over {max_chars} characters will be REJECTED.\n"
            "- Length style ({length}): {length_guidance}\n"
            "- Even if the platform style suggests brevity, you MUST hit the "
            "target range. Add depth, examples, or layered reasoning to fill it.\n\n"
            "OTHER CONSTRAINTS:\n"
            "- Do NOT include hashtags in the post body — those are added separately.\n"
            "- Do NOT wrap the output in quotes or markdown.\n"
            "- Output ONLY the post text, nothing else.\n\n"
            "TONE PROFILE:\n"
            "- Tone: {tone}\n"
            "- Humor level: {humor_level}/10 "
            "(0 = strictly serious, 5 = light wit, 10 = playful and joke-heavy)\n"
            "- Target audience: {audience}\n"
            "- Call-to-action: {cta}",
        ),
        (
            "human",
            "Topic: {topic}\n\n"
            "Reference context (use it for facts and inspiration, do NOT copy verbatim):\n"
            "{context}\n\n"
            "Write the post now. Remember: target {target_min}–{target_max} "
            "characters. Count as you go.",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 2b. REFINEMENT — a separate prompt that shows the previous attempt
# ---------------------------------------------------------------------------
REFINE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are revising a social-media post that an editor rejected. "
            "Your job is to fix the specific problem they flagged.\n\n"
            "PLATFORM: {platform}\n"
            "STYLE GUIDE: {style_notes}\n\n"
            "HARD REQUIREMENT — LENGTH:\n"
            "- Target: {target_min}–{target_max} characters.\n"
            "- Hard ceiling: {max_chars} characters.\n"
            "- Length style ({length}): {length_guidance}\n\n"
            "TONE: {tone}, humor {humor_level}/10, audience {audience}, CTA: {cta}\n\n"
            "Output ONLY the revised post, no preamble, no quotes, no markdown.",
        ),
        (
            "human",
            "Topic: {topic}\n\n"
            "Reference context:\n{context}\n\n"
            "PREVIOUS DRAFT — was {previous_chars} characters "
            "(target was {target_min}–{target_max}, so it missed by "
            "{shortfall} characters):\n"
            "---\n{previous_post}\n---\n\n"
            "EDITOR FEEDBACK: {feedback}\n\n"
            "Now write the revised post. Keep what works in the previous "
            "draft, but EXPAND it with a story, example, or extra layer of "
            "reasoning so it lands inside {target_min}–{target_max} characters. "
            "Do NOT just rephrase the previous draft at the same length.",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 3. Hashtag generation
# ---------------------------------------------------------------------------
HASHTAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You generate hashtags for {platform}. Rules:\n"
            "- Generate exactly {hashtag_count} hashtags.\n"
            "- Mix broad (high-volume) and niche (targeted) tags.\n"
            "- No spaces inside a hashtag. CamelCase for multi-word tags.\n"
            "- Output ONLY the hashtags separated by single spaces, "
            "one line, nothing else.\n"
            "- No numbering, no commas, no explanations.",
        ),
        (
            "human",
            "Topic: {topic}\n\nPost:\n{post}\n\nHashtags:",
        ),
    ]
)

# ---------------------------------------------------------------------------
# 4. Quality check
# ---------------------------------------------------------------------------
QUALITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict social-media editor. Score the post on a 0-1 scale "
            "based on:\n"
            "- Hook strength (does the first line grab attention?)\n"
            "- Clarity and flow\n"
            "- Tone match (target tone: {tone}, humor: {humor_level}/10)\n"
            "- Platform fit ({platform})\n"
            "- LENGTH COMPLIANCE — target {target_min}–{target_max} chars, "
            "actual {actual_chars} chars, hard max {max_chars}. "
            "Heavily penalise (drop below 0.6) if actual is below target_min "
            "or above max_chars.\n\n"
            "Reply in EXACTLY this format on two lines:\n"
            "SCORE: <number between 0 and 1>\n"
            "FEEDBACK: <one short sentence of actionable feedback>",
        ),
        ("human", "Post:\n{post}\n\nEvaluation:"),
    ]
)

# ---------------------------------------------------------------------------
# 5. Image suggestion
# ---------------------------------------------------------------------------
IMAGE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write concise image prompts for social media posts. "
            "Output ONE short, vivid prompt (under 40 words) describing the visual. "
            "No preamble, no quotes.",
        ),
        ("human", "Platform: {platform}\nPost:\n{post}\n\nImage prompt:"),
    ]
)