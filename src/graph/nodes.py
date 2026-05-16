"""Node functions for the LangGraph workflow."""
from __future__ import annotations

from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser

from config.settings import settings
from src.graph.state import CopilotState
from src.llm.groq_llm import get_llm
from src.platforms.constraints import get_platform
from src.prompts.templates import (
    ROUTER_PROMPT,
    POST_GENERATION_PROMPT,
    REFINE_PROMPT,
    HASHTAG_PROMPT,
    QUALITY_PROMPT,
    IMAGE_PROMPT_TEMPLATE,
    LENGTH_GUIDANCE,
)
from src.rag.retriever import retrieve, format_context
from src.tools.web_search import web_search, format_web_results
from src.utils.helpers import (
    parse_hashtags,
    parse_quality_response,
    truncate_to_limit,
)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_query(state: CopilotState) -> CopilotState:
    """Decide whether to use RAG, web search, or both."""
    llm = get_llm(temperature=0.0)
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    try:
        decision = chain.invoke({"topic": state["topic"]}).strip().lower()
    except Exception as exc:
        print(f"[route_query] {exc}")
        decision = "rag"

    if "web" in decision:
        route = "both"
    else:
        route = "rag"
    return {"route": route, "iterations": 0}


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def rag_node(state: CopilotState) -> CopilotState:
    docs = retrieve(state["topic"], platform=state.get("platform"))
    serialised = [
        {"text": d.page_content, "metadata": d.metadata or {}} for d in docs
    ]
    return {"rag_docs": serialised}


def web_node(state: CopilotState) -> CopilotState:
    results = web_search(state["topic"], max_results=4)
    sources = [r["url"] for r in results if r.get("url")]
    return {"web_results": results, "sources": sources}


def build_context(state: CopilotState) -> CopilotState:
    """Merge RAG examples + web findings into a single context blob."""
    parts = []

    web_results = state.get("web_results") or []
    if web_results:
        parts.append("=== Recent web information ===")
        parts.append(format_web_results(web_results))

    rag_docs = state.get("rag_docs") or []
    if rag_docs:
        from langchain_core.documents import Document
        docs = [Document(page_content=d["text"], metadata=d.get("metadata", {}))
                for d in rag_docs]
        parts.append("=== Style examples from past posts ===")
        parts.append(format_context(docs))

    if not parts:
        parts.append("(no external context available — rely on general knowledge)")

    return {"context": "\n\n".join(parts)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _length_for(spec, length: str):
    """Return (target_min, target_max, length_guidance) for a length."""
    target_min, target_max = spec.target_range(length)
    guidance = LENGTH_GUIDANCE.get(length.lower(), LENGTH_GUIDANCE["medium"])
    return target_min, target_max, guidance


# ---------------------------------------------------------------------------
# Post generation
# ---------------------------------------------------------------------------
def generate_post(state: CopilotState) -> CopilotState:
    spec = get_platform(state["platform"])
    length = state.get("length", "medium")
    target_min, target_max, length_guidance = _length_for(spec, length)

    # Translate humor 0-10 onto a sampling temperature 0.2..1.0
    temp = 0.2 + (state.get("humor_level", 5) / 10.0) * 0.8
    llm = get_llm(temperature=round(temp, 2))

    chain = POST_GENERATION_PROMPT | llm | StrOutputParser()
    post = chain.invoke({
        "platform": spec.name,
        "style_notes": spec.style_for(length),
        "max_chars": spec.max_chars,
        "target_min": target_min,
        "target_max": target_max,
        "length": length,
        "length_guidance": length_guidance,
        "tone": state.get("tone", "Casual"),
        "humor_level": state.get("humor_level", 5),
        "audience": state.get("audience", "general audience"),
        "cta": state.get("cta", "encourage engagement"),
        "topic": state["topic"],
        "context": state.get("context", ""),
    }).strip()

    post = post.strip("`").strip()
    if post.startswith('"') and post.endswith('"'):
        post = post[1:-1]
    post = truncate_to_limit(post, spec.max_chars)

    return {"post": post}


# ---------------------------------------------------------------------------
# Hashtags
# ---------------------------------------------------------------------------
def generate_hashtags(state: CopilotState) -> CopilotState:
    spec = get_platform(state["platform"])
    if spec.hashtag_count == 0:
        return {"hashtags": []}

    llm = get_llm(temperature=0.4)
    chain = HASHTAG_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "platform": spec.name,
        "hashtag_count": spec.hashtag_count,
        "topic": state["topic"],
        "post": state.get("post", ""),
    })
    tags = parse_hashtags(raw, max_count=spec.max_hashtags)
    return {"hashtags": tags}


# ---------------------------------------------------------------------------
# Image prompt
# ---------------------------------------------------------------------------
def generate_image_prompt(state: CopilotState) -> CopilotState:
    llm = get_llm(temperature=0.7)
    chain = IMAGE_PROMPT_TEMPLATE | llm | StrOutputParser()
    try:
        prompt = chain.invoke({
            "platform": state["platform"],
            "post": state.get("post", ""),
        }).strip().strip('"').strip("`")
    except Exception:
        prompt = ""
    return {"image_prompt": prompt}


# ---------------------------------------------------------------------------
# Quality check + refinement loop
# ---------------------------------------------------------------------------
def quality_check(state: CopilotState) -> CopilotState:
    spec = get_platform(state["platform"])
    length = state.get("length", "medium")
    target_min, target_max, _ = _length_for(spec, length)
    post = state.get("post", "")

    llm = get_llm(temperature=0.0)
    chain = QUALITY_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({
        "tone": state.get("tone", "Casual"),
        "humor_level": state.get("humor_level", 5),
        "platform": spec.name,
        "max_chars": spec.max_chars,
        "target_min": target_min,
        "target_max": target_max,
        "actual_chars": len(post),
        "post": post,
    })
    score, feedback = parse_quality_response(raw)

    # Programmatic length penalty so we don't trust the judge LLM blindly.
    if len(post) < target_min:
        ratio = len(post) / max(target_min, 1)
        score = min(score, 0.4 + 0.3 * ratio)
        if not feedback:
            feedback = (
                f"Post is too short ({len(post)} chars vs target "
                f"{target_min}-{target_max}). Expand with a concrete example "
                f"or an extra layer of reasoning."
            )

    iterations = state.get("iterations", 0) + 1
    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "iterations": iterations,
    }


def should_refine(state: CopilotState) -> str:
    if state.get("quality_score", 0.0) >= settings.quality_threshold:
        return "finalize"
    if state.get("iterations", 0) >= settings.max_refinement_iterations:
        return "finalize"
    return "refine"


def refine_post(state: CopilotState) -> CopilotState:
    """Re-generate with the previous attempt visible so the model can EXPAND
    instead of regenerating at the same length."""
    spec = get_platform(state["platform"])
    length = state.get("length", "medium")
    target_min, target_max, length_guidance = _length_for(spec, length)
    previous_post = state.get("post", "")
    previous_chars = len(previous_post)
    shortfall = max(0, target_min - previous_chars)

    # Sampling temperature mirrors generate_post
    temp = 0.2 + (state.get("humor_level", 5) / 10.0) * 0.8
    llm = get_llm(temperature=round(temp, 2))

    chain = REFINE_PROMPT | llm | StrOutputParser()
    revised = chain.invoke({
        "platform": spec.name,
        "style_notes": spec.style_for(length),
        "max_chars": spec.max_chars,
        "target_min": target_min,
        "target_max": target_max,
        "length": length,
        "length_guidance": length_guidance,
        "tone": state.get("tone", "Casual"),
        "humor_level": state.get("humor_level", 5),
        "audience": state.get("audience", "general audience"),
        "cta": state.get("cta", "encourage engagement"),
        "topic": state["topic"],
        "context": state.get("context", ""),
        "previous_post": previous_post,
        "previous_chars": previous_chars,
        "shortfall": shortfall,
        "feedback": state.get("quality_feedback", "Expand the post."),
    }).strip()

    revised = revised.strip("`").strip()
    if revised.startswith('"') and revised.endswith('"'):
        revised = revised[1:-1]
    revised = truncate_to_limit(revised, spec.max_chars)

    return {"post": revised}


# ---------------------------------------------------------------------------
# Routing helpers (conditional edges)
# ---------------------------------------------------------------------------
def route_decider(state: CopilotState) -> str:
    return state.get("route", "rag")