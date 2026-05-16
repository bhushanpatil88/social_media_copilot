"""Assemble the LangGraph workflow."""
from __future__ import annotations

from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from src.graph.state import CopilotState
from src.graph.nodes import (
    route_query,
    rag_node,
    web_node,
    build_context,
    generate_post,
    generate_hashtags,
    generate_image_prompt,
    quality_check,
    should_refine,
    refine_post,
    route_decider,
)


def _build_graph():
    g = StateGraph(CopilotState)

    # Nodes
    g.add_node("router", route_query)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("build_context", build_context)
    g.add_node("generate_post", generate_post)
    g.add_node("quality_check", quality_check)
    g.add_node("refine", refine_post)
    g.add_node("hashtags", generate_hashtags)
    g.add_node("image_prompt", generate_image_prompt)

    # Edges
    g.add_edge(START, "router")

    # Router → rag, web, or both (we model 'both' as web → rag)
    g.add_conditional_edges(
        "router",
        route_decider,
        {
            "rag": "rag",
            "both": "web",   # web first, then rag
        },
    )
    g.add_edge("web", "rag")
    g.add_edge("rag", "build_context")
    g.add_edge("build_context", "generate_post")
    g.add_edge("generate_post", "quality_check")

    # Refinement loop
    g.add_conditional_edges(
        "quality_check",
        should_refine,
        {
            "refine": "refine",
            "finalize": "hashtags",
        },
    )
    g.add_edge("refine", "quality_check")

    g.add_edge("hashtags", "image_prompt")
    g.add_edge("image_prompt", END)

    return g.compile()


@lru_cache(maxsize=1)
def get_workflow():
    """Return a singleton compiled LangGraph workflow."""
    return _build_graph()


def run(inputs: dict) -> dict:
    """Run the workflow end-to-end and return the final state dict."""
    workflow = get_workflow()
    return workflow.invoke(inputs)
