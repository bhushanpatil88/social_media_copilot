"""Streamlit frontend for the Social Media Copilot."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make project root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Allow user to set keys in the sidebar BEFORE we import anything that
# reads them. We do this by setting env vars first, then importing.
st.set_page_config(
    page_title="Social Media Copilot",
    page_icon="✍️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — credentials and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Keys")
    st.caption("Your keys never leave this session.")

    groq_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com/keys",
    )
    tavily_key = st.text_input(
        "Tavily API Key (optional)",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Enables web search for fresh topics. Get a free key at tavily.com",
    )

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

    st.divider()
    st.header("📚 Knowledge base")
    st.caption(
        "Seed the vector DB with example posts so the model has style "
        "references to draw on."
    )

    if st.button("Ingest sample posts"):
        try:
            from data.ingest import main as ingest_main
            ingest_main()
            st.success("Sample posts ingested!")
        except Exception as exc:
            st.error(f"Ingest failed: {exc}")

    try:
        from src.rag.vector_store import collection_size
        st.metric("Posts in DB", collection_size())
    except Exception:
        st.metric("Posts in DB", "n/a")

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("✍️ Social Media Copilot")
st.caption(
    "LangGraph + ChatGroq + RAG + Web search. Generate platform-aware posts "
    "with hashtags, an image prompt, and an automatic quality check."
)

if not os.getenv("GROQ_API_KEY"):
    st.warning("Add your Groq API key in the sidebar to start.")
    st.stop()

col_left, col_right = st.columns([1, 1])

with col_left:
    platform = st.selectbox(
        "Platform",
        ["LinkedIn", "Twitter", "Instagram", "Facebook"],
        index=0,
    )
    topic = st.text_area(
        "Topic",
        placeholder="e.g. Why async-first teams ship faster",
        height=100,
    )
    audience = st.text_input(
        "Target audience",
        value="general professional audience",
    )

with col_right:
    tone = st.selectbox(
        "Tone",
        ["Professional", "Casual", "Witty", "Inspirational", "Educational"],
        index=1,
    )
    humor_level = st.slider("Humor level", 0, 10, 4)
    length = st.radio("Length", ["short", "medium", "long"], index=1, horizontal=True)
    cta = st.text_input(
        "Call-to-action",
        value="encourage thoughtful replies",
    )

generate = st.button("🚀 Generate post", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run the workflow
# ---------------------------------------------------------------------------
if generate:
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    with st.spinner("Routing → retrieving → drafting → editing…"):
        try:
            from src.graph.workflow import run

            inputs = {
                "topic": topic.strip(),
                "platform": platform,
                "tone": tone,
                "humor_level": humor_level,
                "length": length,
                "audience": audience,
                "cta": cta,
            }
            result = run(inputs)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()

    # ----- Post output -----
    st.subheader(f"📝 Your {platform} post")
    post = result.get("post", "")
    st.text_area("Post", value=post, height=220, key="post_out")

    char_count = len(post)
    from src.platforms.constraints import get_platform
    spec = get_platform(platform)
    pct = min(1.0, char_count / spec.max_chars)
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Characters", f"{char_count} / {spec.max_chars}")
    cc2.metric(
        "Quality score",
        f"{result.get('quality_score', 0):.2f}",
        help=result.get("quality_feedback", ""),
    )
    cc3.metric("Refinement passes", result.get("iterations", 1))
    st.progress(pct, text="Length usage")

    # ----- Hashtags -----
    hashtags = result.get("hashtags", [])
    if hashtags:
        st.subheader("🏷️ Hashtags")
        st.code(" ".join(hashtags), language=None)

    # ----- Image prompt -----
    img_prompt = result.get("image_prompt", "")
    if img_prompt:
        with st.expander("🎨 Suggested image prompt"):
            st.write(img_prompt)
            st.caption(
                "Drop this into your favourite image generator "
                "(DALL·E, Midjourney, Stable Diffusion, etc.)."
            )

    # ----- Sources -----
    rag_docs = result.get("rag_docs", []) or []
    web_results = result.get("web_results", []) or []
    if rag_docs or web_results:
        with st.expander(
            f"🔍 Context used "
            f"({len(rag_docs)} RAG examples · {len(web_results)} web results)"
        ):
            if web_results:
                st.markdown("**Web sources**")
                for r in web_results:
                    st.markdown(f"- [{r['title']}]({r['url']})")
            if rag_docs:
                st.markdown("**Style examples from past posts**")
                for d in rag_docs:
                    meta = d.get("metadata", {}) or {}
                    st.markdown(
                        f"_{meta.get('platform','?')} · {meta.get('topic','?')}_"
                    )
                    st.markdown(
                        f"> {d['text'][:280]}{'…' if len(d['text']) > 280 else ''}"
                    )

    # ----- Editor feedback -----
    if result.get("quality_feedback"):
        with st.expander("🧐 Editor feedback"):
            st.info(result["quality_feedback"])
