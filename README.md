# ✍️ Social Media Copilot

A production-style **LangGraph** application that generates platform-aware
social media posts using **ChatGroq**, a **RAG** layer over a Chroma vector
store of example posts, and a **web search** tool for fresh topics — wrapped
in a **Streamlit** frontend.

## Features

- 🎯 **Platform-aware generation** — Instagram, Twitter, LinkedIn, Facebook,
  each with its own character limits, hashtag conventions, and style guide.
- 🧠 **LangGraph workflow** with a router, RAG retrieval, web search, post
  generation, hashtag generation, an image-prompt suggestion, and an LLM-as-judge
  quality check that loops back to refine weak posts.
- 🔍 **RAG** over a Chroma vector DB seeded with sample posts. Filters by
  platform first, falls back to cross-platform examples.
- 🌐 **Web search** (Tavily) automatically activates when the topic looks
  recent or niche — controlled by a routing LLM.
- 🎚️ **Tunable controls** — tone, humor (0–10), length, audience, call-to-action.
- 🏷️ **Hashtag generator** with platform-specific counts.
- 🎨 **Image prompt** generator for downstream use with DALL·E / Midjourney / SD.
- 🧐 **Self-correcting** — quality score below threshold triggers a refinement pass.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       Streamlit UI                          │
│   platform · topic · tone · humor · length · audience       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │      LangGraph workflow      │
              └──────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   [router LLM]                                     │
        │                                           │
   web?─┴─rag                                       │
        │                                           │
        ▼                                           ▼
  [Tavily search] ─────────────► [RAG retriever (Chroma)]
                                          │
                                          ▼
                                  [build_context]
                                          │
                                          ▼
                                  [generate_post]  ◄──┐
                                          │           │
                                          ▼           │
                                  [quality_check] ────┘  (refinement loop,
                                          │              max 2 iterations)
                                          ▼
                                  [hashtags] → [image_prompt] → END
```

## Project layout

```
social_media_copilot/
├── app.py                         # Streamlit frontend
├── requirements.txt
├── .env
├── README.md
├── config/settings.py             # Centralised config
├── src/
│   ├── graph/
│   │   ├── state.py               # CopilotState (TypedDict)
│   │   ├── nodes.py               # All node functions
│   │   └── workflow.py            # Graph assembly
│   ├── llm/groq_llm.py            # ChatGroq factory
│   ├── rag/
│   │   ├── vector_store.py        # Chroma + HF embeddings
│   │   └── retriever.py           # Platform-aware retrieval
│   ├── tools/web_search.py        # Tavily wrapper
│   ├── prompts/templates.py       # All prompt templates
│   ├── platforms/constraints.py   # Per-platform rules
│   └── utils/helpers.py
└── data/
    ├── sample_posts.json          # Seed corpus
    └── ingest.py                  # Loader script
```

## Setup

### 1. Install

```bash
git clone <your-repo-url>
cd social_media_copilot
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

fill in your keys:

- `GROQ_API_KEY` — required. Free at [console.groq.com/keys](https://console.groq.com/keys).
- `TAVILY_API_KEY` — optional. Free 1k searches/month at [tavily.com](https://tavily.com).
  Without it, the web-search node simply returns no results and the model
  falls back to RAG + general knowledge.

### 3. Seed the vector DB

```bash
python data/ingest.py
```

This embeds the example posts in `data/sample_posts.json` into a local
Chroma store at `./chroma_db`. First run downloads the embedding model
(~80 MB).

### 4. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

## Adding your own training data

Drop more posts into `data/sample_posts.json` (same schema) and re-run
`python data/ingest.py`. Or call `add_posts()` from `src.rag.vector_store`
directly with your own loader (e.g. from Postgres, S3, an API).

```python
from src.rag.vector_store import add_posts

add_posts([
    {
        "platform": "LinkedIn",
        "topic": "your topic",
        "author": "someone",
        "text": "the post text…",
    },
])
```

## Configuration knobs

All in `config/settings.py` (overridable via env vars):

| Setting | Default | Purpose |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | The Groq model used everywhere |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF embedding model |
| `RAG_TOP_K` | `4` | RAG examples per query |
| `quality_threshold` | `0.75` | Below this → refinement loop |
| `max_refinement_iterations` | `2` | Cap on refinement passes |

## Extending

- **More platforms** — add an entry to `PLATFORMS` in `src/platforms/constraints.py`.
- **More tones** — just edit the dropdown in `app.py`; no code change needed.
- **Image generation** — the `image_prompt` field is ready to feed into any
  T2I API. Add a node in `src/graph/nodes.py` that calls e.g. DALL·E and
  attaches the URL to state.
- **Scheduling** — bolt on the Buffer or Hootsuite API in a final node.
- **Analytics** — log every generation into a SQLite table keyed by
  topic/platform/score for offline analysis.
