# LangChain & LangGraph — Hands-On Learning Lab

> A practical, end-to-end guide to building LLM applications with **[LangChain](https://python.langchain.com/)** and **[LangGraph](https://langchain-ai.github.io/langgraph/)** — from simple chains to production-ready multi-agent systems.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-green?logo=chainlink)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange)](https://langchain-ai.github.io/langgraph/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

This repo is a structured, self-contained learning lab that teaches **LLM application development** through:

- **Concept docs** — clear explanations with diagrams and tables
- **Runnable snippets** — focused, copy-paste-ready Python examples
- **A real project** — a fully working local RAG chatbot you can run today

Whether you're a **developer exploring AI frameworks**, a **data scientist adding LLMs to your stack**, or a **recruiter evaluating AI engineering skills** — every file here is meant to be readable and runnable.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM Framework | [LangChain](https://python.langchain.com/) | Chains, prompts, retrievers, memory |
| Agent Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) | Stateful graphs, multi-agent systems |
| Local LLM | [LM Studio](https://lmstudio.ai/) + `openai/gpt-oss-20b` | Private, offline inference |
| Cloud LLM Fallback | [Google Gemini 2.0 Flash](https://ai.google.dev/) | Fast, capable cloud alternative |
| Vector Store | [Chroma](https://docs.trychroma.com/) | Local semantic search |
| Embeddings | [Google gemini-embedding-001](https://ai.google.dev/) | High-quality text embeddings |
| Package Manager | [uv](https://docs.astral.sh/uv/) | Fast Python dependency management |

---

## Repository Structure

```
.
├── docs/                          # Concept explanations (read these first)
│   ├── langchain/
│   │   ├── 01-lcel-basics.md      # LangChain Expression Language
│   │   └── 02-rag.md              # Retrieval-Augmented Generation
│   └── langgraph/
│       ├── 01-state-graphs.md     # State machines & graph fundamentals
│       ├── 02-checkpointing.md    # Persistence & memory across sessions
│       └── 03-multi-agent.md      # Supervisor, swarm, pipeline patterns
│
├── snippets/                      # Short, runnable code examples
│   ├── langchain/
│   │   ├── 01-lcel-hello.py       # Simplest chain: prompt | model | parser
│   │   ├── 02-lcel-stream.py      # Streaming tokens in real time
│   │   ├── 03-chat-history.py     # Multi-turn memory with RunnableWithMessageHistory
│   │   ├── 04-rag-basic.py        # End-to-end RAG pipeline
│   │   └── 05-rag-with-sources.py # RAG with document citations
│   └── langgraph/
│       ├── 01-minimal-graph.py    # Smallest possible LangGraph
│       ├── 02-conditional-edges.py# Routing between nodes
│       ├── 03-llm-agent.py        # ReAct agent with tool use
│       ├── 04-checkpointing.py    # Multi-turn chat with MemorySaver
│       ├── 05-human-in-loop.py    # Pause & approve before executing
│       ├── 06-supervisor-agent.py # Orchestrator → worker agents
│       └── 07-agent-handoff.py    # Peer-to-peer agent swarm
│
└── projects/
    └── local-doc-chat/            # Full RAG chatbot (see below)
```

---

## Covered Concepts

### LangChain

| Topic | Doc | Snippet |
|---|---|---|
| LCEL — pipe-based chain composition | [01-lcel-basics.md](docs/langchain/01-lcel-basics.md) | [01-lcel-hello.py](snippets/langchain/01-lcel-hello.py) |
| Streaming tokens | — | [02-lcel-stream.py](snippets/langchain/02-lcel-stream.py) |
| Multi-turn chat memory | — | [03-chat-history.py](snippets/langchain/03-chat-history.py) |
| Retrieval-Augmented Generation (RAG) | [02-rag.md](docs/langchain/02-rag.md) | [04-rag-basic.py](snippets/langchain/04-rag-basic.py) |
| RAG with source citations | — | [05-rag-with-sources.py](snippets/langchain/05-rag-with-sources.py) |
| Prompts & output parsers | coming soon | — |
| Tool-using agents (LCEL) | coming soon | — |

### LangGraph

| Topic | Doc | Snippet |
|---|---|---|
| State graphs & TypedDict state | [01-state-graphs.md](docs/langgraph/01-state-graphs.md) | [01-minimal-graph.py](snippets/langgraph/01-minimal-graph.py) |
| Conditional edges & routing | — | [02-conditional-edges.py](snippets/langgraph/02-conditional-edges.py) |
| ReAct agent with tools | — | [03-llm-agent.py](snippets/langgraph/03-llm-agent.py) |
| Checkpointing & session memory | [02-checkpointing.md](docs/langgraph/02-checkpointing.md) | [04-checkpointing.py](snippets/langgraph/04-checkpointing.py) |
| Human-in-the-loop approval | — | [05-human-in-loop.py](snippets/langgraph/05-human-in-loop.py) |
| Multi-agent — supervisor pattern | [03-multi-agent.md](docs/langgraph/03-multi-agent.md) | [06-supervisor-agent.py](snippets/langgraph/06-supervisor-agent.py) |
| Multi-agent — handoff / swarm | — | [07-agent-handoff.py](snippets/langgraph/07-agent-handoff.py) |
| Subgraphs & nested agents | coming soon | — |
| Streaming intermediate steps | coming soon | — |

---

## Featured Project — Local Private Doc Chat

> **Ask questions about your own documents — runs 100% on your machine, no data sent to the cloud.**

Located in [`projects/local-doc-chat/`](projects/local-doc-chat/)

### How It Works

```
Your files (my-docs/)
      ↓
  [Indexer] chunk → embed (Google gemini-embedding-001) → Chroma (local)
      ↓
  User question → similarity search → top-3 chunks
      ↓
  [LangGraph RAG node] context + question → LM Studio / Gemini → answer + sources
      ↓
  Multi-turn memory via MemorySaver checkpointing
```

### Key Engineering Decisions

| Decision | Choice | Why |
|---|---|---|
| Custom LLM wrapper | [`lm_studio_chat.py`](projects/local-doc-chat/lm_studio_chat.py) | Teaches `BaseChatModel` integration for any HTTP LLM API |
| Local-first, cloud fallback | LM Studio → OpenAI → Gemini | Works offline; gracefully degrades to cloud if needed |
| Local embeddings | `gemini-embedding-001` via API | No PyTorch model load on startup — instant cold start |
| Vector store | Chroma (persisted to disk) | No server to run, rebuilds automatically on `--reindex` |
| Session memory | `MemorySaver` → upgradeable to `SqliteSaver` | Conversation history survives across questions |

### Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/shabul/teach-langchain-and-langgraph
cd teach-langchain-and-langgraph/projects/local-doc-chat

# 2. Add your files
cp your-notes.md my-docs/

# 3. Add a Google API key to .env (free at aistudio.google.com/apikey)
echo "GOOGLE_API_KEY=AIza..." >> .env

# 4. Start (installs everything automatically via uv)
./start.sh --reindex

# 5. Ask away
You: Summarise my notes on the Q3 project
Assistant: ...
```

> **No LM Studio?** Add `GOOGLE_API_KEY` or `OPENAI_API_KEY` to `.env` — the app detects it automatically and falls back to the cloud model.

---

## Quick Setup (Snippets)

To run individual snippets:

```bash
pip install langchain langgraph langchain-openai langchain-chroma \
            langchain-community langchain-google-genai python-dotenv

cp .env.example .env   # add your API key
python snippets/langchain/01-lcel-hello.py
```

---

## Skills Demonstrated

For recruiters and engineers evaluating this repo:

- **LLM application architecture** — chaining, retrieval, agents, memory
- **LangChain LCEL** — composable, streamable, type-safe pipelines
- **LangGraph** — stateful graphs, conditional routing, multi-agent orchestration
- **RAG systems** — document ingestion, chunking, embedding, vector search, citation
- **Custom integrations** — building a `BaseChatModel` wrapper for any LLM HTTP API
- **Local-first AI** — private inference with LM Studio, offline-capable
- **Developer tooling** — `uv` for dependency management, `.env`-based config, one-command setup

---

## Learning Path

New to LangChain and LangGraph? Follow this order:

1. Read [`docs/langchain/01-lcel-basics.md`](docs/langchain/01-lcel-basics.md) — understand the pipe model
2. Run [`snippets/langchain/01-lcel-hello.py`](snippets/langchain/01-lcel-hello.py) — first working chain
3. Read [`docs/langgraph/01-state-graphs.md`](docs/langgraph/01-state-graphs.md) — understand stateful graphs
4. Run [`snippets/langgraph/01-minimal-graph.py`](snippets/langgraph/01-minimal-graph.py) — first graph
5. Read [`docs/langchain/02-rag.md`](docs/langchain/02-rag.md) — understand retrieval
6. Run [`snippets/langchain/04-rag-basic.py`](snippets/langchain/04-rag-basic.py) — first RAG chain
7. Read [`docs/langgraph/02-checkpointing.md`](docs/langgraph/02-checkpointing.md) — persistence
8. Read [`docs/langgraph/03-multi-agent.md`](docs/langgraph/03-multi-agent.md) — multi-agent patterns
9. Run the [local-doc-chat project](projects/local-doc-chat/) — everything together

---

## Useful Links

| Resource | URL |
|---|---|
| LangChain docs | https://python.langchain.com/docs/ |
| LangGraph docs | https://langchain-ai.github.io/langgraph/ |
| LangChain LCEL | https://python.langchain.com/docs/concepts/lcel/ |
| Custom chat model guide | https://python.langchain.com/docs/how_to/custom_chat_model/ |
| RAG tutorial | https://python.langchain.com/docs/tutorials/rag/ |
| LangGraph persistence | https://langchain-ai.github.io/langgraph/concepts/persistence/ |
| Google AI Studio (free API key) | https://aistudio.google.com/apikey |
| LM Studio | https://lmstudio.ai/ |
| uv — fast Python package manager | https://docs.astral.sh/uv/ |

---

## Author

Built and maintained by **[Shabul](https://twitter.com/iamshabul)** — Data Scientist at Amazon, working on retail analytics, LLM-based attribute extraction, and return reduction.

- Twitter / X: [@iamshabul](https://twitter.com/iamshabul)
- LinkedIn: [linkedin.com/in/shabul](https://linkedin.com/in/shabul)
