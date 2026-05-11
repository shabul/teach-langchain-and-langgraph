# Local Private Doc/Code Chat

Ask questions about any folder of files using a locally hosted LLM. Nothing leaves your machine.

**Stack:** LM Studio → custom `BaseChatModel` → LangChain RAG → LangGraph + checkpointing → local Chroma + HuggingFace embeddings

## Setup & Usage

`start.sh` handles everything: installs `uv` if missing, creates the virtualenv, syncs deps, checks LM Studio is up, then launches the chat.

```bash
# First run — index the repo and start chatting
./start.sh --dir ../.. --reindex

# Resume (no re-indexing)
./start.sh --dir ../..

# Custom session name (isolates conversation history)
./start.sh --dir ~/my-notes --session notes --reindex

# Adjust retrieval count or model
./start.sh --dir /path/to/code -k 5 --model openai/gpt-oss-20b
```

Make sure LM Studio is running with a model loaded and the REST API server active on `localhost:1234`.

### Manual setup (if you prefer)

```bash
uv sync          # create .venv and install deps
uv run python main.py --dir ../.. --reindex
```

## What each file does

| File | Role |
|---|---|
| `lm_studio_chat.py` | Custom `BaseChatModel` that converts LangChain messages → LM Studio's `{system_prompt, input}` format |
| `indexer.py` | Glob files → split (500-char chunks) → embed with `all-MiniLM-L6-v2` → persist Chroma |
| `graph.py` | LangGraph: retrieve chunks → inject as context → call LLM → append sources |
| `main.py` | CLI loop with argparse; wires everything together |

## Concepts demonstrated

- **Custom LLM integration** — `BaseChatModel` subclass, `_generate`, `ChatResult`
- **Local embeddings** — `HuggingFaceEmbeddings` (no API key needed)
- **RAG pipeline** — Chroma retriever → context injection → generation
- **LangGraph checkpointing** — `MemorySaver` persists conversation across turns
- **Context budget management** — history trimming to stay within 11k token limit

## Swap to production persistence

The default `MemorySaver` resets when the process exits. For persistent history across restarts:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("chat_history.db")
```
