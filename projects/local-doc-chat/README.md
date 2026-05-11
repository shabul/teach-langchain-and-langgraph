# Local Private Doc/Code Chat

Ask questions about any folder of files using a locally hosted LLM. Nothing leaves your machine.

**Stack:** LM Studio → custom `BaseChatModel` → LangChain RAG → LangGraph + checkpointing → local Chroma + HuggingFace embeddings

## Setup

```bash
pip install langchain langgraph langchain-chroma langchain-huggingface \
            langchain-community sentence-transformers requests
```

Make sure LM Studio is running with a model loaded and the REST API server active (`localhost:1234`).

## Usage

```bash
# First run — index files, then chat
python main.py --dir ../..  --reindex

# Resume an existing session (no re-embedding)
python main.py --dir ../..

# Custom session name (isolates conversation history)
python main.py --dir ~/my-notes --session notes --reindex

# Adjust retrieval count and model
python main.py --dir /path/to/code -k 5 --model openai/gpt-oss-20b
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
