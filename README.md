# LangChain & LangGraph — Learning Lab

A hands-on repo for learning LangChain and LangGraph through docs, examples, and runnable code snippets.

## Structure

```
.
├── docs/                  # Concept explanations and notes
│   ├── langchain/
│   └── langgraph/
├── snippets/              # Short, focused code examples
│   ├── langchain/
│   └── langgraph/
└── projects/              # End-to-end mini projects
```

## Topics

### LangChain
- [x] Chains — LCEL basics (`docs/langchain/01-lcel-basics.md`)
- [x] Streaming (`snippets/langchain/02-lcel-stream.py`)
- [x] Memory & chat history (`snippets/langchain/03-chat-history.py`)
- [x] Retrieval — RAG (`docs/langchain/02-rag.md`)
- [x] RAG with source citations (`snippets/langchain/05-rag-with-sources.py`)
- [ ] Prompts & prompt templates
- [ ] Output parsers (structured)
- [ ] Tools & agents (LCEL style)

### LangGraph
- [x] State graphs & state machines (`docs/langgraph/01-state-graphs.md`)
- [x] Conditional edges (`snippets/langgraph/02-conditional-edges.py`)
- [x] ReAct agent with tools (`snippets/langgraph/03-llm-agent.py`)
- [x] Checkpointing & persistence (`docs/langgraph/02-checkpointing.md`)
- [x] Human-in-the-loop (`snippets/langgraph/05-human-in-loop.py`)
- [x] Multi-agent — supervisor pattern (`docs/langgraph/03-multi-agent.md`)
- [x] Multi-agent — handoff / swarm (`snippets/langgraph/07-agent-handoff.py`)
- [ ] Subgraphs (nested agents)
- [ ] Streaming intermediate steps

## Setup

```bash
pip install langchain langgraph langchain-openai langchain-chroma langchain-community python-dotenv
```

Copy `.env.example` to `.env` and fill in your API keys.
