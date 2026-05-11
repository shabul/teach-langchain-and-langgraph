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
- [ ] Chains — LCEL basics
- [ ] Prompts & prompt templates
- [ ] LLM & Chat model wrappers
- [ ] Output parsers
- [ ] Memory & history
- [ ] Retrieval (RAG)
- [ ] Tools & agents

### LangGraph
- [ ] State graphs & state machines
- [ ] Nodes and edges
- [ ] Conditional edges
- [ ] Human-in-the-loop
- [ ] Persistence & checkpointing
- [ ] Multi-agent systems

## Setup

```bash
pip install langchain langgraph langchain-openai python-dotenv
```

Copy `.env.example` to `.env` and fill in your API keys.
