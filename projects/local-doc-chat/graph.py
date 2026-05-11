"""
LangGraph RAG agent: retrieve relevant chunks → call LM Studio → return answer + sources.

State accumulates message history across turns (MemorySaver checkpointing).
A simple trim keeps total history under the 11k-token window.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from lm_studio_chat import ChatLMStudio


def _text(content) -> str:
    """Normalise LLM response content to a plain string.

    Gemini (and some other models) return content as a list of typed parts
    e.g. [{"type": "text", "text": "..."}] instead of a plain str.
    """
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions about a local codebase or document set.
Use ONLY the context below to answer. If the answer is not in the context, say so clearly.
Be concise. Do not repeat the context back verbatim.

--- CONTEXT ---
{context}
--- END CONTEXT ---"""

MAX_HISTORY_MESSAGES = 10  # keep conversation within 11k token budget


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _trim(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep first message + last N to prevent context overflow."""
    if len(messages) <= MAX_HISTORY_MESSAGES:
        return messages
    return [messages[0]] + messages[-(MAX_HISTORY_MESSAGES - 1):]


def make_graph(retriever: VectorStoreRetriever, llm: ChatLMStudio):
    def rag(state: State) -> dict:
        question = _text(state["messages"][-1].content)

        # Retrieve relevant chunks
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        sources = sorted({d.metadata.get("source", "unknown") for d in docs})

        # Build prompt: inject retrieved context into system message
        system = SystemMessage(content=SYSTEM_TEMPLATE.format(context=context))

        # Trim history before sending to stay within the 11k context window
        trimmed = _trim(state["messages"])
        response = llm.invoke([system] + trimmed)

        # Normalise to str (Gemini can return a list of content parts)
        answer = _text(response.content)
        if sources:
            answer += "\n\n> **Sources:** " + ", ".join(f"`{s}`" for s in sources)

        return {"messages": [AIMessage(content=answer)]}

    graph = StateGraph(State)
    graph.add_node("rag", rag)
    graph.add_edge(START, "rag")
    graph.add_edge("rag", END)

    return graph.compile(checkpointer=MemorySaver())
