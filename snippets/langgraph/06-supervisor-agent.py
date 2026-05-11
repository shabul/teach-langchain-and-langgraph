"""
Supervisor multi-agent pattern:
  - supervisor LLM decides which worker to call next
  - researcher: fetches information (mocked)
  - writer: drafts a response based on research
  - supervisor says FINISH when done

Flow:
  START → supervisor → researcher ─┐
              ↑                     │
              └─────── writer  ◄────┘
              ↓
             END
"""
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

Members = Literal["researcher", "writer"]
Next = Literal["researcher", "writer", "FINISH"]


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str


# --- Supervisor ---
SUPERVISOR_PROMPT = (
    "You are a supervisor managing a researcher and a writer.\n"
    "Given the conversation, decide who should act next.\n"
    "Once the writer has produced a final answer, respond with FINISH.\n"
    "Reply with ONLY one word: researcher, writer, or FINISH."
)

def supervisor(state: State) -> dict:
    messages = [SystemMessage(SUPERVISOR_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    next_node = response.content.strip()
    if next_node not in ("researcher", "writer", "FINISH"):
        next_node = "FINISH"
    return {"next": next_node}


# --- Workers ---
def researcher(state: State) -> dict:
    # In a real system this would call a search tool or RAG chain
    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    result = f"[Research] Found relevant info about: '{last_human.content}'. Key facts: it involves LangGraph multi-agent patterns."
    print(f"  [researcher] {result}")
    return {"messages": [AIMessage(result, name="researcher")]}


def writer(state: State) -> dict:
    research = next(
        (m.content for m in reversed(state["messages"]) if getattr(m, "name", None) == "researcher"),
        "No research available.",
    )
    draft = f"[Draft] Based on the research, here is a concise summary: {research.replace('[Research] ', '')}"
    print(f"  [writer] {draft}")
    return {"messages": [AIMessage(draft, name="writer")]}


def route(state: State) -> str:
    return state["next"] if state["next"] != "FINISH" else END


# --- Graph ---
graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")

app = graph.compile()

print("=== Multi-agent run ===")
result = app.invoke({
    "messages": [HumanMessage("Explain how the supervisor pattern works in LangGraph.")],
    "next": "",
})

print("\n=== Final messages ===")
for m in result["messages"]:
    name = getattr(m, "name", type(m).__name__)
    print(f"[{name}] {m.content[:120]}")
