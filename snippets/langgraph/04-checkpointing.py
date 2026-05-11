"""
Checkpointing — persist graph state across multiple invocations on the same thread.

The graph counts how many times the user has sent a message and remembers it
across calls, even though each .invoke() is a separate Python call.
"""
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# MemorySaver stores state in-memory; swap for SqliteSaver to persist to disk
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# All calls with the same thread_id share state
thread = {"configurable": {"thread_id": "alice"}}

print("--- Turn 1 ---")
r1 = app.invoke({"messages": [HumanMessage("Hi! My name is Alice.")]}, config=thread)
print(r1["messages"][-1].content)

print("\n--- Turn 2 ---")
r2 = app.invoke({"messages": [HumanMessage("What's my name?")]}, config=thread)
print(r2["messages"][-1].content)

# Inspect the saved state
snapshot = app.get_state(thread)
print(f"\nCheckpoint has {len(snapshot.values['messages'])} messages total")
print("Next node to run:", snapshot.next or "done")
