"""
Human-in-the-loop: the graph pauses before a "dangerous" node so a human can
review and approve (or abort) before execution continues.

Flow:
  START → draft_action → [PAUSE] → execute_action → END
"""
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    proposed_action: str


def draft_action(state: State) -> dict:
    # Simulate the LLM drafting something that needs approval
    action = "DELETE all records from the 'users' table"
    print(f"[Agent] Proposing action: {action}")
    return {
        "proposed_action": action,
        "messages": [AIMessage(f"I want to: {action}. Awaiting approval.")],
    }


def execute_action(state: State) -> dict:
    action = state["proposed_action"]
    print(f"[Agent] Executing: {action}")
    return {"messages": [AIMessage(f"Done: {action}")]}


graph = StateGraph(State)
graph.add_node("draft_action", draft_action)
graph.add_node("execute_action", execute_action)

graph.add_edge(START, "draft_action")
graph.add_edge("draft_action", "execute_action")
graph.add_edge("execute_action", END)

checkpointer = MemorySaver()
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["execute_action"],  # pause here, before executing
)

thread = {"configurable": {"thread_id": "review-session-1"}}

# --- Step 1: run until the interrupt ---
print("=== Running until interrupt ===")
app.invoke({"messages": [HumanMessage("Please clean up the database.")]}, config=thread)

# --- Step 2: inspect what was proposed ---
snapshot = app.get_state(thread)
print(f"\nPaused before: {snapshot.next}")
print(f"Proposed action: {snapshot.values['proposed_action']}")

# --- Step 3: human decides ---
approved = input("\nApprove action? (y/n): ").strip().lower() == "y"

if approved:
    print("\n=== Resuming after approval ===")
    result = app.invoke(None, config=thread)   # None = resume with existing state
    print(result["messages"][-1].content)
else:
    print("\nAction rejected — graph will not continue.")
