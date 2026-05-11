"""
Agent handoff (swarm) pattern — agents transfer control to each other
using a special handoff tool. No central supervisor.

Agents:
  - triage_agent: decides if the question is technical or billing
  - tech_agent: handles technical questions, can hand off to billing
  - billing_agent: handles billing questions, can hand off to tech
"""
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


# --- Handoff tools (each routes to a different agent) ---
@tool
def transfer_to_tech() -> str:
    """Transfer the user to the technical support agent."""
    return "Transferring to tech agent."

@tool
def transfer_to_billing() -> str:
    """Transfer the user to the billing support agent."""
    return "Transferring to billing agent."

@tool
def transfer_to_triage() -> str:
    """Transfer back to the triage agent."""
    return "Transferring back to triage."


# --- State ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str


# --- Agent factory ---
def make_agent(system_prompt: str, tools: list):
    bound = llm.bind_tools(tools)
    def agent(state: State) -> dict:
        from langchain_core.messages import SystemMessage
        msgs = [SystemMessage(system_prompt)] + state["messages"]
        response = bound.invoke(msgs)
        # detect which agent to hand off to based on tool calls
        next_agent = state["active_agent"]
        if response.tool_calls:
            name = response.tool_calls[0]["name"]
            if name == "transfer_to_tech":
                next_agent = "tech"
            elif name == "transfer_to_billing":
                next_agent = "billing"
            elif name == "transfer_to_triage":
                next_agent = "triage"
        return {"messages": [response], "active_agent": next_agent}
    return agent


triage = make_agent(
    "You are a triage agent. Classify user queries and route: "
    "technical issues → transfer_to_tech, billing issues → transfer_to_billing. "
    "If already resolved, say so.",
    [transfer_to_tech, transfer_to_billing],
)

tech = make_agent(
    "You are a technical support agent. Answer technical questions. "
    "For billing questions, use transfer_to_billing.",
    [transfer_to_billing, transfer_to_triage],
)

billing = make_agent(
    "You are a billing support agent. Answer billing questions. "
    "For technical questions, use transfer_to_tech.",
    [transfer_to_tech, transfer_to_triage],
)

all_tools = ToolNode([transfer_to_tech, transfer_to_billing, transfer_to_triage])


def route(state: State):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def after_tools(state: State):
    return state["active_agent"]


# --- Graph ---
graph = StateGraph(State)
graph.add_node("triage", triage)
graph.add_node("tech", tech)
graph.add_node("billing", billing)
graph.add_node("tools", all_tools)

graph.add_edge(START, "triage")
graph.add_conditional_edges("triage", route)
graph.add_conditional_edges("tech", route)
graph.add_conditional_edges("billing", route)
graph.add_conditional_edges("tools", after_tools)

app = graph.compile()

print("=== Billing question ===")
r = app.invoke({
    "messages": [HumanMessage("I was charged twice this month, can you help?")],
    "active_agent": "triage",
})
for m in r["messages"]:
    if hasattr(m, "content") and m.content:
        print(f"[{type(m).__name__}] {m.content[:120]}")

print("\n=== Technical question ===")
r2 = app.invoke({
    "messages": [HumanMessage("My API key keeps returning a 401 error.")],
    "active_agent": "triage",
})
for m in r2["messages"]:
    if hasattr(m, "content") and m.content:
        print(f"[{type(m).__name__}] {m.content[:120]}")
