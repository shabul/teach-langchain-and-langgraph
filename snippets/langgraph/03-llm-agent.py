"""
A simple ReAct-style agent in LangGraph: think → maybe use a tool → respond.

Tools available: calculator (just eval for demo, don't use in prod).
"""
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 2' or '10 * 5'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


tools = [calculator]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: State):
    last = state["messages"][-1]
    return "tools" if last.tool_calls else END


graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

result = app.invoke({"messages": [HumanMessage("What is 137 * 42?")]})
print(result["messages"][-1].content)
