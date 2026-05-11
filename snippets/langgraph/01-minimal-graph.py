"""
The smallest possible LangGraph: one node, no LLM needed.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    message: str


def greet(state: State) -> dict:
    return {"message": "Hello, " + state["message"] + "!"}


graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

app = graph.compile()

result = app.invoke({"message": "world"})
print(result)  # {'message': 'Hello, world!'}
