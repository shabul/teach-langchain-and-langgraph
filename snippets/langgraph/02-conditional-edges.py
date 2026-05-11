"""
Conditional edges: route to different nodes based on state.

Flow:
  START → classify → [positive | negative] → END
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    sentiment: str


def classify(state: State) -> dict:
    # Toy classifier — swap with an LLM call for real use
    positive_words = {"good", "great", "love", "happy", "awesome"}
    words = set(state["text"].lower().split())
    sentiment = "positive" if words & positive_words else "negative"
    return {"sentiment": sentiment}


def handle_positive(state: State) -> dict:
    print("Positive path:", state["text"])
    return {}


def handle_negative(state: State) -> dict:
    print("Negative path:", state["text"])
    return {}


def route(state: State) -> Literal["positive", "negative"]:
    return state["sentiment"]


graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("positive", handle_positive)
graph.add_node("negative", handle_negative)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("positive", END)
graph.add_edge("negative", END)

app = graph.compile()

app.invoke({"text": "This is great!", "sentiment": ""})
app.invoke({"text": "This is terrible.", "sentiment": ""})
