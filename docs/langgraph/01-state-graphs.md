# LangGraph — State Graphs

LangGraph models your app as a **graph of nodes** where each node reads and writes to a shared **state**.

## Core Concepts

| Term | Meaning |
|---|---|
| **State** | A TypedDict shared across all nodes |
| **Node** | A Python function `(state) -> partial_state` |
| **Edge** | A directed link from one node to another |
| **Conditional edge** | A function that decides which node to go to next |
| **`START` / `END`** | Built-in entry and exit markers |

## Minimal Graph

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def greet(state: State) -> State:
    return {"message": "Hello, " + state["message"]}

graph = StateGraph(State)
graph.add_node("greet", greet)
graph.add_edge(START, "greet")
graph.add_edge("greet", END)

app = graph.compile()
app.invoke({"message": "world"})
# → {"message": "Hello, world"}
```

## State Updates are Partial

Nodes only return the keys they want to update — LangGraph merges the result into the full state automatically.

## See also
- `snippets/langgraph/01-minimal-graph.py`
- `snippets/langgraph/02-conditional-edges.py`
