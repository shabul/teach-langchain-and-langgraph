# Multi-Agent Systems in LangGraph

Multi-agent systems split complex tasks across **specialized agents** that collaborate. LangGraph models this as a graph where each agent is a node.

## Common Patterns

### 1. Supervisor (Orchestrator)

A single LLM decides which worker to call next, then decides when to stop.

```
START → supervisor ──→ researcher
              ↑              ↓
              └─── writer ◄──┘
              ↓
             END
```

### 2. Swarm / Handoff

Agents hand off to each other peer-to-peer (no central orchestrator).  
Each agent can decide to transfer control using a special handoff tool.

### 3. Pipeline (Sequential)

Agents form a linear chain — output of one becomes input of the next.

## Supervisor Pattern (Key Code)

```python
from langchain_core.tools import tool

# The supervisor is an LLM with routing tools
members = ["researcher", "writer"]

@tool
def route(next: Literal["researcher", "writer", "FINISH"]) -> str:
    """Route to the next worker or finish."""
    return next

supervisor_chain = (
    supervisor_prompt
    | llm.bind_tools([route], tool_choice="route")
)
```

## Sharing State Between Agents

All agents share the same `State` TypedDict. Each agent reads what it needs and writes back its contribution:

```python
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_notes: str
    final_report: str
    next: str           # which agent runs next
```

## Agent as a Node

Any LangGraph compiled graph can itself be used as a node inside a larger graph — enabling hierarchical / nested agents.

```python
sub_app = sub_graph.compile()

def researcher_node(state: State) -> dict:
    result = sub_app.invoke({"messages": state["messages"]})
    return {"research_notes": result["output"]}

parent_graph.add_node("researcher", researcher_node)
```

## See also
- `snippets/langgraph/06-supervisor-agent.py`
- `snippets/langgraph/07-agent-handoff.py`
