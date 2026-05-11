# LangGraph Checkpointing & Persistence

Checkpointing lets a graph **save its state after every step** so you can resume, replay, or inspect any point in a run.

## Why It Matters

Without checkpointing, every `invoke` starts fresh.  
With checkpointing, state is saved to a store keyed by `thread_id` — so the same thread picks up exactly where it left off.

## Compiling with a Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()          # in-memory (dev/test)
app = graph.compile(checkpointer=checkpointer)
```

For production use `SqliteSaver` or `PostgresSaver`:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
```

## Thread Config

Every call needs a `thread_id` so the checkpointer knows which conversation to load/save:

```python
config = {"configurable": {"thread_id": "user-42-session-1"}}

app.invoke({"messages": [HumanMessage("Hello")]}, config=config)
app.invoke({"messages": [HumanMessage("Continue")]}, config=config)
# second call resumes from saved state of the first
```

## Inspecting State

```python
# Current state snapshot
snapshot = app.get_state(config)
print(snapshot.values)      # the state dict
print(snapshot.next)        # which node runs next (empty if done)

# Full history of every checkpoint
for step in app.get_state_history(config):
    print(step.values, step.next)
```

## Time Travel — Replaying from a Past Checkpoint

```python
history = list(app.get_state_history(config))
past = history[-2]          # two steps back

# Resume from that point
app.invoke(None, config={"configurable": {"thread_id": "...", "checkpoint_id": past.config["configurable"]["checkpoint_id"]}})
```

## Human-in-the-Loop with `interrupt_before`

```python
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["dangerous_action"],  # pause before this node
)

app.invoke(input, config)          # runs until interrupt
# inspect state, optionally edit it
app.invoke(None, config)           # resume
```

## See also
- `snippets/langgraph/04-checkpointing.py`
- `snippets/langgraph/05-human-in-loop.py`
