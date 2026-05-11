# LCEL — LangChain Expression Language

LCEL is the core way to compose LangChain components using the `|` pipe operator.

## The Pattern

```
input → prompt | model | output_parser → output
```

Every piece is a **Runnable** — anything with `.invoke()`, `.stream()`, and `.batch()`.

## Key Runnables

| Runnable | What it does |
|---|---|
| `ChatPromptTemplate` | Fills variables into a prompt |
| `ChatOpenAI` | Calls the LLM |
| `StrOutputParser` | Pulls the string out of the response |
| `RunnablePassthrough` | Passes input through unchanged |
| `RunnableLambda` | Wraps any Python function |

## Invoke vs Stream vs Batch

```python
chain.invoke({"topic": "black holes"})          # single, blocking
chain.stream({"topic": "black holes"})           # yields tokens
chain.batch([{"topic": "a"}, {"topic": "b"}])   # parallel list
```

## See also
- `snippets/langchain/01-lcel-hello.py`
- `snippets/langchain/02-lcel-chain.py`
