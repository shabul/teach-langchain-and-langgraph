# RAG — Retrieval-Augmented Generation

RAG lets you ground an LLM's answers in your own documents instead of relying on training data.

## The Pipeline

```
Documents → split → embed → vector store
                                  ↓
User question → embed → similarity search → top-k chunks
                                                   ↓
                              prompt (question + chunks) | LLM → answer
```

## Key Components

| Component | Purpose | Common choices |
|---|---|---|
| `TextSplitter` | Chunk documents | `RecursiveCharacterTextSplitter` |
| `Embeddings` | Turn text into vectors | `OpenAIEmbeddings`, `HuggingFaceEmbeddings` |
| `VectorStore` | Store + similarity search | `Chroma`, `FAISS`, `Pinecone` |
| `Retriever` | Fetch relevant chunks | `.as_retriever()` on any VectorStore |

## LCEL RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt          # uses {context} and {question}
    | model
    | StrOutputParser()
)

rag_chain.invoke("What are the return policies?")
```

`RunnablePassthrough()` passes the raw question string straight through to the prompt,
while `retriever` converts it into retrieved chunks automatically.

## Retriever Variants

```python
# Basic similarity search (top 4 by default)
retriever = vectorstore.as_retriever()

# MMR — balances relevance with diversity
retriever = vectorstore.as_retriever(search_type="mmr")

# Filter by metadata
retriever = vectorstore.as_retriever(
    search_kwargs={"filter": {"source": "policy.pdf"}}
)
```

## See also
- `snippets/langchain/04-rag-basic.py`
- `snippets/langchain/05-rag-with-sources.py`
