"""
Basic RAG pipeline:
  1. Load documents
  2. Split into chunks
  3. Embed + store in Chroma
  4. Retrieve + answer with LCEL

pip install langchain-openai langchain-chroma
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- 1. Your documents (replace with a real loader in practice) ---
raw_docs = [
    Document(page_content="LangChain is a framework for building LLM applications. It provides tools for chaining calls, managing prompts, and integrating external data."),
    Document(page_content="LangGraph is built on top of LangChain and lets you model your application as a stateful graph of nodes. Each node is a function that reads and writes to shared state."),
    Document(page_content="Retrieval-Augmented Generation (RAG) combines a retrieval step with LLM generation. The retriever fetches relevant documents; the LLM uses them to answer the question."),
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe operator | to compose Runnables. Every component — prompts, models, parsers — implements the Runnable interface."),
    Document(page_content="Checkpointing in LangGraph saves the graph state after every node execution. This enables resumable conversations, human-in-the-loop workflows, and time travel debugging."),
]

# --- 2. Split ---
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
docs = splitter.split_documents(raw_docs)

# --- 3. Embed + store ---
vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 4. RAG chain ---
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

answer = rag_chain.invoke("What is LangGraph and how does it relate to LangChain?")
print(answer)
