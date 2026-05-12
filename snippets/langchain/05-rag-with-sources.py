"""
RAG with source citations — returns both the answer and which docs were used.

Uses RunnableParallel to run retrieval once but feed two branches:
  - one that formats docs for the prompt
  - one that keeps raw docs for citation
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

docs_data = [
    Document(page_content="The Eiffel Tower is located in Paris, France. It was built in 1889.", metadata={"source": "paris-guide.txt"}),
    Document(page_content="The Colosseum is in Rome, Italy. Construction began in 70 AD under Emperor Vespasian.", metadata={"source": "rome-guide.txt"}),
    Document(page_content="The Great Wall of China stretches over 13,000 miles and was built over many centuries.", metadata={"source": "china-guide.txt"}),
    Document(page_content="The Statue of Liberty stands on Liberty Island in New York Harbor, gifted by France in 1886.", metadata={"source": "nyc-guide.txt"}),
]

splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=10)
docs = splitter.split_documents(docs_data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)

# Run retrieval once, branch into context text + raw docs
setup = RunnableParallel(
    context=retriever | format_docs,
    question=RunnablePassthrough(),
    source_docs=retriever,
)

answer_chain = (
    setup
    | RunnableParallel(
        answer={"context": lambda x: x["context"], "question": lambda x: x["question"]}
                | prompt | ChatGoogleGenerativeAI(model="gemini-2.0-flash") | StrOutputParser(),
        sources=lambda x: list({d.metadata["source"] for d in x["source_docs"]}),
    )
)

result = answer_chain.invoke("Where is the Eiffel Tower?")
print("Answer:", result["answer"])
print("Sources:", result["sources"])
