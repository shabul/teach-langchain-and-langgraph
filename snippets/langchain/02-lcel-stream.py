"""
Streaming tokens from an LCEL chain — great for chatbots.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chain = (
    ChatPromptTemplate.from_template("Write a short poem about {topic}.")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

for chunk in chain.stream({"topic": "rainy days"}):
    print(chunk, end="", flush=True)
print()
