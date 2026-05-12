"""
Streaming tokens from an LCEL chain — great for chatbots.
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

chain = (
    ChatPromptTemplate.from_template("Write a short poem about {topic}.")
    | ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    | StrOutputParser()
)

for chunk in chain.stream({"topic": "rainy days"}):
    print(chunk, end="", flush=True)
print()
