"""
Simplest possible LCEL chain: prompt | model | parser
"""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template("Tell me a fun fact about {topic}.")
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "the moon"})
print(result)
