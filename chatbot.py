from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Simple invocation
messages = [
    ("system", "You are a helpful assistant that translates English to French."),
    ("human", "I love programming."),
]
response = llm.invoke(messages)
print(response.content)  # Output: J'adore la programmation.