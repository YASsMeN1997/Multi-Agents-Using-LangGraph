import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv('.env')

os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg)