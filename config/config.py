import os
from dotenv import load_dotenv

load_dotenv()
class Config:
    os.environ["GROQ_API_KEY"] = os.getenv("GROK_API_KEY")
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
    os.environ["SCRAPGRAPH_API_KEY"] = os.getenv("SCRAPGRAPH_API_KEY")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_PROJECT"] = "Y_Multi_agents_Task"