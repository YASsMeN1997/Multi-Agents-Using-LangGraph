from config import config
from langchain_groq import ChatGroq
import json
config.Config()

from typing import TypedDict, List , Optional
from pydantic import BaseModel, Field
import os
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# agent one : recomended a keywords for search quires
class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(
        ...,
        title="A list of suggested search queries to be passed to the search engine.",
        min_items=1,
        max_items=10  # Will be overridden by input
    )

# Define state schema
class AgentState(TypedDict):
    product_name: str
    websites_list: List[str]
    country_name: str
    language: str
    num_of_keywords: int
    search_queries: Optional[SuggestedSearchQueries]

def search_query_node(state: AgentState):
    # Create JSON parser
    parser = JsonOutputParser(pydantic_object=SuggestedSearchQueries)
    
    # Build prompt with format instructions
    prompt_template = ChatPromptTemplate.from_template(
        """Generate EXACTLY {num_of_keywords} e-commerce search queries in {language} language for:
        Product: {product_name}
        Websites: {websites_list}
        Country: {country_name}
        Queries must: 
        - Be specific to product comparison shopping
        - Include relevant keywords for {country_name}
        - Avoid website prefixes (just the search terms)
        
        Example format for 3 queries:
        {{
            "queries": ["office coffee machine deals", "professional coffee maker price", "commercial espresso machine offers"]
        }}
        
        {format_instructions}"""
    )
    
    # Add format instructions
    prompt = prompt_template.format(
        product_name=state["product_name"],
        websites_list=", ".join(state["websites_list"]),
        country_name=state["country_name"],
        language=state["language"],
        num_of_keywords=state["num_of_keywords"],
        format_instructions=parser.get_format_instructions()
    )
    
    # Create chain 
    chain = llm | parser
    
    try:
        result = chain.invoke(prompt)
       # print(result)
        return {"search_queries": result}
    except Exception as e:
        print(f"Error generating queries: {e}")
        return {"search_queries": None}

# Set up the graph workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("search_queries_recommendation", search_query_node)

# Define entry point
workflow.add_edge(START, "search_queries_recommendation")

# Connect to end
workflow.add_edge("search_queries_recommendation", END)

# Compile the graph
app = workflow.compile()

#display(Image(app.get_graph().draw_mermaid_png()))
# Get the raw image bytes from the graph
graph_image_bytes = app.get_graph().draw_mermaid_png()

# Save to file
with open("workflow_graph.png", "wb") as f:
    f.write(graph_image_bytes)

# Execute the graph
inputs = {
    "product_name": "coffee machine for the office",
    "websites_list": ['www.amazon.com','www.jumia.com','www.noon.com'],
    "country_name": "EGYPT",
    "language": "english",
    "num_of_keywords": 5,
}

result = app.invoke(inputs)

# Save output if needed
output_dir = "out_agent1"
os.makedirs(output_dir, exist_ok=True)
full_output = result["search_queries"]

with open(os.path.join(output_dir, 'suggested_search_queries.json'), "w", encoding="utf-8") as f:
    json.dump(full_output, f, indent=2, ensure_ascii=False)