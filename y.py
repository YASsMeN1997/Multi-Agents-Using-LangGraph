from config import config
from langchain_groq import ChatGroq

config.Config()

from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import StateGraph, START, END

from langchain.chat_models import ChatGroq

# Define the Pydantic model for the output
class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(
        ...,
        title="A list of suggested search queries to be passed to the search engine.",
        min_items=1,
        max_items=100  # Assuming num_of_keywords is 100
    )

# Define the agent's role and behavior
class SearchQueriesRecommendationAgent:
    def __init__(self, llm, verbose=True):
        self.llm = llm
        self.verbose = verbose

    def run(self, context: dict) -> SuggestedSearchQueries:
        # Extract context variables
        product_name = context.get("product_name")
        websites_list = context.get("websites_list")
        country_name = context.get("country_name")
        language = context.get("language")

        # Generate the prompt for the LLM
        prompt = f"""
        YZ company is looking to buy {product_name} at the best prices (value for a price strategy).
        The company targets the following websites: {', '.join(websites_list)}.
        The company wants to reach all available products on the internet to be compared later in another stage.
        The stores must sell the product in {country_name}.
        Generate at most 100 search queries in {language} language.
        The search queries must reach an e-commerce webpage for the product, and not a blog or listing page.
        """

        # Call the LLM to generate search queries
        response = self.llm.invoke(prompt)

        # Parse the LLM response into a list of queries
        queries = response.content.strip().split("\n")

        # Return the output as a Pydantic model
        return SuggestedSearchQueries(queries=queries)

# Define the graph structure
def create_search_queries_graph(llm):
    # Create the agent node
    search_agent = SearchQueriesRecommendationAgent(llm=llm, verbose=True)
    agent_node = AgentNode(
        name="search_queries_recommendation_agent",
        agent=search_agent,
        input_keys=["product_name", "websites_list", "country_name", "language"],
        output_key="suggested_queries"
    )

    # Create the graph
    graph = Graph(nodes=[agent_node])

    # Define the edges (in this case, a single node graph)
    graph.add_edge(agent_node, Edge(end=True))

    return graph

# Example usage
if __name__ == "__main__":
    # Initialize the ChatGroq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    # Create the graph
    search_queries_graph = create_search_queries_graph(llm)

    # Define the input context
    context = {
        "product_name": "smartphone",
        "websites_list": ["amazon", "ebay", "walmart"],
        "country_name": "USA",
        "language": "English"
    }

    # Execute the graph
    result = search_queries_graph.run(context)
    print("Generated Search Queries:")
    for query in result.queries:
        print(f"- {query}")