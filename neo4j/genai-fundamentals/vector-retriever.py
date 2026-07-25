import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth = (
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD")
    )
)

from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

from neo4j_graphrag.retrievers import VectorRetriever
retriever = VectorRetriever(
    driver = driver,
    neo4j_database = os.getenv("NEO4J_DATABASE"),
    index_name="moviePlots",
    embedder = embedder,
    return_properties = ["title", "plot"],
)

result = retriever.search(query_text = "Toys coming alive", top_k=5)

for item in result.items:
    print(item.content, item.metadata["score"])

driver.close()    