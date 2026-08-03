import os

# Disable NLTK's import security hook, which falsely blocks `regex` when the
# virtualenv lives inside the project directory (see nltk.inisec).
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import nest_asyncio

nest_asyncio.apply()

import logging
import sys

from dotenv import load_dotenv

# Load environment variables from a .env file (e.g. PINECONE_API_KEY, OPENAI_API_KEY)
load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# define pinecone index
from pinecone import Pinecone, ServerlessSpec

api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)

# dimensions are for text-embedding-ada-002
# create the index if it doesn't already exist
if "quickstart" not in pc.list_indexes().names():
    pc.create_index(
        name="quickstart",
        dimension=1536,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
pinecone_index = pc.Index("quickstart")

from llama_index.core import StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex


# define pinecone vector index
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace="wiki_cities"
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
vector_index = VectorStoreIndex([], storage_context=storage_context)

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///:memory:", future=True)
metadata_obj = MetaData()

# create city SQL table
table_name = "city_stats"
city_stats_table = Table(
    table_name,
    metadata_obj,
    Column("city_name", String(16), primary_key=True),
    Column("population", Integer),
    Column("country", String(16), nullable=False),
)

metadata_obj.create_all(engine)
# print tables
metadata_obj.tables.keys()


from sqlalchemy import insert

rows = [
    {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
    {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
    {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
]
for row in rows:
    stmt = insert(city_stats_table).values(**row)
    with engine.begin() as connection:
        cursor = connection.execute(stmt)

with engine.connect() as connection:
    cursor = connection.exec_driver_sql("SELECT * FROM city_stats")
    print(cursor.fetchall())

from llama_index.readers.wikipedia import WikipediaReader

cities = ["Toronto", "Berlin", "Tokyo"]
wiki_docs = WikipediaReader().load_data(pages=cities)


from llama_index.core import SQLDatabase

sql_database = SQLDatabase(engine, include_tables=["city_stats"])

from llama_index.core.query_engine import NLSQLTableQueryEngine

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["city_stats"],
)


from llama_index.core import Settings

# Insert documents into vector index
# Each document has metadata of the city attached
for city, wiki_doc in zip(cities, wiki_docs):
    nodes = Settings.node_parser.get_nodes_from_documents([wiki_doc])
    # add metadata to each node
    for node in nodes:
        node.metadata = {"title": city}
    vector_index.insert_nodes(nodes)


from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo
from llama_index.core.query_engine import RetrieverQueryEngine


vector_store_info = VectorStoreInfo(
    content_info="articles about different cities",
    metadata_info=[
        MetadataInfo(
            name="title", type="str", description="The name of the city"
        ),
    ],
)
vector_auto_retriever = VectorIndexAutoRetriever(
    vector_index, vector_store_info=vector_store_info
)

retriever_query_engine = RetrieverQueryEngine.from_args(
    vector_auto_retriever, llm=OpenAI(model="gpt-4")
)

from llama_index.core.tools import QueryEngineTool

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into a SQL query over"
        " a table containing: city_stats, containing the population/country of"
        " each city"
    ),
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=retriever_query_engine,
    description=(
        f"Useful for answering semantic questions about different cities"
    ),
)

from llama_index.core.query_engine import SQLAutoVectorQueryEngine

query_engine = SQLAutoVectorQueryEngine(
    sql_tool, vector_tool, llm=OpenAI(model="gpt-4")
)

response = query_engine.query(
    "Tell me about the arts and culture of the city with the highest"
    " population"
)

print(str(response))

response = query_engine.query("Tell me about the history of Berlin")
print(str(response))

response = query_engine.query(
    "Can you give me the country corresponding to each city?"
)

print(str(response))