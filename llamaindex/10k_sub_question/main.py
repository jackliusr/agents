import os

# NLTK's import-security hook blocks any module it resolves to a path inside
# the current working directory. Because this project keeps its virtualenv at
# .venv/ (inside the repo root), site-packages is "inside the CWD" and the hook
# misfires on legitimate packages (e.g. regex). Disable it before NLTK loads.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator

from llama_index.core import Settings

Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

lyft_index = VectorStoreIndex.from_documents(lyft_docs)
uber_index = VectorStoreIndex.from_documents(uber_docs)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021"
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021"
            ),
        ),
    ),
]

# `SubQuestionQueryEngine.from_defaults()` tries to import the
# `llama-index-question-gen-openai` package, which is unmaintained and pinned to
# `llama-index-llms-openai<0.5` (incompatible with the `>=0.7.10` this project
# needs). Passing an explicit `LLMQuestionGenerator` (which uses `Settings.llm`,
# i.e. OpenAI) bypasses that import entirely.
s_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    question_gen=LLMQuestionGenerator.from_defaults(),
)

response = s_engine.query(
    "Compare and contrast the customer segments and geographies that grew the"
    " fastest"
)

print(response)

response = s_engine.query(
    "Compare revenue growth of Uber and Lyft from 2020 to 2021"
)

print(response)