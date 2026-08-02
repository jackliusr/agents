from dotenv import load_dotenv
load_dotenv()


from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

# Create a web search tool using SerperDevWebSearch
web_tool = ComponentTool(component=SerperDevWebSearch(), name="web_tool")

# Create the agent with the web search tool
agent = Agent(chat_generator=OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M"), tools=[web_tool])

# Run the agent with a query
result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])

# Print the final response
print(result["messages"][-1].text)


from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.core.pipeline import Pipeline

search_pipeline = Pipeline()

search_pipeline.add_component("search", SerperDevWebSearch(top_k=10))
search_pipeline.add_component("fetcher", LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2))
search_pipeline.add_component("converter", HTMLToDocument())
search_pipeline.add_component(
    "output_adapter",
    OutputAdapter(
        template="""
{%- for doc in docs -%}
  {%- if doc.content -%}
  <search-result url=\"{{ doc.meta.url }}\">
  {{ doc.content|truncate(25000) }}
  </search-result>
  {%- endif -%}
{%- endfor -%}
""",
        output_type=str,
    ),
)

search_pipeline.connect("search.links", "fetcher.urls")
search_pipeline.connect("fetcher.streams", "converter.sources")
search_pipeline.connect("converter.documents", "output_adapter.docs")


from haystack.tools import PipelineTool
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

search_tool = PipelineTool(
    name="search",
    description="Use this tool to search for information on the internet.",
    pipeline=search_pipeline,
    input_mapping={"query": ["search.query"]},
    output_mapping={"output_adapter.output": "search_result"},
    outputs_to_string={"source": "search_result"},
)

agent = Agent(
    chat_generator=OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M"),
    tools=[search_tool],
    system_prompt="""
    You are a deep research assistant.
    You create comprehensive research reports to answer the user's questions.
    You use the 'search'-tool to answer any questions.
    You perform multiple searches until you have the information you need to answer the question.
    Make sure you research different aspects of the question.
    Use markdown to format your response.
    When you use information from the websearch results, cite your sources using markdown links.
    It is important that you cite accurately.
    """,
    exit_conditions=["text"],
    max_agent_steps=20,
)

query = "What are the latest updates on the Artemis moon mission?"
messages = [ChatMessage.from_user(query)]

agent_output = agent.run(messages=messages)

print(agent_output["messages"][-1].text)