from dataclasses import Field
from datetime import datetime
import os
from random import randint
from dotenv import load_dotenv
from typing import Annotated
from pydantic import Field
from agent_framework.devui import serve
load_dotenv()  # Loads variables from .env

from agent_framework.openai import OpenAIChatClient
from agent_framework import tool

def get_time(location: str) -> str:
    """Get the current time."""
    return f"The current time in {location} is {datetime.now().strftime('%I:%M %p')}."

@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, Field(description="The location to get the weather for.")],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

agent = OpenAIChatClient(
        api_key="ollama",  # Placeholder, Ollama doesn't require an API key
        base_url=os.environ["OLLAMA_ENDPOINT"],
        model=os.environ["OLLAMA_MODEL"],
    ).as_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent. Use the get_weather tool to answer questions.",
        tools=[get_time, get_weather],
    )

# Launch DevUI
serve(entities=[agent], auto_open=True)
# Opens browser to http://localhost:8080