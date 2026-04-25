import asyncio
from dataclasses import Field
from datetime import datetime
import os
from random import randint

from dotenv import load_dotenv
from typing import Annotated, Any
from pydantic import Field

load_dotenv()  # Loads variables from .env

import asyncio
from agent_framework.openai import OpenAIChatClient
from agent_framework import Agent, workflow, tool,ContextProvider,AgentSession,SessionContext
from agent_framework.azure import AgentFunctionApp


def _create_agent() -> Any:
    client = OpenAIChatClient(
        api_key="ollama",  # Placeholder, Ollama doesn't require an API key
        base_url=os.environ["OLLAMA_ENDPOINT"],
        model=os.environ["OLLAMA_MODEL"],
    )    
    return Agent(
        name="Joker",
        instructions="You are good at telling jokes.",
        client=client,
    )

app = AgentFunctionApp(agents=[_create_agent()], enable_health_check=True, max_poll_retries=50)


