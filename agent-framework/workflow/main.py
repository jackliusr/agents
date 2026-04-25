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

client = OpenAIChatClient(
        api_key="ollama",  # Placeholder, Ollama doesn't require an API key
        base_url=os.environ["OLLAMA_ENDPOINT"],
        model=os.environ["OLLAMA_MODEL"],
    )

writer = Agent(
    name="WriterAgent",
    instructions="Write a short poem (4 lines max) about the given topic.",
    client=client,
)

reviewer = Agent(
    name="ReviewerAgent",
    instructions="Review the given poem in one sentence. Is it good?",
    client=client,
)

@workflow
async def poem_workflow(topic: str) -> str:
    """Write a poem, then review it."""
    poem = (await writer.run(f"Write a poem about: {topic}")).text
    review = (await reviewer.run(f"Review this poem: {poem}")).text
    return f"Poem:\n{poem}\n\nReview: {review}"

async def main():
    topic = "a cat learning to code"
    result = await poem_workflow.run(topic)
    print(result.get_outputs()[0])

if __name__ == "__main__":
    asyncio.run(main())