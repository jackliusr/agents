import asyncio
from agent_framework.openai import OpenAIChatClient
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

import asyncio
from agent_framework.ollama import OllamaChatClient

async def main():
    agent = OllamaChatClient().as_agent(
        name="HelpfulAssistant",
        instructions="You are a helpful assistant running locally via Ollama.",
    )
    result = await agent.run("What is the largest city in France?")
    print(result)

asyncio.run(main())  