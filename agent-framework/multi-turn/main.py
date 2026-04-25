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
from agent_framework import tool,ContextProvider,AgentSession,SessionContext

class UserMemoryProvider(ContextProvider):
    """A context provider that remembers user info in session state."""

    DEFAULT_SOURCE_ID = "user_memory"

    def __init__(self):
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject personalization instructions based on stored user info."""
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Always address them by name.",
            )
        else:
            context.extend_instructions(
                self.source_id,
                "You don't know the user's name yet. Ask for it politely.",
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Extract and store user info in session state after each call."""
        for msg in context.input_messages:
            text = msg.text if hasattr(msg, "text") else ""
            if isinstance(text, str) and "my name is" in text.lower():
                state["user_name"] = text.lower().split("my name is")[-1].strip().split()[0].capitalize()
                
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

async def main():
    agent = OpenAIChatClient(
        api_key="ollama",  # Placeholder, Ollama doesn't require an API key
        base_url=os.environ["OLLAMA_ENDPOINT"],
        model=os.environ["OLLAMA_MODEL"],
    ).as_agent(
        name="ConversationAgent",
        instructions="You are a friendly assistant. Keep your answers brief and to the point.",
        tools=[get_time, get_weather],
        context_providers=[UserMemoryProvider()],
    )
    
    # Create a session to maintain conversation history
    session = agent.create_session()

    # The provider doesn't know the user yet — it will ask for a name
    result = await agent.run("Hello! What's the square root of 9?", session=session)
    print(f"Agent: {result}\n")

    # Now provide the name — the provider stores it in session state
    result = await agent.run("My name is Alice", session=session)
    print(f"Agent: {result}\n")

    # Subsequent calls are personalized — name persists via session state
    result = await agent.run("What is 2 + 2?", session=session)
    print(f"Agent: {result}\n")

    # Inspect session state to see what the provider stored
    provider_state = session.state.get("user_memory", {})
    print(f"[Session State] Stored user name: {provider_state.get('user_name')}")

asyncio.run(main())  