from dotenv import load_dotenv
load_dotenv()

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter

# Chat History components
message_store = InMemoryChatMessageStore()
message_retriever = ChatMessageRetriever(message_store)
message_writer = ChatMessageWriter(message_store)


from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

pipeline = Pipeline()

# components to communicate with an LLM
pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(
        template=[
            ChatMessage.from_system("You are a helpful AI assistant that answers users questions."),
            ChatMessage.from_user("{{query}}"),
        ],
        required_variables="*",
    ),
)

from haystack_integrations.components.generators.ollama import OllamaChatGenerator


pipeline.add_component("llm", OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M"))

# components for chat history retrieval and storage
pipeline.add_component("message_retriever", ChatMessageRetriever(message_store))
pipeline.add_component("message_writer", ChatMessageWriter(message_store))

# connections
pipeline.connect("prompt_builder.prompt", "message_retriever.current_messages")
pipeline.connect("message_retriever.messages", "llm.messages")
pipeline.connect("llm.replies", "message_writer.messages")


chat_history_id = "user_123_session_1"

while True:
    question = input("Enter your question or Q to exit.\n🧑 ")
    if question == "Q":
        break

    res = pipeline.run(
        data={
            "prompt_builder": {"query": question},
            "message_retriever": {"chat_history_id": chat_history_id},
            "message_writer": {"chat_history_id": chat_history_id},
        },
        include_outputs_from={"llm"},
    )
    print(f'🤖 {res["llm"]["replies"][0].text}')
