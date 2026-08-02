from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from dotenv import load_dotenv
load_dotenv()


template = [
    ChatMessage.from_user(
        """
Please create a summary about the following topic:
{{ topic }}
"""
    )
]

builder = ChatPromptBuilder(template=template)


llm = OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M", generation_kwargs={"max_new_tokens": 150})

pipeline = Pipeline()
pipeline.add_component(name="builder", instance=builder)
pipeline.add_component(name="llm", instance=llm)

pipeline.connect("builder.prompt", "llm.messages")

topic = "Climate change"
result = pipeline.run(data={"builder": {"topic": topic}})
print(result["llm"]["replies"][0].text)

yaml_pipeline = pipeline.dumps()

print(yaml_pipeline)


yaml_pipeline = """
components:
  builder:
    init_parameters:
      required_variables: '*'
      template:
      - content:
        - text: '

            Please create a summary about the following topic:

            {{ sentence }}

            '
        meta: {}
        name: null
        role: user
      variables: null
    type: haystack.components.builders.chat_prompt_builder.ChatPromptBuilder
  llm:
    init_parameters:
      generation_kwargs:
        max_new_tokens: 150
      keep_alive: null
      max_retries: 0
      model: qwen3.6:35b-a3b-q4_K_M
      response_format: null
      streaming_callback: null
      think: false
      timeout: 120
      tools: null
      url: http://localhost:11434
    type: haystack_integrations.components.generators.ollama.chat.chat_generator.OllamaChatGenerator
connection_type_validation: true
connections:
- receiver: llm.messages
  sender: builder.prompt
max_runs_per_component: 100
metadata: {}
"""

from haystack import Pipeline

new_pipeline = Pipeline.loads(yaml_pipeline)
result  = new_pipeline.run(data={"builder": {"sentence": "I love capybaras"}})
print(result)
