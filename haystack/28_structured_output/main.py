from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]

from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.generators.chat import OpenAIChatGenerator

#chat_generator = OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M",generation_kwargs={"response_format": CitiesData})
chat_generator = OpenAIChatGenerator(generation_kwargs={"response_format": CitiesData})


from haystack.dataclasses import ChatMessage


text = "Berlin is the capital of Germany. It has a population of 3,850,809. Paris, France's capital, has 2.161 million residents. Lisbon is the capital and the largest city of Portugal with the population of 504,718."
result = chat_generator.run(messages=[ChatMessage.from_user(text)])

import json
valid_reply = result["replies"][0].text
valid_json = json.loads(valid_reply)
print(valid_json)