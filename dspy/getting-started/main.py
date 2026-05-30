import dspy
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')
MODEL = os.getenv('MODEL')
# Pass the key explicitly...
lm = dspy.LM(
    model= MODEL,
    api_key=API_KEY,
    api_base="https://api.deepseek.com")

dspy.configure(lm=lm)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant"
    },
    {
        "role": "user",
        "content": "What is the capital of France?"
    }
]
print('----------------------------------------------------------')
print('direct lm with prompts')
print(lm(messages = messages))


haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print('----------------------------------------------------------')
print('haiku_generator')
print(result.haiku)


haiku_bot = dspy.Predict("location, mood -> haiku, haiku_title")
result = haiku_bot(location="a quiet library", mood="mysterious")
print('----------------------------------------------------------')
print('haiku_bot')
print(result.haiku_title)
print("- - -")
print(result.haiku)


from typing import Literal

Season = Literal[
    "spring", "summer", "autumn", "winter",
]

class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField()
    mood: str = dspy.InputField()
    season: Season = dspy.InputField()
    haiku: str = dspy.OutputField()


haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print('----------------------------------------------------------')
print('haiku_bot: class-based')
print(result.haiku)


reasoning_haiku_bot = dspy.ChainOfThought(HaikuBot)
result = reasoning_haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print('----------------------------------------------------------')
print('reasoning_haiku_bot')
print('------\nhaiku:')
print(result.haiku)
print('------\nreasoning:')
print(result.reasoning)


import wikipedia

def wikipedia_search(query: str) -> list[str]:
    """Search Wikipedia for the given query and return a list of page titles."""
    return wikipedia.search(query)

def get_wikipedia_page(title: str) -> str:
    """Get the content of a Wikipedia page given its title."""
    return wikipedia.page(title).content

haiku_bot = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page], max_iters=4)
result = haiku_bot(location="Camp Meeker", mood="pensive", season="summer")
print('----------------------------------------------------------')
print('ReAct haiku_bot')
print('------\nhaiku:')
print(result.haiku)
print('------\nreasoning:')
print(result.reasoning)

for step, value in result.trajectory.items():
    print(f"{step}: {value}")


class HaikuEnsemble(dspy.Module):
    def __init__(self, n: int = 3):
        super().__init__()
        self.n = n  
        # Module 1 generates several haikus
        self.writer = dspy.ReAct(
            "location, season, mood, num_haikus: int -> haikus: list[str]", 
            tools=[wikipedia_search, get_wikipedia_page],
            max_iters=5
        )
        # Module 2 picks the most evocative
        self.judge = dspy.ChainOfThought(
            "location, season, mood, candidates: list[str] -> most_evocative_index: int"
        )

    def forward(self, location: str, season: str, mood: str):
        candidates = self.writer(
            location=location, season=season, mood=mood, num_haikus=self.n,
        ).haikus
        # Call a much larger model to evaluate our haikus
        with dspy.context(lm=dspy.LM("deepseek/deepseek-v4-pro")):
            verdict = self.judge( 
                location=location, season=season, mood=mood, candidates=candidates,
            )
        return dspy.Prediction(
            haiku=candidates[verdict.most_evocative_index],
            candidates=candidates,
            reasoning=verdict.reasoning,
        )
    
ensemble = HaikuEnsemble(n=5)
result = ensemble(location="Bodega Bay", season="autumn", mood="inspired")
print('----------------------------------------------------------')
print('ensemble')
print('------\nhaiku:')
print(result.haiku)
print('------\nreasoning:' )
print(result.reasoning)