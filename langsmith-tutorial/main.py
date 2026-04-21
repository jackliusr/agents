from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from langsmith import traceable
from langsmith.wrappers import wrap_gemini
from langsmith import traceable, Client, uuid7  

client = wrap_gemini(genai.Client())

docs = [
    "Acme Cloud supports unlimited users on Enterprise plans. Starter plans are limited to 5 users.",
    "To reset your password, click 'Forgot password' on the login page and follow the instructions sent to your email.",
    "API rate limits are 1,000 requests per hour on the Starter plan and 10,000 requests per hour on Enterprise.",
]

@traceable(run_type="retriever")
def retriever(query: str) -> list[str]:
    return docs

@traceable(metadata={"llm": "gemini-2.5-flash"})
def support_bot(question: str) -> str:
    context = retriever(question)
    system_message = (
        "You are a helpful customer support agent. "
        "Answer using only the information provided below:\n\n"
        + "\n".join(context)
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_message),        
        contents=[
           question
        ],
    )
    return response.text

if __name__ == "__main__":
    run_id = uuid7()
    support_bot("How many users can I have on the Starter plan?",
                langsmith_extra={"run_id": run_id},
                )
    ls_client = Client()
    ls_client.create_feedback(run_id, 
                              key="user-score",
                              score=1.0)