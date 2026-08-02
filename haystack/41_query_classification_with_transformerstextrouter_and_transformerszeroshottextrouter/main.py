from dotenv import load_dotenv
load_dotenv()


from haystack_integrations.components.routers.transformers import TransformersTextRouter

text_router = TransformersTextRouter(model="shahrukhx01/bert-mini-finetune-question-detection")
text_router.warm_up()

queries = [
    "Arya Stark father",  # Keyword Query
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

result = text_router.run(text=queries[0])
next(iter(result))


import pandas as pd

results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    results["Query"].append(query)
    results["Output Branch"].append(next(iter(result)))
    results["Class"].append("Keyword Query" if next(iter(result)) == "LABEL_0" else "Question/Statement")

df = pd.DataFrame.from_dict(results)
print(df)



text_router = TransformersTextRouter(model="shahrukhx01/question-vs-statement-classifier")
text_router.warm_up()

queries = [
    "Who was the father of Arya Stark",  # Interrogative Query
    "Lord Eddard was the father of Arya Stark",  # Statement Query
]

results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    results["Query"].append(query)
    results["Output Branch"].append(next(iter(result)))
    results["Class"].append("Question" if next(iter(result)) == "LABEL_1" else "Statement")

print(pd.DataFrame.from_dict(results))


text_router = TransformersTextRouter(model="cardiffnlp/twitter-roberta-base-sentiment")
text_router.warm_up()

queries = [
    "What's the answer?",  # neutral query
    "Would you be so lovely to tell me the answer?",  # positive query
    "Can you give me the damn right answer for once??",  # negative query
]


sent_results = {"Query": [], "Output Branch": [], "Class": []}

for query in queries:
    result = text_router.run(text=query)
    sent_results["Query"].append(query)
    sent_results["Output Branch"].append(next(iter(result)))
    sent_results["Class"].append(
        {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}.get(next(iter(result)), "Unknown")
    )

print(pd.DataFrame.from_dict(sent_results))


from haystack_integrations.components.routers.transformers import TransformersZeroShotTextRouter

text_router = TransformersZeroShotTextRouter(labels=["music", "cinema"])
text_router.warm_up()
queries = [
    "In which films does John Travolta appear?",  # cinema
    "What is the Rolling Stones first album?",  # music
    "Who was Sergio Leone?",  # cinema
]
sent_results = {"Query": [], "Output Branch": []}

for query in queries:
    result = text_router.run(text=query)
    sent_results["Query"].append(query)
    sent_results["Output Branch"].append(next(iter(result)))

print(pd.DataFrame.from_dict(sent_results))





