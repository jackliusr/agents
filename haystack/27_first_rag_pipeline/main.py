from haystack.document_stores.in_memory import InMemoryDocumentStore
from dotenv import load_dotenv
load_dotenv()


document_store = InMemoryDocumentStore()


from datasets import load_dataset
from haystack import Document

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

docs_with_embeddings = doc_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])


from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")


from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store)


from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template, required_variables="*")



import os
from getpass import getpass
from haystack_integrations.components.generators.ollama import OllamaChatGenerator


  
chat_generator = OllamaChatGenerator(model="qwen3.6:35b-a3b-q4_K_M")

from haystack import Pipeline

basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")


examples = [
    "What does Rhodes Statue look like?",

    "Where is Gardens of Babylon?",
    "Why did people build Great Pyramid of Giza?",
    "What does Rhodes Statue look like?",
    "Why did people visit the Temple of Artemis?",
    "What is the importance of Colossus of Rhodes?",
    "What happened to the Tomb of Mausolus?",
    "How did Colossus of Rhodes collapse?",
]

for question in examples:
    print(f"Question: {question}")
    response = basic_rag_pipeline.run({"text_embedder": {"text": question}, 
                                    "prompt_builder": {"question": question}})
    print(response["llm"]["replies"][0].text)