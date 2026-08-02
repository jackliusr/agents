from haystack import Document, Pipeline, super_component
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from datasets import load_dataset


@super_component
class HybridRetriever:
    def __init__(self, document_store: InMemoryDocumentStore, embedder_model: str = "BAAI/bge-small-en-v1.5"):
        # Create the components
        embedding_retriever = InMemoryEmbeddingRetriever(document_store)
        bm25_retriever = InMemoryBM25Retriever(document_store)
        text_embedder = SentenceTransformersTextEmbedder(model=embedder_model)
        document_joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

        # Create the pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component("text_embedder", text_embedder)
        self.pipeline.add_component("embedding_retriever", embedding_retriever)
        self.pipeline.add_component("bm25_retriever", bm25_retriever)
        self.pipeline.add_component("document_joiner", document_joiner)

        # Connect the components
        self.pipeline.connect("text_embedder", "embedding_retriever")
        self.pipeline.connect("bm25_retriever", "document_joiner")
        self.pipeline.connect("embedding_retriever", "document_joiner")

# Load a dataset
dataset = load_dataset("HaystackBot/medrag-pubmed-chunk-with-embeddings", split="train")
docs = [Document(content=doc["contents"], embedding=doc["embedding"]) for doc in dataset]
document_store = InMemoryDocumentStore()
document_store.write_documents(docs)

# Create and run the HybridRetriever
query = "What treatments are available for chronic bronchitis?"
retriever = HybridRetriever(document_store)
result = retriever.run(
    text=query, query=query
)  # `query` variable will match with `text` and `query` inputs of components in the pipeline.


# Print the results
print(f"Found {len(result['documents'])} documents")
for i, doc in enumerate(result["documents"][:3]):  # Show first 3 documents
    print(f"\nDocument {i+1} (Score: {doc.score:.4f}):")
    print(doc.content[:200] + "...")
