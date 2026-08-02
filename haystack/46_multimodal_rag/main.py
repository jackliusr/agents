from dotenv import load_dotenv

load_dotenv()

import shutil
from urllib.request import Request, urlopen

HEADERS = {"User-Agent": "haystack-tutorials"}

def download_image(url, filename):
    req = Request(url, headers=HEADERS)
    with urlopen(req) as r, open(filename, "wb") as f:
        shutil.copyfileobj(r, f)

# download_image(
#     "https://upload.wikimedia.org/wikipedia/commons/2/26/Pink_Lady_Apple_%284107712628%29.jpg",
#     "apple.jpg",
# )
# download_image(
#     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg/960px-Cattle_tyrant_%28Machetornis_rixosa%29_on_Capybara.jpg",
#     "capybara.jpg",
# )

from PIL import Image

img = Image.open("apple.jpg")
# We resize the image here just to avoid it taking up too much space in the notebook
img_resized = img.resize((img.width // 6, img.height // 6))
img_resized

from haystack.components.converters.image import ImageFileToDocument

image_file_converter = ImageFileToDocument()
image_docs = image_file_converter.run(sources=["apple.jpg", "capybara.jpg"])["documents"]
print(image_docs)


from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersTextEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentImageEmbedder

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/clip-ViT-L-14", progress_bar=False)
image_embedder = SentenceTransformersDocumentImageEmbedder(
    model="sentence-transformers/clip-ViT-L-14", progress_bar=False
)

# Warm up the models to load them
text_embedder.warm_up()
image_embedder.warm_up()

import torch
from sentence_transformers import util

query = "A red apple on a white background"
text_embedding = text_embedder.run(text=query)["embedding"]
image_docs_with_embeddings = image_embedder.run(image_docs)["documents"]

# Compare the similarities between the query and two image documents
for doc in image_docs_with_embeddings:
    similarity = util.cos_sim(torch.tensor(text_embedding), torch.tensor(doc.embedding))
    print(f"Similarity with {doc.meta['file_path'].split('/')[-1]}: {similarity.item():.2f}")

# download_image("https://arxiv.org/pdf/1706.03762", "attention_is_all_you_need.pdf")

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.image import ImageFileToDocument
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentImageEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.routers.file_type_router import FileTypeRouter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

# Create our document store
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Define our components
file_type_router = FileTypeRouter(mime_types=["application/pdf", "image/jpeg"])
image_converter = ImageFileToDocument()
pdf_converter = PyPDFToDocument()
pdf_splitter = DocumentSplitter(split_by="page", split_length=1)
text_doc_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/clip-ViT-L-14", progress_bar=False
)
image_embedder = SentenceTransformersDocumentImageEmbedder(
    model="sentence-transformers/clip-ViT-L-14", progress_bar=False
)
document_writer = DocumentWriter(doc_store)

# Create the Indexing Pipeline
indexing_pipe = Pipeline()
indexing_pipe.add_component("file_type_router", file_type_router)
indexing_pipe.add_component("pdf_converter", pdf_converter)
indexing_pipe.add_component("pdf_splitter", pdf_splitter)
indexing_pipe.add_component("image_converter", image_converter)
indexing_pipe.add_component("text_doc_embedder", text_doc_embedder)
indexing_pipe.add_component("image_doc_embedder", image_embedder)
indexing_pipe.add_component("document_writer", document_writer)

indexing_pipe.connect("file_type_router.application/pdf", "pdf_converter.sources")
indexing_pipe.connect("pdf_converter.documents", "pdf_splitter.documents")
indexing_pipe.connect("pdf_splitter.documents", "text_doc_embedder.documents")
indexing_pipe.connect("file_type_router.image/jpeg", "image_converter.sources")
indexing_pipe.connect("image_converter.documents", "image_doc_embedder.documents")
indexing_pipe.connect("text_doc_embedder.documents", "document_writer.documents")
indexing_pipe.connect("image_doc_embedder.documents", "document_writer.documents")

indexing_result = indexing_pipe.run(
    data={"file_type_router": {"sources": ["attention_is_all_you_need.pdf", "apple.jpg"]}}
)

indexed_documents = doc_store.filter_documents()
print(f"Indexed {len(indexed_documents)} documents")

# Searching Image + Text

from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

retriever = InMemoryEmbeddingRetriever(document_store=doc_store)
text_embedding = text_embedder.run(text="An image of an apple")["embedding"]
results = retriever.run(text_embedding)["documents"]

for idx, doc in enumerate(results[:5]):
    print(f"Document {idx+1}:")
    print(f"Score: {doc.score}")
    print(f"File Path: {doc.meta['file_path']}")
    print("")

# Building an Image + Text Indexing Pipeline using the LLMDocumentContentExtractor

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.converters.image import ImageFileToDocument
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentEmbedder
from haystack_integrations.components.embedders.sentence_transformers import SentenceTransformersDocumentImageEmbedder
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.routers.file_type_router import FileTypeRouter
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.extractors.image import LLMDocumentContentExtractor

# Create our document store
doc_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

# Define our components
file_type_router = FileTypeRouter(mime_types=["application/pdf", "image/jpeg"])
image_converter = ImageFileToDocument()
pdf_converter = PyPDFToDocument()
pdf_splitter = DocumentSplitter(split_by="page", split_length=1)
document_writer = DocumentWriter(doc_store)

# Now we use high-performing text-only embedders
doc_embedder = SentenceTransformersDocumentEmbedder(model="mixedbread-ai/mxbai-embed-large-v1", progress_bar=False)

# New LLMDocumentContentExtractor
llm_content_extractor = LLMDocumentContentExtractor(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), # you can replace this with other chat generators that support vision
    max_workers=1,  # This can be used to parallelize the content extraction
)