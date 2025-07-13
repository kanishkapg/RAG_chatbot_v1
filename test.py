from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from dotenv import load_dotenv

load_dotenv('.env')

loader = TextLoader("output_tesseract.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(documents)

# Load the model (auto-downloads on first run)
model = SentenceTransformer("BAAI/bge-m3")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
collection_name = "rag_qna"
vector_size = 1024

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

client.get_collection(collection_name)

# Example: Embed a single sentence
text = """Eligible employees can avail of paid Maternity leave for 
a continuous period of 26 weeks, of which 8 weeks can be availed 
for the pre-natal period. """

embedding = model.encode(text)

points = [
    PointStruct(
        id = 1,
        vector = embedding.tolist(),
        payload = {"text": text, "source": "company_policy"}
    )
]

client.upsert(
    collection_name=collection_name,
    points=points
)

print(f"Embedding shape: {embedding.shape}")
print(embedding)