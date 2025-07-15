from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq
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

model = SentenceTransformer("BAAI/bge-m3")

#Initialize Qdrant client
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

#Initialize Groq model
groq_model = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

points = []
for i, doc in enumerate(docs):
    embedding = model.encode(doc.page_content)
    points.append(
        PointStruct(
            id=i + 1,
            vector=embedding.tolist(),
            payload={"text": doc.page_content, "source": "output_tesseract.txt"}
        )
    )

client.upsert(
    collection_name=collection_name,
    points=points
)

query = "Which act governs the appointment of Directors for Institutes or Centres for Higher Learning?"
query_embedding = model.encode(query).tolist()

hits = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    limit=3,
    with_payload=True,
    with_vectors=False
).points

for hit in hits:
    print(f"ID: {hit.id}, Score: {hit.score}")

context  = " ".join([hit.payload['text'] for hit in hits])
sources  = [hit.payload['source'] for hit in hits]

prompt = f"""
Answer the question based on the context provided below:
Context: {context}
Question: {query}
Provide a concise answer and cite the sources if applicable.
"""

response = groq_model.chat.completions.create(
    messages=[
        {"role": "user", "content": prompt}
    ],
    model="llama-3.1-8b-instant",
    max_tokens=512,
    temperature=0.3
)

answer = response.choices[0].message.content.strip()
print(f"Answer: {answer}")
print(f"Sources: {', '.join(sources)}")
