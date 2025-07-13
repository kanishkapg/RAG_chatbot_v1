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

# Load the model (auto-downloads on first run)
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

# Example: Embed a single sentence
text = """The company offers a comprehensive leave policy designed to support employee 
well-being and work-life balance. Employees are entitled to 18 days of paid annual 
leave per year, accrued monthly at a rate of 1.5 days. Additionally, the company 
provides 10 paid sick leaves, which can be utilized with prior notification and a 
medical certificate if absent for more than three consecutive days. Maternity and 
paternity leaves are granted as per statutory requirements—26 weeks for maternity 
and 15 days for paternity. Unused annual leaves can be carried forward to the next year, 
up to a maximum of 5 days, while sick leaves lapse annually. Leave requests must be 
submitted through the HR portal at least seven days in advance for planned leaves, 
except in emergencies. Approval depends on workload and team availability, with 
managers required to respond within three working days. Unplanned absences without 
prior notice may result in disciplinary action. Employees can also avail of unpaid 
leaves for exceptional circumstances, subject to management approval. The HR department 
conducts quarterly audits to ensure compliance and address discrepancies. For extended 
medical leaves beyond the sick leave quota, employees may apply for a special leave 
arrangement with supporting documentation. The policy is reviewed annually to align 
with industry standards and legal regulations. """

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

query = "How may days that a man can take as paid leaves per year?"
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
