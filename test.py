from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from groq import Groq
import os
import json
import time
import glob
import re
from dotenv import load_dotenv
from test_tesseract import process_all_pdfs

load_dotenv('.env', override=True)

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Process all PDFs or use existing files
all_metadata = {}
combined_text_path = "output/combined_text.txt"
all_metadata_path = "output/all_metadata.json"

# Check if we need to process PDFs
process_pdfs = False
if not os.path.exists(combined_text_path) or not os.path.exists(all_metadata_path):
    process_pdfs = True

if process_pdfs:
    print("Processing PDFs...")
    all_metadata, combined_text_path = process_all_pdfs('./data')
else:
    print("Using existing processed files")
    # Load combined metadata
    with open(all_metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)

# Load the combined text content
loader = TextLoader(combined_text_path)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
docs = text_splitter.split_documents(documents)

# Identify source document for each chunk
for i, doc in enumerate(docs):
    content = doc.page_content
    
    # Find which PDF this chunk belongs to
    source_pdf = None
    
    # Check for document markers in this chunk
    begin_match = re.search(r'--- BEGIN DOCUMENT: (.+?) ---', content)
    if begin_match:
        source_pdf = begin_match.group(1)
        
    end_match = re.search(r'--- END DOCUMENT: (.+?) ---', content)
    if end_match:
        source_pdf = end_match.group(1)
        
    page_match = re.search(r'----- (.+?): Page \d+ -----', content)
    if page_match:
        source_pdf = page_match.group(1)
        
    metadata_match = re.search(r'--- Document Metadata: (.+?) ---', content)
    if metadata_match:
        source_pdf = metadata_match.group(1)
    
    # If no match found in this chunk but previous chunk has a source, use that
    if not source_pdf and i > 0 and 'source_pdf' in docs[i-1].metadata:
        source_pdf = docs[i-1].metadata['source_pdf']

    if source_pdf:
        doc.metadata['source_pdf'] = source_pdf
    # No need to set an attribute directly on the Document object

model = SentenceTransformer("BAAI/bge-m3")

# Initialize Qdrant client
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=300
)
collection_name = "rag_qna"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)
    
# Create collection without payload_schema
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)

# Create payload indexes separately
client.create_payload_index(
    collection_name=collection_name,
    field_name="source_pdf",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_title",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_author",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_circular_number",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name=collection_name,
    field_name="doc_date",
    field_schema=PayloadSchemaType.KEYWORD
)

# Initialize Groq model
groq_model = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Create points with document metadata
points = []
for i, doc in enumerate(docs):
    embedding = model.encode(doc.page_content)
    
    # Add appropriate metadata to each chunk
    chunk_metadata = {
        "text": doc.page_content,
        "source": "combined_text.txt"
    }
    
    # Add source PDF if available
    source_pdf = doc.metadata.get('source_pdf')
    if source_pdf:
        chunk_metadata["source_pdf"] = source_pdf
        
        # Add document-specific metadata if available
        if source_pdf in all_metadata:
            pdf_metadata = all_metadata[source_pdf]
            chunk_metadata.update({
                "doc_title": pdf_metadata.get("title", ""),
                "doc_author": pdf_metadata.get("author", ""),
                "doc_circular_number": pdf_metadata.get("circular_number", ""),
                "doc_date": pdf_metadata.get("creation_date", "")
            })
    
    points.append(
        PointStruct(
            id=i + 1,
            vector=embedding.tolist(),
            payload=chunk_metadata
        )
    )

# Batch upsert points to Qdrant (in case there are many documents)
batch_size = 25
max_retries = 3

for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    retry_count = 0
    success = False
    
    while not success and retry_count < max_retries:
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True  # Ensure operation completes before continuing
            )
            print(f"Uploaded batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}")
            success = True
        except Exception as e:
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            print(f"Error uploading batch: {str(e)}")
            print(f"Retrying in {wait_time} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(wait_time)
    
    if not success:
        print(f"Failed to upload batch after {max_retries} attempts")

# Enhanced query function to use metadata
def query_with_metadata(query_text, metadata_filters=None):
    query_embedding = model.encode(query_text).tolist()
    
    # Apply metadata filtering if provided
    filter_query = None
    if metadata_filters:
        filter_conditions = []
        for key, value in metadata_filters.items():
            filter_key = key
            if key in ["title", "author", "circular_number", "date"]:
                filter_key = f"doc_{key}"
            filter_conditions.append({
                "key": filter_key,
                "match": {"value": value}
            })
        filter_query = {"must": filter_conditions}
    
    # Execute search with optional filters
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3,
        query_filter=filter_query
    )
    
    for hit in hits:
        print(f"ID: {hit.id}, Score: {hit.score}")
        # Print source PDF if available
        source_pdf = hit.payload.get("source_pdf", "")
        if source_pdf:
            print(f"  Source PDF: {source_pdf}")
        # Print metadata if available
        for key in hit.payload:
            if key.startswith("doc_") and hit.payload[key]:
                print(f"  {key}: {hit.payload[key]}")
        print(f"  Text: {hit.payload['text'][:100]}...")
    
    return hits

# Generate response with metadata-enhanced context
def generate_response(query, hits):
    # Include metadata in the context
    context_parts = []
    for hit in hits:
        # Add metadata prefix if available
        metadata_prefix = ""
        source_pdf = hit.payload.get("source_pdf", "")
        if source_pdf:
            metadata_prefix += f"Source document: {source_pdf}. "
        
        for key in hit.payload:
            if key.startswith("doc_") and hit.payload[key]:
                metadata_prefix += f"{key.replace('doc_', '')}: {hit.payload[key]}. "
        
        if metadata_prefix:
            context_parts.append(f"{metadata_prefix}\n{hit.payload['text']}")
        else:
            context_parts.append(hit.payload['text'])
    
    context = "\n\n".join(context_parts)
    sources = [hit.payload.get('source_pdf', 'unknown') for hit in hits]
    
    # Enhanced prompt with metadata awareness
    prompt = f"""
    Answer the question based on the context provided below:
    Context: {context}
    Question: {query}
    
    When answering, please reference any relevant document metadata like source documents, 
    circular numbers, dates, or document titles if they help provide context to your answer.
    
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
    print(f"\nAnswer: {answer}")
    print(f"Sources: {', '.join(set(sources))}")
    return answer

# Show available documents
print("\n=== Available Documents ===")
for pdf_name, metadata in all_metadata.items():
    print(f"- {pdf_name}")
    print(f"  Title: {metadata.get('title', 'N/A')}")
    if metadata.get('circular_number'):
        print(f"  Circular: {metadata.get('circular_number')}")
    print(f"  Date: {metadata.get('creation_date', 'N/A')}")
    print()

# Example query across all documents
print("\n=== Basic Query (All Documents) ===")
query = "What are the citizenship and age requirements for applicants to the post of Director?"
hits = query_with_metadata(query)
generate_response(query, hits)

# Query specific document if multiple documents exist
if len(all_metadata) > 1:
    print("\n=== Document-Specific Query ===")
    first_pdf = list(all_metadata.keys())[0]
    query = "What are the main points in this document?"
    hits = query_with_metadata(query, metadata_filters={"source_pdf": first_pdf})
    generate_response(query, hits)