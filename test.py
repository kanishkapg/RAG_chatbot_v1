from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import json
import httpx
import re

load_dotenv('.env', override=True)

class DocumentProcessor:
    
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"
    
    def _preprocess_text_for_metadata(self, text: str) -> str:
        """Preprocess text to focus on the header/metadata section."""
        # Take first 2000 characters which usually contain the metadata
        header_text = text[:2000]
        
        # Look for common patterns in circulars
        lines = header_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in [
                'circular', 'number', 'date', 'subject', 'title', 'issued', 
                'effective', 'supersede', 'repeal', 'cancel', 'department'
            ]):
                relevant_lines.append(line)
        
        # If we found relevant lines, use them; otherwise use the header
        if relevant_lines:
            return '\n'.join(relevant_lines)
        else:
            return header_text
    
    def _extract_metadata_with_regex(self, text: str) -> Dict:
        metadata = {
            "circular_number": None,
            "title": None,
            "issued_date": None,
            "effective_date": None,
            "repealed_circulars": [],
            "other": {}
        }
        
        circular_patterns = [
            r'circular\s+no\.?\s*:?\s*([A-Z0-9/-]+)',
            r'circular\s+number\s*:?\s*([A-Z0-9/-]+)',
            r'cir\.\s*no\.?\s*:?\s*([A-Z0-9/-]+)',
            r'reference\s*:?\s*([A-Z0-9/-]+)',
        ]
        
        for pattern in circular_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["circular_number"] = match.group(1).strip()
                break
        
        date_patterns = [
            r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'issued\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+\w+\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["issued_date"] = match.group(1).strip()
                break
        
        title_patterns = [
            r'subject\s*:?\s*([^\n]+)',
            r'title\s*:?\s*([^\n]+)',
            r're\s*:?\s*([^\n]+)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) > 10:  # Ensure it's a meaningful title
                    metadata["title"] = title
                break
        
        return metadata
    
    def _extract_metadata(self, text: str) -> Dict:
        processed_text = self._preprocess_text_for_metadata(text)
        
        prompt = f"""
        You are an expert at extracting metadata from official documents and circulars. 
        
        Analyze the following text and extract metadata in VALID JSON format. Look carefully for:
        1. Document/circular numbers (often after "Circular No:", "Reference:", "No:", etc.)
        2. Titles or subjects (often after "Subject:", "Re:", "Title:", etc.)  
        3. Dates (issued date, effective date, etc.)
        4. Any references to superseded/repealed documents
        5. Department or issuing authority information
        
        Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null", 
            "issued_date": "string or null",
            "effective_date": "string or null",
            "repealed_circulars": ["array of strings"],
            "other": {{"key": "value"}}
        }}
        
        Document text:
        {processed_text}
        
        JSON Response:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=512,
                temperature=0.1
            )
            
            metadata_str = response.choices[0].message.content.strip()
            print(f"LLM Response: {metadata_str}")
            
            json_start = metadata_str.find('{')
            json_end = metadata_str.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = metadata_str[json_start:json_end]
                metadata = json.loads(json_str)
                
                if isinstance(metadata, dict):
                    return self._validate_metadata(metadata)
            
            print("LLM extraction failed, falling back to regex...")
            
        except Exception as e:
            print(f"Error with LLM extraction: {e}")
        
        regex_metadata = self._extract_metadata_with_regex(text)
        print(f"Regex extracted metadata: {regex_metadata}")
        return regex_metadata
    
    def _validate_metadata(self, metadata: Dict) -> Dict:
        default_metadata = {
            "circular_number": None,
            "title": None,
            "issued_date": None,
            "effective_date": None,
            "repealed_circulars": [],
            "other": {}
        }
        
        for key in default_metadata:
            if key not in metadata:
                metadata[key] = default_metadata[key]
        
        for key, value in metadata.items():
            if isinstance(value, str) and (value.strip() == "" or value.lower() in ["null", "none", "n/a"]):
                metadata[key] = None
            elif isinstance(value, list) and len(value) == 0:
                metadata[key] = []
        
        return metadata
    
    def load_and_split_documents(self) -> List[Dict]:
        loader = TextLoader(self.file_path)
        documents = loader.load()
        
        full_text = documents[0].page_content
        
        metadata = self._extract_metadata(full_text)
        print(f"Extracted metadata: {metadata}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = text_splitter.split_documents(documents)
        
        chunked_docs = []
        for i, doc in enumerate(docs):
            chunked_docs.append({
                "page_content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    **metadata,
                    "chunk_id": i + 1,
                    "total_chunks": len(docs)
                }
            })
        
        return chunked_docs

class VectorDatabaseManager:
    
    def __init__(self, collection_name: str = "rag_qna", vector_size: int = 1024):
        http_client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0, read=10.0, write=10.0)
        )
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_model = SentenceTransformer("BAAI/bge-m3")
    
    def initialize_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
    
    def upsert_documents(self, documents: List[Dict], source: str = "unknown") -> None:
        points = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc["page_content"])
            payload = {
                "text": doc["page_content"],
                "source": source,
                **doc["metadata"]
            }
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Upserted {len(points)} document chunks to Qdrant")
    
    def search_similar_documents(self, query: str, limit: int = 3) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query).tolist()
        
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_payload=True,
            with_vectors=False
        ).points
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "source"]}
            }
            for hit in hits
        ]

class LLMResponseGenerator:
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
    
    def generate_response(self, query: str, context: str, sources: List[str], metadata_list: List[Dict]) -> Dict:
        prompt = self._build_prompt(query, context, sources, metadata_list)
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=512,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": sources,
            "metadata": metadata_list
        }
    
    def _build_prompt(self, query: str, context: str, sources: List[str], metadata_list: List[Dict]) -> str:
        metadata_info = ""
        for meta in metadata_list:
            if meta.get('circular_number'):
                metadata_info += f"Circular Number: {meta['circular_number']}\n"
            if meta.get('title'):
                metadata_info += f"Title: {meta['title']}\n"
            if meta.get('issued_date'):
                metadata_info += f"Issued Date: {meta['issued_date']}\n"
        
        return f"""
        Answer the question based on the context provided below. Use the document metadata to provide accurate references.

        Document Metadata:
        {metadata_info}

        Context: {context}
        
        Question: {query}
        
        Provide a concise answer and cite the sources with circular numbers and dates where available.
        """

class RAGSystem:
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.document_processor = DocumentProcessor(document_path)
        self.vector_db = VectorDatabaseManager()
        self.llm = LLMResponseGenerator()
    
    def initialize_system(self) -> None:
        print("Processing documents and extracting metadata...")
        documents = self.document_processor.load_and_split_documents()
        print(f"Split document into {len(documents)} chunks")
        
        print("Initializing vector database...")
        self.vector_db.initialize_collection()
        self.vector_db.upsert_documents(documents, source=self.document_path)
        print("RAG system initialization complete!")
    
    def query(self, question: str) -> Dict:
        hits = self.vector_db.search_similar_documents(question)
        
        context = " ".join([hit["text"] for hit in hits])
        sources = list(set([hit["source"] for hit in hits]))
        metadata_list = [hit["metadata"] for hit in hits]
        
        response = self.llm.generate_response(question, context, sources, metadata_list)
        
        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"],
            "relevant_documents": hits
        }

if __name__ == "__main__":
    rag_system = RAGSystem("output_tesseract.txt")
    rag_system.initialize_system()
    
    query = "What are the citizenship and age requirements for applicants to the post of Director?"
    result = rag_system.query(query)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
        print(f"\nID: {doc['id']}, Score: {doc['score']:.4f}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"Metadata: {doc['metadata']}")