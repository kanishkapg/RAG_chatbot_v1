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
from test_tesseract import process_all_pdfs
from config import DATA_DIR

load_dotenv('.env', override=True)

class DocumentProcessor:
    
    def __init__(self, text_dir: str = "extracted_texts", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_dir = text_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"
    
    def _extract_metadata(self, text: str, source_file: str) -> Dict:
        """Extract metadata using LLM only - no regex fallback."""
        
        prompt = f"""
        You are an expert at extracting metadata from official documents and circulars. 
        
        Analyze the following text from {source_file}and extract metadata in VALID JSON format. Look carefully for:
        1. Document/circular numbers (often after "Circular No:", "Reference:", "No:", etc.)
        2. Titles or subjects (often after "Subject:", "Re:", "Title:", etc.)  
        3. Dates (issued date, effective date, etc.)
        4. Any references to superseded/repealed/cancelled documents
        5. Department or issuing authority information
        
        Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null", 
            "issued_date": "string or null",
            "effective_date": "string or null",
            "repealed_circulars": ["array of strings"],
            "other": {{"key": "value"}},
            "source_file": "string"
        }}
        
        Document text:
        {text}
        
        JSON Response:"""
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a precise metadata extractor. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=800,
                temperature=0.1
            )
            
            metadata_str = response.choices[0].message.content.strip()
            print(f"LLM Response for {source_file}: {metadata_str}")
            
            json_start = metadata_str.find('{')
            json_end = metadata_str.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = metadata_str[json_start:json_end]
                metadata = json.loads(json_str)
                
                if isinstance(metadata, dict):
                    validated_metadata = self._validate_metadata(metadata)
                    print(f"Successfully extracted metadata for {source_file}: {validated_metadata}")
                    return validated_metadata
            
            print(f"Failed to parse JSON for {source_file}, returning default metadata")
            return self._get_default_metadata(source_file)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error for {source_file}: {e}, returning default metadata")
            return self._get_default_metadata(source_file)
        except Exception as e:
            print(f"Error with LLM extraction for {source_file}: {e}, returning default metadata")
            return self._get_default_metadata(source_file)
    
    def _get_default_metadata(self, source_file: str) -> Dict:
        """Return default metadata when extraction fails completely."""
        return {
            "circular_number": None,
            "title": None,
            "issued_date": None,
            "effective_date": None,
            "repealed_circulars": [],
            "other": {},
            "source_file": source_file
        }
    
    def _validate_metadata(self, metadata: Dict) -> Dict:
        default_metadata = self._get_default_metadata(metadata.get("source_file", "unknown"))
        
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
        
        chunked_docs = []
        
        if not os.path.exists(self.text_dir):
            raise FileNotFoundError(f"Text directory {self.text_dir} does not exist.")
        
        text_files = [f for f in os.listdir(self.text_dir) if f.endswith('.txt')]
        
        for text_file in text_files:
            self.text_file_path = os.path.join(self.text_dir, text_file)
            loader = TextLoader(self.text_file_path)
            documents = loader.load()
            
            full_text = documents[0].page_content
            metadata = self._extract_metadata(full_text, text_file)
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
            docs = text_splitter.split_documents(documents)
        
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
            timeout=httpx.Timeout(1140.0, connect=300.0, read=420.0, write=420.0)
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
    
    def upsert_documents(self, documents: List[Dict]) -> None:
        points = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_model.encode(doc["page_content"])
            payload = {
                "text": doc["page_content"],
                "source": doc["metadata"]["source_file"],
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
    
    def search_similar_documents(self, query: str, limit: int = 3, prefer_effective: bool = True) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query).tolist()
        
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit*2,
            with_payload=True,
            with_vectors=False
        ).points
        
        # Filter and prioritize effective circulars
        filtered_hits = []
        repealed_circulars = set()
        for hit in hits:
            if hit.payload.get("repealed_circulars"):
                repealed_circulars.update(hit.payload["repealed_circulars"])
        
        for hit in hits:
            circular_number = hit.payload.get("circular_number")
            if prefer_effective and circular_number in repealed_circulars:
                continue  # Skip repealed circulars
            filtered_hits.append(hit)
        
        # Sort by score and effective_date (newer first)
        filtered_hits.sort(key=lambda x: (
            x.score,
            x.payload.get("effective_date", "") or ""
        ), reverse=True)
        
        # Limit to requested number
        filtered_hits = filtered_hits[:limit]
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload["text"],
                "source": hit.payload["source"],
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "source"]}            }
            for hit in filtered_hits
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
            metadata_info += f"\nDocument from {meta['source_file']}:\n"
            if meta.get('circular_number'):
                metadata_info += f"Circular Number: {meta['circular_number']}\n"
            if meta.get('title'):
                metadata_info += f"Title: {meta['title']}\n"
            if meta.get('issued_date'):
                metadata_info += f"Issued Date: {meta['issued_date']}\n"
            if meta.get('effective_date'):
                metadata_info += f"Effective Date: {meta['effective_date']}\n"
            if meta.get('repealed_circulars'):
                metadata_info += f"Repealed Circulars: {', '.join(meta['repealed_circulars'])}\n"
                
        return f"""
        Answer the question based on the context provided below. Prioritize information from documents that are effective 
        (not listed in any 'repealed_circulars'). If multiple documents are relevant, prefer the one with the most recent effective_date. 
        Cite the circular number, source file, and effective date in your response.

        Document Metadata:
        {metadata_info}

        Context: {context}
        
        Question: {query}
        
        Provide a concise answer and include citations with circular numbers, source files, and effective dates where applicable.
        """

class RAGSystem:

    def __init__(self, data_dir: str = DATA_DIR, text_dir: str = "extracted_texts"):
        self.data_dir = data_dir
        self.text_dir = text_dir
        self.document_processor = DocumentProcessor(text_dir=self.text_dir)
        self.vector_db = VectorDatabaseManager()
        self.llm = LLMResponseGenerator()
    
    def initialize_system(self) -> None:
        print("Processing documents and extracting metadata...")
        extracted_texts = process_all_pdfs(self.data_dir, self.text_dir)
        
        print("Processing documents and extracting metadata...")
        documents = self.document_processor.load_and_split_documents()
        print(f"Split documents into {len(documents)} chunks")
        
        print("Initializing vector database...")
        self.vector_db.initialize_collection()
        self.vector_db.upsert_documents(documents)
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
    rag_system = RAGSystem(data_dir=DATA_DIR, text_dir="extracted_texts")
    rag_system.initialize_system()
    
    query = "How should applications and documents be submitted according to the new procedure?"
    result = rag_system.query(query)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
        print(f"\nID: {doc['id']}, Score: {doc['score']:.4f}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"Metadata: {doc['metadata']}")