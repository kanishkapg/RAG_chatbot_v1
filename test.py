from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from groq import Groq
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv('.env', override=True)

class DocumentProcessor:
    
    def __init__(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_and_split_documents(self) -> List[Dict]:
        loader = TextLoader(self.file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        docs = text_splitter.split_documents(documents)
        return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]


class VectorDatabaseManager:
    
    def __init__(self, collection_name: str = "rag_qna", vector_size: int = 1024):
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
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=embedding.tolist(),
                    payload={"text": doc["page_content"], "source": source}
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
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
                "source": hit.payload["source"]
            }
            for hit in hits
        ]


class LLMResponseGenerator:
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
    
    def generate_response(self, query: str, context: str, sources: List[str]) -> Dict:
        prompt = self._build_prompt(query, context, sources)
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=512,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "sources": sources
        }
    
    def _build_prompt(self, query: str, context: str, sources: List[str]) -> str:
        """Construct the prompt for the LLM"""
        return f"""
        Answer the question based on the context provided below:
        Context: {context}
        Question: {query}
        Provide a concise answer and cite the sources if applicable.
        """


class RAGSystem:
    
    def __init__(self, document_path: str):
        self.document_path = document_path
        self.document_processor = DocumentProcessor(document_path)
        self.vector_db = VectorDatabaseManager()
        self.llm = LLMResponseGenerator()
    
    def initialize_system(self) -> None:
        documents = self.document_processor.load_and_split_documents()
        self.vector_db.initialize_collection()
        self.vector_db.upsert_documents(documents, source=self.document_path)
    
    def query(self, question: str) -> Dict:
        hits = self.vector_db.search_similar_documents(question)
        
        context = " ".join([hit["text"] for hit in hits])
        sources = list(set([hit["source"] for hit in hits]))
        
        response = self.llm.generate_response(question, context, sources)
        
        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"],
            "relevant_documents": hits
        }


if __name__ == "__main__":
    rag_system = RAGSystem("output_tesseract.txt")
    rag_system.initialize_system()
    
    query = "In which languages should the notice calling for applications be published?"
    result = rag_system.query(query)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
        print(f"\nID: {doc['id']}, Score: {doc['score']}")
        print(f"Text: {doc['text'][:200]}...")