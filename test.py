from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, Filter, MatchValue
from groq import Groq
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import Json
import json
import httpx
from test_tesseract import process_all_pdfs
from config import DATA_DIR, POSTGRES_CONFIG
import logging
from typing import List, Dict, Optional, Set

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('.env', override=True)

def get_db_connection():
    """Establish PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "llama-3.1-8b-instant"

    def _extract_metadata(self, text: str, source_file: str) -> Dict:
        """Extract metadata using LLM and store in PostgreSQL."""
        prompt = f"""
        You are an expert at extracting metadata from official documents and circulars. 
        
        Analyze the following text from {source_file} and extract metadata in VALID JSON format. Look carefully for:
        1. Document/circular numbers (often after "Circular No:", "Reference:", "No:", etc.)
        2. Titles or subjects (often after "Subject:", "Re:", "Title:", etc.)  
        3. Dates (issued date, effective date, etc.)
        4. Any references to superseded/repealed/amended/canceled documents. Classify them based on the exact language:
            Repealed: If the document explicitly repeals, revokes, withdraws, terminates, or cancels another.
            Superseded: If the document supersedes, replaces, overrides, or overrules another.
            Amended: If the document amends, modifies, revises, extends, or updates another.
        5. Department or issuing authority information
        
        Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null",
            "issued_date": "string or null",
            "effective_date": "string or null",
            "repealed_circulars": ["array of strings"],
            "superseded_circulars": ["array of strings"],
            "amended_circulars": ["array of strings"],
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
            logger.info(f"LLM Response for {source_file}: {metadata_str}")
            
            json_start = metadata_str.find('{')
            json_end = metadata_str.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = metadata_str[json_start:json_end]
                metadata = json.loads(json_str)
                
                if isinstance(metadata, dict):
                    validated_metadata = self._validate_metadata(metadata)
                    # Store metadata in PostgreSQL
                    self._store_metadata(validated_metadata)
                    logger.info(f"Successfully extracted and stored metadata for {source_file}: {validated_metadata}")
                    return validated_metadata
            
            logger.warning(f"Failed to parse JSON for {source_file}, returning default metadata")
            default_metadata = self._get_default_metadata(source_file)
            self._store_metadata(default_metadata)
            return default_metadata
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {source_file}: {e}, returning default metadata")
            default_metadata = self._get_default_metadata(source_file)
            self._store_metadata(default_metadata)
            return default_metadata
        except Exception as e:
            logger.error(f"Error with LLM extraction for {source_file}: {e}, returning default metadata")
            default_metadata = self._get_default_metadata(source_file)
            self._store_metadata(default_metadata)
            return default_metadata

    def _store_metadata(self, metadata: Dict) -> None:
        """Store metadata in PostgreSQL."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO document_metadata (filename, metadata, extraction_date)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (filename) DO UPDATE
                    SET metadata = %s, extraction_date = CURRENT_TIMESTAMP
                    """,
                    (metadata["source_file"], Json(metadata), Json(metadata))
                )
            conn.commit()
            logger.info(f"Stored metadata for {metadata['source_file']} in PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata['source_file']}: {e}")
        finally:
            conn.close()

    def _get_default_metadata(self, source_file: str) -> Dict:
        """Return default metadata when extraction fails."""
        return {
            "circular_number": None,
            "title": None,
            "issued_date": None,
            "effective_date": None,
            "repealed_circulars": [],
            "superseded_circulars": [],
            "amended_circulars": [],
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
        """Load documents from PostgreSQL and split into chunks."""
        chunked_docs = []
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pdf_files.filename, extracted_text, document_metadata.metadata 
                    FROM pdf_files 
                    LEFT JOIN document_metadata 
                    ON pdf_files.filename = document_metadata.filename
                    """
                )
                rows = cur.fetchall()
                
                for filename, extracted_text, metadata in rows:
                    if not extracted_text:
                        logger.warning(f"No extracted text for {filename}, skipping")
                        continue
                        
                    # If no metadata exists, extract it
                    if not metadata:
                        metadata = self._extract_metadata(extracted_text, filename)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    # Create a mock document for splitting
                    from langchain.docstore.document import Document
                    doc = Document(page_content=extracted_text, metadata={"source_file": filename})
                    docs = text_splitter.split_documents([doc])
                    
                    for i, doc_chunk in enumerate(docs):
                        chunked_docs.append({
                            "page_content": doc_chunk.page_content,
                            "metadata": {
                                **metadata,
                                "chunk_id": i + 1,
                                "total_chunks": len(docs)
                            }
                        })
        finally:
            conn.close()
        
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

    def _ensure_payload_indexes(self):
        """Create needed payload indexes for filtering."""
        # Each of these fields is used in filters / conditions
        index_fields = [
            "circular_number",
            "repealed_circulars",
            "superseded_circulars",
            "amended_circulars"
        ]
        for field in index_fields:
            try:
                # 'keyword' works for single string or array of strings
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema="keyword"
                )
                logger.info(f"Created/verified payload index on '{field}'")
            except Exception as e:
                # If it already exists or field absent yet, just log debug
                logger.debug(f"Index create attempt for '{field}' result: {e}")

    def initialize_collection(self, force_recreate: bool = False) -> None:
        """Initialize Qdrant collection only if it doesn't exist, then ensure indexes."""
        if force_recreate or not self.client.collection_exists(self.collection_name):
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
            logger.info(f"Initialized Qdrant collection {self.collection_name}")
        else:
            logger.info(f"Qdrant collection {self.collection_name} already exists, skipping creation")
        # Always ensure indexes (idempotent)
        self._ensure_payload_indexes()

    def get_existing_documents(self) -> set:
        """Get set of existing filenames in Qdrant."""
        try:
            if self.client.collection_exists(self.collection_name):
                # Scroll through all points to get existing filenames
                points = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,  # Adjust based on your data size
                    with_payload=True,
                    with_vectors=False
                )
                existing_filenames = {point.payload["source"] for point in points[0]}
                return existing_filenames
            return set()
        except Exception as e:
            logger.error(f"Failed to get existing documents from Qdrant: {e}")
            return set()
    
    def upsert_documents(self, documents: List[Dict]) -> None:
        """Upsert only new documents based on existing filenames."""
        if not documents:
            logger.info("No new documents to upsert")
            return

        existing_filenames = self.get_existing_documents()
        new_documents = [doc for doc in documents if doc["metadata"]["source_file"] not in existing_filenames]

        if not new_documents:
            logger.info("No new or updated documents to upsert")
            return
        
        points = []
        for i, doc in enumerate(new_documents):
            try:
                # Normalize metadata fields to avoid None (use empty list or omit)
                md = doc["metadata"].copy()
                for list_field in ["repealed_circulars", "superseded_circulars", "amended_circulars"]:
                    if md.get(list_field) is None:
                        md[list_field] = []
                if md.get("circular_number") is None:
                    # Optionally remove if None to reduce noise
                    md.pop("circular_number", None)

                embedding = self.embedding_model.encode(doc["page_content"])
                payload = {
                    "text": doc["page_content"],
                    "source": md.get("source_file"),
                    **md
                }
                # Stable point id
                import hashlib
                pid = int(hashlib.md5(f"{payload['source']}|{md.get('chunk_id')}".encode()).hexdigest()[:12], 16)
                points.append(
                    PointStruct(
                        id=pid,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )
            except Exception as e:
                logger.error(f"Failed to encode document chunk for {doc['metadata'].get('source_file')}: {e}")
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} new document chunks to Qdrant")
            
    def _get_all_successors(self, circular_nums: Set[str]) -> Set[str]:
        """Recursively find all successor circulars (newer ones that repeal/supersede/amend the given ones)."""
        all_related = set(circular_nums)
        to_process = set(circular_nums)
        
        while to_process:
            new_successors = set()
            for circ in to_process:
                if not circ:
                    continue
                filter = Filter(
                    should=[
                        FieldCondition(
                            key="repealed_circulars",
                            match=MatchValue(value=circ)
                        ),
                        FieldCondition(
                            key="superseded_circulars",
                            match=MatchValue(value=circ)
                        ),
                        FieldCondition(
                            key="amended_circulars",
                            match=MatchValue(value=circ)
                        )
                    ]
                )
                
                try:
                    scroll_result = self.client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=filter,
                        limit=100,
                        with_payload=True,
                        with_vectors=False
                    )
                    for point in scroll_result[0]:
                        successor_circ = point.payload.get("circular_number")
                        if successor_circ and successor_circ not in all_related:
                            new_successors.add(successor_circ)
                except Exception as e:
                    logger.error(f"Failed to scroll for successors of {circ}: {e}")
        
            all_related.update(new_successors)
            to_process = new_successors
        
        return all_related
                
                    
    def search_similar_documents(self, query: str, limit: int = 3, prefer_effective: bool = True) -> List[Dict]:
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Get initial search results
            initial_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=20, 
                with_payload=True,
                with_vectors=False
            ).points
            
            # Extract circular numbers from initial hits
            circular_nums = {
                hit.payload.get("circular_number") 
                for hit in initial_hits 
                if hit.payload.get("circular_number")
            }
            
            if circular_nums:
                # Find all related successors
                all_related = self._get_all_successors(circular_nums)
                
                if all_related:
                    # Filter for related documents
                    related_filter = Filter(
                        should=[
                            FieldCondition(
                                key="circular_number", 
                                match=MatchValue(value=c)
                            ) 
                            for c in all_related if c
                        ]
                    )
                    
                    hits = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        query_filter=related_filter,
                        limit=limit * 5,  # e.g., 15 if limit=3
                        with_payload=True,
                        with_vectors=False
                    ).points
                else:
                    hits = initial_hits
            else:
                hits = initial_hits
            
            # Identify invalid circulars (repealed or superseded)
            repealed_circulars = set()
            superseded_circulars = set()
            
            for hit in hits:
                if hit.payload.get("repealed_circulars"):
                    repealed_circulars.update(hit.payload["repealed_circulars"])
                if hit.payload.get("superseded_circulars"):
                    superseded_circulars.update(hit.payload["superseded_circulars"])
            
            invalid_circulars = repealed_circulars | superseded_circulars
            
            # Filter and sort hits
            filtered_hits = []
            for hit in hits:
                circular_number = hit.payload.get("circular_number")
                if prefer_effective and circular_number in invalid_circulars:
                    continue
                filtered_hits.append(hit)
            
            filtered_hits.sort(
                key=lambda x: (x.score, x.payload.get("effective_date", "0000-00-00")),
                reverse=True
            )
            filtered_hits = filtered_hits[:limit]
            
            # Format results
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "metadata": {
                        k: v 
                        for k, v in hit.payload.items() 
                        if k not in ["text", "source"]
                    }
                }
                for hit in filtered_hits
            ]
    
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []
class LLMResponseGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def generate_response(self, query: str, context: str, sources: List[str], metadata_list: List[Dict]) -> Dict:
        try:
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
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return {
                "answer": "Error generating response",
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
            if meta.get('superseded_circulars'):
                metadata_info += f"Superseded Circulars: {', '.join(meta['superseded_circulars'])}\n"
            if meta.get('amended_circulars'):
                metadata_info += f"Amended Circulars: {', '.join(meta['amended_circulars'])}\n"

        return f"""
        Answer the question based on the context provided below. Prioritize information from documents that are effective
        (not listed in any 'repealed_circulars' or 'superseded_circulars'). For documents listed in 'amended_circulars', 
        include information from both the original and the amendments, preferring updates from the amendments.

        If multiple documents are relevant, prefer the one with the most recent effective_date.
        Cite the circular number, source file, and effective date in your response.
        Important: Always provide answer strictly based on the context provided, do not make assumptions or provide information not present in the documents.

        Document Metadata:
        {metadata_info}

        Context: {context}

        Question: {query}

        Provide a concise answer and include citations with circular numbers, source files, and effective dates where applicable.
        """

class RAGSystem:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabaseManager()
        self.llm = LLMResponseGenerator()

    def initialize_system(self) -> None:
        logger.info("Processing PDFs and extracting text...")
        extracted_texts = process_all_pdfs(self.data_dir)
        
        logger.info("Processing documents and extracting metadata...")
        documents = self.document_processor.load_and_split_documents()
        logger.info(f"Split documents into {len(documents)} chunks")
        
        logger.info("Initializing vector database...")
        self.vector_db.initialize_collection()
        self.vector_db.upsert_documents(documents)
        logger.info("RAG system initialization complete!")

    def query(self, question: str) -> Dict:
        try:
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
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "question": question,
                "answer": "Error processing query",
                "sources": [],
                "relevant_documents": []
            }

if __name__ == "__main__":
    # Create PostgreSQL tables if they don't exist
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files (
                    filename VARCHAR(255) PRIMARY KEY,
                    file_hash VARCHAR(64),
                    extracted_text TEXT,
                    extraction_date TIMESTAMP,
                    UNIQUE (filename)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    filename VARCHAR(255) PRIMARY KEY,
                    metadata JSONB,
                    extraction_date TIMESTAMP,
                    FOREIGN KEY (filename) REFERENCES pdf_files(filename)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON pdf_files(file_hash)")
        conn.commit()
        logger.info("PostgreSQL tables created or verified")
    finally:
        conn.close()

    rag_system = RAGSystem(data_dir=DATA_DIR)
    # Force recreate once to add indexes; later set to False
    rag_system.vector_db.initialize_collection(force_recreate=True)
    rag_system.initialize_system()
    
    query = "How many casual leaves are allowed per month?"
    result = rag_system.query(query)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print("\nRelevant documents:")
    for doc in result["relevant_documents"]:
        print(f"\nID: {doc['id']}, Score: {doc['score']:.4f}")
        print(f"Text: {doc['text'][:200]}...")
        print(f"Metadata: {doc['metadata']}")