from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
from typing import List, Dict
import json
import logging

from src.db import get_db_connection
from src.graph_manager import Neo4jManager
from src.taxonomy import TaxonomyManager

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-120b"
        self.graph_manager = Neo4jManager()
        self.taxonomy = TaxonomyManager()

    def _extract_metadata(self, text: str, source_file: str) -> Dict:
        """Extract metadata using LLM and store in PostgreSQL."""
        prompt = f"""
        You are an expert at extracting metadata from official documents and circulars. 
        
        Analyze the following text from {source_file} and extract metadata in VALID JSON format. Look carefully for:
        1. Document/circular numbers
        2. Titles or subjects
        3. Dates (issued date, effective date, etc.)
        4. Department or issuing authority information
        5. Policy category/area that this circular addresses (both raw label and hierarchical category path if possible)
        6. Document Relationships (REPEALS, AMENDS, SUPERSEDES, REFERENCES, EXTENDS)
        7. Versioning Information
        
        Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null", 
            "issued_date": "string or null",
            "effective_date": "string or null",
            "policy_category": "string or null",   // legacy single label
            "policy_category_raw": "string or null", // raw extracted label
            "category_path": ["array of strings"], // hierarchical path if known
            "department": "string or null",
            "departments": ["array of strings"],
            "roles": ["array of strings"],
            "global": true or false,
            "document_relationships": {{
                "repeals": ["array of circular numbers"],
                "amends": ["array of circular numbers"],
                "supersedes": ["array of circular numbers"],
                "references": ["array of circular numbers"],
                "extends": ["array of circular numbers"]
            }},
            "other": {{"key": "value"}},
            "version": "string or null",
            "previous_versions": ["array of strings"],
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
                metadata = json.loads(metadata_str[json_start:json_end])
                metadata["source_file"] = source_file

                # ---- Category Normalization ----
                raw_cat = metadata.get("policy_category_raw") or metadata.get("policy_category")
                metadata["policy_category_raw"] = raw_cat

                # ---- Category Normalization & Multi-Detection ----
                cat_paths = []

                if raw_cat:
                    normalized = self.taxonomy.normalize(raw_cat)
                    if normalized:
                        cat_paths.append(normalized)
                    else:
                        try:
                            conn = get_db_connection()
                            self.taxonomy.suggest_unknown(raw_cat, conn)
                        except Exception:
                            pass

                # Fallback: also scan document text for multiple categories
                detected = self.taxonomy.match_multiple(text)
                for path in detected:
                    if path not in cat_paths:
                        cat_paths.append(path)

                metadata["category_path"] = cat_paths if cat_paths else []


                # ---- Department Normalization ----
                if not isinstance(metadata.get("departments"), list):
                    dept = metadata.get("department")
                    metadata["departments"] = [dept] if dept else []

                # ---- Roles ----
                if not isinstance(metadata.get("roles"), list):
                    metadata["roles"] = []

                # ---- Global Flag ----
                metadata["global"] = bool(metadata.get("global", False))

                # ---- Implicit supersession detection ----
                lower_text = text.lower()
                implicit_phrases = [
                    "in this respect",
                    "in this regard",
                    "insofar as",
                    "in so far as",
                    "to the extent",
                    "in that respect",
                    "in this behalf"
                ]
                metadata["_implicit_actions"] = any(ph in lower_text for ph in implicit_phrases)

                validated_metadata = self._validate_metadata(metadata)
                self._store_metadata(validated_metadata)
                return validated_metadata
            
            logger.warning(f"Failed to parse JSON for {source_file}, returning default metadata")
            default_metadata = self._get_default_metadata(source_file)
            self._store_metadata(default_metadata)
            return default_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata for {source_file}: {e}")
            default_metadata = self._get_default_metadata(source_file)
            self._store_metadata(default_metadata)
            return default_metadata

    def _store_metadata(self, metadata: Dict) -> None:
        """Store metadata in PostgreSQL and Neo4j."""
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
                    (metadata["source_file"], json.dumps(metadata), json.dumps(metadata))
                )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata['source_file']}: {e}")
        finally:
            conn.close()
            
        try:
            self.graph_manager.add_document(metadata)
        except Exception as e:
            logger.error(f"Failed to add document to Neo4j: {e}")
            
    def _get_default_metadata(self, source_file: str) -> Dict:
        """Return default metadata when extraction fails."""
        return {
            "circular_number": None,
            "title": None,
            "issued_date": None,
            "effective_date": None,
            "policy_category": None,
            "policy_category_raw": None,
            "category_path": [],
            "department": None,
            "departments": [],
            "roles": [],
            "global": False,
            "document_relationships": {
                "repeals": [],
                "amends": [],
                "supersedes": [],
                "references": [],
                "extends": [],
            },
            "other": {},
            "version": None,
            "previous_versions": [],
            "source_file": source_file
        }

    def _validate_metadata(self, metadata: Dict) -> Dict:
        defaults = self._get_default_metadata(metadata.get("source_file", "unknown"))
        for key, value in defaults.items():
            if key not in metadata:
                metadata[key] = value
        for key, value in metadata.items():
            if isinstance(value, str) and value.strip().lower() in ["", "null", "none", "n/a"]:
                metadata[key] = None
        return metadata

    def load_and_split_documents(self) -> List[Dict]:
        """Load documents from PostgreSQL and split into chunks."""
        chunked_docs = []
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT pdf_files.filename, extracted_text, document_metadata.metadata 
                    FROM pdf_files 
                    LEFT JOIN document_metadata 
                    ON pdf_files.filename = document_metadata.filename
                """)
                rows = cur.fetchall()
                for filename, extracted_text, metadata in rows:
                    if not extracted_text:
                        continue
                    if not metadata:
                        metadata = self._extract_metadata(extracted_text, filename)
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    from langchain.docstore.document import Document
                    doc = Document(page_content=extracted_text, metadata={"source_file": filename})
                    docs = text_splitter.split_documents([doc])
                    for i, doc_chunk in enumerate(docs):
                        chunked_docs.append({
                            "page_content": doc_chunk.page_content,
                            "metadata": {**metadata, "chunk_id": i + 1, "total_chunks": len(docs)}
                        })
        finally:
            conn.close()
        return chunked_docs
