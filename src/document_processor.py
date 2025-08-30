from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
import os
from typing import List, Dict
import json
import logging
from datetime import datetime
import re

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
        4. Department or issuing authority information (CHRO,GCEO,CAO,CFO etc.)
        5. Policy category/area that this circular addresses (both raw label and hierarchical category path if possible)
        6. Document Relationships (REPEALS, AMENDS, SUPERSEDES, REFERENCES, EXTENDS)
        7. Versioning Information
        
    Additionally, detect whether the text contains a narrative supersede (for example "This circular supersedes all prior maternity/paternity policies" or "will supersede all circulars issued in this respect") where no explicit circular numbers are listed. If such a narrative supersede exists, set "implicit_supersedes": true and provide an optional short "supersedes_scope" string describing the scope (e.g. "maternity/paternity policies" or "policies in this category for HR").

    Return ONLY a valid JSON object with these exact fields:
        {{
            "circular_number": "string or null",
            "title": "string or null", 
            "issued_date": "string or null", //YYYY−MM−DD format
            "effective_date": "date or null",  //YYYY−MM−DD format
            "policy_category": "date or null",   // legacy single label
            "policy_category_raw": "string or null", // raw extracted label
            "category_path": ["array of strings"], // hierarchical path if known
            "department": "string or null", // CHRO,GCEO,CAO,CFO etc. Don't include any other things
            "roles": ["array of strings"],
            "implicit_supersedes": true or false,
            "supersedes_scope": "string or null",
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


                # ---- Roles ----
                if not isinstance(metadata.get("roles"), list):
                    metadata["roles"] = []

                # ---- Implicit supersession detection ----
                # look for sentences that indicate superseding without explicit circular numbers
                # e.g. "This circular supersedes all prior maternity/paternity policies, effective 1 February 2024." 
                # We'll set `implicit_supersedes` boolean and try to capture a short `supersedes_scope` phrase.
                metadata["implicit_supersedes"] = False
                metadata["supersedes_scope"] = None

                supersede_regex = re.compile(r"(?:this\s+circular\s+)?(supersede|supersedes|will\s+supersede|shall\s+supersede|superseding)\s+(all\s+prior\s+)?(?P<scope>[^.,;\n]+)", re.IGNORECASE)
                m = supersede_regex.search(text)
                if m:
                    # heuristically accept matches that mention policies, circulars, or similar
                    scope = m.groupdict().get("scope")
                    if scope and re.search(r"policy|policies|circular|circulars|notice|guideline|procedure|rule", scope, re.IGNORECASE):
                        metadata["implicit_supersedes"] = True
                        # trim trailing words and punctuation
                        scope = scope.strip().strip(' .;:,')
                        # limit length
                        if len(scope) > 150:
                            scope = scope[:150].rsplit(' ', 1)[0]
                        metadata["supersedes_scope"] = scope

                validated_metadata = self._validate_metadata(metadata)
                # store initial metadata
                self._store_metadata(validated_metadata)

                # If implicit supersession phrases were detected by the LLM/text detector, try to resolve them
                try:
                    if validated_metadata.get("implicit_supersedes") and not validated_metadata.get("_implicit_supersedes_applied"):
                        self._apply_implicit_supersedes(validated_metadata)
                except Exception as e:
                    logger.error(f"Error applying implicit supersedes for {source_file}: {e}")

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
            "roles": [],
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

    def _apply_implicit_supersedes(self, metadata: Dict) -> None:
        """
        Resolve narrative/implicit supersedes by finding older documents in the same
        category family (partial/ancestor category matching allowed), department/roles/global overlap,
        and issued_date < current issued_date. Update metadata.document_relationships['supersedes']
        and persist the updated metadata.
        """
        # basic checks
        src = metadata.get("source_file")
        if not src:
            logger.debug("_no source_file on metadata, skipping implicit supersedes")
            return

        # Only run this resolution when the metadata explicitly flagged implicit supersedes
        if not metadata.get("implicit_supersedes"):
            logger.debug(f"implicit_supersedes not set for {src}; skipping implicit supersedes")
            return

        # require issued_date to compare; if missing, skip
        cur_issued = metadata.get("issued_date")
        if not cur_issued:
            logger.debug(f"No issued_date for {src}; skipping implicit supersedes")
            return

        try:
            cur_dt = datetime.strptime(cur_issued, "%Y-%m-%d")
        except Exception:
            logger.debug(f"Could not parse issued_date '{cur_issued}' for {src}; skipping")
            return

        category_paths = metadata.get("category_path") or []
        if not category_paths:
            logger.debug(f"No category_path for {src}; skipping implicit supersedes")
            return

        # helper to check ancestor/partial match between two category paths
        def category_matches(target_paths, candidate_paths):
            # return True if any candidate path is a prefix of any target path or vice-versa
            for t in target_paths:
                for c in candidate_paths:
                    # both are lists of strings
                    if not isinstance(t, list) or not isinstance(c, list):
                        continue
                    # normalize lowercase
                    t_norm = [p.lower() for p in t]
                    c_norm = [p.lower() for p in c]
                    # candidate is ancestor of target
                    if len(c_norm) <= len(t_norm) and t_norm[:len(c_norm)] == c_norm:
                        return True
                    # target is ancestor of candidate
                    if len(t_norm) <= len(c_norm) and c_norm[:len(t_norm)] == t_norm:
                        return True
            return False

        candidates = []
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Fetch metadata for all documents; we'll filter in Python for flexible matching.
                cur.execute("SELECT filename, metadata FROM document_metadata")
                rows = cur.fetchall()

            for filename, row_meta in rows:
                # skip self
                if filename == src:
                    continue

                try:
                    if isinstance(row_meta, str):
                        other = json.loads(row_meta)
                    else:
                        other = row_meta
                except Exception:
                    continue

                # prefer circular_number, fallback to source_file so we don't drop candidates
                other_cn = other.get("circular_number") or other.get("source_file")
                other_issued = other.get("issued_date")
                other_paths = other.get("category_path") or []

                # require date and category_path; allow missing circular_number (we use source_file)
                if not other_issued or not other_paths:
                    logger.debug(f"Skipping {filename}: missing issued_date or category_path")
                    continue

                # try multiple common date formats for other document
                other_dt = None
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                    try:
                        other_dt = datetime.strptime(other_issued, fmt)
                        break
                    except Exception:
                        pass
                if other_dt is None:
                    logger.debug(f"Skipping {filename}: could not parse issued_date '{other_issued}'")
                    continue

                # only consider older documents
                if other_dt >= cur_dt:
                    logger.debug(f"Skipping {filename}: not older (other_dt >= cur_dt)")
                    continue

                # category partial/ancestor matching
                if not category_matches(category_paths, other_paths):
                    continue

                # department checks: require exact, non-empty department match (case-insensitive)
                # as per new rule: do NOT fall back to roles when department is missing.
                cur_dept = (metadata.get("department") or "").strip().lower()
                other_dept = (other.get("department") or "").strip().lower()

                if not cur_dept or not other_dept:
                    # one or both documents lack department metadata; skip candidate
                    continue
                if cur_dept != other_dept:
                    # departments differ; skip candidate
                    continue

                candidates.append(other_cn)

            logger.debug(f"Implicit supersedes candidates for {src} (pre-dedupe): {candidates}")

            # dedupe
            candidates = sorted(set(candidates))
            logger.debug(f"Implicit supersedes candidates for {src} (deduped): {candidates}")

            if candidates:
                metadata.setdefault("document_relationships", {}).setdefault("supersedes", [])
                existing = set(metadata["document_relationships"].get("supersedes") or [])
                merged = sorted(existing.union(candidates))
                metadata["document_relationships"]["supersedes"] = merged
                # mark applied to avoid loops
                metadata["_implicit_supersedes_applied"] = True
                # persist updated metadata
                logger.info(f"Resolved implicit supersedes for {src}: {merged}")
                self._store_metadata(metadata)

                # --- Verification: read back and log stored value ---
                try:
                    verify_conn = get_db_connection()
                    with verify_conn.cursor() as vcur:
                        vcur.execute("SELECT metadata FROM document_metadata WHERE filename = %s", (metadata["source_file"],))
                        row = vcur.fetchone()
                        if row:
                            try:
                                stored = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                                stored_sup = stored.get("document_relationships", {}).get("supersedes")
                                logger.info(f"Post-write check for {metadata['source_file']}: stored.supersedes={stored_sup}")
                            except Exception as e:
                                logger.error(f"Post-write parse failed for {metadata['source_file']}: {e}")
                        else:
                            logger.error(f"Post-write: no row found for filename={metadata['source_file']}")
                except Exception as e:
                    logger.error(f"Post-write verification failed for {metadata['source_file']}: {e}")
                finally:
                    try:
                        verify_conn.close()
                    except Exception:
                        pass
            else:
                # still mark applied to avoid reprocessing
                metadata["_implicit_supersedes_applied"] = True
                self._store_metadata(metadata)
                logger.debug(f"No candidates found; marked _implicit_supersedes_applied for {src}")

        except Exception as e:
            logger.error(f"Failed to compute implicit supersedes for {src}: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
