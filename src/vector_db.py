from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import httpx
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


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

    def initialize_collection(self, force_recreate: bool = False) -> None:
        """Initialize Qdrant collection only if it doesn't exist."""
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
        """Upsert only new or updated documents based on existing filenames."""
        if not documents:
            logger.info("No new documents to upsert")
            return

        existing_filenames = self.get_existing_documents()
        new_documents = [doc for doc in documents if doc["metadata"]["source_file"] not in existing_filenames]

        if not new_documents:
            logger.info("No new or updated documents to upsert")
            return
        
        points = []
        for i, doc in enumerate(documents):
            try:
                embedding = self.embedding_model.encode(doc["page_content"])
                payload = {
                    "text": doc["page_content"],
                    "source": doc["metadata"]["source_file"],
                    **doc["metadata"]
                }
                points.append(
                    PointStruct(
                        id=i,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )
            except Exception as e:
                logger.error(f"Failed to encode document chunk {i+1} for {doc['metadata']['source_file']}: {e}")
        
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} document chunks to Qdrant")

    def search_similar_documents(self, query: str, limit: int = 3, prefer_effective: bool = True) -> List[Dict]:
        try:
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
                # Get all repealed circulars from document relationships
                if hit.payload.get("document_relationships"):
                    relationships = hit.payload["document_relationships"]
                    if relationships.get("repeals"):
                        repealed_circulars.update(relationships["repeals"])

            for hit in hits:
                circular_number = hit.payload.get("circular_number")
                if prefer_effective and circular_number in repealed_circulars:
                    continue
                filtered_hits.append(hit)
            
            # Sort by score and effective_date (newer first)
            filtered_hits.sort(key=lambda x: (
                x.score,
                x.payload.get("effective_date", "") or ""
            ), reverse=True)
            
            filtered_hits = filtered_hits[:limit]
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "source"]}
                }
                for hit in filtered_hits
            ]
        except Exception as e:
            logger.error(f"Failed to search similar documents: {e}")
            return []

    def search_similar_documents_with_validation(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Enhanced search that includes both original and replacement documents.
        This ensures historical context is preserved while also providing current information.
        """
        try:
            # Step 1: Get initial semantic matches (larger pool for validation)
            query_embedding = self.embedding_model.encode(query).tolist()
            initial_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            ).points
            
            # Step 2: Import graph manager and validate documents
            from src.graph_manager import Neo4jManager
            graph_manager = Neo4jManager()
            graph_manager.connect()
            
            all_hits = []  # Keep ALL relevant hits
            replacement_candidates = set()
            processed_circulars = set()
            superseded_circulars = set()
            
            # First pass: Keep all initial hits and identify replacements
            for hit in initial_hits:
                circular_number = hit.payload.get("circular_number")
                if not circular_number:
                    # No circular number, keep as is
                    all_hits.append(hit)
                    continue
                
                # Always keep the original hit for historical context
                all_hits.append(hit)
                
                # Skip validation if we already processed this circular
                if circular_number in processed_circulars:
                    continue
                processed_circulars.add(circular_number)
                
                # Check if document has been replaced using Neo4j
                effectiveness_info = graph_manager.find_effective_document(circular_number)
                
                if effectiveness_info and not effectiveness_info["is_effective"]:
                    # Document has been replaced - track it and add replacements to candidates
                    superseded_circulars.add(circular_number)
                    if effectiveness_info["replaced_by"]:
                        replacement_candidates.update(effectiveness_info["replaced_by"])
                        logger.debug(f"Document {circular_number} replaced by: {effectiveness_info['replaced_by']}")
            
            # Step 3: Query vector DB for replacement documents
            if replacement_candidates:
                logger.info(f"Searching for replacement documents: {replacement_candidates}")
                
                for replacement_circular in replacement_candidates:
                    # Skip if we already have chunks from this replacement document
                    if replacement_circular in processed_circulars:
                        continue
                    
                    replacement_hits = self.client.query_points(
                        collection_name=self.collection_name,
                        query=query_embedding,
                        limit=3,  # Get more chunks from replacement documents
                        query_filter={
                            "must": [{"key": "circular_number", "match": {"value": replacement_circular}}]
                        },
                        with_payload=True,
                        with_vectors=False
                    ).points
                    
                    if replacement_hits:
                        all_hits.extend(replacement_hits)
                        logger.debug(f"Found {len(replacement_hits)} chunks from replacement document {replacement_circular}")
            
            # Step 4: Enhance sorting to prioritize current documents while preserving historical ones
            def get_sort_key(hit):
                circular_num = hit.payload.get("circular_number")
                is_superseded = circular_num in superseded_circulars
                issued_date = hit.payload.get("issued_date", "")
                
                # Prioritize: 1) Relevance score, 2) Current documents, 3) Newer documents
                return (
                    hit.score,  # Primary: relevance score
                    not is_superseded,  # Secondary: current documents first
                    issued_date or ""  # Tertiary: newer documents first
                )
            
            all_hits.sort(key=get_sort_key, reverse=True)
            
            # Step 5: Intelligent limiting - ensure we have both historical and current context
            final_hits = []
            current_count = 0
            historical_count = 0
            max_per_type = max(1, limit // 2)  # At least 1 of each type if available
            
            for hit in all_hits:
                if len(final_hits) >= limit:
                    break
                    
                circular_num = hit.payload.get("circular_number")
                is_superseded = circular_num in superseded_circulars
                
                if is_superseded:
                    if historical_count < max_per_type or current_count >= max_per_type:
                        final_hits.append(hit)
                        historical_count += 1
                else:
                    if current_count < max_per_type or historical_count >= max_per_type:
                        final_hits.append(hit)
                        current_count += 1
            
            # If we still have space and didn't get enough variety, fill remaining slots
            remaining_hits = [hit for hit in all_hits if hit not in final_hits]
            while len(final_hits) < limit and remaining_hits:
                final_hits.append(remaining_hits.pop(0))
            
            # Close the graph manager connection
            graph_manager.close()
            
            logger.info(f"Enhanced search returned {len(final_hits)} chunks ({current_count} current, {historical_count} historical) from {len(set(hit.payload.get('circular_number', 'unknown') for hit in final_hits))} documents")
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "metadata": {
                        **{k: v for k, v in hit.payload.items() if k not in ["text", "source"]},
                        "is_superseded": hit.payload.get("circular_number") in superseded_circulars
                    }
                }
                for hit in final_hits
            ]
            
        except Exception as e:
            logger.error(f"Failed to search with validation: {e}")
            # Make sure to close the connection even on error
            try:
                graph_manager.close()
            except:
                pass
            # Fallback to original search method
            return self.search_similar_documents(query, limit, prefer_effective=True)

    def get_documents_by_circular(self, circular_number: str, limit: int = 10) -> List[Dict]:
        """
        Get all document chunks for a specific circular number.
        Useful for debugging and understanding what content is available.
        """
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query_filter={
                    "must": [{"key": "circular_number", "match": {"value": circular_number}}]
                },
                limit=limit,
                with_payload=True,
                with_vectors=False
            ).points
            
            return [
                {
                    "id": hit.id,
                    "text": hit.payload["text"],
                    "source": hit.payload["source"],
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "source"]}
                }
                for hit in hits
            ]
        except Exception as e:
            logger.error(f"Failed to get documents for circular {circular_number}: {e}")
            return []

    def get_collection_stats(self) -> Dict:
        """Get statistics about the vector database collection."""
        try:
            if not self.client.collection_exists(self.collection_name):
                return {"error": "Collection does not exist"}
            
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get a sample of documents to analyze circular distribution
            sample_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )[0]
            
            circular_counts = {}
            total_points = len(sample_points)
            
            for point in sample_points:
                circular_num = point.payload.get("circular_number", "Unknown")
                circular_counts[circular_num] = circular_counts.get(circular_num, 0) + 1
            
            return {
                "total_vectors": collection_info.vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "sample_size": total_points,
                "unique_circulars": len(circular_counts),
                "circular_distribution": dict(sorted(circular_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
