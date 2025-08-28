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
