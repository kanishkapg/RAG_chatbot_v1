from typing import List, Dict
import logging
from src.document_processor import DocumentProcessor
from src.vector_db import VectorDatabaseManager
from src.llm_response import LLMResponseGenerator
from src.db import ensure_tables_exist
from src.graph_manager import Neo4jManager
from test_tesseract import process_all_pdfs
from config import DATA_DIR

logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabaseManager()
        self.llm = LLMResponseGenerator()
        self.graph_manager = Neo4jManager()

    def initialize_system(self) -> None:
        logger.info("Processing PDFs and extracting text...")
        extracted_texts = process_all_pdfs(self.data_dir)
        
        logger.info("Initializing Neo4j constraints...")
        self.graph_manager.create_constraints()
        
        logger.info("Processing documents and extracting metadata...")
        documents = self.document_processor.load_and_split_documents()
        logger.info(f"Split documents into {len(documents)} chunks")
        
        logger.info("Initializing vector database...")
        self.vector_db.initialize_collection()
        self.vector_db.upsert_documents(documents)
        
        logger.info("Visualizing document relationships...")
        visualization_path = self.graph_manager.visualize_document_relationships()
        logger.info(f"Graph visualization saved to {visualization_path}")
        
        logger.info("RAG system initialization complete!")
        
    
    def find_effective_circular(self, circular_number: str) -> Dict:
        """Find if a circular is still effective or has been replaced."""
        return self.graph_manager.find_effective_document(circular_number)
    
        
    def generate_visualization(self) -> str:
        """Generate and return the path to the visualization file."""
        return self.graph_manager.visualize_document_relationships()
    

    def query(self, question: str, use_graph_validation: bool = True) -> Dict:
        """
        Query the RAG system with optional graph validation.
        
        Args:
            question: The user's question
            use_graph_validation: Whether to use Neo4j graph validation for ensuring 
                                document effectiveness (default: True)
        """
        try:
            if use_graph_validation:
                # Use the enhanced search with graph validation
                hits = self.vector_db.search_similar_documents_with_validation(question)
                logger.info("Using graph-validated document retrieval")
            else:
                # Use the original search method
                hits = self.vector_db.search_similar_documents(question)
                logger.info("Using standard document retrieval")
            
            context = " ".join([hit["text"] for hit in hits])
            sources = list(set([hit["source"] for hit in hits]))
            metadata_list = [hit["metadata"] for hit in hits]
            
            response = self.llm.generate_response(question, context, sources, metadata_list)
            
            return {
                "question": question,
                "answer": response["answer"],
                "sources": response["sources"],
                "relevant_documents": hits,
                "validation_used": use_graph_validation
            }
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "question": question,
                "answer": "Error processing query",
                "sources": [],
                "relevant_documents": [],
                "validation_used": use_graph_validation
            }


def build_and_initialize():
    # Ensure DB tables
    ensure_tables_exist()

    rag_system = RAGSystem(data_dir=DATA_DIR)
    rag_system.initialize_system()
    return rag_system
