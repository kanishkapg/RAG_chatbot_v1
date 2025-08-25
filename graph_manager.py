import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Dict, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('.env', override=True)

class Neo4jManager:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_constraints(self):
        """Create necessary constraints for the graph database."""
        with self.driver.session() as session:
            # Create constraint on circular_number for Document nodes
            try:
                session.run("""
                    CREATE CONSTRAINT document_circular_number IF NOT EXISTS 
                    FOR (d:Document) REQUIRE d.circular_number IS UNIQUE
                """)
                logger.info("Created constraint on Document.circular_number")
            except Exception as e:
                logger.warning(f"Constraint creation failed (might already exist): {e}")

    def add_document(self, metadata: Dict):
        """Add a document node to Neo4j."""
        circular_number = metadata.get("circular_number")
        if not circular_number:
            logger.warning(f"No circular number for {metadata.get('source_file')}, using filename as identifier")
            circular_number = metadata.get("source_file")
        
        with self.driver.session() as session:
            # Create or update Document node
            session.run("""
                MERGE (d:Document {circular_number: $circular_number})
                SET d.title = $title,
                    d.issued_date = $issued_date,
                    d.effective_date = $effective_date,
                    d.policy_category = $policy_category,
                    d.department = $department,
                    d.source_file = $source_file,
                    d.version = $version,
                    d.filename = $filename
            """, {
                "circular_number": circular_number,
                "title": metadata.get("title"),
                "issued_date": metadata.get("issued_date"),
                "effective_date": metadata.get("effective_date"),
                "policy_category": metadata.get("policy_category"),
                "department": metadata.get("department"),
                "source_file": metadata.get("source_file"),
                "version": metadata.get("version"),
                "filename": metadata.get("source_file")
            })
            
            logger.info(f"Added/updated document node for {circular_number}")
            
            # Add relationships based on document_relationships
            relationships = metadata.get("document_relationships", {})
            
            # Process REPEALS relationships
            for repealed_doc in relationships.get("repeals", []):
                session.run("""
                    MATCH (a:Document {circular_number: $doc_a})
                    MERGE (b:Document {circular_number: $doc_b})
                    MERGE (a)-[r:REPEALS]->(b)
                """, {"doc_a": circular_number, "doc_b": repealed_doc})
                logger.info(f"Added REPEALS relationship: {circular_number} -> {repealed_doc}")
            
            # Process AMENDS relationships
            for amended_doc in relationships.get("amends", []):
                session.run("""
                    MATCH (a:Document {circular_number: $doc_a})
                    MERGE (b:Document {circular_number: $doc_b})
                    MERGE (a)-[r:AMENDS]->(b)
                """, {"doc_a": circular_number, "doc_b": amended_doc})
                logger.info(f"Added AMENDS relationship: {circular_number} -> {amended_doc}")
            
            # Process SUPERSEDES relationships
            for superseded_doc in relationships.get("supersedes", []):
                session.run("""
                    MATCH (a:Document {circular_number: $doc_a})
                    MERGE (b:Document {circular_number: $doc_b})
                    MERGE (a)-[r:SUPERSEDES]->(b)
                """, {"doc_a": circular_number, "doc_b": superseded_doc})
                logger.info(f"Added SUPERSEDES relationship: {circular_number} -> {superseded_doc}")
            
            # Process REFERENCES relationships
            for referenced_doc in relationships.get("references", []):
                session.run("""
                    MATCH (a:Document {circular_number: $doc_a})
                    MERGE (b:Document {circular_number: $doc_b})
                    MERGE (a)-[r:REFERENCES]->(b)
                """, {"doc_a": circular_number, "doc_b": referenced_doc})
                logger.info(f"Added REFERENCES relationship: {circular_number} -> {referenced_doc}")
    
    def get_document_chain(self, circular_number: str):
        """Get the chain of documents related to a specific circular."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {circular_number: $circular_number})
                OPTIONAL MATCH path1 = (d)-[:REPEALS|AMENDS|SUPERSEDES*1..3]->(related)
                OPTIONAL MATCH path2 = (previous)-[:REPEALS|AMENDS|SUPERSEDES*1..3]->(d)
                RETURN d, related, previous, 
                       relationships(path1) as outgoing_rels,
                       relationships(path2) as incoming_rels
            """, {"circular_number": circular_number})
            
            records = list(result)
            return records
    
    def visualize_document_relationships(self, output_file="document_graph.html"):
        """Create a visualization of the document relationships."""
        from pyvis.network import Network
        
        # Create a new network
        net = Network(height="800px", width="100%", notebook=False, directed=True)
        
        with self.driver.session() as session:
            # Get all documents
            documents = session.run("""
                MATCH (d:Document)
                RETURN d.circular_number as id, d.title as title, 
                       d.issued_date as issued_date, d.source_file as source
            """)
            
            # Add nodes
            for doc in documents:
                label = f"{doc['id']}\n{doc['title'] or ''}\n{doc['issued_date'] or ''}"
                net.add_node(doc["id"], label=label, title=doc["source"])
            
            # Get all relationships
            relationships = session.run("""
                MATCH (a:Document)-[r]->(b:Document)
                RETURN a.circular_number as source, b.circular_number as target, 
                       type(r) as type
            """)
            
            # Add edges with different colors based on relationship type
            rel_colors = {
                "REPEALS": "red",
                "AMENDS": "blue",
                "SUPERSEDES": "green",
                "REFERENCES": "gray"
            }
            
            for rel in relationships:
                net.add_edge(
                    rel["source"], 
                    rel["target"], 
                    title=rel["type"],
                    color=rel_colors.get(rel["type"], "black"),
                    arrows="to"
                )
        
        # Configure physics
        net.toggle_physics(True)
        net.barnes_hut()
        
        # Save the visualization
        net.save_graph(output_file)
        logger.info(f"Graph visualization saved to {output_file}")
        
        return output_file

    def find_effective_document(self, circular_number: str):
        """
        Check if a document has been repealed, amended, or superseded
        and return the most recent effective version.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {circular_number: $circular_number})
                OPTIONAL MATCH (newer:Document)-[:REPEALS|SUPERSEDES]->(d)
                WITH d, collect(newer) as newer_docs
                RETURN d.circular_number as original,
                       [doc IN newer_docs | doc.circular_number] as replaced_by,
                       CASE WHEN size(newer_docs) > 0 THEN false ELSE true END as is_effective
            """, {"circular_number": circular_number})
            
            record = result.single()
            if record:
                return {
                    "original": record["original"],
                    "replaced_by": record["replaced_by"],
                    "is_effective": record["is_effective"]
                }
            return None