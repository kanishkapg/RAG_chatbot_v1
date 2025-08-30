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
        # sanitize inputs to avoid nested-collections (Neo4j doesn't allow list-of-lists in properties)
        def _normalize_list(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            if isinstance(x, list):
                out = []
                for item in x:
                    if isinstance(item, list):
                        # join hierarchical path into a single string
                        out.append(" > ".join([str(p) for p in item if p is not None]))
                    else:
                        out.append(str(item))
                return out
            return [str(x)]

        category_path_raw = metadata.get("category_path") or []
        # convert possible list-of-lists into list of strings
        category_path_safe = []
        if isinstance(category_path_raw, list):
            for p in category_path_raw:
                if isinstance(p, list):
                    category_path_safe.append(" > ".join([str(part) for part in p if part is not None]))
                else:
                    category_path_safe.append(str(p))
        else:
            category_path_safe = [str(category_path_raw)]

        departments = _normalize_list(metadata.get("departments"))
        roles = _normalize_list(metadata.get("roles"))

        try:
            with self.driver.session() as session:
                # Create or update Document node (use sanitized properties)
                session.run("""
                    MERGE (d:Document {circular_number: $circular_number})
                    SET d.title = $title,
                        d.issued_date = $issued_date,
                        d.effective_date = $effective_date,
                        d.policy_category_raw = $policy_category_raw,
                        d.category_path = $category_path,
                        d.policy_category = $policy_category,
                        d.departments = $departments,
                        d.roles = $roles,
                        d.global = $global,
                        d.source_file = $source_file,
                        d.version = $version,
                        d.filename = $filename
                """, {
                    "circular_number": circular_number,
                    "title": metadata.get("title"),
                    "issued_date": metadata.get("issued_date"),
                    "effective_date": metadata.get("effective_date"),
                    "policy_category_raw": metadata.get("policy_category_raw"),
                    "category_path": category_path_safe,
                    "policy_category": metadata.get("policy_category"),
                    "departments": departments,
                    "roles": roles,
                    "global": metadata.get("global", False),
                    "source_file": metadata.get("source_file"),
                    "version": metadata.get("version"),
                    "filename": metadata.get("source_file")
                })

                logger.info(f"Added/updated document node for {circular_number}")

                # Add relationships based on document_relationships (validate targets are scalars)
                relationships = metadata.get("document_relationships", {}) or {}

                def _iter_valid_targets(key):
                    vals = relationships.get(key, [])
                    if not isinstance(vals, list):
                        return []
                    for v in vals:
                        if v is None:
                            continue
                        # skip nested collections
                        if isinstance(v, (list, dict)):
                            logger.warning(f"Skipping malformed relationship target for {key}: {v}")
                            continue
                        yield str(v)

                # Process REPEALS relationships
                for repealed_doc in _iter_valid_targets("repeals"):
                    session.run("""
                        MATCH (a:Document {circular_number: $doc_a})
                        MERGE (b:Document {circular_number: $doc_b})
                        MERGE (a)-[r:REPEALS]->(b)
                    """, {"doc_a": circular_number, "doc_b": repealed_doc})
                    logger.info(f"Added REPEALS relationship: {circular_number} -> {repealed_doc}")

                # Process AMENDS relationships
                for amended_doc in _iter_valid_targets("amends"):
                    session.run("""
                        MATCH (a:Document {circular_number: $doc_a})
                        MERGE (b:Document {circular_number: $doc_b})
                        MERGE (a)-[r:AMENDS]->(b)
                    """, {"doc_a": circular_number, "doc_b": amended_doc})
                    logger.info(f"Added AMENDS relationship: {circular_number} -> {amended_doc}")

                # Process EXTENDS relationships
                for extended_doc in _iter_valid_targets("extends"):
                    session.run("""
                        MATCH (a:Document {circular_number: $doc_a})
                        MERGE (b:Document {circular_number: $doc_b})
                        MERGE (a)-[r:EXTENDS]->(b)
                    """, {"doc_a": circular_number, "doc_b": extended_doc})
                    logger.info(f"Added EXTENDS relationship: {circular_number} -> {extended_doc}")

                # Process SUPERSEDES relationships
                for superseded_doc in _iter_valid_targets("supersedes"):
                    session.run("""
                        MATCH (a:Document {circular_number: $doc_a})
                        MERGE (b:Document {circular_number: $doc_b})
                        MERGE (a)-[r:SUPERSEDES]->(b)
                    """, {"doc_a": circular_number, "doc_b": superseded_doc})
                    logger.info(f"Added SUPERSEDES relationship: {circular_number} -> {superseded_doc}")

                # Process REFERENCES relationships
                for referenced_doc in _iter_valid_targets("references"):
                    session.run("""
                        MATCH (a:Document {circular_number: $doc_a})
                        MERGE (b:Document {circular_number: $doc_b})
                        MERGE (a)-[r:REFERENCES]->(b)
                    """, {"doc_a": circular_number, "doc_b": referenced_doc})
                    logger.info(f"Added REFERENCES relationship: {circular_number} -> {referenced_doc}")

        except Exception as e:
            logger.error(f"Failed to add/update document {circular_number} in Neo4j: {e}")
            return False

        # After node creation, optionally expand implicit edges if flagged
        try:
            if metadata.get("_implicit_actions"):
                self.expand_implicit_supersedes(circular_number, metadata)
        except Exception as e:
            logger.error(f"Failed implicit expansion for {circular_number}: {e}")
    
    def get_document_chain(self, circular_number: str):
        """Get the chain of documents related to a specific circular."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {circular_number: $circular_number})
                OPTIONAL MATCH path1 = (d)-[:REPEALS|AMENDS|SUPERSEDES*1..3]->(related)
                OPTIONAL MATCH path1 = (d)-[:REPEALS|AMENDS|SUPERSEDES|EXTENDS*1..3]->(related)
                OPTIONAL MATCH path2 = (previous)-[:REPEALS|AMENDS|SUPERSEDES|EXTENDS*1..3]->(d)
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
                "EXTENDS": "purple",
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

    def expand_implicit_supersedes(self, circular_number: str, metadata: Dict):
        """
        Create implicit SUPERSEDES relationships based on category_path, departments/roles/global and issued_date.
        Implicit relationships are marked with `implicit=true` so they can be audited or removed when taxonomy updates.
        This function is conservative: it only acts when `category_path` is non-empty and only on older documents.
        """
        category_path = metadata.get("category_path") or []
        if not category_path:
            logger.debug(f"No category_path for {circular_number}; skipping implicit expansion")
            return

        departments = metadata.get("departments") or []
        roles = metadata.get("roles") or []
        is_global = metadata.get("global", False)
        issued_date = metadata.get("issued_date")

        with self.driver.session() as session:
            # conservative candidate selection: exact category_path match
            results = session.run("""
                MATCH (d_new:Document {circular_number: $new_id})
                MATCH (d_old:Document)
                WHERE d_old.circular_number <> $new_id
                  AND d_old.category_path = $category_path
                  AND ( $is_global = true
                        OR size([x IN d_old.departments WHERE x IN $departments]) > 0
                        OR size([x IN d_old.roles WHERE x IN $roles]) > 0 )
                  AND (d_old.issued_date IS NULL OR d_old.issued_date < $issued_date)
                RETURN d_old.circular_number AS old_id
            """, {
                "new_id": circular_number,
                "category_path": category_path,
                "departments": departments,
                "roles": roles,
                "is_global": is_global,
                "issued_date": issued_date
            })

            for rec in results:
                old_id = rec["old_id"]
                # Idempotent creation of implicit SUPERSEDES relationship
                session.run("""
                    MATCH (a:Document {circular_number: $new_id})
                    MATCH (b:Document {circular_number: $old_id})
                    MERGE (a)-[r:SUPERSEDES {implicit: true}]->(b)
                """, {"new_id": circular_number, "old_id": old_id})
                logger.info(f"Implicit SUPERSEDES created: {circular_number} -> {old_id}")
