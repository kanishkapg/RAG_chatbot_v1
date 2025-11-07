#!/usr/bin/env python3
"""One-off sync: read all metadata from PostgreSQL and add to Neo4j using Neo4jManager.add_document().

Run from repo root:

    python sync_metadata_to_neo4j.py

This will log what it adds. It is safe to run multiple times (MERGE used in graph_manager).
"""
import json
import logging
from dotenv import load_dotenv
load_dotenv('.env', override=True)

from src.graph_manager import Neo4jManager
from config import POSTGRES_CONFIG
import psycopg2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def fetch_all_metadata():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT filename, metadata FROM document_metadata;")
            return cur.fetchall()
    finally:
        conn.close()


def main():
    gm = Neo4jManager()
    gm.create_constraints()
    rows = fetch_all_metadata()
    if not rows:
        logger.info("No metadata rows found in document_metadata table.")
        return

    added = 0
    for filename, metadata in rows:
        if metadata is None:
            logger.warning(f"Skipping {filename}: metadata is NULL")
            continue
        # if stored as text, try parse
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception as e:
                logger.warning(f"Skipping {filename}: metadata is not valid JSON: {e}")
                continue

        # ensure required keys
        metadata.setdefault("source_file", filename)
        metadata.setdefault("document_relationships", {"repeals": [], "amends": [], "supersedes": [], "references": []})

        try:
            gm.add_document(metadata)
            logger.info(f"Synced {metadata.get('circular_number') or filename} to Neo4j")
            added += 1
        except Exception as e:
            logger.error(f"Failed to add {filename} to Neo4j: {e}")

    logger.info(f"Sync complete. Attempted to add {added} documents.")


if __name__ == '__main__':
    main()
