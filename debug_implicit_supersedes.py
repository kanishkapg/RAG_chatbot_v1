#!/usr/bin/env python3

import sys
import os
import logging
import json
from datetime import datetime

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor
from db import get_db_connection

def test_actual_implicit_supersedes():
    """Test the actual _apply_implicit_supersedes method with debug logging"""
    
    # Get the CHRO's Circular No. 03-2024 metadata from the database
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT metadata FROM document_metadata WHERE filename = %s",
                ("CHRO's Circular No. 03-2024",)
            )
            row = cur.fetchone()
            
            if not row:
                print("Document not found in database!")
                return
                
            metadata = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            print(f"Found metadata for: {metadata.get('source_file')}")
            print(f"Implicit supersedes: {metadata.get('implicit_supersedes')}")
            print(f"Current supersedes: {metadata.get('document_relationships', {}).get('supersedes', [])}")
            print()
            
            # Create document processor instance
            processor = DocumentProcessor()
            
            # Clear any existing supersedes to test from scratch
            if 'document_relationships' not in metadata:
                metadata['document_relationships'] = {}
            metadata['document_relationships']['supersedes'] = []
            
            print("=== RUNNING ACTUAL _apply_implicit_supersedes METHOD ===")
            
            # Run the actual method with debug logging
            processor._apply_implicit_supersedes(metadata)
            
            print()
            print("=== FINAL RESULT ===")
            final_supersedes = metadata.get('document_relationships', {}).get('supersedes', [])
            print(f"Final supersedes list: {final_supersedes}")
            print(f"Count: {len(final_supersedes)}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    test_actual_implicit_supersedes()
