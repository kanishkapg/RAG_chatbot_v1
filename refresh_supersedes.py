#!/usr/bin/env python3

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.document_processor import DocumentProcessor

def main():
    """Refresh all implicit supersedes relationships in the database."""
    
    print("🔄 Starting refresh of all implicit supersedes relationships...")
    print("This will recalculate all supersedes relationships from scratch.\n")
    
    try:
        processor = DocumentProcessor()
        processor.refresh_all_implicit_supersedes()
        print("\n✅ Successfully refreshed all implicit supersedes relationships!")
        
    except Exception as e:
        logger.error(f"❌ Error during refresh: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
