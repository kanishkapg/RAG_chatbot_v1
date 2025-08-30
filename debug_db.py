#!/usr/bin/env python3

import json
from src.db import get_db_connection

def debug_database():
    """Debug the database to see what documents are stored."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT filename, metadata FROM document_metadata')
            rows = cur.fetchall()
            
        print(f'Total documents in DB: {len(rows)}')
        print('\nDocument details:')
        for filename, metadata in rows:
            if isinstance(metadata, dict):
                meta = metadata
            else:
                meta = json.loads(metadata)
            
            circular_num = meta.get('circular_number', 'N/A')
            issued_date = meta.get('issued_date', 'N/A')
            department = meta.get('department', 'N/A')
            category_path = meta.get('category_path', 'N/A')
            
            print(f'  {filename}:')
            print(f'    Circular: {circular_num}')
            print(f'    Date: {issued_date}')
            print(f'    Dept: {department}')
            print(f'    Category: {str(category_path)[:100]}...')
            print()
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    debug_database()
