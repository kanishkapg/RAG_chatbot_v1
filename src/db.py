from dotenv import load_dotenv
import logging
import psycopg2
from config import POSTGRES_CONFIG

load_dotenv('.env', override=True)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Establish PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def ensure_tables_exist():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pdf_files (
                    filename VARCHAR(255) PRIMARY KEY,
                    file_hash VARCHAR(64),
                    extracted_text TEXT,
                    extraction_date TIMESTAMP,
                    UNIQUE (filename)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    filename VARCHAR(255) PRIMARY KEY,
                    metadata JSONB,
                    extraction_date TIMESTAMP,
                    FOREIGN KEY (filename) REFERENCES pdf_files(filename)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON pdf_files(file_hash)")
            # suggestion queue for taxonomy/aliases human review
            cur.execute("""
                CREATE TABLE IF NOT EXISTS category_suggestions (
                    id SERIAL PRIMARY KEY,
                    label TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    reviewed BOOLEAN NOT NULL DEFAULT FALSE,
                    reviewer TEXT,
                    resolved_path TEXT,
                    notes TEXT
                )
            """)
        conn.commit()
        logger.info("PostgreSQL tables created or verified")
    finally:
        conn.close()
