import pytesseract
from pdf2image import convert_from_path
import os
import hashlib
import psycopg2
from psycopg2.extras import Json
from config import DATA_DIR, POSTGRES_CONFIG
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_db_connection():
    """Establish PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> tuple[list[str], str]:
    """Extract text from a PDF and store in PostgreSQL without saving to file."""
    # Compute file hash
    file_hash = compute_file_hash(pdf_path)
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Check if PDF is already processed
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT file_hash FROM pdf_files WHERE filename = %s",
                (pdf_filename,)
            )
            result = cur.fetchone()
            if result and result[0] == file_hash:
                logger.info(f"Skipping {pdf_path} - already processed with same hash")
                return [], file_hash
    finally:
        conn.close()

    # Convert PDF to images
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path}: {e}")
        return [], file_hash
    
    full_text = []
    for i, image in enumerate(images):
        try:
            text = pytesseract.image_to_string(image, lang='eng')
            full_text.append(f"----- Page {i+1} -----\n{text}\n")
        except Exception as e:
            logger.error(f"Failed to extract text from page {i+1} of {pdf_path}: {e}")
    
    # Store in PostgreSQL
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO pdf_files (filename, file_hash, extracted_text, extraction_date)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (filename) DO UPDATE
                SET file_hash = %s, extracted_text = %s, extraction_date = CURRENT_TIMESTAMP
                """,
                (pdf_filename, file_hash, "\n".join(full_text), file_hash, "\n".join(full_text))
            )
        conn.commit()
        logger.info(f"Stored extracted text for {pdf_path} in PostgreSQL")
    except Exception as e:
        logger.error(f"Failed to store extracted text in PostgreSQL for {pdf_path}: {e}")
    finally:
        conn.close()
    
    return full_text, file_hash

def process_all_pdfs(data_dir: str = DATA_DIR) -> dict:
    """Process all PDFs in the data directory, skipping unchanged files."""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    extracted_texts = {}
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(data_dir, pdf_file)
        logger.info(f"Processing {pdf_path}...")
        text, file_hash = extract_text_from_pdf(pdf_path)
        if text:  # Only include if text was extracted (i.e., not skipped)
            extracted_texts[pdf_file] = {
                "text": text,
                "file_hash": file_hash
            }
    
    return extracted_texts