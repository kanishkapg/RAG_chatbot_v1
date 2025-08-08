import os

DATA_DIR="./data/dummy" # change into 'dummy' or 'original'
POSTGRES_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB", "slt_circulars_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "12345678"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}