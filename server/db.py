"""server/db.py — EYE-D Postgres(pgvector) 접속. 자격증명은 .env/환경변수에서."""
import os
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

def _dsn():
    dsn = os.environ.get("EYED_DB_DSN")
    if dsn:
        return dsn
    return (
        f"host={os.environ.get('EYED_DB_HOST', 'localhost')} "
        f"port={os.environ.get('EYED_DB_PORT', '5433')} "
        f"dbname={os.environ.get('EYED_DB_NAME', 'eyed')} "
        f"user={os.environ.get('EYED_DB_USER', 'eyed')} "
        f"password={os.environ.get('EYED_DB_PASSWORD', '')}"
    )

def get_conn():
    return psycopg2.connect(_dsn(), cursor_factory=RealDictCursor)
