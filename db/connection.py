"""Database connection and initialization."""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'semantic.db')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

def get_connection(db_path: str = None) -> sqlite3.Connection:
    """Get a database connection with proper settings."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db(db_path: str = None) -> sqlite3.Connection:
    """Initialize the database from schema.sql."""
    conn = get_connection(db_path)
    with open(SCHEMA_PATH, 'r') as f:
        conn.executescript(f.read())
    return conn
