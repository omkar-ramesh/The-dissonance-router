"""SQLite connection helpers."""
import sqlite3
from pathlib import Path
from typing import Optional

from src import config

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection with dict-like rows, creating the parent folder if needed."""
    path = db_path or config.DB_PATH
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Create all tables if they don't exist yet."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()
