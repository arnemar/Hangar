import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

from hangar.config import DB_PATH


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Open a transaction and close its connection on exit."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            yield conn


def init_chat_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                mode TEXT DEFAULT 'chat',
                model TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

        """)


def init_db() -> None:
    """Initialize the database schema for compiler tables and rebuild the search index."""
    init_chat_db()
    schema = Path(__file__).resolve().parents[1] / "compiler" / "schema.sql"
    with get_db() as conn:
        conn.executescript(schema.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES('rebuild')")
