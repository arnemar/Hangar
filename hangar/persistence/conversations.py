"""Conversation CRUD and the additive migration for the local workspace."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from hangar.persistence.db import get_db

_SESSION = "id, title, model, project_name, created_at, updated_at"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def migrate_workspace() -> None:
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for table, column in (("sessions", "project_name"), ("messages", "attachments")):
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_changes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                project_root TEXT NOT NULL,
                path TEXT NOT NULL,
                before_hash TEXT,
                after_text TEXT NOT NULL,
                diff TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_changes_session ON file_changes(session_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)")


def find_session(session_id: str, conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(f"SELECT {_SESSION} FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions() -> list[dict]:
    with get_db() as conn:
        return [dict(row) for row in conn.execute(f"SELECT {_SESSION} FROM sessions ORDER BY updated_at DESC")]


def create_session(model: str, project_name: str | None) -> dict:
    session_id = str(uuid.uuid4())
    timestamp = now()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id,title,mode,model,created_at,updated_at,project_name) "
            "VALUES (?, 'New chat', 'chat', ?, ?, ?, ?)",
            (session_id, model, timestamp, timestamp, project_name),
        )
        return find_session(session_id, conn)


def update_session(session_id: str, fields: dict) -> dict | None:
    with get_db() as conn:
        assignments = ", ".join(f"{name}=?" for name in fields)
        if fields:
            conn.execute(f"UPDATE sessions SET {assignments} WHERE id=?", (*fields.values(), session_id))
        return find_session(session_id, conn)


def messages_for(session_id: str) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,role,content,created_at,attachments FROM messages "
            "WHERE session_id=? ORDER BY created_at, rowid", (session_id,),
        ).fetchall()
    return [{**dict(row), "attachments": json.loads(row["attachments"] or "[]")} for row in rows]


def record_message(
    session_id: str, role: str, content: str, attachments: list[str] | None = None,
) -> None:
    timestamp = now()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (id,session_id,role,content,created_at,attachments) VALUES (?,?,?,?,?,?)",
            (str(uuid.uuid4()), session_id, role, content, timestamp, json.dumps(attachments or [])),
        )
        if role == "user":
            title = " ".join(content.split())[:70] or "Files"
            conn.execute(
                "UPDATE sessions SET updated_at=?, "
                "title=CASE WHEN title IN ('New Chat','New chat') THEN ? ELSE title END WHERE id=?",
                (timestamp, title, session_id),
            )


def delete_session(session_id: str) -> bool:
    with get_db() as conn:
        removed = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,)).rowcount
        if removed:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM file_changes WHERE session_id=?", (session_id,))
        return bool(removed)
