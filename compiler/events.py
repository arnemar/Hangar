"""Append-only event store.

No component may mutate the graph without first emitting an event.
The event log is the authoritative history; the graph is derived state.

Event types
-----------
FILE_DISCOVERED     file found during walk
FILE_CHANGED        file hash changed since last snapshot
FILE_REMOVED        file no longer present
NODE_CREATED        new node inserted into graph
NODE_UPDATED        existing node content changed (same identity)
NODE_REMOVED        node deleted (symbol no longer exists)
EDGE_CREATED        new edge inserted (may be unresolved)
EDGE_RESOLVED       previously unresolved edge linked to a node_id
EDGE_REMOVED        edge deleted
SUMMARY_GENERATED   summary written for a node
EMBEDDING_GENERATED embedding written for a node
IMPORTANCE_RECALCULATED
                    fan_in/fan_out/importance scores updated
COMPILATION_STARTED
COMPILATION_COMPLETED
RETRIEVAL_TRACE     one record per query() call — for debugging, weight tuning,
                    and future automatic optimization
                    payload: {
                      query, intent, session_id,
                      retrieved: [{name, kind, path, score, sources}],
                      source_counts: {fts, graph, hot},
                      result_count, budget, budget_ratio,
                      confidence, latency_ms
                    }
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from db import get_db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(
    project_id: str,
    event_type: str,
    entity_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    """Append one event to the log. Returns the new event id."""
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO events (project_id, type, entity_id, payload, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                project_id,
                event_type,
                entity_id,
                json.dumps(payload) if payload else None,
                _now(),
            ),
        )
        return cur.lastrowid


def emit_many(project_id: str, events: list[tuple[str, str | None, dict | None]]) -> None:
    """Bulk-append events. Each tuple is (type, entity_id, payload)."""
    now = _now()
    rows = [
        (project_id, t, eid, json.dumps(p) if p else None, now)
        for t, eid, p in events
    ]
    with get_db() as conn:
        conn.executemany(
            "INSERT INTO events (project_id, type, entity_id, payload, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def get_events(
    project_id: str,
    event_type: str | None = None,
    since: str | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Read events from the log for replay or inspection."""
    query = "SELECT * FROM events WHERE project_id = ?"
    params: list[Any] = [project_id]
    if event_type:
        query += " AND type = ?"
        params.append(event_type)
    if since:
        query += " AND timestamp > ?"
        params.append(since)
    query += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]
