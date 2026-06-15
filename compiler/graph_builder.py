"""Graph builder — IRFile → knowledge graph mutations with event emission.

Contract
--------
* Input:  IRFile (pure data from canonicalizer)
* Output: side effects — DB writes to nodes/edges + event log
* Enforces the last_ir_hash invariant (updates file_snapshots.last_ir_hash)
* Handles node lifecycle: CREATE, UPDATE (content changed), REMOVE (symbol gone)
* Handles edge lifecycle: CREATE unresolved, RESOLVE when to_id found

Invariant
---------
    if file_snapshots.last_ir_hash == ir_file.hash → this function must NOT
    have been called (canonicalizer returns None in that case).
    If it IS called, we always write through.

Node lifecycle
--------------
    NEW symbol:     INSERT node → emit NODE_CREATED
    SAME content:   no-op (content_hash identical)
    CHANGED content: UPDATE node → emit NODE_UPDATED
    GONE symbol:    DELETE node → emit NODE_REMOVED

Edge lifecycle
--------------
    NEW edge (unresolved):  INSERT with resolved=0, to_id=NULL → emit EDGE_CREATED
    NEW edge (resolved):    INSERT with resolved=1, to_id set → emit EDGE_CREATED
    RESOLVE existing edge:  UPDATE to_id + resolved=1 → emit EDGE_RESOLVED
    GONE edge:              DELETE → emit EDGE_REMOVED

The linker (separate module) handles bulk resolution of unresolved edges across files.
This module only resolves intra-file edges (containment) that arrive already resolved.

Concurrency
-----------
Single-threaded per file. The caller (pipeline coordinator) is responsible
for not calling build() on the same file concurrently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from compiler.ir import IRFile, IRNode, IREdge, content_hash
from compiler.events import emit, emit_many
from db import get_db


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_existing_nodes(project_id: str, path: str) -> dict[str, dict]:
    """Load all nodes for this file from the DB, keyed by node id."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, kind, name, content_hash, signature, start_line, end_line "
            "FROM nodes WHERE project_id = ? AND path = ?",
            (project_id, path),
        ).fetchall()
    return {r["id"]: dict(r) for r in rows}


def _load_existing_edges(project_id: str, path: str) -> dict[tuple, dict]:
    """Load all edges whose from_id belongs to a node in this file, keyed by (from_id, raw_ref, kind)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT e.id, e.from_id, e.to_id, e.raw_ref, e.kind, e.resolved
               FROM edges e
               JOIN nodes n ON e.from_id = n.id
               WHERE n.project_id = ? AND n.path = ?""",
            (project_id, path),
        ).fetchall()
    return {(r["from_id"], r["raw_ref"], r["kind"]): dict(r) for r in rows}


def _node_content_hash(node: IRNode) -> str:
    """Hash the mutable content of a node (not its identity)."""
    blob = f"{node.signature}|{node.source}|{node.start_line}|{node.end_line}|{node.visibility}"
    return content_hash(blob)


# ── Graph mutations ───────────────────────────────────────────────────────────

@dataclass
class BuildResult:
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_removed: int = 0
    edges_created: int = 0
    edges_removed: int = 0


def build(ir: IRFile) -> BuildResult:
    """Write IRFile to the knowledge graph.

    Computes diff against existing DB state, applies minimal mutations,
    emits events, and updates last_ir_hash in file_snapshots.
    """
    result = BuildResult()
    now = _now()
    project_id = ir.project_id
    path = ir.path

    existing_nodes = _load_existing_nodes(project_id, path)
    existing_edges = _load_existing_edges(project_id, path)

    incoming_node_ids: set[str] = {n.id for n in ir.nodes}
    incoming_edge_keys: set[tuple] = {(e.from_id, e.raw_ref, e.kind) for e in ir.edges}

    node_events: list[tuple[str, str | None, dict | None]] = []
    edge_events: list[tuple[str, str | None, dict | None]] = []

    with get_db() as conn:

        # ── Node diff ────────────────────────────────────────────────────────

        for node in ir.nodes:
            ch = _node_content_hash(node)
            existing = existing_nodes.get(node.id)

            if existing is None:
                # INSERT new node
                conn.execute(
                    """INSERT INTO nodes
                       (id, project_id, path, kind, name, parent_id, signature,
                        visibility, language, start_line, end_line, source,
                        content_hash, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        node.id, project_id, path, node.kind, node.name,
                        node.parent_id, node.signature, node.visibility,
                        node.language, node.start_line, node.end_line,
                        node.source, ch, now, now,
                    ),
                )
                node_events.append((
                    "NODE_CREATED", node.id,
                    {"kind": node.kind, "name": node.name, "path": path},
                ))
                result.nodes_created += 1

            elif existing["content_hash"] != ch:
                # UPDATE changed content
                conn.execute(
                    """UPDATE nodes SET
                       signature=?, visibility=?, start_line=?, end_line=?,
                       source=?, content_hash=?, updated_at=?
                       WHERE id=?""",
                    (
                        node.signature, node.visibility,
                        node.start_line, node.end_line,
                        node.source, ch, now, node.id,
                    ),
                )
                node_events.append((
                    "NODE_UPDATED", node.id,
                    {"kind": node.kind, "name": node.name, "path": path},
                ))
                result.nodes_updated += 1
            # else: content identical → no-op

        # ── Node removals (symbols that no longer exist) ──────────────────────

        for nid in existing_nodes:
            if nid not in incoming_node_ids:
                conn.execute("DELETE FROM edges WHERE from_id = ?", (nid,))
                conn.execute("DELETE FROM nodes WHERE id = ?", (nid,))
                existing = existing_nodes[nid]
                node_events.append((
                    "NODE_REMOVED", nid,
                    {"kind": existing["kind"], "name": existing["name"], "path": path},
                ))
                result.nodes_removed += 1

        # ── Edge diff ────────────────────────────────────────────────────────

        for edge in ir.edges:
            key = (edge.from_id, edge.raw_ref, edge.kind)
            existing_edge = existing_edges.get(key)

            if existing_edge is None:
                # INSERT new edge
                conn.execute(
                    """INSERT INTO edges (project_id, from_id, to_id, raw_ref, kind, resolved, confidence, created_at)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        project_id, edge.from_id, edge.to_id, edge.raw_ref,
                        edge.kind, 1 if edge.resolved else 0,
                        edge.confidence, now,
                    ),
                )
                edge_events.append((
                    "EDGE_CREATED", None,
                    {
                        "from_id": edge.from_id, "to_id": edge.to_id,
                        "raw_ref": edge.raw_ref, "kind": edge.kind,
                        "resolved": edge.resolved,
                    },
                ))
                result.edges_created += 1

            elif not existing_edge["resolved"] and edge.resolved and edge.to_id:
                # RESOLVE a previously unresolved edge
                conn.execute(
                    "UPDATE edges SET to_id=?, resolved=1 WHERE id=?",
                    (edge.to_id, existing_edge["id"]),
                )
                edge_events.append((
                    "EDGE_RESOLVED", None,
                    {"from_id": edge.from_id, "to_id": edge.to_id, "kind": edge.kind},
                ))
            # else: already exists and consistent → no-op

        # ── Edge removals ─────────────────────────────────────────────────────

        for key, ex_edge in existing_edges.items():
            if key not in incoming_edge_keys:
                conn.execute("DELETE FROM edges WHERE id = ?", (ex_edge["id"],))
                edge_events.append((
                    "EDGE_REMOVED", None,
                    {"from_id": ex_edge["from_id"], "kind": ex_edge["kind"]},
                ))
                result.edges_removed += 1

        # ── Update last_ir_hash ───────────────────────────────────────────────
        conn.execute(
            """UPDATE file_snapshots
               SET last_ir_hash=?, last_compiled_at=?
               WHERE project_id=? AND path=?""",
            (ir.hash, now, project_id, path),
        )

    # ── Bulk emit events (outside the write transaction) ──────────────────────
    if node_events or edge_events:
        emit_many(project_id, node_events + edge_events)

    return result
