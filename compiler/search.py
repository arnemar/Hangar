"""Public search API — thin orchestration over the read model.

This module is the only surface area consumers (agent tools, FastAPI routes,
CLI) should import. It translates high-level intent into QueryContext and
delegates everything else to read_model.

Available functions
-------------------
find_symbols     — text search + optional graph expansion
expand_from      — graph-only traversal from known node IDs
symbol_context   — fetch full source for a specific node_id
"""

from __future__ import annotations

from compiler.read_model import QueryContext, SearchResult, query, record_access
from db import get_db


def find_symbols(
    project_id: str,
    query_text: str,
    *,
    intent: str = "find",
    focus_nodes: list[str] | None = None,
    kinds: list[str] | None = None,
    max_depth: int = 2,
    max_results: int = 20,
    session_id: str | None = None,
) -> list[SearchResult]:
    """Search for symbols by name/signature, optionally anchored to focus nodes.

    Args:
        project_id:  project to search
        query_text:  symbol name, signature fragment, or description
        intent:      find | impact | edit | debug | explain
        focus_nodes: seed node IDs for graph expansion (e.g. currently open file's node)
        kinds:       restrict to specific node kinds (e.g. ["function", "class"])
        max_depth:   graph traversal depth from focus_nodes
        max_results: cap on returned results
        session_id:  for hot symbol tracking

    Returns:
        Ranked list of SearchResult, best match first.
    """
    ctx = QueryContext(
        project_id=project_id,
        query_text=query_text,
        intent=intent,
        focus_nodes=focus_nodes or [],
        max_depth=max_depth,
        retrieval_budget=max_results,
        session_id=session_id,
        kinds=kinds,
    )
    results = query(ctx)
    # Track access for top results
    for r in results[:5]:
        record_access(project_id, r.node_id, session_id)
    return results


def expand_from(
    project_id: str,
    node_ids: list[str],
    *,
    intent: str = "find",
    max_depth: int = 3,
    max_results: int = 30,
    session_id: str | None = None,
) -> list[SearchResult]:
    """Graph-only expansion from known node IDs. No text query needed.

    Use this when you already know which nodes to start from (e.g. the agent
    identified a function, now expand to see what it calls and what calls it).
    """
    ctx = QueryContext(
        project_id=project_id,
        query_text="",
        intent=intent,
        focus_nodes=node_ids,
        max_depth=max_depth,
        retrieval_budget=max_results,
        session_id=session_id,
    )
    return query(ctx)


def symbol_context(
    project_id: str,
    node_id: str,
    session_id: str | None = None,
) -> dict | None:
    """Fetch full node details (including source) for a single node_id.

    Returns None if the node doesn't exist. Records an access event.
    """
    with get_db() as conn:
        row = conn.execute(
            """SELECT id, path, kind, name, parent_id, signature,
                      source, start_line, end_line, language, visibility,
                      fan_in, fan_out, importance, summary
               FROM nodes
               WHERE id = ? AND project_id = ?""",
            (node_id, project_id),
        ).fetchone()

    if row is None:
        return None

    record_access(project_id, node_id, session_id)
    return dict(row)
