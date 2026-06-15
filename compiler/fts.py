"""FTS5 full-text search over the knowledge graph.

Wraps the nodes_fts virtual table. All functions are pure SQL — no ranking,
no business logic. Results are raw node dicts with an added `fts_score` float
normalized to [0, 1] where 1.0 is the best match.

FTS5 bm25() returns negative values (more negative = better). Normalization:
    fts_score = abs(bm25_raw) / (1.0 + abs(bm25_raw))
This maps 0..-∞ → 0..1 monotonically: stronger matches get higher scores.

Query sanitization: special FTS5 operators (", *, :, ^, ~, (, )) are stripped
so raw user input and symbol names never break the MATCH expression.
Each token becomes a prefix search (token*) so "getUserBy" finds "getUserById".
"""

from __future__ import annotations

import re

from db import get_db


def _build_fts_query(raw: str) -> str:
    """Sanitize user input → FTS5 MATCH expression with prefix matching.

    Uses OR semantics so that natural-language queries like "login authentication"
    still find symbols whose name matches any token. AND would require all tokens
    to appear in the same node record, which almost never happens for sparse
    name/signature/summary content.
    """
    tokens = re.sub(r'[^\w\s]', ' ', raw).split()
    valid = [t for t in tokens if len(t) >= 3]
    if not valid:
        return ""
    return " OR ".join(f"{t}*" for t in valid)


def fts_query(
    project_id: str,
    query: str,
    kinds: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    """Search the FTS5 index.

    Args:
        project_id: project to search within
        query:      raw search string (sanitized internally)
        kinds:      optional filter on node.kind (e.g. ["function", "class"])
        limit:      max rows to return

    Returns:
        List of node dicts with `fts_score` in [0, 1].
        Empty list if the query is blank after sanitization.
    """
    fts_expr = _build_fts_query(query)
    if not fts_expr:
        return []

    kind_filter = ""
    params: list = [fts_expr, project_id]
    if kinds:
        placeholders = ",".join("?" * len(kinds))
        kind_filter = f"AND n.kind IN ({placeholders})"
        params.extend(kinds)
    params.append(limit)

    sql = f"""
        SELECT
            n.id, n.path, n.kind, n.name, n.parent_id,
            n.signature, n.start_line, n.end_line, n.language,
            n.visibility, n.fan_in, n.importance, n.summary,
            bm25(nodes_fts) AS bm25_raw
        FROM nodes_fts
        JOIN nodes n ON nodes_fts.rowid = n.rowid
        WHERE nodes_fts MATCH ?
          AND n.project_id = ?
          {kind_filter}
        ORDER BY bm25_raw          -- most negative = best
        LIMIT ?
    """

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        row = dict(r)
        raw_score = row.pop("bm25_raw", 0.0) or 0.0
        row["fts_score"] = abs(raw_score) / (1.0 + abs(raw_score))
        results.append(row)
    return results
