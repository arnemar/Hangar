"""Graph traversal over the knowledge graph using recursive CTEs.

Two traversal directions:
  forward   — from seed nodes outward (what does this symbol use/call?)
  backward  — from seed nodes inward (what uses/calls this symbol?) → impact analysis

Each result carries a `graph_depth` (0 = seed node, 1 = directly connected, etc.)
and a `graph_score` in [0, 1] using exponential decay:
    graph_score = exp(-0.5 * depth)
so depth 0 → 1.0, depth 1 → 0.61, depth 2 → 0.37, depth 3 → 0.22.

Edge kinds can be filtered to control traversal. For example, passing
kinds=["calls", "imports"] gives a call-graph walk; ["contains"] gives
a structural tree walk.

The CTE uses UNION (not UNION ALL) to guarantee each node is visited once
at its minimum depth — we GROUP BY id and take MIN(depth) to collapse
any duplicates SQLite might surface.
"""

from __future__ import annotations

import math

from db import get_db


def _graph_score(depth: int) -> float:
    return math.exp(-0.5 * depth)


def graph_expand(
    project_id: str,
    seed_ids: list[str],
    direction: str = "forward",
    max_depth: int = 3,
    edge_kinds: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Walk the graph from seed_ids, returning reachable nodes with depth info.

    Args:
        project_id:  project to search within
        seed_ids:    starting node IDs (must belong to project_id)
        direction:   "forward" (seed → uses) or "backward" (seed ← callers)
        max_depth:   maximum traversal depth
        edge_kinds:  optional filter on edge.kind; None = all resolved edges
        limit:       max nodes to return

    Returns:
        List of node dicts with `graph_depth` int and `graph_score` float.
    """
    if not seed_ids:
        return []

    direction = direction.lower()
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    seed_placeholders = ",".join("?" * len(seed_ids))

    # Edge join direction
    if direction == "forward":
        # traverse FROM seed: follow edges where from_id is current
        cte_join = "JOIN edges e ON e.from_id = reach.id AND e.to_id IS NOT NULL"
        next_col = "e.to_id"
    else:
        # traverse TO seed: follow edges where to_id is current (reverse)
        cte_join = "JOIN edges e ON e.to_id = reach.id AND e.from_id IS NOT NULL"
        next_col = "e.from_id"

    kind_filter = ""
    extra_params: list = []
    if edge_kinds:
        kp = ",".join("?" * len(edge_kinds))
        kind_filter = f"AND e.kind IN ({kp})"
        extra_params.extend(edge_kinds)

    sql = f"""
        WITH RECURSIVE reach(id, depth) AS (
            SELECT id, 0
            FROM nodes
            WHERE id IN ({seed_placeholders})
              AND project_id = ?

            UNION

            SELECT {next_col}, reach.depth + 1
            FROM reach
            {cte_join}
            JOIN nodes n2 ON n2.id = {next_col}
            WHERE e.resolved = 1
              {kind_filter}
              AND reach.depth < ?
              AND n2.project_id = ?
        )
        SELECT
            n.id, n.path, n.kind, n.name, n.parent_id,
            n.signature, n.start_line, n.end_line, n.language,
            n.visibility, n.fan_in, n.importance, n.summary,
            MIN(reach.depth) AS graph_depth
        FROM reach
        JOIN nodes n ON n.id = reach.id
        WHERE n.project_id = ?
        GROUP BY n.id
        ORDER BY graph_depth ASC, n.fan_in DESC
        LIMIT ?
    """

    # Parameter order:
    # seed_ids (for IN clause), project_id (seed filter),
    # extra_params (edge kind filter, if any), max_depth, project_id (n2 filter),
    # project_id (outer WHERE), limit
    params: list = (
        list(seed_ids)
        + [project_id]
        + extra_params
        + [max_depth, project_id, project_id, limit]
    )

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        row = dict(r)
        depth = row.get("graph_depth") or 0
        row["graph_score"] = _graph_score(depth)
        results.append(row)
    return results


def find_neighbors(
    project_id: str,
    node_id: str,
    max_depth: int = 1,
) -> list[dict]:
    """Convenience: get immediate neighbors (callers + callees) of a single node."""
    fwd = graph_expand(project_id, [node_id], "forward", max_depth=max_depth)
    bwd = graph_expand(project_id, [node_id], "backward", max_depth=max_depth)
    seen: dict[str, dict] = {}
    for row in fwd + bwd:
        nid = row["id"]
        if nid not in seen or row["graph_depth"] < seen[nid]["graph_depth"]:
            seen[nid] = row
    return list(seen.values())
