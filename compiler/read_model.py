"""Read model — unified query interface for the knowledge graph.

This is the single entry point for ALL retrieval. No other module should
query the nodes/edges tables directly for retrieval purposes.

Architecture
------------
  QueryContext    — captures intent, constraints, and budget
  SearchResult    — normalized output: one per matched node
  query()         — dispatches to strategies, merges, ranks, returns results

Merge rules (when a node appears in multiple strategies)
---------------------------------------------------------
  score signals:  take MAX of each signal independently
  source set:     union ("fts", "graph", "semantic")
  explanation:    concatenate non-empty explanations

Deduplication
-------------
  Keyed by node_id. All merging happens before ranking.

Hot symbols
-----------
  Loaded once per query from hot_symbols table (project + session).
  Used as a boolean boost signal in ranking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from compiler.fts import fts_query
from compiler.graph import graph_expand
from compiler.ranking import RankInput, rank as rank_inputs
from db import get_db


# ── Query context ─────────────────────────────────────────────────────────────

@dataclass
class QueryContext:
    project_id: str
    query_text: str = ""
    intent: str = "find"          # find | impact | edit | debug | explain
    focus_nodes: list[str] = field(default_factory=list)
    max_depth: int = 3            # graph traversal depth
    retrieval_budget: int = 20    # max results returned
    session_id: str | None = None # for hot symbol lookup
    kinds: list[str] | None = None  # optional node kind filter


# ── Search result ─────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    node_id: str
    score: float
    sources: set[str]       # e.g. {"fts", "graph"}
    explanation: str

    # Denormalized fields (avoid second DB round-trip for display)
    kind: str
    name: str
    path: str
    signature: str
    start_line: int
    end_line: int
    language: str
    fan_in: int
    summary: str | None


# ── Hot symbol loader ─────────────────────────────────────────────────────────

def _load_hot_ids(project_id: str, session_id: str | None) -> set[str]:
    """Return node_ids that are in the hot_symbols table for this project."""
    with get_db() as conn:
        if session_id:
            rows = conn.execute(
                "SELECT node_id FROM hot_symbols WHERE project_id = ? AND session_id = ?",
                (project_id, session_id),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT node_id FROM hot_symbols WHERE project_id = ?",
                (project_id,),
            ).fetchall()
    return {r["node_id"] for r in rows}


# ── Intermediate accumulator ──────────────────────────────────────────────────

@dataclass
class _Accumulator:
    """Per-node accumulator for merging signals across strategies."""
    node: dict                    # raw node row (first seen wins for fields)
    fts_score: float = 0.0
    graph_score: float = 0.0
    semantic_score: float = 0.0
    sources: set[str] = field(default_factory=set)
    explanations: list[str] = field(default_factory=list)


def _merge(accum: dict[str, _Accumulator], rows: list[dict], source: str) -> None:
    """Merge a batch of strategy results into the accumulator map."""
    score_key = f"{source}_score"
    for row in rows:
        nid = row["id"]
        if nid not in accum:
            accum[nid] = _Accumulator(node=row)
        acc = accum[nid]
        acc.sources.add(source)
        # Take max of each signal independently
        current = getattr(acc, score_key, 0.0)
        setattr(acc, score_key, max(current, row.get(score_key, 0.0)))


def _top_importance_nodes(project_id: str, limit: int) -> list[dict]:
    """Return the most-connected non-trivial nodes by fan_in, excluding files/folders."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, path, kind, name, parent_id, signature, start_line, end_line,
                      language, visibility, fan_in, importance, summary, fan_in AS graph_score
               FROM nodes
               WHERE project_id = ?
                 AND kind NOT IN ('file', 'folder', 'project')
                 AND length(name) > 1
                 AND path NOT LIKE 'test%'
                 AND path NOT LIKE 'tests%'
                 AND path NOT LIKE '%test%'
               ORDER BY fan_in DESC
               LIMIT ?""",
            (project_id, limit),
        ).fetchall()
    result = []
    for r in rows:
        row = dict(r)
        max_fan = max(row["fan_in"], 1)
        row["graph_score"] = row["fan_in"] / max_fan  # will be renormalized after merge
        result.append(row)
    return result


# ── Main query function ───────────────────────────────────────────────────────

def query(ctx: QueryContext) -> list[SearchResult]:
    """Unified retrieval entry point.

    Dispatches to active strategies, merges results by node_id,
    ranks with intent-aware scoring, returns up to retrieval_budget results.
    Emits a RETRIEVAL_TRACE event for every call (fire-and-forget, never blocks).
    """
    _t0 = time.monotonic()
    accum: dict[str, _Accumulator] = {}

    # ── Strategy 1: FTS5 text search ─────────────────────────────────────────
    if ctx.query_text.strip():
        fts_rows = fts_query(
            ctx.project_id,
            ctx.query_text,
            kinds=ctx.kinds,
            limit=ctx.retrieval_budget * 3,
        )
        _merge(accum, fts_rows, "fts")

    # ── Strategy 2: Graph traversal from focus nodes ──────────────────────────
    if ctx.focus_nodes:
        direction = "backward" if ctx.intent == "impact" else "forward"
        graph_rows = graph_expand(
            ctx.project_id,
            ctx.focus_nodes,
            direction=direction,
            max_depth=ctx.max_depth,
            limit=ctx.retrieval_budget * 3,
        )
        _merge(accum, graph_rows, "graph")

    # ── Strategy 3: Importance fallback for orient / sparse FTS ─────────────────
    # When orient intent or FTS found almost nothing, pull top-fan_in nodes so
    # "what is this project about?" gets a real answer even with no good search terms.
    if ctx.intent == "orient" or len(accum) < 3:
        _merge(accum, _top_importance_nodes(ctx.project_id, ctx.retrieval_budget * 2), "graph")

    if not accum:
        return []

    # ── Hot symbol lookup ─────────────────────────────────────────────────────
    hot_ids = _load_hot_ids(ctx.project_id, ctx.session_id)

    # ── Build RankInputs ──────────────────────────────────────────────────────
    rank_list: list[tuple[str, RankInput]] = []
    for nid, acc in accum.items():
        ri = RankInput(
            node_id=nid,
            fts_score=acc.fts_score,
            graph_score=acc.graph_score,
            semantic_score=acc.semantic_score,
            fan_in=acc.node.get("fan_in") or 0,
            is_hot=nid in hot_ids,
        )
        rank_list.append((nid, ri))

    ranked = rank_inputs([ri for _, ri in rank_list], intent=ctx.intent)

    # ── Build SearchResults ───────────────────────────────────────────────────
    nid_to_acc = {nid: acc for nid, _ in rank_list for acc in [accum[nid]]}
    results: list[SearchResult] = []

    for final_score, ri in ranked[: ctx.retrieval_budget]:
        acc = nid_to_acc[ri.node_id]
        n = acc.node
        explanation_parts = []
        if acc.fts_score > 0:
            explanation_parts.append(f"text match ({acc.fts_score:.2f})")
        if acc.graph_score > 0:
            explanation_parts.append(f"graph proximity ({acc.graph_score:.2f})")
        if ri.is_hot:
            explanation_parts.append("recently accessed")

        results.append(SearchResult(
            node_id=ri.node_id,
            score=final_score,
            sources=acc.sources,
            explanation=", ".join(explanation_parts),
            kind=n.get("kind", ""),
            name=n.get("name", ""),
            path=n.get("path", ""),
            signature=n.get("signature", ""),
            start_line=n.get("start_line") or 0,
            end_line=n.get("end_line") or 0,
            language=n.get("language", ""),
            fan_in=n.get("fan_in") or 0,
            summary=n.get("summary"),
        ))

    # ── Emit retrieval trace (fire-and-forget) ────────────────────────────────
    try:
        latency_ms = round((time.monotonic() - _t0) * 1000)
        source_counts: dict[str, int] = {"fts": 0, "graph": 0, "hot": 0}
        for r in results:
            for s in r.sources:
                source_counts[s] = source_counts.get(s, 0) + 1
            if r.node_id in hot_ids:
                source_counts["hot"] += 1
        confidence = round(sum(r.score for r in results) / len(results), 3) if results else 0.0
        from compiler.events import emit
        emit(
            ctx.project_id,
            "RETRIEVAL_TRACE",
            entity_id=ctx.session_id,
            payload={
                "query": ctx.query_text,
                "intent": ctx.intent,
                "session_id": ctx.session_id,
                "retrieved": [
                    {"name": r.name, "kind": r.kind, "path": r.path,
                     "score": round(r.score, 3), "sources": sorted(r.sources)}
                    for r in results
                ],
                "source_counts": source_counts,
                "result_count": len(results),
                "budget": ctx.retrieval_budget,
                "budget_ratio": round(len(results) / max(ctx.retrieval_budget, 1), 2),
                "confidence": confidence,
                "latency_ms": latency_ms,
            },
        )
    except Exception:
        pass  # trace failures must never break retrieval

    return results


# ── Hot symbol tracking ───────────────────────────────────────────────────────

def record_access(project_id: str, node_id: str, session_id: str | None = None) -> None:
    """Record that a node was accessed (for hot symbol ranking)."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            """INSERT INTO hot_symbols (project_id, node_id, session_id, last_accessed, access_count)
               VALUES (?, ?, ?, ?, 1)
               ON CONFLICT(project_id, node_id) DO UPDATE SET
                   last_accessed = excluded.last_accessed,
                   access_count = access_count + 1,
                   session_id = excluded.session_id""",
            (project_id, node_id, session_id, now),
        )
