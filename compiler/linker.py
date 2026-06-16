"""Cross-file edge linker — resolves unresolved edges after compilation.

The AST extractor emits edges with resolved=False and a raw_ref string
for all cross-file references (imports, calls to external symbols).
The linker runs after all files are compiled and attempts to match
raw_ref strings against the node index.

Resolution strategy (in priority order)
----------------------------------------
1. Direct name match:    raw_ref == node.name, same project
2. Qualified match:      raw_ref == "module:symbol" → resolve module path,
                         find symbol in that file's nodes
3. Fuzzy fallback:       raw_ref contains the name (substring, case-insensitive)
                         with a confidence penalty

Resolution is idempotent: already-resolved edges are skipped.
Unresolved edges that cannot be matched are left with resolved=0.

raw_ref formats (from ast_extractor)
--------------------------------------
  "mod/path:Symbol"    → named import from a relative module
  "mod/path:default"   → default import
  "mod/path"           → namespace import (entire module)
  "Symbol"             → direct call/reference with no path hint

The linker normalizes relative paths against the importer's directory
before looking up file nodes.
"""

from __future__ import annotations

import logging
import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timezone

from compiler.events import emit_many
from db import get_db

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class LinkResult:
    resolved: int = 0
    skipped_already_resolved: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


# ── Path normalization ────────────────────────────────────────────────────────

def _normalize_ref_path(raw_ref: str, importer_path: str) -> tuple[str, str]:
    """Parse raw_ref → (module_path, symbol_name).

    If raw_ref contains ":", splits on first ":".
    Module path is normalized relative to the importer's directory.
    Returns ("", raw_ref) if no path component found.
    """
    if ":" in raw_ref:
        module_part, symbol = raw_ref.split(":", 1)
    else:
        module_part = raw_ref
        symbol = ""

    if not module_part or not module_part.startswith("."):
        # Not a relative import; symbol resolution only
        return ("", symbol or module_part)

    # Resolve relative path from importer's directory
    importer_dir = posixpath.dirname(importer_path.replace("\\", "/"))
    # Remove leading ./ or ../
    joined = posixpath.normpath(posixpath.join(importer_dir, module_part))
    # Try with common extensions (without extension first — match file node by prefix)
    return (joined, symbol)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_unresolved(project_id: str) -> list[dict]:
    """Load all unresolved edges for the project with importer path info."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT e.id, e.from_id, e.raw_ref, e.kind,
                      n.path AS importer_path, n.name AS importer_name
               FROM edges e
               JOIN nodes n ON e.from_id = n.id
               WHERE n.project_id = ? AND e.resolved = 0 AND e.raw_ref IS NOT NULL""",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _build_symbol_index(project_id: str) -> dict[str, list[dict]]:
    """Build name → [node_rows] index for fast symbol lookup."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, path, kind, name FROM nodes WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    index: dict[str, list[dict]] = {}
    for r in rows:
        name = r["name"].lower()
        index.setdefault(name, []).append(dict(r))
    return index


def _build_path_index(project_id: str) -> dict[str, list[dict]]:
    """Build normalized-path-prefix → [node_rows] index for file/module lookup."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, path, kind, name FROM nodes WHERE project_id = ? AND kind = 'file'",
            (project_id,),
        ).fetchall()
    index: dict[str, list[dict]] = {}
    for r in rows:
        # Normalize path: strip extension variants for lookup
        p = r["path"].replace("\\", "/")
        for suffix in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", "/index.ts", "/index.js"):
            if p.endswith(suffix):
                key = p[: -len(suffix)]
                index.setdefault(key, []).append(dict(r))
        index.setdefault(p, []).append(dict(r))
    return index


# ── Resolution logic ──────────────────────────────────────────────────────────

def _resolve_edge(
    edge: dict,
    symbol_index: dict[str, list[dict]],
    path_index: dict[str, list[dict]],
) -> tuple[str | None, float]:
    """Attempt to resolve one unresolved edge.

    Returns (to_id, confidence) or (None, 0.0) if unresolvable.
    """
    raw_ref = edge["raw_ref"]
    importer_path = edge["importer_path"]

    module_path, symbol = _normalize_ref_path(raw_ref, importer_path)

    # ── Strategy 1: qualified path + symbol ───────────────────────────────────
    if module_path and symbol and symbol not in ("default", "namespace", "*"):
        candidates = path_index.get(module_path, [])
        if candidates:
            file_node = candidates[0]
            # Find symbol in that file
            hits = symbol_index.get(symbol.lower(), [])
            file_hits = [h for h in hits if h["path"] == file_node["path"]]
            if file_hits:
                # Prefer functions/classes/methods over generic nodes
                preferred = [h for h in file_hits if h["kind"] in (
                    "function", "class", "method", "interface", "type_alias", "enum", "constant"
                )]
                best = (preferred or file_hits)[0]
                return best["id"], 1.0

    # ── Strategy 2: namespace / default import → resolve to file node ─────────
    if module_path and symbol in ("", "default", "namespace", "*"):
        candidates = path_index.get(module_path, [])
        if candidates:
            return candidates[0]["id"], 0.9

    # ── Strategy 3: bare name match (no path hint) ────────────────────────────
    if symbol:
        hits = symbol_index.get(symbol.lower(), [])
        if len(hits) == 1:
            return hits[0]["id"], 0.85
        if len(hits) > 1:
            # Ambiguous — pick by kind priority, penalize confidence
            preferred = [h for h in hits if h["kind"] in ("function", "class", "method")]
            if preferred:
                return preferred[0]["id"], 0.6
            return hits[0]["id"], 0.5

    # ── Strategy 4: bare raw_ref as name (no colon split) ─────────────────────
    bare = raw_ref.rsplit(":", 1)[-1].lower()
    if bare and bare != raw_ref.lower():
        hits = symbol_index.get(bare, [])
        if hits:
            return hits[0]["id"], 0.4

    return None, 0.0


# ── Public API ────────────────────────────────────────────────────────────────

def link(project_id: str, min_confidence: float = 0.4) -> LinkResult:
    """Resolve all unresolved edges for a project.

    Args:
        project_id:      project to link
        min_confidence:  minimum confidence score to accept a resolution

    Returns:
        LinkResult with counts and any errors.
    """
    result = LinkResult()
    unresolved = _load_unresolved(project_id)

    if not unresolved:
        logger.info("[linker] %s: no unresolved edges", project_id)
        return result

    logger.info("[linker] %s: %d unresolved edges to process", project_id, len(unresolved))

    symbol_index = _build_symbol_index(project_id)
    path_index = _build_path_index(project_id)

    now = datetime.now(timezone.utc).isoformat()
    resolve_events: list[tuple[str, str | None, dict | None]] = []

    with get_db() as conn:
        for edge in unresolved:
            try:
                to_id, confidence = _resolve_edge(edge, symbol_index, path_index)
            except Exception as exc:
                result.failed += 1
                result.errors.append(f"edge {edge['id']}: {exc}")
                continue

            if to_id is None or confidence < min_confidence:
                result.failed += 1
                continue

            conn.execute(
                "UPDATE edges SET to_id = ?, resolved = 1, confidence = ? WHERE id = ?",
                (to_id, confidence, edge["id"]),
            )
            resolve_events.append((
                "EDGE_RESOLVED", None,
                {
                    "edge_id": edge["id"],
                    "from_id": edge["from_id"],
                    "to_id": to_id,
                    "raw_ref": edge["raw_ref"],
                    "kind": edge["kind"],
                    "confidence": confidence,
                },
            ))
            result.resolved += 1

    if resolve_events:
        emit_many(project_id, resolve_events)

    # Update project_state.unresolved_edges + recompute fan_in
    with get_db() as conn:
        still_unresolved = conn.execute(
            """SELECT COUNT(*) FROM edges e
               JOIN nodes n ON e.from_id = n.id
               WHERE n.project_id = ? AND e.resolved = 0""",
            (project_id,),
        ).fetchone()[0]
        conn.execute(
            "UPDATE project_state SET unresolved_edges = ? WHERE project_id = ?",
            (still_unresolved, project_id),
        )
        # Recompute fan_in = count of resolved calls/imports edges pointing TO each node
        conn.execute(
            """UPDATE nodes SET fan_in = (
                SELECT COUNT(*) FROM edges
                WHERE edges.to_id = nodes.id
                  AND edges.resolved = 1
                  AND edges.kind IN ('calls', 'imports')
            ) WHERE project_id = ?""",
            (project_id,),
        )

    logger.info(
        "[linker] %s: resolved=%d failed=%d remaining=%d",
        project_id, result.resolved, result.failed, still_unresolved,
    )
    return result
