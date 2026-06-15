"""Canonicalizer — AST → IRFile compilation with snapshot-invariant enforcement.

Contract
--------
* Input:  project_id, path, raw source bytes/str, language, optional last_ir_hash
* Output: IRFile (pure, frozen) OR None (if snapshot invariant holds — hash unchanged)
* NO database access
* NO event emission
* Deterministic: same input always produces identical IRFile

Snapshot invariant
------------------
The canonicalizer enforces the *IR-level* invariant:
    if content_hash(source) == last_ir_hash → return None (no recompilation)

The caller (graph_builder) holds the last_ir_hash; we receive it as a parameter.
The graph_builder writes it back after successful graph mutation.

Merge rules
-----------
When the same symbol appears more than once in a file (e.g. overloaded functions,
re-opened interfaces), we merge them:
    - Use the FIRST declaration's start_line
    - Use the LAST declaration's end_line
    - Concatenate signatures, deduplicate, join with " | "
    - Concatenate source bodies
    - Keep identity (id) stable — it is hash of (project:path:kind:parent:name)

Edge deduplication
------------------
Edges are keyed by (from_id, raw_ref, kind).
Duplicate edges are dropped; first occurrence wins (highest confidence).

Node deduplication
------------------
Nodes are keyed by id (deterministic SHA-256).
Duplicate nodes (same identity) are merged, not duplicated.

Unsupported languages
---------------------
Returns an IRFile with no nodes/edges for unsupported languages.
This is intentional — the graph builder will store the file node only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from compiler.ir import (
    IRFile, IRNode, IREdge,
    node_id as make_node_id,
    file_id as make_file_id,
    content_hash,
)
from compiler.ast_extractor import extract as ast_extract, ASTResult, supported_languages


# ── Merge helpers ─────────────────────────────────────────────────────────────

def _merge_nodes(existing: IRNode, incoming: IRNode) -> IRNode:
    """Merge two nodes with the same identity (e.g. repeated interface declarations)."""
    # Merge signatures: deduplicate parts, preserve order
    existing_sigs = [s.strip() for s in existing.signature.split("|") if s.strip()]
    new_sigs = [s.strip() for s in incoming.signature.split("|") if s.strip()]
    merged_sig_parts: list[str] = list(existing_sigs)
    for s in new_sigs:
        if s not in merged_sig_parts:
            merged_sig_parts.append(s)
    merged_sig = " | ".join(merged_sig_parts) if merged_sig_parts else ""

    # Merge source: concatenate with separator
    merged_source = existing.source
    if incoming.source and incoming.source != existing.source:
        merged_source = existing.source + "\n\n" + incoming.source

    return IRNode(
        id=existing.id,
        kind=existing.kind,
        name=existing.name,
        path=existing.path,
        project_id=existing.project_id,
        parent_id=existing.parent_id,
        signature=merged_sig,
        source=merged_source,
        start_line=min(existing.start_line, incoming.start_line),
        end_line=max(existing.end_line, incoming.end_line),
        language=existing.language,
        visibility=existing.visibility,
    )


def _dedup_nodes(raw: list[IRNode]) -> list[IRNode]:
    """Merge nodes with identical IDs (same identity, different occurrences)."""
    seen: dict[str, IRNode] = {}
    for node in raw:
        if node.id in seen:
            seen[node.id] = _merge_nodes(seen[node.id], node)
        else:
            seen[node.id] = node
    return list(seen.values())


def _dedup_edges(raw: list[IREdge]) -> list[IREdge]:
    """Drop duplicate edges. Key: (from_id, raw_ref or to_id, kind). First wins."""
    seen: set[tuple] = set()
    result: list[IREdge] = []
    for edge in raw:
        key = (edge.from_id, edge.raw_ref or edge.to_id, edge.kind)
        if key not in seen:
            seen.add(key)
            result.append(edge)
    return result


def _inject_file_node(project_id: str, path: str, language: str) -> IRNode:
    """Every IRFile includes an explicit file-level node."""
    fid = make_file_id(project_id, path)
    # File node name is the filename portion
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return IRNode(
        id=fid,
        kind="file",
        name=name,
        path=path,
        project_id=project_id,
        parent_id=None,
        signature="",
        source="",
        start_line=0,
        end_line=0,
        language=language,
        visibility="public",
    )


def _inject_contains_edges(file_node_id: str, nodes: list[IRNode]) -> list[IREdge]:
    """Add file→top-level-symbol containment edges for symbols with no parent."""
    edges = []
    for node in nodes:
        if node.kind == "file":
            continue
        if node.parent_id is None:
            edges.append(IREdge(
                from_id=file_node_id,
                to_id=node.id,
                raw_ref=None,
                kind="contains",
                resolved=True,
                confidence=1.0,
            ))
    return edges


# ── Public API ────────────────────────────────────────────────────────────────

def canonicalize(
    project_id: str,
    path: str,
    source: str,
    language: str,
    last_ir_hash: str | None = None,
) -> IRFile | None:
    """Compile source → IRFile, or None if content is unchanged.

    Args:
        project_id:   project identifier
        path:         repo-relative path (used in node IDs)
        source:       raw file content (text)
        language:     language string from discovery (e.g. "typescript")
        last_ir_hash: sha256 of source from last successful compilation.
                      If it matches current content_hash, return None.

    Returns:
        IRFile ready for graph_builder consumption, or None if hash stable.
    """
    src_hash = content_hash(source)

    # ── Snapshot invariant: IR level ──────────────────────────────────────────
    if last_ir_hash and last_ir_hash == src_hash:
        return None

    file_node = _inject_file_node(project_id, path, language)

    # ── Unsupported language: file node only ──────────────────────────────────
    if language not in supported_languages():
        return IRFile(
            project_id=project_id,
            path=path,
            language=language,
            hash=src_hash,
            nodes=(file_node,),
            edges=(),
        )

    # ── AST extraction (pure, no IO) ──────────────────────────────────────────
    result: ASTResult = ast_extract(project_id, path, source, language)

    # ── Merge duplicate nodes ──────────────────────────────────────────────────
    nodes = _dedup_nodes(result.nodes)

    # ── Add file node first, contains-edges for top-level symbols ─────────────
    contains_edges = _inject_contains_edges(file_node.id, nodes)
    all_nodes = [file_node] + nodes

    # ── Merge duplicate edges (AST edges + contains edges) ────────────────────
    all_edges = _dedup_edges(result.edges + contains_edges)

    return IRFile(
        project_id=project_id,
        path=path,
        language=language,
        hash=src_hash,
        nodes=tuple(all_nodes),
        edges=tuple(all_edges),
    )
