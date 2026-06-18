"""Compilation pipeline — orchestrates discovery → canonicalize → graph_build.

Entry point: compile_project(project_id, root)

Flow
----
1. scan(root)               — filesystem diff, emits FILE_* events
2. for each changed file:
     canonicalize()         — AST → IRFile, checks IR-level hash
     build(ir)              — IRFile → graph mutations + events
3. Update file_snapshots.last_ir_hash after each successful build

The snapshot invariants ensure idempotency:
    * File hash stable          → skip entirely (discovery handles this)
    * IR hash (content) stable  → canonicalizer returns None, skip graph write
    * Both must change          → full recompile for that file

Concurrency: single-threaded. Files are compiled sequentially.
Parallelism is safe but not implemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from compiler.discovery import scan, ScanResult
from compiler.canonicalizer import canonicalize
from compiler.graph_builder import build, BuildResult
from compiler.linker import link, LinkResult
from compiler.events import emit
from db import get_db

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    scan: ScanResult | None = None
    files_compiled: int = 0
    files_skipped_ir: int = 0
    files_failed: int = 0
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_removed: int = 0
    edges_created: int = 0
    edges_resolved: int = 0
    errors: list[str] = field(default_factory=list)
    # Rich symbol deltas (populated by compile_file)
    added_names: list[str] = field(default_factory=list)
    removed_names: list[str] = field(default_factory=list)
    updated_names: list[str] = field(default_factory=list)
    # Graph consistency report (populated when verification="full")
    graph_health: dict = field(default_factory=dict)


def _project_state_start(project_id: str, now: str) -> None:
    with get_db() as conn:
        conn.execute(
            """INSERT INTO project_state (project_id, is_compiling, last_compile_started)
               VALUES (?, 1, ?)
               ON CONFLICT(project_id) DO UPDATE SET
                   is_compiling = 1,
                   last_compile_started = excluded.last_compile_started""",
            (project_id, now),
        )


def _project_state_finish(project_id: str, now: str) -> None:
    with get_db() as conn:
        total_nodes = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        total_edges = conn.execute(
            "SELECT COUNT(*) FROM edges e JOIN nodes n ON e.from_id = n.id WHERE n.project_id = ?",
            (project_id,)
        ).fetchone()[0]
        unresolved = conn.execute(
            "SELECT COUNT(*) FROM edges e JOIN nodes n ON e.from_id = n.id WHERE n.project_id = ? AND e.resolved = 0",
            (project_id,)
        ).fetchone()[0]
        conn.execute(
            """UPDATE project_state SET
               is_compiling = 0,
               last_compile_finished = ?,
               graph_version = graph_version + 1,
               total_nodes = ?,
               total_edges = ?,
               unresolved_edges = ?
               WHERE project_id = ?""",
            (now, total_nodes, total_edges, unresolved, project_id),
        )


def _check_graph_health(project_id: str) -> dict:
    """Lightweight consistency checks: dangling edges, FTS/node count parity."""
    with get_db() as conn:
        dangling = conn.execute(
            """SELECT COUNT(*) FROM edges e
               JOIN nodes fn ON e.from_id = fn.id
               WHERE fn.project_id = ?
                 AND e.to_id IS NOT NULL
                 AND e.to_id NOT IN (SELECT id FROM nodes)""",
            (project_id,),
        ).fetchone()[0]
        node_count = conn.execute(
            "SELECT COUNT(*) FROM nodes WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM nodes_fts WHERE project_id = ?", (project_id,)
        ).fetchone()[0]
    return {
        "dangling_edges": dangling,
        "fts_node_mismatch": abs(node_count - fts_count),
        "node_count": node_count,
    }


def _load_changed_files(project_id: str) -> list[dict]:
    """Return file_snapshots rows that need IR recompilation.

    A file needs recompilation if it was discovered/changed (no last_ir_hash yet
    or last_ir_hash != current hash).
    """
    with get_db() as conn:
        rows = conn.execute(
            """SELECT path, language, hash, last_ir_hash
               FROM file_snapshots
               WHERE project_id = ?
               AND (last_ir_hash IS NULL OR last_ir_hash != hash)""",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _read_file(root: Path, rel_path: str) -> str | None:
    """Read file content. Returns None on IO error."""
    try:
        return (root / rel_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def compile_file(
    project_id: str, root: str | Path, rel_path: str, check_health: bool = False
) -> PipelineResult:
    """Incrementally recompile a single file and relink affected edges.

    Used by tool_edit_file after a successful edit to keep the graph current
    without scanning the entire project.

    Args:
        check_health: if True, run _check_graph_health after linking and attach
                      the report to result.graph_health.
    """
    from compiler.discovery import detect_language, hash_file
    root = Path(root).resolve()
    result = PipelineResult()

    filepath = root / rel_path
    language = detect_language(filepath)
    file_hash, _ = hash_file(filepath)

    # Snapshot symbol names before the build so we can compute rich deltas
    with get_db() as conn:
        before_names: set[str] = {
            r["name"] for r in conn.execute(
                "SELECT name FROM nodes WHERE project_id = ? AND path = ?",
                (project_id, rel_path),
            ).fetchall()
        }

    # Update snapshot hash so the file is treated as changed
    from compiler.ir import file_id
    fid = file_id(project_id, rel_path)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO file_snapshots (id, project_id, path, language, hash, size)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id, path) DO UPDATE SET
                   hash = excluded.hash,
                   last_ir_hash = NULL""",
            (fid, project_id, rel_path, language, file_hash, filepath.stat().st_size),
        )

    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        result.errors.append(str(e))
        result.files_failed = 1
        return result

    try:
        ir = canonicalize(project_id=project_id, path=rel_path, source=source,
                          language=language, last_ir_hash=None)
    except Exception as e:
        result.errors.append(f"canonicalize: {e}")
        result.files_failed = 1
        return result

    if ir is None:
        result.files_skipped_ir = 1
        return result

    try:
        br: BuildResult = build(ir)
        result.files_compiled = 1
        result.nodes_created = br.nodes_created
        result.nodes_updated = br.nodes_updated
        result.nodes_removed = br.nodes_removed
        result.edges_created = br.edges_created
    except Exception as e:
        result.errors.append(f"graph_build: {e}")
        result.files_failed = 1
        return result

    # Relink edges touching this file's nodes
    lr = link(project_id)
    result.edges_resolved = lr.resolved

    # Compute rich symbol deltas
    with get_db() as conn:
        after_names: set[str] = {
            r["name"] for r in conn.execute(
                "SELECT name FROM nodes WHERE project_id = ? AND path = ?",
                (project_id, rel_path),
            ).fetchall()
        }
    result.added_names = sorted(after_names - before_names)
    result.removed_names = sorted(before_names - after_names)
    result.updated_names = sorted(before_names & after_names)

    if check_health:
        result.graph_health = _check_graph_health(project_id)

    return result


def compile_project(project_id: str, root: str | Path) -> PipelineResult:
    """Full compilation: scan → canonicalize → graph_build.

    Args:
        project_id: project identifier
        root:       repository root (absolute path)

    Returns:
        PipelineResult with counters and any errors.
    """
    from datetime import datetime, timezone

    root = Path(root).resolve()
    result = PipelineResult()
    now = datetime.now(timezone.utc).isoformat()

    _project_state_start(project_id, now)

    # ── Phase 1: filesystem scan ──────────────────────────────────────────────
    logger.info("[pipeline] scanning %s", root)
    result.scan = scan(project_id, root)
    logger.info(
        "[pipeline] scan complete — discovered=%d changed=%d removed=%d stable=%d",
        result.scan.discovered, result.scan.changed,
        result.scan.removed, result.scan.stable,
    )

    # ── Phase 2: compile changed files ────────────────────────────────────────
    pending = _load_changed_files(project_id)
    logger.info("[pipeline] %d files need IR compilation", len(pending))

    for row in pending:
        rel_path = row["path"]
        language = row["language"]
        last_ir_hash = row.get("last_ir_hash")

        source = _read_file(root, rel_path)
        if source is None:
            result.files_failed += 1
            result.errors.append(f"read error: {rel_path}")
            continue

        try:
            ir = canonicalize(
                project_id=project_id,
                path=rel_path,
                source=source,
                language=language,
                last_ir_hash=last_ir_hash,
            )
        except Exception as exc:
            result.files_failed += 1
            result.errors.append(f"canonicalize error: {rel_path}: {exc}")
            logger.exception("[pipeline] canonicalize failed: %s", rel_path)
            continue

        if ir is None:
            # IR hash stable — no graph changes needed
            result.files_skipped_ir += 1
            continue

        try:
            br: BuildResult = build(ir)
        except Exception as exc:
            result.files_failed += 1
            result.errors.append(f"graph_build error: {rel_path}: {exc}")
            logger.exception("[pipeline] graph_build failed: %s", rel_path)
            continue

        result.files_compiled += 1
        result.nodes_created += br.nodes_created
        result.nodes_updated += br.nodes_updated
        result.nodes_removed += br.nodes_removed
        result.edges_created += br.edges_created

        logger.debug(
            "[pipeline] %s — +%d nodes, ~%d updated, -%d removed, +%d edges",
            rel_path, br.nodes_created, br.nodes_updated,
            br.nodes_removed, br.edges_created,
        )

    if result.errors:
        logger.warning("[pipeline] %d files failed", result.files_failed)

    # ── Phase 3: link cross-file edges ────────────────────────────────────────
    if result.files_compiled > 0 or result.scan.discovered > 0:
        logger.info("[pipeline] running linker")
        lr = link(project_id)
        result.edges_resolved = lr.resolved
        result.errors.extend(lr.errors)

    finish_time = datetime.now(timezone.utc).isoformat()
    _project_state_finish(project_id, finish_time)

    logger.info(
        "[pipeline] done — compiled=%d skipped_ir=%d failed=%d "
        "nodes(+%d ~%d -%d) edges(+%d)",
        result.files_compiled, result.files_skipped_ir, result.files_failed,
        result.nodes_created, result.nodes_updated, result.nodes_removed,
        result.edges_created,
    )

    return result
