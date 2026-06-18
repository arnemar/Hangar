"""File system discovery — emits filesystem state transitions into the event log.

Responsibilities
----------------
* Walk the repository tree
* Compute content hashes
* Compare against file_snapshots
* Emit FILE_DISCOVERED / FILE_CHANGED / FILE_REMOVED / FILE_HASH_STABLE
* Update file_snapshots (hash, language, size) — NOT last_ir_hash
  (that is owned by the graph builder after IR compilation succeeds)

Does NOT
--------
* Parse AST
* Touch nodes or edges
* Generate summaries or embeddings
* Interpret file content in any way

Snapshot invariant (must hold everywhere in the system)
-------------------------------------------------------
  If snapshot.hash == current file hash → emit FILE_HASH_STABLE, stop.
  Compilation pipeline MUST NOT proceed for unchanged files.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from compiler.ir import file_id
from compiler.events import emit, emit_many
from db import get_db


# ── Ignore lists ──────────────────────────────────────────────────────────────

IGNORE_DIRS: frozenset[str] = frozenset({
    "node_modules", ".git", "dist", "build", "target", ".next", ".nuxt",
    "__pycache__", ".cache", "coverage", "vendor", ".venv", "venv", "venv_win", "env",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".hypothesis",
    "generated", ".gradle", ".idea", ".vs", "out", "bin", "obj",
    ".terraform", ".serverless", "cdk.out",
})

IGNORE_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".exe",
    ".lock",       # lockfiles are rarely useful to parse
    ".min.js", ".min.css",
    ".map",        # source maps
    ".svg",        # binary-ish, skip for now
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".avif",
    ".mp4", ".mp3", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".pdf", ".docx", ".xlsx",
    ".wasm",
})

IGNORE_FILES: frozenset[str] = frozenset({
    ".DS_Store", "Thumbs.db", "desktop.ini",
    "package-lock.json",   # too noisy, use package.json only
    "yarn.lock", "pnpm-lock.yaml", "bun.lockb",
    "Cargo.lock",          # keep Cargo.toml; Cargo.lock is generated
    "poetry.lock", "Pipfile.lock",
    "composer.lock",
})


# ── Language detection ────────────────────────────────────────────────────────

LANG_BY_EXT: dict[str, str] = {
    ".ts": "typescript",  ".tsx": "typescript",
    ".js": "javascript",  ".jsx": "javascript",  ".mjs": "javascript",  ".cjs": "javascript",
    ".json": "json",      ".jsonc": "json",
    ".yaml": "yaml",      ".yml": "yaml",
    ".sql": "sql",
    ".md": "markdown",    ".mdx": "markdown",    ".rst": "markdown",
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",      ".kts": "kotlin",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".cpp": "cpp",        ".cc": "cpp",           ".cxx": "cpp",
    ".h": "cpp",          ".hpp": "cpp",
    ".sh": "shell",       ".bash": "shell",       ".zsh": "shell",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",      ".htm": "html",
    ".css": "css",        ".scss": "css",         ".sass": "css",
    ".graphql": "graphql", ".gql": "graphql",
    ".proto": "protobuf",
    ".tf": "terraform",   ".tfvars": "terraform",
    ".dockerfile": "docker",
    ".env": "env",        ".env.example": "env",
}

LANG_BY_NAME: dict[str, str] = {
    "Dockerfile": "docker",
    "Makefile": "makefile",
    "Gemfile": "ruby",
    ".gitignore": "gitignore",
    ".prettierrc": "json",
    ".eslintrc": "json",
    "tsconfig.json": "json",
    "package.json": "json",
}


def detect_language(path: Path) -> str:
    name = path.name
    if name in LANG_BY_NAME:
        return LANG_BY_NAME[name]
    ext = "".join(path.suffixes)  # handles .env.example
    return LANG_BY_EXT.get(ext, LANG_BY_EXT.get(path.suffix, "unknown"))


# ── Hashing ───────────────────────────────────────────────────────────────────

def hash_file(path: Path) -> tuple[str, int]:
    """Return (sha256_hex, byte_size). Reads in chunks to handle large files."""
    h = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
                size += len(chunk)
    except OSError:
        return "", 0
    return h.hexdigest(), size


# ── Snapshot helpers ──────────────────────────────────────────────────────────

def _load_snapshots(project_id: str) -> dict[str, dict]:
    """Load all file snapshots for a project keyed by path."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT path, hash, language, size FROM file_snapshots WHERE project_id = ?",
            (project_id,),
        ).fetchall()
    return {r["path"]: dict(r) for r in rows}


def _upsert_snapshot(project_id: str, path: str, hash_: str, language: str, size: int) -> None:
    fid = file_id(project_id, path)
    with get_db() as conn:
        conn.execute(
            """INSERT INTO file_snapshots (id, project_id, path, language, hash, size)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id, path) DO UPDATE SET
                   hash = excluded.hash,
                   language = excluded.language,
                   size = excluded.size""",
            (fid, project_id, path, language, hash_, size),
        )


def _delete_snapshot(project_id: str, path: str) -> None:
    with get_db() as conn:
        conn.execute(
            "DELETE FROM file_snapshots WHERE project_id = ? AND path = ?",
            (project_id, path),
        )


# ── Scan result ───────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    discovered: int = 0
    changed: int = 0
    removed: int = 0
    stable: int = 0
    skipped: int = 0

    @property
    def total_files(self) -> int:
        return self.discovered + self.changed + self.stable

    @property
    def needs_compilation(self) -> int:
        return self.discovered + self.changed


# ── Public API ────────────────────────────────────────────────────────────────

def scan(project_id: str, root: str | Path, max_file_size: int = 5 * 1024 * 1024) -> ScanResult:
    """Walk root, diff against snapshots, emit filesystem events.

    Emits only state-transition events — no parsing, no graph mutations.
    Files that have not changed emit FILE_HASH_STABLE and are NOT passed to
    the compilation pipeline.

    Args:
        project_id:    project identifier
        root:          repository root directory
        max_file_size: files larger than this (bytes) are skipped
    """
    root = Path(root).resolve()
    existing = _load_snapshots(project_id)
    seen_paths: set[str] = set()
    result = ScanResult()
    pending_events: list[tuple[str, str | None, dict | None]] = []

    emit(project_id, "COMPILATION_STARTED", payload={"root": str(root)})

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune ignored directories in-place (modifies os.walk iteration)
        dirnames[:] = [
            d for d in dirnames
            if d not in IGNORE_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if filename in IGNORE_FILES:
                result.skipped += 1
                continue

            filepath = Path(dirpath) / filename
            ext = "".join(filepath.suffixes)
            # Check full suffix chain AND single suffix AND .min.js/.min.css by name ending
            if (ext in IGNORE_EXTENSIONS
                    or filepath.suffix in IGNORE_EXTENSIONS
                    or filename.endswith(".min.js")
                    or filename.endswith(".min.css")):
                result.skipped += 1
                continue

            rel_path = str(filepath.relative_to(root))
            seen_paths.add(rel_path)
            language = detect_language(filepath)
            file_hash, size = hash_file(filepath)

            if not file_hash:
                result.skipped += 1
                continue

            if size > max_file_size:
                result.skipped += 1
                continue

            snapshot = existing.get(rel_path)

            if snapshot is None:
                # New file
                _upsert_snapshot(project_id, rel_path, file_hash, language, size)
                pending_events.append((
                    "FILE_DISCOVERED",
                    file_id(project_id, rel_path),
                    {"path": rel_path, "language": language, "hash": file_hash, "size": size},
                ))
                result.discovered += 1

            elif snapshot["hash"] != file_hash:
                # Content changed
                _upsert_snapshot(project_id, rel_path, file_hash, language, size)
                pending_events.append((
                    "FILE_CHANGED",
                    file_id(project_id, rel_path),
                    {
                        "path": rel_path,
                        "language": language,
                        "old_hash": snapshot["hash"],
                        "new_hash": file_hash,
                        "size": size,
                    },
                ))
                result.changed += 1

            else:
                # Hash stable — do not recompile
                pending_events.append((
                    "FILE_HASH_STABLE",
                    file_id(project_id, rel_path),
                    {"path": rel_path, "hash": file_hash},
                ))
                result.stable += 1

    # Detect removed files
    for path, _ in existing.items():
        if path not in seen_paths:
            _delete_snapshot(project_id, path)
            pending_events.append((
                "FILE_REMOVED",
                file_id(project_id, path),
                {"path": path},
            ))
            result.removed += 1

    emit_many(project_id, pending_events)
    emit(project_id, "COMPILATION_COMPLETED", payload={
        "discovered": result.discovered,
        "changed": result.changed,
        "removed": result.removed,
        "stable": result.stable,
        "skipped": result.skipped,
    })

    return result
