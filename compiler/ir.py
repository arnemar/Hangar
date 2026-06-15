"""Canonical Intermediate Representation.

Pure data types — no DB access, no IO, no side effects.
The canonicalizer produces IRFile; the graph builder consumes it.

Phase A (frontend): AST → IRFile           deterministic, stateless
Phase B (backend):  IRFile → graph + events stateful, async
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


# ── Canonical ID ──────────────────────────────────────────────────────────────

def node_id(project_id: str, path: str, kind: str, name: str, parent: str = "") -> str:
    """Deterministic full SHA-256 node identity.

    Identity factors: project, file path, symbol kind, parent name, symbol name.
    Content (body, signature) is NOT part of identity — it can change without
    changing what the symbol IS.
    """
    key = f"{project_id}:{path}:{kind}:{parent}:{name}"
    return hashlib.sha256(key.encode()).hexdigest()


def file_id(project_id: str, path: str) -> str:
    return hashlib.sha256(f"{project_id}:{path}:file:".encode()).hexdigest()


def content_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


# ── IR node kinds ─────────────────────────────────────────────────────────────

# Physical (filesystem / AST)
PHYSICAL_KINDS = {
    "project", "folder", "file",
    "class", "interface", "enum", "struct", "type_alias",
    "function", "method", "constructor", "property", "getter", "setter",
    "variable", "constant", "parameter",
    "import", "export", "namespace", "module",
    "decorator", "annotation",
}

# Semantic (application-level, extracted by language-specific logic)
SEMANTIC_KINDS = {
    "api_endpoint", "route", "middleware",
    "db_table", "db_column", "migration", "schema",
    "config_key", "env_var",
    "test_case", "test_suite",
    "docker_service", "ci_job", "cli_command",
}

# Document (README, specs, docs)
DOCUMENT_KINDS = {
    "readme_section", "doc_section", "api_spec",
}

# Edge relationship types
EDGE_KINDS = {
    "contains",    # parent → child (structural)
    "calls",       # function → function
    "imports",     # file → file/symbol
    "inherits",    # class → class
    "implements",  # class → interface
    "uses",        # general dependency
    "reads",       # reads a field/variable
    "writes",      # writes a field/variable
    "creates",     # instantiates
    "returns",     # returns type
    "throws",      # throws exception type
    "tests",       # test → subject
    "references",  # loose reference
    "depends_on",  # build/config dependency
    "configured_by",
}


# ── IR types (immutable, hashable) ────────────────────────────────────────────

@dataclass(frozen=True)
class IRNode:
    id: str              # full SHA-256 from node_id()
    kind: str
    name: str
    path: str
    project_id: str
    parent_id: str | None
    signature: str
    source: str
    start_line: int
    end_line: int
    language: str
    visibility: str      # public / private / protected / internal


@dataclass(frozen=True)
class IREdge:
    from_id: str         # always a resolved node ID
    to_id: str | None    # resolved node ID, or None if unresolved
    raw_ref: str | None  # original reference string e.g. "AuthService.login"
    kind: str            # one of EDGE_KINDS
    resolved: bool
    confidence: float = 1.0


@dataclass(frozen=True)
class IRFile:
    """Output of the canonicalizer for a single file. Pure data, no side effects."""
    project_id: str
    path: str
    language: str
    hash: str            # sha256 of raw file content
    nodes: tuple[IRNode, ...]
    edges: tuple[IREdge, ...]
