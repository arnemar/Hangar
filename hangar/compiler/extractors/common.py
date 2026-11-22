"""Shared syntactic extraction result."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hangar.compiler.ir import IRNode, IREdge


@dataclass
class ASTResult:
    nodes: list[IRNode] = field(default_factory=list)
    edges: list[IREdge] = field(default_factory=list)
    # name → node map for intra-file reference hints (not cross-file)
    local_symbols: dict[str, IRNode] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
