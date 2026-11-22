"""Dispatch syntactic extraction to the appropriate language parser."""

from __future__ import annotations
from hangar.compiler.extractors.common import ASTResult
from hangar.compiler.extractors.python import _PythonExtractor
from hangar.compiler.extractors.javascript import _TSExtractor


_SUPPORTED: dict[str, str] = {
    "typescript": "typescript",
    "javascript": "javascript",
    "python":     "python",
}


def extract(
    project_id: str,
    path: str,
    source: str,
    language: str,
) -> ASTResult:
    """Pure syntactic extraction. No DB, no events, no side effects.

    Returns unresolved IRNodes and IREdges.
    All cross-file edges have resolved=False and to_id=None.
    Containment edges have resolved=True.
    """
    if language not in _SUPPORTED:
        return ASTResult()
    if language == "python":
        return _PythonExtractor().extract(project_id, path, source, language)
    extractor = _TSExtractor()
    return extractor.extract(project_id, path, source, language)


def supported_languages() -> frozenset[str]:
    return frozenset(_SUPPORTED)
