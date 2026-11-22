"""Tree-sitter parsers and JavaScript syntax helpers."""

from __future__ import annotations
import re


try:
    import tree_sitter_typescript as _ts_typescript
    import tree_sitter_javascript as _ts_javascript
    from tree_sitter import Language, Parser, Node
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False


_parsers: dict[str, "Parser"] = {}


def _get_parser(language: str) -> "Parser | None":
    if not _TREE_SITTER_AVAILABLE:
        return None
    if language not in _parsers:
        try:
            if language == "typescript":
                lang = Language(_ts_typescript.language_typescript())
            elif language == "javascript":
                lang = Language(_ts_javascript.language())
            else:
                return None
            _parsers[language] = Parser(lang)
        except Exception:
            return None
    return _parsers[language]


# Utility

def _text(node: "Node", source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _line(node: "Node") -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _child_text(node: "Node", field_name: str, source_bytes: bytes) -> str:
    child = node.child_by_field_name(field_name)
    return _text(child, source_bytes) if child else ""


def _signature(node: "Node", source_bytes: bytes) -> str:
    """Extract signature: source up to first { or => body, stripped."""
    raw = _text(node, source_bytes)
    # Cut at opening brace or arrow body
    for sep in ("{", "=>"):
        idx = raw.find(sep)
        if idx != -1:
            raw = raw[:idx]
    return re.sub(r"\s+", " ", raw).strip()[:200]


def _visibility(node: "Node", source_bytes: bytes, is_exported: bool) -> str:
    """Determine visibility from export status and TS accessibility modifier."""
    for child in node.children:
        if child.type == "accessibility_modifier":
            return _text(child, source_bytes).strip()
    return "public" if is_exported else "internal"
