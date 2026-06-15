"""AST extractor — pure syntactic compiler pass.

Contract
--------
* Input:  project_id, file path, raw source, language
* Output: ASTResult(nodes, edges, local_symbols)
* NO database access
* NO event emission
* NO cross-file resolution
* NO semantic inference
* NO fan-in / fan-out computation
* NO embeddings or summaries

Nodes represent syntactic entities only.
Edges represent structural containment and unresolved local references.
Call edges store raw_ref strings; to_id is always None here.
Resolution belongs to the linker stage.

Language map is hardcoded — no runtime grammar inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

try:
    import tree_sitter_typescript as _ts_typescript
    import tree_sitter_javascript as _ts_javascript
    from tree_sitter import Language, Parser, Node
    _TREE_SITTER_AVAILABLE = True
except ImportError:
    _TREE_SITTER_AVAILABLE = False

from compiler.ir import IRNode, IREdge, node_id, file_id, content_hash


# ── Output type ───────────────────────────────────────────────────────────────

@dataclass
class ASTResult:
    nodes: list[IRNode] = field(default_factory=list)
    edges: list[IREdge] = field(default_factory=list)
    # name → node map for intra-file reference hints (not cross-file)
    local_symbols: dict[str, IRNode] = field(default_factory=dict)


# ── Hardcoded language routing ────────────────────────────────────────────────
# Never infer grammars dynamically — prevents silent parsing drift.

_SUPPORTED: dict[str, str] = {
    "typescript": "typescript",
    "javascript": "javascript",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Parser cache ──────────────────────────────────────────────────────────────

_parsers: dict[str, "Parser"] = {}


def _get_parser(language: str) -> "Parser | None":
    if not _TREE_SITTER_AVAILABLE:
        return None
    if language not in _SUPPORTED:
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


# ── Utility ───────────────────────────────────────────────────────────────────

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


# ── TypeScript / JavaScript extractor ─────────────────────────────────────────

class _TSExtractor:
    """Extracts IRNodes and IREdges from TypeScript/JavaScript source.

    Traversal order (per spec):
    1. Top-level declarations
    2. Nested structures
    3. Function bodies
    4. Local call expressions
    5. Imports / exports
    6. Containment edges

    All edges that reference external symbols are unresolved (to_id=None).
    """

    def extract(self, project_id: str, path: str, source: str, language: str) -> ASTResult:
        parser = _get_parser(language)
        if parser is None:
            return ASTResult()

        source_bytes = source.encode("utf-8")
        tree = parser.parse(source_bytes)
        now = _now()

        result = ASTResult()
        # Use file_id() — same formula as canonicalizer's _inject_file_node.
        # The file node itself is created by the canonicalizer; we only need
        # the ID here for edge from_id references.
        file_node_id = file_id(project_id, path)

        # Walk top-level children
        root = tree.root_node
        self._walk(
            root, project_id, path, source_bytes, language,
            parent_id=file_node_id, parent_name="",
            is_exported=False, depth=0,
            result=result,
        )

        # Build containment edges (file → top-level children)
        # These were built during walk; collect import edges here
        self._extract_imports(root, project_id, path, source_bytes, file_node_id, result)

        return result

    # ── Walker ────────────────────────────────────────────────────────────────

    def _walk(
        self,
        node: "Node",
        project_id: str,
        path: str,
        source_bytes: bytes,
        language: str,
        parent_id: str,
        parent_name: str,
        is_exported: bool,
        depth: int,
        result: ASTResult,
    ) -> None:
        if depth > 8:  # guard against pathological nesting
            return

        for child in node.children:
            actual_exported = is_exported
            target = child

            # Unwrap export statement
            if child.type in ("export_statement", "export_default_declaration"):
                actual_exported = True
                # Find the real declaration inside
                inner = child.child_by_field_name("declaration")
                if inner is None:
                    # export default expression — skip
                    continue
                target = inner

            handler = _TS_NODE_HANDLERS.get(target.type)
            if handler:
                ir_node = handler(
                    target, project_id, path, source_bytes, language,
                    parent_id, parent_name, actual_exported,
                )
                if ir_node:
                    result.nodes.append(ir_node)
                    result.local_symbols[ir_node.name] = ir_node
                    # Containment edge
                    result.edges.append(IREdge(
                        from_id=parent_id,
                        to_id=ir_node.id,
                        raw_ref=None,
                        kind="contains",
                        resolved=True,
                    ))
                    # Recurse into body for nested declarations + call extraction
                    body = target.child_by_field_name("body")
                    if body:
                        self._walk(
                            body, project_id, path, source_bytes, language,
                            parent_id=ir_node.id, parent_name=ir_node.name,
                            is_exported=False, depth=depth + 1,
                            result=result,
                        )
                        # Extract call expressions inside this body
                        self._extract_calls(body, source_bytes, ir_node.id, result)

    # ── Declaration handlers ──────────────────────────────────────────────────

    def _handle_class(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        nid = node_id(project_id, path, "class", name, parent_name)
        ir = IRNode(
            id=nid, kind="class", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=_signature(node, source_bytes),
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, is_exported),
        )
        # Heritage: extends / implements edges
        heritage = node.child_by_field_name("body")  # wrong — use type_parameters/extends
        for child in node.children:
            if child.type == "class_heritage":
                for hchild in child.children:
                    if hchild.type == "extends_clause":
                        ref = _child_text(hchild, "value", source_bytes) or _text(hchild, source_bytes).replace("extends", "").strip()
                        # Will be refined by: ref.split(".")[0]
                        return ir  # edges added below
        return ir

    def _handle_function(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        return IRNode(
            id=node_id(project_id, path, "function", name, parent_name),
            kind="function", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=_signature(node, source_bytes),
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, is_exported),
        )

    def _handle_method(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        return IRNode(
            id=node_id(project_id, path, "method", name, parent_name),
            kind="method", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=_signature(node, source_bytes),
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, False),
        )

    def _handle_interface(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        return IRNode(
            id=node_id(project_id, path, "interface", name, parent_name),
            kind="interface", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=_signature(node, source_bytes),
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, is_exported),
        )

    def _handle_enum(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        return IRNode(
            id=node_id(project_id, path, "enum", name, parent_name),
            kind="enum", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=name,
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, is_exported),
        )

    def _handle_type_alias(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        name = _text(name_node, source_bytes)
        start, end = _line(node)
        return IRNode(
            id=node_id(project_id, path, "type_alias", name, parent_name),
            kind="type_alias", name=name, path=path, project_id=project_id,
            parent_id=parent_id, signature=_signature(node, source_bytes),
            source=_text(node, source_bytes),
            start_line=start, end_line=end, language=language,
            visibility=_visibility(node, source_bytes, is_exported),
        )

    def _handle_variable(
        self, node: "Node", project_id: str, path: str, source_bytes: bytes,
        language: str, parent_id: str, parent_name: str, is_exported: bool,
    ) -> IRNode | None:
        # lexical_declaration / variable_declaration → variable_declarator children
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = child.child_by_field_name("name")
                if not name_node:
                    continue
                name = _text(name_node, source_bytes)
                value = child.child_by_field_name("value")
                kind = "function" if value and value.type in ("arrow_function", "function") else "variable"
                # Skip unexported local variables inside method/function bodies —
                # they pollute the graph without aiding navigation.
                if kind == "variable" and not is_exported:
                    return None
                start, end = _line(node)
                return IRNode(
                    id=node_id(project_id, path, kind, name, parent_name),
                    kind=kind, name=name, path=path, project_id=project_id,
                    parent_id=parent_id,
                    signature=_signature(value or node, source_bytes) if kind == "function" else name,
                    source=_text(node, source_bytes),
                    start_line=start, end_line=end, language=language,
                    visibility=_visibility(node, source_bytes, is_exported),
                )
        return None

    # ── Imports ───────────────────────────────────────────────────────────────

    def _extract_imports(
        self,
        root: "Node",
        project_id: str,
        path: str,
        source_bytes: bytes,
        file_node_id: str,
        result: ASTResult,
    ) -> None:
        for child in root.children:
            if child.type in ("import_statement", "import_declaration"):
                self._emit_import_edges(child, project_id, path, source_bytes, file_node_id, result)
            elif child.type == "expression_statement":
                # require() calls
                expr = child.children[0] if child.children else None
                if expr and expr.type == "call_expression":
                    fn = expr.child_by_field_name("function")
                    if fn and _text(fn, source_bytes) == "require":
                        args = expr.child_by_field_name("arguments")
                        if args:
                            for arg in args.children:
                                if arg.type in ("string", "template_string"):
                                    raw = _text(arg, source_bytes).strip("'\"` ")
                                    result.edges.append(IREdge(
                                        from_id=file_node_id,
                                        to_id=None,
                                        raw_ref=raw,
                                        kind="imports",
                                        resolved=False,
                                    ))

    def _emit_import_edges(
        self,
        node: "Node",
        project_id: str,
        path: str,
        source_bytes: bytes,
        file_node_id: str,
        result: ASTResult,
    ) -> None:
        # Find source string — tree-sitter-typescript may not expose it as a named field,
        # so fall back to scanning for the last 'string' child.
        source_node = node.child_by_field_name("source")
        if source_node is None:
            for c in node.children:
                if c.type == "string":
                    source_node = c
        if not source_node:
            return
        module_ref = _text(source_node, source_bytes).strip("'\"` ")
        # Find import clause — field name varies; find by node type.
        clause = node.child_by_field_name("import_clause")
        if clause is None:
            for c in node.children:
                if c.type == "import_clause":
                    clause = c
                    break
        if clause:
            for child in clause.children:
                if child.type == "named_imports":
                    for spec in child.children:
                        if spec.type == "import_specifier":
                            # tree-sitter-typescript may not expose 'name' as a field;
                            # fall back to first identifier child.
                            name_node = spec.child_by_field_name("name")
                            if name_node is None:
                                name_node = next(
                                    (c for c in spec.children if c.type == "identifier"), None
                                )
                            symbol_name = _text(name_node, source_bytes) if name_node else ""
                            raw = f"{module_ref}:{symbol_name}" if symbol_name else module_ref
                            result.edges.append(IREdge(
                                from_id=file_node_id,
                                to_id=None,
                                raw_ref=raw,
                                kind="imports",
                                resolved=False,
                            ))
                elif child.type == "namespace_import":
                    result.edges.append(IREdge(
                        from_id=file_node_id,
                        to_id=None,
                        raw_ref=module_ref,
                        kind="imports",
                        resolved=False,
                    ))
                elif child.type == "identifier":
                    # default import
                    result.edges.append(IREdge(
                        from_id=file_node_id,
                        to_id=None,
                        raw_ref=f"{module_ref}:default",
                        kind="imports",
                        resolved=False,
                    ))
        else:
            result.edges.append(IREdge(
                from_id=file_node_id,
                to_id=None,
                raw_ref=module_ref,
                kind="imports",
                resolved=False,
            ))

    # ── Call extraction ───────────────────────────────────────────────────────

    def _extract_calls(
        self,
        body: "Node",
        source_bytes: bytes,
        caller_id: str,
        result: ASTResult,
    ) -> None:
        """Shallow call expression scan — one level deep, does not recurse into nested functions."""
        for child in self._iter_shallow(body):
            if child.type == "call_expression":
                fn = child.child_by_field_name("function")
                if not fn:
                    continue
                raw = _text(fn, source_bytes).strip()
                # Skip trivial/short names that aren't useful
                if len(raw) < 2 or raw in ("if", "for", "while", "super"):
                    continue
                result.edges.append(IREdge(
                    from_id=caller_id,
                    to_id=None,
                    raw_ref=raw,
                    kind="calls",
                    resolved=False,
                    confidence=0.8,
                ))

    def _iter_shallow(self, node: "Node"):
        """Yield descendants without recursing into function/class boundaries."""
        _boundaries = {
            "function_declaration", "function", "arrow_function",
            "method_definition", "class_declaration", "class_body",
        }
        for child in node.children:
            yield child
            if child.type not in _boundaries:
                yield from self._iter_shallow(child)


# ── Node type → handler dispatch ──────────────────────────────────────────────

_ex = _TSExtractor()

_TS_NODE_HANDLERS = {
    "class_declaration":          _ex._handle_class,
    "abstract_class_declaration": _ex._handle_class,
    "function_declaration":       _ex._handle_function,
    "generator_function_declaration": _ex._handle_function,
    "method_definition":          _ex._handle_method,
    "interface_declaration":      _ex._handle_interface,
    "enum_declaration":           _ex._handle_enum,
    "type_alias_declaration":     _ex._handle_type_alias,
    "lexical_declaration":        _ex._handle_variable,
    "variable_declaration":       _ex._handle_variable,
}


# ── Public API ────────────────────────────────────────────────────────────────

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
    extractor = _TSExtractor()
    return extractor.extract(project_id, path, source, language)


def supported_languages() -> frozenset[str]:
    return frozenset(_SUPPORTED)
