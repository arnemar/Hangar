"""Extract local calls and import references from JavaScript syntax."""

from __future__ import annotations
from hangar.compiler.ir import IREdge
from hangar.compiler.extractors.common import ASTResult
from hangar.compiler.extractors.javascript_syntax import _text


try:
    from tree_sitter import Node
except ImportError:
    pass


def _extract_imports(
    root: "Node",
    project_id: str,
    path: str,
    source_bytes: bytes,
    file_node_id: str,
    result: ASTResult,
) -> None:
    for child in root.children:
        if child.type in ("import_statement", "import_declaration"):
            _emit_import_edges(child, project_id, path, source_bytes, file_node_id, result)
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

# Call extraction

def _extract_calls(
    body: "Node",
    source_bytes: bytes,
    caller_id: str,
    result: ASTResult,
) -> None:
    """Shallow call expression scan — one level deep, does not recurse into nested functions."""
    for child in _iter_shallow(body):
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

def _iter_shallow(node: "Node"):
    """Yield descendants without recursing into function/class boundaries."""
    _boundaries = {
        "function_declaration", "function", "arrow_function",
        "method_definition", "class_declaration", "class_body",
    }
    for child in node.children:
        yield child
        if child.type not in _boundaries:
            yield from _iter_shallow(child)
