"""Walk JavaScript and TypeScript declarations."""

from __future__ import annotations
from hangar.compiler.ir import IRNode, IREdge, node_id, file_id
from hangar.compiler.extractors.common import ASTResult, _now
from hangar.compiler.extractors.javascript_syntax import _get_parser, _text, _line, _child_text, _signature, _visibility
from hangar.compiler.extractors.javascript_references import _extract_imports, _extract_calls


try:
    from tree_sitter import Node
except ImportError:
    pass


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
        _extract_imports(root, project_id, path, source_bytes, file_node_id, result)

        return result

    # Walker

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
                        _extract_calls(body, source_bytes, ir_node.id, result)

    # Declaration handlers

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
