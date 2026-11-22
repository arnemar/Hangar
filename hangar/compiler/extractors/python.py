"""Extract Python symbols and imports using the standard library."""

from __future__ import annotations
import ast as _ast
from hangar.compiler.ir import IRNode, IREdge, node_id, file_id
from hangar.compiler.extractors.common import ASTResult


class _PythonExtractor:
    """Extracts functions, async functions, classes, and methods from Python source."""

    def extract(self, project_id: str, path: str, source: str, language: str) -> ASTResult:
        try:
            tree = _ast.parse(source, filename=path)
        except SyntaxError:
            return ASTResult()

        result = ASTResult()
        file_node_id = file_id(project_id, path)
        lines = source.splitlines()

        for stmt in _ast.iter_child_nodes(tree):
            if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                ir = self._make_func(stmt, project_id, path, source, lines,
                                     file_node_id, "", "function")
                if ir:
                    result.nodes.append(ir)
                    result.local_symbols[ir.name] = ir
                    result.edges.append(IREdge(
                        from_id=file_node_id, to_id=ir.id,
                        raw_ref=None, kind="contains", resolved=True,
                    ))

            elif isinstance(stmt, _ast.ClassDef):
                ir = self._make_class(stmt, project_id, path, source, lines,
                                      file_node_id, "")
                if ir:
                    result.nodes.append(ir)
                    result.local_symbols[ir.name] = ir
                    result.edges.append(IREdge(
                        from_id=file_node_id, to_id=ir.id,
                        raw_ref=None, kind="contains", resolved=True,
                    ))
                    for child in _ast.iter_child_nodes(stmt):
                        if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                            method = self._make_func(child, project_id, path, source, lines,
                                                     ir.id, ir.name, "method")
                            if method:
                                result.nodes.append(method)
                                result.edges.append(IREdge(
                                    from_id=ir.id, to_id=method.id,
                                    raw_ref=None, kind="contains", resolved=True,
                                ))

            elif isinstance(stmt, (_ast.Import, _ast.ImportFrom)):
                self._extract_import(stmt, file_node_id, result)

        return result

    def _vis(self, name: str) -> str:
        if name.startswith("__") and name.endswith("__"):
            return "public"   # dunder is part of the public protocol
        if name.startswith("__"):
            return "private"
        if name.startswith("_"):
            return "internal"
        return "public"

    def _sig(self, stmt: "_ast.stmt", lines: list[str]) -> str:
        """Full function/class header (name + all params), up to the body colon."""
        start = stmt.lineno - 1
        if hasattr(stmt, "body") and stmt.body:
            end = stmt.body[0].lineno - 1
        else:
            end = start + 1
        end = min(end, start + 30, len(lines))
        seg = " ".join(l.strip() for l in lines[start:end])
        return seg[:300]

    def _src(self, stmt: "_ast.stmt", source: str) -> str:
        seg = _ast.get_source_segment(source, stmt)
        return seg or ""

    def _make_func(self, stmt, project_id, path, source, lines,
                   parent_id, parent_name, kind):
        name = stmt.name
        nid = node_id(project_id, path, kind, name, parent_name)
        return IRNode(
            id=nid, kind=kind, name=name, path=path,
            project_id=project_id, parent_id=parent_id,
            signature=self._sig(stmt, lines),
            source=self._src(stmt, source),
            start_line=stmt.lineno,
            end_line=getattr(stmt, "end_lineno", stmt.lineno),
            language="python",
            visibility=self._vis(name),
        )

    def _make_class(self, stmt, project_id, path, source, lines,
                    parent_id, parent_name):
        name = stmt.name
        nid = node_id(project_id, path, "class", name, parent_name)
        return IRNode(
            id=nid, kind="class", name=name, path=path,
            project_id=project_id, parent_id=parent_id,
            signature=self._sig(stmt, lines),
            source=self._src(stmt, source),
            start_line=stmt.lineno,
            end_line=getattr(stmt, "end_lineno", stmt.lineno),
            language="python",
            visibility=self._vis(name),
        )

    def _extract_import(self, stmt, file_node_id: str, result: ASTResult) -> None:
        if isinstance(stmt, _ast.Import):
            for alias in stmt.names:
                result.edges.append(IREdge(
                    from_id=file_node_id, to_id=None,
                    raw_ref=alias.name, kind="imports", resolved=False,
                ))
        elif isinstance(stmt, _ast.ImportFrom):
            module = stmt.module or ""
            prefix = "." * (stmt.level or 0) + module
            for alias in stmt.names:
                raw = f"{prefix}:{alias.name}" if prefix else alias.name
                result.edges.append(IREdge(
                    from_id=file_node_id, to_id=None,
                    raw_ref=raw, kind="imports", resolved=False,
                ))
