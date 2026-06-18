"""Tool registry and implementations.

Cross-platform notes
--------------------
- _run_cmd() uses SHELL_PREFIX (PowerShell on Windows, bash on Linux/macOS).
- Blocking tools (shell, git, ssh) are async and run inside a ThreadPoolExecutor
  so they never block the FastAPI event loop while Ollama is also streaming.
- SSH uses paramiko when installed; falls back to subprocess + sshpass on Linux/macOS.
  On Windows paramiko is required for password auth (sshpass is unavailable).
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from config import IS_WINDOWS, SHELL_PREFIX
from storage import get_remote, load_memory, load_settings, save_memory


def _expand_path(path: str) -> Path:
    """Resolve ~, %VAR%, and PowerShell $env:VAR in paths before handing to pathlib."""
    if IS_WINDOWS:
        path = re.sub(r'\$env:(\w+)', lambda m: os.environ.get(m.group(1), m.group(0)), path)
    return Path(os.path.expandvars(path)).expanduser().resolve()

# ── Thread pool for blocking I/O ──────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="tool-worker")

# ── Registry ──────────────────────────────────────────────────────────────────

TOOLS: dict = {}


from dataclasses import dataclass as _dc, field as _field
from typing import Literal as _Lit


@_dc
class ToolMetadata:
    """Declares the semantic effect of a tool on Plan state.

    Checked by Plan.advance() — no hard-coded inference in plan.py.
    """
    affects_working_set: bool = True
    working_set_reason: _Lit["retrieved", "edited", "dependency", "hypothesis"] = "retrieved"
    contributes_evidence: bool = False   # tool output can serve as investigation evidence
    contributes_hypothesis: bool = False # tool output can generate hypotheses


# Shorthand presets used in @tool() calls below
_WS_EVIDENCE = ToolMetadata(affects_working_set=False, contributes_evidence=True)
_WS_NONE     = ToolMetadata(affects_working_set=False)
_WS_EDITED   = ToolMetadata(working_set_reason="edited")
_WS_DEP      = ToolMetadata(working_set_reason="dependency")
_WS_DEFAULT  = ToolMetadata()  # retrieved, affects_working_set=True


def tool(name: str, description: str, params: dict, *, meta: ToolMetadata | None = None):
    def decorator(fn):
        TOOLS[name] = {
            "function": fn,
            "meta": meta or _WS_DEFAULT,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": list(params.keys()),
                    },
                },
            },
        }
        return fn
    return decorator


READ_ONLY_TOOLS = {"web_search", "browse_page", "read_file", "list_dir", "http_request", "git", "ssh"}


def get_tool_schemas(mode: str = "agent") -> list:
    if mode == "research":
        return [t["schema"] for name, t in TOOLS.items() if name in READ_ONLY_TOOLS]
    return [t["schema"] for t in TOOLS.values()]


async def execute_tool(name: str, args: dict) -> str:
    if name not in TOOLS:
        return f"[error]: Unknown tool: {name}"
    fn = TOOLS[name]["function"]
    try:
        if asyncio.iscoroutinefunction(fn):
            return await fn(**args)
        return fn(**args)
    except Exception as e:
        return f"[error]: Tool execution failed: {e}"


# ── Shell helper ──────────────────────────────────────────────────────────────

def _run_cmd(
    command: str,
    cwd: Optional[str] = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    """Run *command* in the platform shell (PowerShell / bash / sh)."""
    if SHELL_PREFIX:
        return subprocess.run(
            SHELL_PREFIX + [command],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=cwd,
        )
    return subprocess.run(
        command, shell=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, cwd=cwd,
    )


# ── Sync worker functions (called inside the thread pool) ─────────────────────

def _shell_worker(command: str) -> str:
    try:
        result = _run_cmd(command, cwd=str(Path.home()))
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error]: Command timed out after 30 seconds"
    except Exception as e:
        return f"[error]: {e}"


def _git_worker(action: str, path: str, args: str) -> str:
    try:
        p = _expand_path(path)
        if not p.exists():
            return f"[error]: Path not found: {path}"
        cmd_map = {
            "status": "git status --short",
            "log": f"git log --oneline -20 {args}".strip(),
            "diff": f"git diff {args}".strip(),
            "branch": "git branch -a",
            "show": f"git show --stat {args}".strip(),
        }
        if action not in cmd_map:
            return "[error]: Unknown action. Use: status, log, diff, branch, show"
        result = _run_cmd(cmd_map[action], cwd=str(p), timeout=15)
        output = result.stdout or result.stderr or ""
        return output.strip() or f"(no output for git {action})"
    except Exception as e:
        return f"[error]: {e}"


def _ssh_worker(remote: str, command: str, path: str) -> str:
    """Full SSH resolution + execution — runs in the thread pool."""
    try:
        host = remote
        user = ""
        key_path = ""
        password = ""
        port = 22

        if "@" not in remote:
            r = get_remote(remote)
            if not r:
                names = [x.get("name") for x in load_settings().get("remotes", [])]
                return f"[error]: Remote '{remote}' not found. Available: {names or ['none configured']}"
            host = r.get("host", "")
            user = r.get("user", "")
            key_path = r.get("key_path", "")
            password = r.get("password", "")
            port = r.get("port", 22)
        else:
            user, host = remote.split("@", 1)

        if path:
            command = f"cd {path} && {command}"

        if _HAS_PARAMIKO:
            return _ssh_paramiko(host, user, command, port=port, key_path=key_path, password=password)
        return _ssh_subprocess(host, user, command, port=port, key_path=key_path, password=password)

    except Exception as e:
        return f"[error]: {e}"


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool("shell", "Execute a shell command and return the output", {
    "command": {"type": "string", "description": "The shell command to execute"},
}, meta=_WS_EVIDENCE)
async def tool_shell(command: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _shell_worker, command)


@tool("read_file", "Read the contents of a file", {
    "path": {"type": "string", "description": "Absolute or home-relative path to the file"},
})
def tool_read_file(path: str) -> str:
    try:
        p = _expand_path(path)
        if not p.exists():
            return f"[error]: File not found: {path}"
        if p.stat().st_size > 500_000:
            return "[error]: File too large (>500 KB). Use shell with head/tail instead."
        return p.read_text(errors="replace")
    except Exception as e:
        return f"[error]: {e}"


@tool("write_file", "Write content to a file, creating it if it doesn't exist", {
    "path": {"type": "string", "description": "Path to write to"},
    "content": {"type": "string", "description": "Content to write"},
    "mode": {"type": "string", "description": "Write mode: 'overwrite' or 'append'"},
}, meta=_WS_EDITED)
def tool_write_file(path: str, content: str, mode: str = "overwrite", project_id: str = "") -> str:
    import json as _json
    try:
        p = _expand_path(path)
        original = p.read_text(encoding="utf-8", errors="replace") if p.exists() else None
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")

        # Syntax check for Python files
        if p.suffix == ".py":
            import py_compile, tempfile
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
                    tmp.write(p.read_text(encoding="utf-8"))
                    tmp_path = tmp.name
                py_compile.compile(tmp_path, doraise=True)
                Path(tmp_path).unlink(missing_ok=True)
            except py_compile.PyCompileError as e:
                # Rollback
                if original is not None:
                    p.write_text(original, encoding="utf-8")
                else:
                    p.unlink(missing_ok=True)
                return _json.dumps({"success": False, "syntax_ok": False,
                                    "rollback_performed": True, "diagnostics": [f"SyntaxError: {e}", "File restored."]})

        return _json.dumps({"success": True, "syntax_ok": True, "diagnostics": [f"Written to {p}"]})
    except Exception as e:
        return _json.dumps({"success": False, "diagnostics": [str(e)]})


# ── Batch transaction state ───────────────────────────────────────────────────
# Maps tx_id → {abs_path_str: original_content}.  Lives for the process lifetime.
_transactions: dict[str, dict[str, str]] = {}


@tool(
    "begin_transaction",
    "Start a batch edit transaction. Returns a tx_id. Pass it to edit_file calls to "
    "stage edits without per-edit verification. Call commit_transaction when done to "
    "verify all staged edits atomically and recompile, or rollback_transaction to undo.",
    {},
    meta=_WS_NONE,
)
def tool_begin_transaction() -> str:
    import uuid
    tx_id = str(uuid.uuid4())[:8]
    _transactions[tx_id] = {}
    return json.dumps({"tx_id": tx_id, "diagnostics": ["Transaction started. Pass tx_id to edit_file calls."]})


@tool(
    "commit_transaction",
    "Verify and commit all staged edits in a transaction. Runs syntax checks and "
    "incremental graph recompile for every changed file. Rolls back ALL files on "
    "any failure so the codebase stays consistent.",
    {
        "tx_id": {"type": "string", "description": "Transaction ID from begin_transaction"},
        "project_id": {"type": "string", "description": "Project name for graph recompilation (optional)"},
    },
    meta=_WS_NONE,
)
def tool_commit_transaction(tx_id: str, project_id: str = "") -> str:
    if tx_id not in _transactions:
        return json.dumps({"success": False, "diagnostics": [f"Transaction '{tx_id}' not found or already closed"]})

    tx = _transactions[tx_id]
    if not tx:
        del _transactions[tx_id]
        return json.dumps({"success": True, "files_committed": 0, "diagnostics": ["Empty transaction — nothing to commit"]})

    diagnostics: list[str] = []
    syntax_ok = True

    # ── Phase 1: syntax check all changed .py files ───────────────────────────
    import py_compile, tempfile
    for abs_path in tx:
        p = Path(abs_path)
        if p.suffix.lower() == ".py":
            try:
                current = p.read_text(encoding="utf-8", errors="replace")
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
                    tmp.write(current)
                    tmp_path = tmp.name
                py_compile.compile(tmp_path, doraise=True)
                Path(tmp_path).unlink(missing_ok=True)
            except py_compile.PyCompileError as e:
                syntax_ok = False
                diagnostics.append(f"{p.name}: SyntaxError: {e}")

    # ── Rollback all on syntax failure ────────────────────────────────────────
    if not syntax_ok:
        for abs_path, original in tx.items():
            Path(abs_path).write_text(original, encoding="utf-8")
        del _transactions[tx_id]
        diagnostics.append(f"All {len(tx)} file(s) restored to original.")
        return json.dumps({"success": False, "syntax_ok": False, "rollback_performed": True,
                           "files_committed": 0, "diagnostics": diagnostics})

    # ── Phase 2: graph recompile each changed file ────────────────────────────
    compile_ok = True
    total_nodes = 0
    total_edges = 0

    if project_id:
        from storage import get_project
        from compiler.pipeline import compile_file
        proj = get_project(project_id)
        if proj:
            root = proj.get("root", "")
            for abs_path in tx:
                p = Path(abs_path)
                try:
                    rel = str(p.relative_to(Path(root).resolve()))
                    cr = compile_file(project_id, root, rel, check_health=False)
                    total_nodes += cr.nodes_created + cr.nodes_updated
                    total_edges += cr.edges_created
                    if cr.errors:
                        compile_ok = False
                        diagnostics.extend([f"{p.name}: {e}" for e in cr.errors[:2]])
                except Exception as ce:
                    diagnostics.append(f"{p.name}: compile skipped: {ce}")

    if not compile_ok:
        for abs_path, original in tx.items():
            Path(abs_path).write_text(original, encoding="utf-8")
        del _transactions[tx_id]
        diagnostics.append(f"All {len(tx)} file(s) restored to original.")
        return json.dumps({"success": False, "compile_ok": False, "rollback_performed": True,
                           "files_committed": 0, "diagnostics": diagnostics})

    files_committed = len(tx)
    del _transactions[tx_id]
    return json.dumps({
        "success": True,
        "files_committed": files_committed,
        "changed_nodes": total_nodes,
        "changed_edges": total_edges,
        "diagnostics": diagnostics or [f"{files_committed} file(s) committed and verified."],
    })


@tool(
    "rollback_transaction",
    "Restore all files staged in a transaction to their original content and discard the transaction.",
    {
        "tx_id": {"type": "string", "description": "Transaction ID from begin_transaction"},
    },
    meta=_WS_NONE,
)
def tool_rollback_transaction(tx_id: str) -> str:
    if tx_id not in _transactions:
        return json.dumps({"success": False, "diagnostics": [f"Transaction '{tx_id}' not found or already closed"]})
    tx = _transactions.pop(tx_id)
    for abs_path, original in tx.items():
        Path(abs_path).write_text(original, encoding="utf-8")
    return json.dumps({
        "success": True,
        "files_restored": len(tx),
        "diagnostics": [f"{len(tx)} file(s) restored."],
    })


@tool(
    "edit_file",
    (
        "Find and replace text in a file. Verifies syntax, recompiles the graph, and "
        "rolls back on failure. Pass tx_id from begin_transaction to stage edits for "
        "atomic multi-file commits. Use verification='full' to also run graph consistency checks."
    ),
    {
        "path": {"type": "string", "description": "Path to the file to edit"},
        "old_text": {"type": "string", "description": "Exact text to find (whitespace must match exactly)"},
        "new_text": {"type": "string", "description": "Replacement text"},
        "project_id": {"type": "string", "description": "Project name for graph recompilation (optional)"},
        "verification": {
            "type": "string",
            "description": (
                "Verification level: "
                "'syntax' (default) — syntax check only; "
                "'graph' — syntax + incremental graph recompile; "
                "'full' — graph + consistency checks (dangling edges, FTS parity)"
            ),
        },
        "transaction_id": {
            "type": "string",
            "description": (
                "Optional tx_id from begin_transaction. When set, the edit is staged "
                "in the transaction without per-edit verification — call commit_transaction "
                "to verify and finalize all staged edits atomically."
            ),
        },
    },
    meta=_WS_EDITED,
)
def tool_edit_file(
    path: str,
    old_text: str,
    new_text: str,
    project_id: str = "",
    verification: str = "syntax",
    transaction_id: str = "",
) -> str:
    import json as _json
    try:
        p = _expand_path(path)
        if not p.exists():
            return _json.dumps({"success": False, "diagnostics": [f"File not found: {path}"]})
        original = p.read_text(encoding="utf-8", errors="replace")
        if old_text not in original:
            return _json.dumps({"success": False, "diagnostics": ["Text not found in file — check exact whitespace and indentation"]})

        count = original.count(old_text)
        updated = original.replace(old_text, new_text)

        # ── Transaction staging (no immediate verification) ────────────────────
        if transaction_id:
            if transaction_id not in _transactions:
                return _json.dumps({"success": False, "diagnostics": [f"Transaction '{transaction_id}' not found"]})
            abs_str = str(p)
            # Store the ORIGINAL content (first write wins — preserves pre-tx state)
            if abs_str not in _transactions[transaction_id]:
                _transactions[transaction_id][abs_str] = original
            p.write_text(updated, encoding="utf-8")
            return _json.dumps({
                "success": True,
                "staged": True,
                "transaction_id": transaction_id,
                "occurrences_replaced": count,
                "diagnostics": [f"Staged in transaction {transaction_id}. Call commit_transaction to finalize."],
            })

        # ── Atomic backup (written before any mutation) ───────────────────────
        tmp_path = p.with_name(p.name + ".hangar_tmp")
        tmp_path.write_text(original, encoding="utf-8")

        try:
            p.write_text(updated, encoding="utf-8")

            diagnostics: list[str] = []
            syntax_ok = True
            compile_ok = True
            changed_nodes = 0
            changed_edges = 0
            added_names: list[str] = []
            removed_names: list[str] = []
            graph_health: dict = {}

            # ── Syntax check ─────────────────────────────────────────────────
            if p.suffix.lower() == ".py":
                import py_compile, tempfile
                try:
                    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as tmp:
                        tmp.write(updated)
                        check_tmp = tmp.name
                    py_compile.compile(check_tmp, doraise=True)
                    Path(check_tmp).unlink(missing_ok=True)
                except py_compile.PyCompileError as e:
                    syntax_ok = False
                    diagnostics.append(f"SyntaxError: {e}")

            # ── Graph recompilation (graph / full) ────────────────────────────
            if syntax_ok and project_id and verification in ("graph", "full"):
                try:
                    from storage import get_project
                    proj = get_project(project_id)
                    if proj:
                        root = proj.get("root", "")
                        rel = str(p.resolve().relative_to(Path(root).resolve()))
                        from compiler.pipeline import compile_file
                        cr = compile_file(
                            project_id, root, rel,
                            check_health=(verification == "full"),
                        )
                        changed_nodes = cr.nodes_created + cr.nodes_updated
                        changed_edges = cr.edges_created
                        added_names = cr.added_names
                        removed_names = cr.removed_names
                        graph_health = cr.graph_health
                        if cr.errors:
                            compile_ok = False
                            diagnostics.extend(cr.errors[:3])
                except Exception as ce:
                    diagnostics.append(f"Graph recompile skipped: {ce}")
            elif syntax_ok and project_id and verification == "syntax":
                # Still do graph recompile to keep graph current, but don't gate on it
                try:
                    from storage import get_project
                    proj = get_project(project_id)
                    if proj:
                        root = proj.get("root", "")
                        rel = str(p.resolve().relative_to(Path(root).resolve()))
                        from compiler.pipeline import compile_file
                        cr = compile_file(project_id, root, rel)
                        changed_nodes = cr.nodes_created + cr.nodes_updated
                        changed_edges = cr.edges_created
                        added_names = cr.added_names
                        removed_names = cr.removed_names
                except Exception:
                    pass  # best-effort for syntax-level verification

            # ── Rollback if verification failed ───────────────────────────────
            if not syntax_ok or not compile_ok:
                shutil.copy2(str(tmp_path), str(p))
                tmp_path.unlink(missing_ok=True)
                diagnostics.append("File restored from backup.")
                return _json.dumps({
                    "success": False,
                    "syntax_ok": syntax_ok,
                    "compile_ok": compile_ok,
                    "rollback_performed": True,
                    "changed_nodes": 0,
                    "changed_edges": 0,
                    "diagnostics": diagnostics,
                })

            tmp_path.unlink(missing_ok=True)
            result = {
                "success": True,
                "syntax_ok": True,
                "compile_ok": True,
                "rollback_performed": False,
                "occurrences_replaced": count,
                "changed_nodes": changed_nodes,
                "changed_edges": changed_edges,
                "diagnostics": diagnostics or ["Edit applied and verified."],
            }
            if added_names:
                result["added_symbols"] = added_names
            if removed_names:
                result["removed_symbols"] = removed_names
            if graph_health:
                result["graph_health"] = graph_health
            return _json.dumps(result)

        except Exception:
            # Ensure backup is always restored on unexpected error
            if tmp_path.exists():
                shutil.copy2(str(tmp_path), str(p))
                tmp_path.unlink(missing_ok=True)
            raise

    except Exception as e:
        return _json.dumps({"success": False, "diagnostics": [str(e)]})


@tool("list_dir", "List files and directories at a path", {
    "path": {"type": "string", "description": "Directory path to list, defaults to home directory"},
}, meta=_WS_NONE)
def tool_list_dir(path: str = "~") -> str:
    try:
        p = _expand_path(path)
        if not p.exists():
            return f"[error]: Path not found: {path}"
        if not p.is_dir():
            return f"[error]: Not a directory: {path}"
        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        lines = []
        for item in items:
            prefix = "📁" if item.is_dir() else "📄"
            size = ""
            if item.is_file():
                s = item.stat().st_size
                size = f" ({s:,} B)" if s < 1_000_000 else f" ({s / 1_000_000:.1f} MB)"
            lines.append(f"{prefix} {item.name}{size}")
        return f"{p}\n" + "\n".join(lines) if lines else f"{p}\n(empty)"
    except Exception as e:
        return f"[error]: {e}"


@tool("web_search", "Search the web using DuckDuckGo. Returns real search results with titles, URLs, and snippets.", {
    "query": {"type": "string", "description": "Search query"},
    "max_results": {"type": "integer", "description": "Number of results to return (default 6, max 15)"},
}, meta=_WS_EVIDENCE)
async def tool_web_search(query: str, max_results: int = 6) -> str:
    try:
        from duckduckgo_search import DDGS
        limit = max(1, min(15, max_results))
        loop = asyncio.get_running_loop()

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=limit))

        results = await loop.run_in_executor(_executor, _search)
        if not results:
            return f"No results found for: {query}"
        lines = [f"Search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '(no title)')}")
            lines.append(f"   URL: {r.get('href', '')}")
            lines.append(f"   {r.get('body', '')[:200]}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error]: {e}"


@tool(
    "browse_page",
    "Fetch a web page and return its main readable content, stripping ads, navigation, and boilerplate. "
    "Use after web_search to read the full content of a result.",
    {
        "url": {"type": "string", "description": "Full URL to fetch"},
        "max_chars": {"type": "integer", "description": "Max characters to return (default 4000)"},
    },
    meta=_WS_EVIDENCE,
)
async def tool_browse_page(url: str, max_chars: int = 4000) -> str:
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "Mozilla/5.0 (compatible; HangarBot/1.0)"}) as client:
            r = await client.get(url)
            r.raise_for_status()
            html = r.text

        loop = asyncio.get_running_loop()

        def _extract():
            try:
                import trafilatura
                text = trafilatura.extract(html, include_links=False, include_images=False,
                                           no_fallback=False, favor_recall=True)
                return text or ""
            except Exception:
                # Fallback: strip tags manually
                import re
                text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
                text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text)
                return text.strip()

        content = await loop.run_in_executor(_executor, _extract)
        limit = max(500, min(8000, max_chars))
        if not content:
            return f"[error]: Could not extract content from {url}"
        return f"Content from {url}:\n\n{content[:limit]}" + (" [truncated]" if len(content) > limit else "")
    except Exception as e:
        return f"[error]: {e}"


@tool("manage_memory", "Add, list, or delete memory entries", {
    "action": {"type": "string", "description": "Action: 'add', 'list', or 'delete'"},
    "content": {"type": "string", "description": "Content to add (for 'add' action)"},
    "index": {"type": "integer", "description": "Index to delete (for 'delete' action)"},
}, meta=_WS_NONE)
def tool_manage_memory(action: str, content: str = "", index: int = -1) -> str:
    memory = load_memory()
    if action == "add":
        if not content:
            return "[error]: Content required for add action"
        memory["facts"].append({"content": content, "created_at": datetime.now().isoformat()})
        save_memory(memory)
        return f"[ok]: Memory saved: {content}"
    elif action == "list":
        if not memory["facts"]:
            return "No memories stored yet."
        return "\n".join(f"{i}: {f['content']}" for i, f in enumerate(memory["facts"]))
    elif action == "delete":
        if index < 0 or index >= len(memory["facts"]):
            return f"[error]: Invalid index {index}"
        removed = memory["facts"].pop(index)
        save_memory(memory)
        return f"[ok]: Deleted memory: {removed['content']}"
    return f"[error]: Unknown action: {action}"


@tool("git", "Run git commands on a repository", {
    "action": {"type": "string", "description": "Git action: status, log, diff, branch, show"},
    "path": {"type": "string", "description": "Path to the git repository"},
    "args": {"type": "string", "description": "Extra args e.g. commit hash"},
}, meta=_WS_EVIDENCE)
async def tool_git(action: str, path: str = "~", args: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _git_worker, action, path, args)


@tool("http_request", "Make an HTTP request", {
    "method": {"type": "string", "description": "HTTP method: GET, POST, PUT, DELETE"},
    "url": {"type": "string", "description": "URL to request"},
    "body": {"type": "string", "description": "JSON body for POST/PUT (optional)"},
    "headers": {"type": "string", "description": "JSON headers dict (optional)"},
}, meta=_WS_EVIDENCE)
async def tool_http_request(method: str, url: str, body: str = "", headers: str = "") -> str:
    try:
        h = json.loads(headers) if headers else {}
        b = None
        if body:
            try:
                b = json.loads(body)
                h.setdefault("Content-Type", "application/json")
            except Exception:
                b = body
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.request(
                method.upper(), url,
                json=b if isinstance(b, dict) else None,
                content=b.encode() if isinstance(b, str) else None,
                headers=h,
            )
            ct = r.headers.get("content-type", "")
            text = json.dumps(r.json(), indent=2)[:2000] if "json" in ct else r.text[:2000]
            return f"Status: {r.status_code}\n{text}"
    except Exception as e:
        return f"[error]: {e}"


# ── SSH ───────────────────────────────────────────────────────────────────────

try:
    import paramiko as _paramiko
    _HAS_PARAMIKO = True
except ImportError:
    _paramiko = None
    _HAS_PARAMIKO = False


def _ssh_paramiko(
    host: str, user: str, command: str,
    port: int = 22, key_path: str = "", password: str = "", timeout: int = 30,
) -> str:
    client = _paramiko.SSHClient()
    client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
    try:
        kw: dict = dict(hostname=host, username=user or None, port=port, timeout=10)
        if key_path:
            kw["key_filename"] = str(Path(key_path).expanduser())
        elif password:
            kw["password"] = password
        client.connect(**kw)
        _, stdout, stderr = client.exec_command(command, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        if err:
            clean = "\n".join(l for l in err.strip().splitlines() if "known hosts" not in l.lower())
            if clean:
                out += f"\n[stderr]: {clean}"
        return out.strip() or "(no output)"
    finally:
        client.close()


def _ssh_subprocess(
    host: str, user: str, command: str,
    port: int = 22, key_path: str = "", password: str = "", timeout: int = 30,
) -> str:
    if password and IS_WINDOWS:
        return "[error]: Password-based SSH on Windows requires paramiko (pip install paramiko)"
    if password and not shutil.which("sshpass"):
        return "[error]: sshpass not installed. Run: sudo apt install sshpass"

    args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=5",
        "-o", "ServerAliveCountMax=2",
        "-p", str(port),
    ]
    if key_path:
        args += ["-i", str(Path(key_path).expanduser())]
    args += [f"{user}@{host}" if user else host, command]
    if password:
        args = ["sshpass", "-p", password] + args

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return "[error]: SSH command timed out"
    finally:
        if proc.poll() is None:
            proc.kill()

    output = stdout or ""
    if stderr:
        clean = "\n".join(l for l in stderr.strip().splitlines() if "known hosts" not in l.lower())
        if clean:
            output += f"\n[stderr]: {clean}"
    return output.strip() or "(no output)"


# ── Knowledge graph ───────────────────────────────────────────────────────────

@tool(
    "search_symbols",
    (
        "Search the compiled knowledge graph for functions, classes, types, or any symbol "
        "by name or description. Returns ranked results with file locations, signatures, "
        "and relevance scores. Use this instead of read_file/list_dir when looking for "
        "a specific symbol or understanding what a codebase exports."
    ),
    {
        "project": {
            "type": "string",
            "description": "Project name (must be registered in settings)",
        },
        "query": {
            "type": "string",
            "description": "Symbol name, partial name, or short description",
        },
        "intent": {
            "type": "string",
            "description": (
                "Search intent — controls ranking weights. "
                "find (default): locate a symbol; "
                "edit: find high-impact symbols safe to change; "
                "debug: surface recently active code; "
                "explain: prioritise docs/signatures"
            ),
        },
        "kinds": {
            "type": "string",
            "description": (
                "Comma-separated node kinds to restrict results. "
                "Valid: function, method, class, interface, type_alias, enum, variable, constant. "
                "Leave empty for all kinds."
            ),
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return (default 15, max 50)",
        },
    },
)
def tool_search_symbols(
    project: str,
    query: str,
    intent: str = "find",
    kinds: str = "",
    max_results: int = 15,
) -> str:
    try:
        from compiler.search import find_symbols
        from storage import get_project

        p = get_project(project)
        if not p:
            from storage import load_projects
            names = [x.get("name") for x in load_projects()]
            return f"[error]: Project '{project}' not found. Available: {names or ['none']}"

        kind_list = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
        limit = max(1, min(50, max_results))

        results = find_symbols(
            project_id=project,
            query_text=query,
            intent=intent,
            kinds=kind_list,
            max_results=limit,
        )

        if not results:
            return f"No symbols found for '{query}' in project '{project}'."

        lines = [f"Found {len(results)} result(s) for '{query}':"]
        for r in results:
            loc = f"{r.path}:{r.start_line}"
            sig = f" — {r.signature}" if r.signature else ""
            src_tags = "+".join(sorted(r.sources))
            lines.append(
                f"  [{r.kind}] {r.name}{sig}\n"
                f"    {loc} (score={r.score:.3f}, via={src_tags})\n"
                f"    node_id: {r.node_id}"
            )
            if r.summary:
                lines.append(f"    summary: {r.summary}")

        lines.append(
            "\nTo fetch full source: get_symbol_source(project, node_id)"
            "\nTo trace dependencies: graph_expand(project, node_ids)"
        )
        return "\n".join(lines)

    except Exception as e:
        return f"[error]: search_symbols failed: {e}"


@tool(
    "graph_expand",
    (
        "Expand the knowledge graph from one or more known node IDs to discover "
        "connected symbols. Use 'forward' to find what a symbol depends on (calls, imports), "
        "or 'backward' (impact) to find everything that depends on it. "
        "Combine with search_symbols: first find a node_id, then expand from it."
    ),
    {
        "project": {
            "type": "string",
            "description": "Project name",
        },
        "node_ids": {
            "type": "string",
            "description": "Comma-separated node IDs to start from (from search_symbols results)",
        },
        "intent": {
            "type": "string",
            "description": (
                "Traversal intent. "
                "find: expand forward (what does this use?); "
                "impact: expand backward (what uses this?)"
            ),
        },
        "max_depth": {
            "type": "integer",
            "description": "Traversal depth (1 = direct neighbors, 3 = default, max 5)",
        },
    },
    meta=_WS_DEP,
)
def tool_graph_expand(
    project: str,
    node_ids: str,
    intent: str = "find",
    max_depth: int = 3,
) -> str:
    try:
        from compiler.search import expand_from
        from storage import get_project

        p = get_project(project)
        if not p:
            return f"[error]: Project '{project}' not found."

        ids = [x.strip() for x in node_ids.split(",") if x.strip()]
        if not ids:
            return "[error]: node_ids must not be empty."

        depth = max(1, min(5, max_depth))
        results = expand_from(project, ids, intent=intent, max_depth=depth)

        if not results:
            return "No connected nodes found."

        lines = [f"Graph expansion from {len(ids)} seed(s), depth={depth}:"]
        for r in results:
            loc = f"{r.path}:{r.start_line}"
            sig = f" — {r.signature}" if r.signature else ""
            lines.append(
                f"  [{r.kind}] {r.name}{sig}\n"
                f"    {loc} (score={r.score:.3f})"
            )

        return "\n".join(lines)

    except Exception as e:
        return f"[error]: graph_expand failed: {e}"


@tool(
    "get_symbol_source",
    (
        "Fetch the full source code and metadata for a specific symbol by node_id. "
        "Use this after search_symbols gives you a node_id — it returns the complete "
        "implementation without needing to know the file path or line numbers."
    ),
    {
        "project": {
            "type": "string",
            "description": "Project name",
        },
        "node_id": {
            "type": "string",
            "description": "Node ID from search_symbols or graph_expand results",
        },
    },
)
def tool_get_symbol_source(project: str, node_id: str) -> str:
    try:
        from compiler.search import symbol_context
        from storage import get_project

        p = get_project(project)
        if not p:
            return f"[error]: Project '{project}' not found."

        ctx = symbol_context(project, node_id)
        if ctx is None:
            return f"[error]: Node '{node_id}' not found in project '{project}'."

        lines = [
            f"[{ctx['kind']}] {ctx['name']}",
            f"File: {ctx['path']}:{ctx['start_line']}-{ctx['end_line']}",
        ]
        if ctx.get("signature"):
            lines.append(f"Signature: {ctx['signature']}")
        if ctx.get("summary"):
            lines.append(f"Summary: {ctx['summary']}")
        if ctx.get("fan_in") or ctx.get("fan_out"):
            lines.append(f"Fan-in: {ctx.get('fan_in', 0)}  Fan-out: {ctx.get('fan_out', 0)}")
        if ctx.get("source"):
            lang = ctx.get("language", "")
            lines.append(f"```{lang}")
            lines.append(ctx["source"].strip())
            lines.append("```")
        else:
            lines.append("(source not available — recompile to extract source)")

        return "\n".join(lines)

    except Exception as e:
        return f"[error]: get_symbol_source failed: {e}"


@tool("ssh", "Run a command on a remote machine via SSH", {
    "remote": {"type": "string", "description": "Remote name from settings (e.g. mac-mini) or user@host"},
    "command": {"type": "string", "description": "Shell command to run on the remote machine"},
    "path": {"type": "string", "description": "Working directory on the remote machine (optional)"},
}, meta=_WS_EVIDENCE)
async def tool_ssh(remote: str, command: str, path: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _ssh_worker, remote, command, path)
