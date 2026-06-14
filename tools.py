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
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from config import IS_WINDOWS, SHELL_PREFIX
from storage import get_remote, load_memory, load_settings, save_memory

# ── Thread pool for blocking I/O ──────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="tool-worker")

# ── Registry ──────────────────────────────────────────────────────────────────

TOOLS: dict = {}


def tool(name: str, description: str, params: dict):
    def decorator(fn):
        TOOLS[name] = {
            "function": fn,
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


READ_ONLY_TOOLS = {"web_search", "read_file", "list_dir", "http_request", "git", "ssh"}


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
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
    return subprocess.run(
        command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd,
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
        p = Path(path).expanduser().resolve()
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
})
async def tool_shell(command: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _shell_worker, command)


@tool("read_file", "Read the contents of a file", {
    "path": {"type": "string", "description": "Absolute or home-relative path to the file"},
})
def tool_read_file(path: str) -> str:
    try:
        p = Path(path).expanduser().resolve()
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
})
def tool_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(p, "a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"[ok]: Written to {p}"
    except Exception as e:
        return f"[error]: {e}"


@tool("edit_file", "Find and replace text in a file", {
    "path": {"type": "string", "description": "Path to the file to edit"},
    "old_text": {"type": "string", "description": "Exact text to find"},
    "new_text": {"type": "string", "description": "Replacement text"},
})
def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"[error]: File not found: {path}"
        content = p.read_text(errors="replace")
        if old_text not in content:
            return "[error]: Text not found in file"
        count = content.count(old_text)
        p.write_text(content.replace(old_text, new_text), encoding="utf-8")
        return f"[ok]: Replaced {count} occurrence(s) in {p}"
    except Exception as e:
        return f"[error]: {e}"


@tool("list_dir", "List files and directories at a path", {
    "path": {"type": "string", "description": "Directory path to list, defaults to home directory"},
})
def tool_list_dir(path: str = "~") -> str:
    try:
        p = Path(path).expanduser().resolve()
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


@tool("web_search", "Search the web using DuckDuckGo", {
    "query": {"type": "string", "description": "Search query"},
})
async def tool_web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText']}")
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")
            return "\n".join(results) if results else f"No results found for: {query}"
    except Exception as e:
        return f"[error]: {e}"


@tool("manage_memory", "Add, list, or delete memory entries", {
    "action": {"type": "string", "description": "Action: 'add', 'list', or 'delete'"},
    "content": {"type": "string", "description": "Content to add (for 'add' action)"},
    "index": {"type": "integer", "description": "Index to delete (for 'delete' action)"},
})
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
})
async def tool_git(action: str, path: str = "~", args: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _git_worker, action, path, args)


@tool("http_request", "Make an HTTP request", {
    "method": {"type": "string", "description": "HTTP method: GET, POST, PUT, DELETE"},
    "url": {"type": "string", "description": "URL to request"},
    "body": {"type": "string", "description": "JSON body for POST/PUT (optional)"},
    "headers": {"type": "string", "description": "JSON headers dict (optional)"},
})
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


@tool("ssh", "Run a command on a remote machine via SSH", {
    "remote": {"type": "string", "description": "Remote name from settings (e.g. mac-mini) or user@host"},
    "command": {"type": "string", "description": "Shell command to run on the remote machine"},
    "path": {"type": "string", "description": "Working directory on the remote machine (optional)"},
})
async def tool_ssh(remote: str, command: str, path: str = "") -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _ssh_worker, remote, command, path)
