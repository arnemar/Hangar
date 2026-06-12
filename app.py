"""
AI Workspace - Lightweight local AI agent
FastAPI backend with streaming, agent tools, and persistent memory
"""

import asyncio
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "workspace.db"
MEMORY_PATH = DATA_DIR / "memory.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-coder-v2:16b")
MAX_TOOL_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                mode TEXT DEFAULT 'chat',
                model TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT,
                content TEXT,
                tool_calls TEXT,
                tool_results TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def load_memory() -> dict:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text())
        except Exception:
            pass
    return {"facts": [], "preferences": {}, "updated_at": None}

def save_memory(memory: dict):
    memory["updated_at"] = datetime.now().isoformat()
    MEMORY_PATH.write_text(json.dumps(memory, indent=2))

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = {}

def tool(name: str, description: str, params: dict):
    """Decorator to register a tool."""
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
                    }
                }
            }
        }
        return fn
    return decorator


@tool("shell", "Execute a shell command and return the output", {
    "command": {"type": "string", "description": "The shell command to execute"}
})
def tool_shell(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
            cwd=str(Path.home())
        )
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error]: Command timed out after 30 seconds"
    except Exception as e:
        return f"[error]: {e}"


@tool("read_file", "Read the contents of a file", {
    "path": {"type": "string", "description": "Absolute or home-relative path to the file"}
})
def tool_read_file(path: str) -> str:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"[error]: File not found: {path}"
        if p.stat().st_size > 500_000:
            return f"[error]: File too large (>{500_000} bytes). Use shell with head/tail instead."
        return p.read_text(errors="replace")
    except Exception as e:
        return f"[error]: {e}"


@tool("write_file", "Write content to a file, creating it if it doesn't exist", {
    "path": {"type": "string", "description": "Path to write to"},
    "content": {"type": "string", "description": "Content to write"},
    "mode": {"type": "string", "description": "Write mode: 'overwrite' or 'append'"}
})
def tool_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "append":
            with open(p, "a") as f:
                f.write(content)
        else:
            p.write_text(content)
        return f"[ok]: Written to {p}"
    except Exception as e:
        return f"[error]: {e}"


@tool("list_dir", "List files and directories at a path", {
    "path": {"type": "string", "description": "Directory path to list, defaults to home directory"}
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
                size = f" ({s:,} bytes)" if s < 1_000_000 else f" ({s/1_000_000:.1f} MB)"
            lines.append(f"{prefix} {item.name}{size}")
        return f"{p}\n" + "\n".join(lines) if lines else f"{p}\n(empty)"
    except Exception as e:
        return f"[error]: {e}"


@tool("web_search", "Search the web using DuckDuckGo", {
    "query": {"type": "string", "description": "Search query"}
})
async def tool_web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
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
    "index": {"type": "integer", "description": "Index to delete (for 'delete' action)"}
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
    "args": {"type": "string", "description": "Extra args e.g. commit hash"}
})
def tool_git(action: str, path: str = "~", args: str = "") -> str:
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
            return f"[error]: Unknown action. Use: status, log, diff, branch, show"
        result = subprocess.run(cmd_map[action], shell=True, capture_output=True, text=True, timeout=15, cwd=str(p))
        output = result.stdout or result.stderr or ""
        return output.strip() or f"(no output for git {action})"
    except Exception as e:
        return f"[error]: {e}"


@tool("edit_file", "Find and replace text in a file", {
    "path": {"type": "string", "description": "Path to the file to edit"},
    "old_text": {"type": "string", "description": "Exact text to find"},
    "new_text": {"type": "string", "description": "Replacement text"}
})
def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"[error]: File not found: {path}"
        content = p.read_text(errors="replace")
        if old_text not in content:
            return f"[error]: Text not found in file"
        count = content.count(old_text)
        p.write_text(content.replace(old_text, new_text))
        return f"[ok]: Replaced {count} occurrence(s) in {p}"
    except Exception as e:
        return f"[error]: {e}"


@tool("http_request", "Make an HTTP request", {
    "method": {"type": "string", "description": "HTTP method: GET, POST, PUT, DELETE"},
    "url": {"type": "string", "description": "URL to request"},
    "body": {"type": "string", "description": "JSON body for POST/PUT (optional)"},
    "headers": {"type": "string", "description": "JSON headers dict (optional)"}
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
            r = await client.request(method.upper(), url,
                json=b if isinstance(b, dict) else None,
                content=b.encode() if isinstance(b, str) else None, headers=h)
            ct = r.headers.get("content-type", "")
            text = json.dumps(r.json(), indent=2)[:2000] if "json" in ct else r.text[:2000]
            return f"Status: {r.status_code}\n{text}"
    except Exception as e:
        return f"[error]: {e}"


# ---------------------------------------------------------------------------
# Settings store
# ---------------------------------------------------------------------------

SETTINGS_PATH = DATA_DIR / "settings.json"

def load_settings() -> dict:
    defaults = {
        "ollama_url": "http://localhost:11434",
        "default_model": "deepseek-coder-v2:16b",
        "agent_max_iterations": 10,
        "agent_timeout": 120,
        "remotes": [],
        "user_name": "",
        "system_prompt_chat": "",
        "system_prompt_agent": "",
    }
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text())
            defaults.update(saved)
        except Exception:
            pass
    return defaults

def save_settings(s: dict):
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))

def get_remote(name: str) -> Optional[dict]:
    s = load_settings()
    for r in s.get("remotes", []):
        if r.get("name") == name:
            return r
    return None


# ---------------------------------------------------------------------------
# SSH tool
# ---------------------------------------------------------------------------

@tool("ssh", "Run a command on a remote machine via SSH", {
    "remote": {"type": "string", "description": "Remote name from settings (e.g. mac-mini) or user@host directly"},
    "command": {"type": "string", "description": "Shell command to run on the remote machine"},
    "path": {"type": "string", "description": "Working directory on the remote machine (optional)"}
})
def tool_ssh(remote: str, command: str, path: str = "") -> str:
    try:
        # Resolve remote from settings or use direct user@host
        host = remote
        user = None
        key_path = None
        password = None
        port = 22

        if "@" not in remote:
            r = get_remote(remote)
            if not r:
                remotes = load_settings().get("remotes", [])
                names = [x.get("name") for x in remotes]
                return f"[error]: Remote '{remote}' not found in settings. Available: {names or ['none configured']}"
            host = r.get("host", "")
            user = r.get("user", "")
            key_path = r.get("key_path", "")
            password = r.get("password", "")
            port = r.get("port", 22)
        else:
            user, host = remote.split("@", 1)

        if path:
            command = f"cd {path} && {command}"

        # Build SSH command
        ssh_base = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
            "-p", str(port),
        ]
        if key_path:
            ssh_base += ["-i", str(Path(key_path).expanduser())]
        if user:
            ssh_base += [f"{user}@{host}"]
        else:
            ssh_base += [host]
        ssh_base += [command]

        if password:
            # sshpass must come before the entire ssh command
            ssh_opts = ["sshpass", "-p", password] + ssh_base
        else:
            ssh_opts = ssh_base

        proc = subprocess.Popen(ssh_opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()  # Clean up zombie process
            return "[error]: SSH command timed out after 30s. Try a faster command (e.g. use ls instead of find, or add -maxdepth 2 to find)"
        finally:
            if proc.poll() is None:
                proc.kill()

        output = stdout or ""
        # Filter known_hosts warnings from stderr
        if stderr:
            stderr_clean = "\n".join(l for l in stderr.strip().splitlines() if "known hosts" not in l.lower())
            if stderr_clean:
                output += f"\n[stderr]: {stderr_clean}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "[error]: SSH command timed out"
    except FileNotFoundError as e:
        if "sshpass" in str(e):
            return "[error]: sshpass not installed. Run: sudo apt install sshpass"
        return f"[error]: {e}"
    except Exception as e:
        return f"[error]: {e}"


def get_tool_schemas():
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

# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

async def get_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


async def stream_ollama(messages: list, model: str, tools: list = None) -> AsyncGenerator[str, None]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    yield line


async def chat_ollama(messages: list, model: str, tools: list = None) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        return r.json()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="AI Workspace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ---------------------------------------------------------------------------
# Routes - Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    models = await get_models()
    return {"models": models, "default": DEFAULT_MODEL}

# ---------------------------------------------------------------------------
# Routes - Sessions
# ---------------------------------------------------------------------------

@app.get("/api/sessions")
async def get_sessions():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
        return {"sessions": [dict(r) for r in rows]}


@app.post("/api/sessions")
async def create_session(request: Request):
    data = await request.json()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (session_id, data.get("title", "New Chat"), data.get("mode", "chat"),
             data.get("model", DEFAULT_MODEL), now, now)
        )
    return {"id": session_id}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        return {"messages": [dict(r) for r in rows]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return {"ok": True}

# ---------------------------------------------------------------------------
# Routes - Chat (streaming)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None

@app.post("/api/chat")
async def chat(req: ChatRequest):
    s = load_settings()
    model = req.model or s.get("default_model", DEFAULT_MODEL)
    custom_chat_prompt = s.get("system_prompt_chat", "")
    user_name = s.get("user_name", "")

    # Load session history
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (req.session_id,)
        ).fetchall()
        session = conn.execute("SELECT * FROM sessions WHERE id=?", (req.session_id,)).fetchone()

    # Build memory context
    memory = load_memory()
    system_parts = ["You are a helpful AI assistant running locally. Be concise and direct."]
    if user_name:
        system_parts.append(f"The user's name is {user_name}.")
    if custom_chat_prompt:
        system_parts.append(custom_chat_prompt)
    if memory["facts"]:
        facts = "\n".join(f"- {f['content']}" for f in memory["facts"])
        system_parts.append(f"\nThings you know about the user:\n{facts}")

    messages = [{"role": "system", "content": "\n".join(system_parts)}]
    for row in rows:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": req.message})

    # Save user message
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), req.session_id, "user", req.message, None, None, now)
        )
        conn.execute("UPDATE sessions SET updated_at=?, title=? WHERE id=?",
                     (now, req.message[:50] if session and session["title"] == "New Chat" else (session["title"] if session else req.message[:50]), req.session_id))

    async def generate():
        full_response = ""
        try:
            async for line in stream_ollama(messages, model):
                try:
                    data = json.loads(line)
                    if data.get("message", {}).get("content"):
                        chunk = data["message"]["content"]
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    if data.get("done"):
                        # Save assistant message
                        with get_db() as conn:
                            conn.execute(
                                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                                (str(uuid.uuid4()), req.session_id, "assistant",
                                 full_response, None, None, datetime.now().isoformat())
                            )
                        stats = {
                            "type": "done",
                            "eval_count": data.get("eval_count", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "eval_duration": data.get("eval_duration", 0),
                            "total_duration": data.get("total_duration", 0),
                        }
                        yield f"data: {json.dumps(stats)}\n\n"
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Routes - Agent
# ---------------------------------------------------------------------------

class AgentRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None
    project: Optional[str] = None  # Active project name

@app.post("/api/agent")
async def agent(req: AgentRequest):
    s = load_settings()
    model = req.model or s.get("default_model", DEFAULT_MODEL)
    memory = load_memory()
    max_iterations = s.get("agent_max_iterations", MAX_TOOL_ITERATIONS)
    agent_timeout = s.get("agent_timeout", 120)
    user_name = s.get("user_name", "")
    custom_agent_prompt = s.get("system_prompt_agent", "")

    # Build tool descriptions for the system prompt
    tool_descriptions = []
    for name, t in TOOLS.items():
        schema = t["schema"]["function"]
        props = schema["parameters"]["properties"]
        params_desc = ", ".join(f'{k}: {v.get("type","string")}' for k, v in props.items())
        tool_descriptions.append(f'- {name}({params_desc}): {schema["description"]}')

    tools_text = "\n".join(tool_descriptions)
    home_dir = str(Path.home())

    system_prompt = f"""You are an AI agent with access to tools. Use tools to accomplish tasks.

The user home directory is: {home_dir}
When listing the home directory always use path: {home_dir}

Available tools:
{tools_text}

When you need to use a tool, respond with ONLY this JSON format (nothing else):
{{"tool": "tool_name", "args": {{"param": "value"}}}}

When you have gathered enough information and are ready to give a final answer, respond normally in plain text WITHOUT any JSON.

Rules:
- Always use tools when the task requires real data — never make up file contents or command outputs
- After getting tool results, either use another tool or give your final answer
- Keep final answers concise and clear
- Never wrap your final answer in JSON
- For SSH and shell commands: avoid slow commands like bare `find /` without -maxdepth — always add -maxdepth 2 or 3 to stay within timeouts
- For finding folders on a remote machine: explore systematically — first `ls /` to see top level, then `ls /home`, `ls /mnt`, `ls /media` etc., then drill into likely locations. Never assume the folder is in the home directory.
- When looking for a specific named folder like "photos": use `find / -maxdepth 4 -type d -iname "*photo*" 2>/dev/null` — this is safe with maxdepth 4
- Run multiple tool calls if needed to fully explore — don't give up after one ls
- When asked to "look at", "describe", or "understand" a folder or project: first list the directory, then read key files (README, main script, config) before answering
- Never describe files you haven't read — always read_file on relevant files before giving a description
- A directory listing alone is never enough to describe what something does — you must read the actual files"""

    # Inject project context if active
    if req.project:
        proj = get_project(req.project)
        if proj:
            proj_root = proj.get("root", "")
            proj_remote = proj.get("remote", "")
            proj_loc = f"{proj_remote}:{proj_root}" if proj_remote else proj_root
            system_prompt += f"""

Active project: {proj.get("name")}
Location: {proj_loc}
{"Remote: use ssh tool with remote=\"" + proj_remote + "\" for all file operations in this project" if proj_remote else "Local: use shell/read_file/write_file/edit_file for file operations"}
Stay within {proj_root} unless explicitly asked to go elsewhere.
When editing files, read the current content first then use edit_file.
When asked about the project structure, use {"ssh to run ls/find in " + proj_root if proj_remote else "list_dir on " + proj_root}."""

    if user_name:
        system_prompt += f"\n\nThe user's name is {user_name}."
    if memory["facts"]:
        facts = "\n".join(f"- {f['content']}" for f in memory["facts"])
        system_prompt += f"\n\nUser context:\n{facts}"
    if custom_agent_prompt:
        system_prompt += f"\n\nAdditional instructions:\n{custom_agent_prompt}"

    # Load history
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (req.session_id,)
        ).fetchall()

    messages = [{"role": "system", "content": system_prompt}]
    for row in rows:
        if row["role"] in ("user", "assistant"):
            messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": req.message})

    # Save user message
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), req.session_id, "user", req.message, None, None, now)
        )

    def strip_deepseek_tokens(text: str) -> str:
        """Strip DeepSeek native tool call tokens that leak into text responses."""
        # Remove <｜tool_calls_begin｜>...<｜tool_calls_end｜> blocks entirely
        text = re.sub(r'<｜tool[▁_]calls[▁_]begin｜>[\s\S]*?<｜tool[▁_]calls[▁_]end｜>', '', text, flags=re.IGNORECASE)
        # Remove individual tool call markers
        text = re.sub(r'<｜tool[▁_]call[▁_]begin｜>.*?<｜tool[▁_]call[▁_]end｜>', '', text, flags=re.DOTALL|re.IGNORECASE)
        # Remove tool output blocks
        text = re.sub(r'<｜tool[▁_]outputs[▁_]begin｜>[\s\S]*?<｜tool[▁_]outputs[▁_]end｜>', '', text, flags=re.IGNORECASE)
        # Remove any remaining pipe-delimited markers
        text = re.sub(r'<｜[^｜]+｜>', '', text)
        return text.strip()

    def parse_tool_call(text: str) -> Optional[dict]:
        """Extract JSON tool call from model response."""
        # First strip any DeepSeek native tokens
        text = strip_deepseek_tokens(text).strip()
        if not text:
            return None

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if "tool" in data and "args" in data:
                return data
        except Exception:
            pass
        # Try extracting JSON from text with tool/args keys
        match = re.search(r'\{[^{}]*"tool"[^{}]*"args"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if "tool" in data and "args" in data:
                    return data
            except Exception:
                pass
        # Try nested JSON (args may contain nested objects)
        matches = re.findall(r'\{(?:[^{}]|\{[^{}]*\})*\}', text)
        for m in matches:
            try:
                data = json.loads(m)
                if "tool" in data and "args" in data:
                    return data
            except Exception:
                pass
        # Try ```json blocks
        match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
        if match:
            try:
                data = json.loads(match.group(1))
                if "tool" in data and "args" in data:
                    return data
            except Exception:
                pass
        return None

    async def run_agent():
        all_tool_calls = []
        all_tool_results = []
        final_response = ""
        iteration = 0
        total_tokens = 0
        start_time = time.time()

        while iteration < max_iterations:
            iteration += 1
            yield f"data: {json.dumps({'type': 'thinking', 'iteration': iteration})}\n\n"

            try:
                response = await chat_ollama(messages, model)  # No tools= param
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                return

            raw_content = response.get("message", {}).get("content", "").strip()
            total_tokens += response.get("eval_count", 0) + response.get("prompt_eval_count", 0)

            if not raw_content:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Empty response from model'})}\n\n"
                return

            # Strip DeepSeek native tokens before processing
            content = strip_deepseek_tokens(raw_content)
            if not content:
                # Model only output tokens with no actual content — treat as done
                break

            # Try to parse as tool call
            tool_call = parse_tool_call(content)

            if tool_call:
                tool_name = tool_call.get("tool", "")
                tool_args = tool_call.get("args", {})

                yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'args': tool_args})}\n\n"

                result = await execute_tool(tool_name, tool_args)

                yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'result': result[:1000]})}\n\n"

                all_tool_calls.append({"tool": tool_name, "args": tool_args})
                all_tool_results.append(result)

                # Feed result back into conversation
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"Tool result for {tool_name}:\n{result}\n\nContinue working toward the user's goal. If you need more information, use another tool. Only give your final answer when you have enough real data to answer properly — do not summarize or describe things you haven't actually read."
                })
            else:
                # No tool call — this is the final response
                final_response = content
                yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"
                break

        if not final_response and all_tool_results:
            # Hit iteration limit — ask for summary
            messages.append({
                "role": "user",
                "content": "Please summarize what you found and give your final answer."
            })
            try:
                response = await chat_ollama(messages, model)
                final_response = response.get("message", {}).get("content", "Max iterations reached.")
            except Exception:
                final_response = "Max iterations reached."
            yield f"data: {json.dumps({'type': 'response', 'content': final_response})}\n\n"

        # Save to DB
        tool_call_json = json.dumps(all_tool_calls) if all_tool_calls else None
        tool_result_json = json.dumps(all_tool_results) if all_tool_results else None
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), req.session_id, "assistant",
                 final_response, tool_call_json, tool_result_json,
                 datetime.now().isoformat())
            )
            conn.execute("UPDATE sessions SET updated_at=? WHERE id=?",
                         (datetime.now().isoformat(), req.session_id))

        elapsed = time.time() - start_time
        yield f"data: {json.dumps({'type': 'done', 'total_tokens': total_tokens, 'elapsed': round(elapsed, 1), 'iterations': iteration})}\n\n"

    return StreamingResponse(run_agent(), media_type="text/event-stream")

# ---------------------------------------------------------------------------
# Routes - Memory
# ---------------------------------------------------------------------------

@app.get("/api/memory")
async def get_memory():
    return load_memory()

@app.post("/api/memory")
async def add_memory(request: Request):
    data = await request.json()
    memory = load_memory()
    memory["facts"].append({
        "content": data["content"],
        "created_at": datetime.now().isoformat()
    })
    save_memory(memory)
    return {"ok": True}

@app.delete("/api/memory/{index}")
async def delete_memory(index: int):
    memory = load_memory()
    if index < 0 or index >= len(memory["facts"]):
        raise HTTPException(status_code=404, detail="Memory not found")
    memory["facts"].pop(index)
    save_memory(memory)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Routes - Settings
# ---------------------------------------------------------------------------

@app.get("/api/settings")
async def get_settings():
    return load_settings()

@app.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    current = load_settings()
    current.update(data)
    save_settings(current)
    return {"ok": True}

@app.get("/api/remotes")
async def get_remotes():
    s = load_settings()
    # Return remotes without passwords for security
    remotes = []
    for r in s.get("remotes", []):
        safe = {k: v for k, v in r.items() if k != "password"}
        safe["has_password"] = bool(r.get("password"))
        safe["has_key"] = bool(r.get("key_path"))
        remotes.append(safe)
    return {"remotes": remotes}

@app.post("/api/remotes")
async def add_remote(request: Request):
    data = await request.json()
    if not data.get("name") or not data.get("host"):
        raise HTTPException(status_code=400, detail="name and host required")
    s = load_settings()
    remotes = s.get("remotes", [])
    # Update if exists
    for i, r in enumerate(remotes):
        if r.get("name") == data["name"]:
            remotes[i] = data
            save_settings(s)
            return {"ok": True}
    remotes.append(data)
    s["remotes"] = remotes
    save_settings(s)
    return {"ok": True}

@app.delete("/api/remotes/{name}")
async def delete_remote(name: str):
    s = load_settings()
    s["remotes"] = [r for r in s.get("remotes", []) if r.get("name") != name]
    save_settings(s)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

PROJECTS_PATH = DATA_DIR / "projects.json"

def load_projects() -> list:
    if PROJECTS_PATH.exists():
        try:
            return json.loads(PROJECTS_PATH.read_text())
        except Exception:
            pass
    return []

def save_projects(projects: list):
    PROJECTS_PATH.write_text(json.dumps(projects, indent=2))

def get_project(name: str) -> Optional[dict]:
    for p in load_projects():
        if p.get("name") == name:
            return p
    return None


def run_project_cmd(project: dict, cmd: str) -> str:
    """Run a command in the project root, routing through SSH if remote."""
    remote = project.get("remote")
    root = project.get("root", "")
    if remote:
        # Route through SSH
        return tool_ssh(remote, cmd, path=root)
    else:
        # Local execution
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=root
            )
            return (result.stdout or result.stderr or "(no output)").strip()
        except subprocess.TimeoutExpired:
            return "[error]: Command timed out"
        except Exception as e:
            return f"[error]: {e}"


def read_project_file(project: dict, rel_path: str) -> str:
    """Read a file from the project, routing through SSH if remote."""
    remote = project.get("remote")
    root = project.get("root", "").rstrip("/")
    full_path = f"{root}/{rel_path.lstrip('/')}"
    if remote:
        return tool_ssh(remote, f"cat {full_path}")
    else:
        return tool_read_file(full_path)


def scan_project_structure(project: dict) -> str:
    """Scan project directory structure up to depth 2."""
    ignore = project.get("ignore", ["venv", "node_modules", "__pycache__", ".git", ".next", "dist", "build"])
    ignore_str = " ".join(f"-not -path '*/{i}/*' -not -name '{i}'" for i in ignore)
    cmd = f"find . -maxdepth 2 -type f {ignore_str} 2>/dev/null | sort | head -80"
    result = run_project_cmd(project, cmd)
    return result or "(empty project)"


def find_key_files(project: dict) -> list:
    """Find README and main entry point files."""
    key_names = ["README.md", "README.txt", "readme.md",
                 "app.py", "main.py", "index.py", "server.py",
                 "index.js", "index.ts", "main.js", "app.js",
                 "package.json", "pyproject.toml", "Cargo.toml",
                 "go.mod", "pom.xml", "build.gradle"]
    root = project.get("root", "")
    remote = project.get("remote")
    found = []
    for name in key_names:
        if remote:
            result = tool_ssh(remote, f"test -f {root}/{name} && echo exists")
            if "exists" in result:
                found.append(name)
        else:
            if (Path(root) / name).exists():
                found.append(name)
        if len(found) >= 4:
            break
    return found


# ---------------------------------------------------------------------------
# Routes - Projects
# ---------------------------------------------------------------------------

@app.get("/api/projects")
async def get_projects():
    return {"projects": load_projects()}

@app.post("/api/projects")
async def create_project(request: Request):
    data = await request.json()
    if not data.get("name") or not data.get("root"):
        raise HTTPException(status_code=400, detail="name and root required")
    projects = load_projects()
    # Update if exists
    for i, p in enumerate(projects):
        if p.get("name") == data["name"]:
            projects[i] = data
            save_projects(projects)
            return {"ok": True, "updated": True}
    projects.append(data)
    save_projects(projects)
    return {"ok": True}

@app.delete("/api/projects/{name}")
async def delete_project(name: str):
    projects = [p for p in load_projects() if p.get("name") != name]
    save_projects(projects)
    return {"ok": True}

@app.get("/api/projects/{name}/scan")
async def scan_project(name: str):
    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    structure = scan_project_structure(p)
    key_files = find_key_files(p)
    # Read key file contents (first 2 only to avoid context overflow)
    key_contents = {}
    for f in key_files[:2]:
        content = read_project_file(p, f)
        if not content.startswith("[error]"):
            key_contents[f] = content[:3000]
    return {
        "structure": structure,
        "key_files": key_files,
        "key_contents": key_contents
    }

@app.post("/api/projects/{name}/file")
async def read_project_file_route(name: str, request: Request):
    data = await request.json()
    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    content = read_project_file(p, data.get("path", ""))
    return {"content": content}

# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
