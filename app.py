"""AI Workspace — FastAPI application and route definitions."""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import cancel_request, run_agent_stream
from config import BASE_DIR, DEFAULT_MODEL, to_api_path
from db import get_db, init_db
from mcp_client import mcp_manager
from ollama_client import get_models, stream_ollama
from projects import find_key_files, read_project_file, scan_project_structure
from skills import get_all_skills
from storage import (
    get_project,
    get_remote,
    load_memory,
    load_projects,
    load_settings,
    save_memory,
    save_projects,
    save_settings,
)
from tools import tool_ssh

# ── Lifespan (startup / shutdown) ────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    s = load_settings()
    for srv in s.get("mcp_servers", []):
        if srv.get("enabled", True) and srv.get("command"):
            cmd = srv["command"] if isinstance(srv["command"], list) else srv["command"].split()
            ok, msg = await mcp_manager.connect(srv["name"], cmd, srv.get("env", {}))
            status = "connected" if ok else "failed"
            print(f"[MCP] {srv['name']}: {status} — {msg}")
    yield
    await mcp_manager.disconnect_all()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="AI Workspace", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────


@app.get("/api/models")
async def list_models():
    s = load_settings()
    models = await get_models(s.get("ollama_url", ""))
    return {"models": models, "default": s.get("default_model", DEFAULT_MODEL)}


# ── Sessions ──────────────────────────────────────────────────────────────────


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
    s = load_settings()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?,?,?,?,?,?)",
            (
                session_id,
                data.get("title", "New Chat"),
                data.get("mode", "chat"),
                data.get("model", s.get("default_model", DEFAULT_MODEL)),
                now, now,
            ),
        )
    return {"id": session_id}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    return {"messages": [dict(r) for r in rows]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return {"ok": True}


# ── Chat ──────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None


@app.post("/api/chat")
async def chat(req: ChatRequest):
    s = load_settings()
    model = req.model or s.get("default_model", DEFAULT_MODEL)
    num_ctx = s.get("context_length", 8192)
    ollama_url = s.get("ollama_url", "")

    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY created_at",
            (req.session_id,),
        ).fetchall()
        session = conn.execute(
            "SELECT * FROM sessions WHERE id=?", (req.session_id,)
        ).fetchone()

    memory = load_memory()
    parts = ["You are a helpful AI assistant running locally. Be concise and direct."]
    if s.get("user_name"):
        parts.append(f"The user's name is {s['user_name']}.")
    if s.get("system_prompt_chat"):
        parts.append(s["system_prompt_chat"])
    if memory["facts"]:
        facts = "\n".join(f"- {f['content']}" for f in memory["facts"])
        parts.append(f"\nThings you know about the user:\n{facts}")

    messages = [{"role": "system", "content": "\n".join(parts)}]
    for row in rows:
        messages.append({"role": row["role"], "content": row["content"]})
    messages.append({"role": "user", "content": req.message})

    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), req.session_id, "user", req.message, None, None, now),
        )
        old_title = session["title"] if session else "New Chat"
        title = req.message[:50] if old_title == "New Chat" else old_title
        conn.execute(
            "UPDATE sessions SET updated_at=?, title=? WHERE id=?",
            (now, title, req.session_id),
        )

    async def generate():
        full = ""
        try:
            async for line in stream_ollama(messages, model, num_ctx=num_ctx, ollama_url=ollama_url):
                try:
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content")
                    if chunk:
                        full += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                    if data.get("done"):
                        with get_db() as conn:
                            conn.execute(
                                "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
                                (str(uuid.uuid4()), req.session_id, "assistant",
                                 full, None, None, datetime.now().isoformat()),
                            )
                        yield f"data: {json.dumps({'type': 'done', 'eval_count': data.get('eval_count', 0), 'prompt_eval_count': data.get('prompt_eval_count', 0), 'eval_duration': data.get('eval_duration', 0), 'total_duration': data.get('total_duration', 0)})}\n\n"
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Agent ─────────────────────────────────────────────────────────────────────


class AgentRequest(BaseModel):
    session_id: str
    message: str
    model: Optional[str] = None
    project: Optional[str] = None
    request_id: Optional[str] = None


@app.post("/api/agent")
async def agent_endpoint(req: AgentRequest):
    s = load_settings()
    model = req.model or s.get("default_model", DEFAULT_MODEL)
    req_id = req.request_id or str(uuid.uuid4())

    now = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), req.session_id, "user", req.message, None, None, now),
        )

    async def stream():
        yield f"data: {json.dumps({'type': 'request_id', 'id': req_id})}\n\n"
        async for chunk in run_agent_stream(
            session_id=req.session_id,
            message=req.message,
            model=model,
            project_name=req.project,
            request_id=req_id,
            settings=s,
        ):
            yield chunk

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Memory ────────────────────────────────────────────────────────────────────


@app.get("/api/memory")
async def get_memory():
    return load_memory()


@app.post("/api/memory")
async def add_memory(request: Request):
    data = await request.json()
    memory = load_memory()
    memory["facts"].append({"content": data["content"], "created_at": datetime.now().isoformat()})
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


# ── Settings ──────────────────────────────────────────────────────────────────


@app.get("/api/settings")
async def get_settings_route():
    return load_settings()


@app.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    s = load_settings()
    s.update(data)
    save_settings(s)
    return {"ok": True}


# ── Remotes ───────────────────────────────────────────────────────────────────


@app.get("/api/remotes")
async def get_remotes():
    remotes = []
    for r in load_settings().get("remotes", []):
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


# ── MCP servers ───────────────────────────────────────────────────────────────


@app.get("/api/mcp")
async def get_mcp_servers():
    return {"servers": mcp_manager.get_status()}


@app.post("/api/mcp")
async def add_mcp_server(request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    command_raw = data.get("command", "")
    if not name or not command_raw:
        raise HTTPException(status_code=400, detail="name and command required")

    cmd = command_raw if isinstance(command_raw, list) else command_raw.split()
    env = data.get("env", {})

    # Persist to settings
    s = load_settings()
    servers = s.get("mcp_servers", [])
    servers = [x for x in servers if x.get("name") != name]
    servers.append({"name": name, "command": cmd, "env": env, "enabled": True})
    s["mcp_servers"] = servers
    save_settings(s)

    # Connect immediately
    ok, msg = await mcp_manager.connect(name, cmd, env)
    return {"ok": ok, "message": msg, "servers": mcp_manager.get_status()}


@app.delete("/api/mcp/{name}")
async def delete_mcp_server(name: str):
    await mcp_manager.disconnect(name)
    s = load_settings()
    s["mcp_servers"] = [x for x in s.get("mcp_servers", []) if x.get("name") != name]
    save_settings(s)
    return {"ok": True}


@app.post("/api/mcp/{name}/reconnect")
async def reconnect_mcp_server(name: str):
    s = load_settings()
    srv = next((x for x in s.get("mcp_servers", []) if x.get("name") == name), None)
    if not srv:
        raise HTTPException(status_code=404, detail=f"MCP server '{name}' not found")
    cmd = srv["command"] if isinstance(srv["command"], list) else srv["command"].split()
    ok, msg = await mcp_manager.connect(name, cmd, srv.get("env", {}))
    return {"ok": ok, "message": msg, "servers": mcp_manager.get_status()}


# ── Skills ────────────────────────────────────────────────────────────────────


@app.get("/api/skills")
async def get_skills():
    s = load_settings()
    return {"skills": get_all_skills(s.get("custom_skills", []))}


@app.post("/api/skills")
async def add_custom_skill(request: Request):
    data = await request.json()
    if not data.get("name") or not data.get("prompt"):
        raise HTTPException(status_code=400, detail="name and prompt required")
    s = load_settings()
    customs = s.get("custom_skills", [])
    customs = [x for x in customs if x.get("name") != data["name"]]
    customs.append({
        "name": data["name"],
        "description": data.get("description", ""),
        "prompt": data["prompt"],
        "icon": data.get("icon", "⚡"),
    })
    s["custom_skills"] = customs
    save_settings(s)
    return {"ok": True}


@app.delete("/api/skills/{name}")
async def delete_custom_skill(name: str):
    s = load_settings()
    s["custom_skills"] = [x for x in s.get("custom_skills", []) if x.get("name") != name]
    save_settings(s)
    return {"ok": True}


# ── File read (for @mention expansion) ───────────────────────────────────────


@app.get("/api/file")
async def read_file_route(path: str = ""):
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail="File not found")
        if not p.is_file():
            raise HTTPException(status_code=400, detail="Not a file")
        if p.stat().st_size > 200_000:
            raise HTTPException(status_code=400, detail="File too large (>200 KB)")
        return {"content": p.read_text(errors="replace"), "path": to_api_path(p)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Projects ──────────────────────────────────────────────────────────────────


@app.get("/api/projects")
async def get_projects():
    return {"projects": load_projects()}


@app.post("/api/projects")
async def create_project(request: Request):
    data = await request.json()
    if not data.get("name") or not data.get("root"):
        raise HTTPException(status_code=400, detail="name and root required")
    projects = load_projects()
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
    save_projects([p for p in load_projects() if p.get("name") != name])
    return {"ok": True}


@app.get("/api/projects/{name}/scan")
async def scan_project(name: str):
    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    structure = scan_project_structure(p)
    key_files = find_key_files(p)
    key_contents = {}
    for f in key_files[:2]:
        content = read_project_file(p, f)
        if not content.startswith("[error]"):
            key_contents[f] = content[:3000]
    return {"structure": structure, "key_files": key_files, "key_contents": key_contents}


@app.post("/api/projects/{name}/file")
async def read_project_file_route(name: str, request: Request):
    data = await request.json()
    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"content": read_project_file(p, data.get("path", ""))}


# ── Compiler ──────────────────────────────────────────────────────────────────


@app.post("/api/compiler/{name}/compile")
async def compile_project_route(name: str):
    """Trigger a full compilation pass for a project (scan → canonicalize → graph_build)."""
    import asyncio
    from compiler.pipeline import compile_project

    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    root = p.get("root", "")
    if not root:
        raise HTTPException(status_code=400, detail="Project has no root path")

    result = await asyncio.to_thread(compile_project, name, root)
    scan = result.scan
    return {
        "ok": True,
        "project": name,
        "scan": {
            "discovered": scan.discovered if scan else 0,
            "changed": scan.changed if scan else 0,
            "removed": scan.removed if scan else 0,
            "stable": scan.stable if scan else 0,
            "skipped": scan.skipped if scan else 0,
        } if scan else None,
        "compiled": result.files_compiled,
        "skipped_ir": result.files_skipped_ir,
        "failed": result.files_failed,
        "nodes": {
            "created": result.nodes_created,
            "updated": result.nodes_updated,
            "removed": result.nodes_removed,
        },
        "edges_created": result.edges_created,
        "errors": result.errors,
    }


@app.get("/api/compiler/{name}/nodes")
async def get_project_nodes(name: str, kind: str = "", limit: int = 100):
    """Query graph nodes for a project, optionally filtered by kind."""
    from db import get_db

    p = get_project(name)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")

    with get_db() as conn:
        if kind:
            rows = conn.execute(
                "SELECT id, path, kind, name, signature, start_line, end_line, language "
                "FROM nodes WHERE project_id = ? AND kind = ? ORDER BY path, start_line LIMIT ?",
                (name, kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, path, kind, name, signature, start_line, end_line, language "
                "FROM nodes WHERE project_id = ? ORDER BY path, start_line LIMIT ?",
                (name, limit),
            ).fetchall()

    return {"nodes": [dict(r) for r in rows], "count": len(rows)}


# ── Browse ────────────────────────────────────────────────────────────────────


@app.get("/api/browse")
async def browse_directory(path: str = "", remote: str = ""):
    if not path:
        path = to_api_path(Path.home()) if not remote else "/home"
    try:
        if remote:
            r = get_remote(remote)
            if not r:
                raise HTTPException(status_code=404, detail=f"Remote '{remote}' not found")
            result = await tool_ssh(remote, f"ls -la {path} 2>/dev/null")
            if result.startswith("[error]"):
                raise HTTPException(status_code=400, detail=result)
            entries = []
            for line in result.strip().splitlines():
                parts = line.split(None, 8)
                if len(parts) < 9:
                    continue
                perms, _, _, _, _, _, _, _, name = parts
                if name in (".", ".."):
                    continue
                entries.append({
                    "name": name,
                    "is_dir": perms.startswith("d"),
                    "path": f"{path.rstrip('/')}/{name}",
                })
            entries.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
            parent = str(Path(path).parent).replace("\\", "/") if path != "/" else None
            return {"path": path, "entries": entries, "parent": parent, "remote": remote}
        else:
            p = Path(path).expanduser().resolve()
            if not p.exists() or not p.is_dir():
                raise HTTPException(status_code=400, detail=f"Not a directory: {path}")
            entries = []
            try:
                for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    if item.name.startswith("."):
                        continue
                    entries.append({
                        "name": item.name,
                        "is_dir": item.is_dir(),
                        "path": to_api_path(item),
                    })
            except PermissionError:
                pass
            parent = to_api_path(p.parent) if p != p.parent else None
            return {"path": to_api_path(p), "entries": entries, "parent": parent, "remote": ""}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Cancel ────────────────────────────────────────────────────────────────────


@app.post("/api/cancel/{request_id}")
async def cancel_agent(request_id: str):
    cancel_request(request_id)
    return {"ok": True, "cancelled": request_id}


# ── Static ────────────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
