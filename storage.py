import json
from datetime import datetime
from typing import Optional

from config import DEFAULT_MODEL, MEMORY_PATH, PROJECTS_PATH, SETTINGS_PATH

# ── Memory ────────────────────────────────────────────────────────────────────

def load_memory() -> dict:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"facts": [], "preferences": {}, "updated_at": None}


def save_memory(memory: dict) -> None:
    memory["updated_at"] = datetime.now().isoformat()
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")


# ── Settings ──────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "ollama_url": "http://localhost:11434",
    "default_model": DEFAULT_MODEL,
    "agent_max_iterations": 12,
    "agent_timeout": 120,
    "planner_model": "",
    "context_length": 8192,
    "remotes": [],
    "mcp_servers": [],
    "custom_skills": [],
    "user_name": "",
    "system_prompt_chat": "",
    "system_prompt_agent": "",
}


def load_settings() -> dict:
    s = dict(_DEFAULTS)
    if SETTINGS_PATH.exists():
        try:
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")


def get_remote(name: str) -> Optional[dict]:
    for r in load_settings().get("remotes", []):
        if r.get("name") == name:
            return r
    return None


# ── Projects ──────────────────────────────────────────────────────────────────

def load_projects() -> list:
    if PROJECTS_PATH.exists():
        try:
            return json.loads(PROJECTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def save_projects(projects: list) -> None:
    PROJECTS_PATH.write_text(json.dumps(projects, indent=2), encoding="utf-8")


def get_project(name: str) -> Optional[dict]:
    for p in load_projects():
        if p.get("name") == name:
            return p
    return None
