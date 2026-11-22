"""Locations and defaults shared by the app and the standalone compiler."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HANGAR_DATA_DIR") or BASE_DIR / "data").expanduser().resolve()
DB_PATH = DATA_DIR / "workspace.db"
MEMORY_PATH = DATA_DIR / "memory.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROJECTS_PATH = DATA_DIR / "projects.json"

OLLAMA_URL = os.getenv("OLLAMA_URL") or "http://localhost:11434"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL") or "deepseek-coder-v2:16b"
