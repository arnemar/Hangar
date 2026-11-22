import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("HANGAR_DATA_DIR") or BASE_DIR / "data").expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "workspace.db"
MEMORY_PATH = DATA_DIR / "memory.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROJECTS_PATH = DATA_DIR / "projects.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-coder-v2:16b")
