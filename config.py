import os
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "workspace.db"
MEMORY_PATH = DATA_DIR / "memory.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROJECTS_PATH = DATA_DIR / "projects.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-coder-v2:16b")
MAX_TOOL_ITERATIONS = 5

IS_WINDOWS = sys.platform == "win32"
OS_NAME = "Windows" if IS_WINDOWS else ("macOS" if sys.platform == "darwin" else "Linux")


def _find_shell() -> list[str] | None:
    if IS_WINDOWS:
        for exe in ("pwsh", "powershell"):
            if shutil.which(exe):
                return [exe, "-NoProfile", "-NonInteractive", "-Command"]
        return None  # falls back to shell=True (cmd.exe)
    else:
        for exe in ("bash", "sh"):
            p = shutil.which(exe)
            if p:
                return [p, "-c"]
        return None


SHELL_PREFIX: list[str] | None = _find_shell()


def shell_name() -> str:
    if IS_WINDOWS:
        return SHELL_PREFIX[0] if SHELL_PREFIX else "cmd.exe"
    return SHELL_PREFIX[0] if SHELL_PREFIX else "sh"


def to_api_path(p) -> str:
    """Normalize path separators to forward slashes for API responses."""
    return str(p).replace("\\", "/")
