"""Project utilities: directory scanning, file reading, command execution.

Local scanning uses pure Python (pathlib) so it works identically on
Windows, macOS, and Linux without shelling out to `find`.
Remote scanning delegates to SSH and expects a Linux/macOS remote.
"""

import subprocess
from pathlib import Path
from typing import Optional

from config import SHELL_PREFIX


# ── Command execution ─────────────────────────────────────────────────────────

def run_project_cmd(project: dict, cmd: str) -> str:
    """Run *cmd* in the project root, routing through SSH if remote."""
    from tools import tool_ssh  # local import avoids circular dependency

    remote = project.get("remote")
    root = project.get("root", "")

    if remote:
        return tool_ssh(remote, cmd, path=root)

    try:
        if SHELL_PREFIX:
            result = subprocess.run(
                SHELL_PREFIX + [cmd],
                capture_output=True, text=True, timeout=30, cwd=root,
            )
        else:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=root,
            )
        return (result.stdout or result.stderr or "(no output)").strip()
    except subprocess.TimeoutExpired:
        return "[error]: Command timed out"
    except Exception as e:
        return f"[error]: {e}"


# ── File reading ──────────────────────────────────────────────────────────────

def read_project_file(project: dict, rel_path: str) -> str:
    from tools import tool_read_file, tool_ssh

    remote = project.get("remote")
    root = project.get("root", "").rstrip("/\\")

    if remote:
        full = f"{root}/{rel_path.lstrip('/')}"
        return tool_ssh(remote, f"cat {full}")

    full = Path(root) / rel_path.lstrip("/\\")
    return tool_read_file(str(full))


# ── Directory scanning ────────────────────────────────────────────────────────

_DEFAULT_IGNORE = frozenset(
    ["venv", "node_modules", "__pycache__", ".git", ".next", "dist", "build", ".tox", ".mypy_cache"]
)


def scan_project_structure(project: dict) -> str:
    """Return a compact tree of the project (max depth 2, max 80 entries).

    Local projects: pure Python — no shell, works on all platforms.
    Remote projects: SSH `find` (target is always Linux/macOS).
    """
    ignore = _DEFAULT_IGNORE | set(project.get("ignore", []))
    remote = project.get("remote")
    root_str = project.get("root", "")

    if remote:
        ignore_args = " ".join(
            f"-not -path '*/{i}/*' -not -name '{i}'" for i in sorted(ignore)
        )
        cmd = f"find . -maxdepth 2 -type f {ignore_args} 2>/dev/null | sort | head -80"
        return run_project_cmd(project, cmd)

    # Local: pure Python
    root = Path(root_str).expanduser().resolve()
    if not root.exists():
        return "(project root not found)"

    lines: list[str] = []

    def _scan(path: Path, depth: int) -> None:
        if depth > 2 or len(lines) >= 80:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return
        for item in entries:
            if item.name.startswith(".") or item.name in ignore:
                continue
            indent = "  " * (depth)
            suffix = "/" if item.is_dir() else ""
            lines.append(f"{indent}{item.name}{suffix}")
            if len(lines) >= 80:
                return
            if item.is_dir():
                _scan(item, depth + 1)

    _scan(root, 0)
    return "\n".join(lines) or "(empty project)"


# ── Key file detection ────────────────────────────────────────────────────────

_KEY_FILE_NAMES = [
    "README.md", "readme.md", "README.txt",
    "app.py", "main.py", "index.py", "server.py",
    "index.js", "index.ts", "main.js", "app.js",
    "package.json", "pyproject.toml", "Cargo.toml",
    "go.mod", "pom.xml", "build.gradle",
]


def find_key_files(project: dict) -> list[str]:
    from tools import tool_ssh

    root = project.get("root", "")
    remote = project.get("remote")
    found: list[str] = []

    for name in _KEY_FILE_NAMES:
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
