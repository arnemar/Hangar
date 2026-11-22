"""Find folders for the local user before they register a project."""

import os
import stat
from collections import deque
from pathlib import Path

from hangar.workspace.paths import WorkspaceError, hidden_name, is_link, valid_part, validate_root

MAX_LISTED = 300
MAX_RESULTS = 100
MAX_SCAN_ENTRIES = 12000
MAX_SEARCH_FOLDERS = 600


def _scan(folder: Path, protected: Path, budget: int) -> tuple[list[Path], int, bool]:
    paths = []
    scanned = 0
    with os.scandir(folder) as candidates:
        for entry in candidates:
            if scanned == budget:
                return paths, scanned, True
            scanned += 1
            if entry.name.startswith(".") or hidden_name(entry.name) or not valid_part(entry.name):
                continue
            path = folder / entry.name
            if path.is_relative_to(protected):
                continue
            try:
                entry.name.encode("utf-8")
                info = entry.stat(follow_symlinks=False)
            except (OSError, UnicodeError):
                continue
            if stat.S_ISDIR(info.st_mode) and not is_link(info):
                paths.append(path)
    return paths, scanned, False


def _search(root: Path, protected: Path, query: str) -> tuple[list[Path], bool]:
    pending = deque([root])
    matches = []
    budget = MAX_SCAN_ENTRIES
    for _ in range(MAX_SEARCH_FOLDERS):
        if not pending or not budget:
            break
        folder = pending.popleft()
        try:
            if folder != root and folder.resolve(strict=True) != folder:
                continue
            paths, scanned, limited = _scan(folder, protected, budget)
        except (OSError, RuntimeError):
            if folder == root:
                raise
            continue
        budget -= scanned
        matches.extend(path for path in paths if query in path.name.casefold())
        if len(matches) > MAX_RESULTS or limited:
            return matches[:MAX_RESULTS], True
        pending.extend(paths)
    return matches, bool(pending)


def _entry(path: Path) -> dict[str, str]:
    return {"name": path.name or path.anchor, "path": str(path)}


def browse_folders(path: str, query: str, data_dir: Path) -> dict:
    protected = data_dir.resolve()
    root = validate_root(path or str(Path.home()), protected)
    try:
        if query:
            folders, limited = _search(root, protected, query.casefold())
        else:
            folders, _, limited = _scan(root, protected, MAX_SCAN_ENTRIES)
    except (OSError, RuntimeError) as exc:
        raise WorkspaceError("Hangar cannot list this folder.", 403) from exc

    folders.sort(key=lambda path: (path.name.casefold(), str(path).casefold()))
    return {
        "folder": _entry(root),
        "parent": str(root.parent) if root != root.parent else None,
        "folders": [_entry(folder) for folder in folders[:MAX_LISTED]],
        "truncated": limited or len(folders) > MAX_LISTED,
    }
