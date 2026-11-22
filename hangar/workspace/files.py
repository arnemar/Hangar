"""Operations on text files within one explicitly registered local project."""

import os
import stat
from pathlib import Path

from hangar.config import DATA_DIR
from hangar.workspace.paths import WorkspaceError, hidden_name, is_link, project_path, valid_part, validate_root

MAX_FILE_BYTES = 256 * 1024
MAX_DIRECTORY_ENTRIES = 300


def register_root(raw: str) -> Path:
    return validate_root(raw, DATA_DIR)


class LocalProject:
    def __init__(self, record: dict):
        if record.get("remote"):
            raise WorkspaceError("This project uses SSH. Attach a local project folder.", 409)
        self.name = record["name"]
        self.root = register_root(record.get("root", ""))
        if record.get("scope_version") == 1 and self.root != Path(record["root"]):
            raise WorkspaceError("The configured project folder has changed. Reattach it.", 409)

    def path(self, relative: str, *, missing_leaf: bool = False) -> Path:
        return project_path(self.root, DATA_DIR.resolve(), relative, missing_leaf=missing_leaf)

    def entries(self, relative: str = "") -> dict:
        relative = "" if relative == "." else relative
        folder = self.path(relative)
        if not folder.is_dir():
            raise WorkspaceError("This path is not a folder.")
        entries = []
        protected = DATA_DIR.resolve()
        try:
            with os.scandir(folder) as children:
                for child in children:
                    if hidden_name(child.name) or not valid_part(child.name):
                        continue
                    if Path(child.path).is_relative_to(protected):
                        continue
                    try:
                        info = Path(child.path).lstat()
                    except FileNotFoundError:
                        continue
                    if is_link(info) or not (
                        stat.S_ISDIR(info.st_mode) or (stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
                    ):
                        continue
                    entries.append({
                        "name": child.name,
                        "path": f"{relative}/{child.name}" if relative else child.name,
                        "directory": stat.S_ISDIR(info.st_mode),
                    })
                    if len(entries) > MAX_DIRECTORY_ENTRIES:
                        break
        except OSError as exc:
            raise WorkspaceError("Hangar cannot read this folder.", 403) from exc
        entries.sort(key=lambda item: (not item["directory"], item["name"].casefold()))
        return {
            "path": relative,
            "entries": entries[:MAX_DIRECTORY_ENTRIES],
            "truncated": len(entries) > MAX_DIRECTORY_ENTRIES,
        }

    def snapshot(self, relative: str, *, new: bool = False) -> tuple[Path, bytes | None]:
        path = self.path(relative, missing_leaf=new)
        try:
            info = path.lstat()
        except FileNotFoundError:
            if new:
                return path, None
            raise WorkspaceError("The file was not found.", 404) from None
        except OSError as exc:
            raise WorkspaceError("Hangar cannot open this file.", 403) from exc
        if not stat.S_ISREG(info.st_mode) or is_link(info) or info.st_nlink > 1:
            raise WorkspaceError("Only regular text files can be opened.")
        if info.st_size > MAX_FILE_BYTES:
            raise WorkspaceError("This file is larger than 256 KiB.", 413)
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
            with os.fdopen(os.open(path, flags), "rb") as f:
                opened = os.fstat(f.fileno())
                if not os.path.samestat(info, opened) or not stat.S_ISREG(opened.st_mode) or opened.st_nlink > 1:
                    raise WorkspaceError("The file changed while opening. Try again.", 409)
                data = f.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise WorkspaceError("Hangar could not read this file.", 403) from exc
        if len(data) > MAX_FILE_BYTES:
            raise WorkspaceError("This file is larger than 256 KiB.", 413)
        return path, data

    def read(self, relative: str) -> str:
        _, data = self.snapshot(relative)
        return decode_text(data)


def decode_text(data: bytes) -> str:
    try:
        if b"\0" in data:
            raise UnicodeError
        return data.decode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceError("Only UTF-8 text files can be opened.", 415) from exc
