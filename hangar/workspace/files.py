"""Operations on text files within one explicitly registered local project."""

import os
import stat
from pathlib import Path, PureWindowsPath

from hangar.config import DATA_DIR

MAX_FILE_BYTES = 256 * 1024
MAX_DIRECTORY_ENTRIES = 300
_HIDDEN = {
    ".git", ".hg", ".svn", ".venv", "venv", "venv_win",
    "node_modules", "__pycache__", ".ssh", ".aws", ".gnupg",
    ".codex", ".claude", ".kube", ".docker", ".npmrc", ".pypirc", ".netrc",
    ".git-credentials",
}
_PRIVATE_SUFFIXES = (".pem", ".key", ".pfx", ".p12", ".kdbx")


class WorkspaceError(ValueError):
    def __init__(self, detail: str, status: int = 400):
        super().__init__(detail)
        self.status = status


def _hidden(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered in _HIDDEN or lowered.startswith(".hangar-")
        or (lowered == ".env" or lowered.startswith(".env."))
        and lowered not in {".env.example", ".env.sample", ".env.template"}
        or lowered.endswith(_PRIVATE_SUFFIXES)
    )


def _valid_part(part: str) -> bool:
    return bool(
        part and part not in {".", ".."} and part == part.rstrip(" .")
        and not any(character in part for character in ":\\\\")
        and not PureWindowsPath(part).is_reserved()
    )


def _link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def register_root(raw: str) -> Path:
    if not raw.strip():
        raise WorkspaceError("Enter the absolute path of a folder.")
    try:
        path = Path(raw.strip()).expanduser()
        if not path.is_absolute():
            raise WorkspaceError("Use an absolute folder path on the computer running Hangar.")
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise WorkspaceError("This path is not a folder.")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise WorkspaceError("That folder was not found on the computer running Hangar.", 404) from exc
    except (OSError, RuntimeError) as exc:
        raise WorkspaceError("Hangar cannot access that folder.", 403) from exc
    if resolved.is_relative_to(DATA_DIR.resolve()) or any(_hidden(part) for part in resolved.parts):
        raise WorkspaceError("This folder contains protected application or credential files.", 403)
    return resolved


class LocalProject:
    def __init__(self, record: dict):
        if record.get("remote"):
            raise WorkspaceError("This project uses SSH. Attach a local project folder.", 409)
        self.name = record["name"]
        self.root = register_root(record.get("root", ""))
        if record.get("scope_version") == 1 and self.root != Path(record["root"]):
            raise WorkspaceError("The configured project folder has changed. Reattach it.", 409)

    def path(self, relative: str, *, missing_leaf: bool = False) -> Path:
        if self.root.resolve() != self.root:
            raise WorkspaceError("The configured project folder has changed. Reattach it.", 409)
        if not relative:
            return self.root
        windows = PureWindowsPath(relative)
        if (
            windows.drive or windows.root or relative.startswith("/")
            or "\\" in relative or "\0" in relative
        ):
            raise WorkspaceError("Use a path relative to the project with forward slashes.", 403)
        parts = relative.split("/")
        if not all(_valid_part(part) for part in parts):
            raise WorkspaceError("This is not a valid project-relative path.", 403)
        current = self.root
        for index, part in enumerate(parts):
            if _hidden(part):
                raise WorkspaceError("This file or folder is protected.", 403)
            current = current / part
            if current.is_relative_to(DATA_DIR.resolve()):
                raise WorkspaceError("This file or folder is protected.", 403)
            try:
                info = current.lstat()
            except FileNotFoundError as exc:
                if missing_leaf and index == len(parts) - 1:
                    return current
                raise WorkspaceError("The file or folder was not found.", 404) from exc
            except NotADirectoryError as exc:
                raise WorkspaceError("Part of this path is not a folder.", 404) from exc
            except OSError as exc:
                raise WorkspaceError("Hangar cannot access this path.", 403) from exc
            if _link(info) or not current.resolve().is_relative_to(self.root):
                raise WorkspaceError("Linked files and folders are outside project access.", 403)
            if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                raise WorkspaceError("Hard-linked files are outside project access.", 403)
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise WorkspaceError("Part of this path is not a folder.", 404)
        return current

    def entries(self, relative: str = "") -> dict:
        relative = "" if relative == "." else relative
        folder = self.path(relative)
        if not folder.is_dir():
            raise WorkspaceError("This path is not a folder.")
        entries = []
        try:
            with os.scandir(folder) as children:
                for child in children:
                    if not _valid_part(child.name) or _hidden(child.name) or Path(child.path).is_relative_to(DATA_DIR.resolve()):
                        continue
                    try:
                        info = Path(child.path).lstat()
                    except FileNotFoundError:
                        continue
                    if _link(info) or not (
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
        if not stat.S_ISREG(info.st_mode) or _link(info) or info.st_nlink > 1:
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
