"""Validate project roots, path components, and filesystem boundaries."""

import os
import stat
from pathlib import Path, PureWindowsPath

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


def hidden_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered in _HIDDEN or lowered.startswith(".hangar-")
        or (lowered == ".env" or lowered.startswith(".env."))
        and lowered not in {".env.example", ".env.sample", ".env.template"}
        or lowered.endswith(_PRIVATE_SUFFIXES)
    )


def valid_part(part: str) -> bool:
    return bool(
        part and part not in {".", ".."} and part == part.rstrip(" .")
        and not any(character in part for character in ":\\\\")
        and not PureWindowsPath(part).is_reserved()
    )


def is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def validate_root(raw: str, data_dir: Path) -> Path:
    if not raw.strip() or "\0" in raw:
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
    if resolved.is_relative_to(data_dir.resolve()) or any(hidden_name(part) for part in resolved.parts):
        raise WorkspaceError("This folder contains protected application or credential files.", 403)
    return resolved


def project_path(root: Path, data_dir: Path, relative: str, *, missing_leaf: bool = False) -> Path:
    if root.resolve() != root:
        raise WorkspaceError("The configured project folder has changed. Reattach it.", 409)
    if not relative:
        return root
    windows = PureWindowsPath(relative)
    if (
        windows.drive or windows.root or relative.startswith("/")
        or "\\" in relative or "\0" in relative
    ):
        raise WorkspaceError("Use a path relative to the project with forward slashes.", 403)
    parts = relative.split("/")
    if not all(valid_part(part) for part in parts):
        raise WorkspaceError("This is not a valid project-relative path.", 403)
    current = root
    for index, part in enumerate(parts):
        if hidden_name(part):
            raise WorkspaceError("This file or folder is protected.", 403)
        current = current / part
        if current.is_relative_to(data_dir):
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
        if is_link(info) or not current.resolve().is_relative_to(root):
            raise WorkspaceError("Linked files and folders are outside project access.", 403)
        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
            raise WorkspaceError("Hard-linked files are outside project access.", 403)
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise WorkspaceError("Part of this path is not a folder.", 404)
    return current
