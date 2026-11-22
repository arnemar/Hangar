"""Stage exact text changes and apply them only against the reviewed snapshot."""

import difflib
import hashlib
import os
import stat
import tempfile
import uuid

from hangar.persistence.conversations import now
from hangar.persistence.db import get_db
from hangar.persistence.storage import get_project
from hangar.workspace.files import MAX_FILE_BYTES, LocalProject, WorkspaceError, decode_text

_PUBLIC = "id,session_id,project_name,path,diff,status,created_at"


def _digest(data: bytes | None) -> str | None:
    return hashlib.sha256(data).hexdigest() if data is not None else None


def _diff(path: str, before: str, after: str, create: bool) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile="/dev/null" if create else f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines)


def _line_endings(value: str, ending: str) -> str:
    return value.replace("\r\n", "\n").replace("\n", ending)


def propose_change(
    session_id: str, project: LocalProject, path: str, operation: str, new_text: str, old_text: str = "",
) -> dict:
    if operation not in {"replace", "create"}:
        raise WorkspaceError("Choose replace or create.")
    creating = operation == "create"
    _, original = project.snapshot(path, new=creating)
    if creating:
        if original is not None:
            raise WorkspaceError("That file already exists. Use replace for an existing file.", 409)
        before, after = "", new_text
    else:
        before = decode_text(original)
        if not old_text:
            raise WorkspaceError("Read the file and supply the exact text to replace.")
        ending = "\r\n" if "\r\n" in before else "\n"
        old_text = _line_endings(old_text, ending)
        start = before.find(old_text)
        if start < 0 or before.find(old_text, start + 1) >= 0:
            raise WorkspaceError("The text to replace must match exactly once. Read the file again.", 409)
        after = before[:start] + _line_endings(new_text, ending) + before[start + len(old_text):]

    try:
        encoded = after.encode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceError("The replacement must be valid UTF-8.") from exc
    decode_text(encoded)
    if len(encoded) > MAX_FILE_BYTES:
        raise WorkspaceError("The proposed file would be larger than 256 KiB.", 413)
    if encoded == original:
        raise WorkspaceError("The proposal contains no changes.")

    change_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO file_changes "
            "(id,session_id,project_name,project_root,path,before_hash,after_text,diff,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (change_id, session_id, project.name, str(project.root), path, _digest(original),
             after, _diff(path, before, after, creating), now()),
        )
        return dict(conn.execute(f"SELECT {_PUBLIC} FROM file_changes WHERE id=?", (change_id,)).fetchone())


def changes_for(session_id: str) -> list[dict]:
    with get_db() as conn:
        return [dict(row) for row in conn.execute(
            f"SELECT {_PUBLIC} FROM file_changes WHERE session_id=? ORDER BY created_at", (session_id,),
        )]


def _apply(row) -> None:
    record = get_project(row["project_name"])
    if not record:
        raise WorkspaceError("This project is no longer registered.", 409)
    project = LocalProject(record)
    if str(project.root) != row["project_root"]:
        raise WorkspaceError("The project folder has changed. Ask for a new proposal.", 409)

    target, current = project.snapshot(row["path"], new=True)
    desired = row["after_text"].encode("utf-8")
    if current == desired:
        return
    if _digest(current) != row["before_hash"]:
        raise WorkspaceError("The file changed since this proposal. Ask for a new proposal.", 409)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".hangar-", delete=False) as temp:
            temp_path = type(target)(temp.name)
            temp.write(desired)
            temp.flush()
            os.fsync(temp.fileno())
        _, latest = project.snapshot(row["path"], new=True)
        if _digest(latest) != row["before_hash"]:
            raise WorkspaceError("The file changed since this proposal. Ask for a new proposal.", 409)
        if current is None:
            if os.name == "nt":
                os.rename(temp_path, target)
            else:
                os.link(temp_path, target)
        else:
            os.chmod(temp_path, stat.S_IMODE(target.lstat().st_mode))
            os.replace(temp_path, target)
    except FileExistsError as exc:
        raise WorkspaceError("The file was created after this proposal. Ask for a new proposal.", 409) from exc
    except OSError as exc:
        raise WorkspaceError("Hangar could not write this file. Check the folder permissions.", 403) from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def decide_change(session_id: str, change_id: str, decision: str) -> dict:
    with get_db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM file_changes WHERE session_id=? AND id=?", (session_id, change_id),
        ).fetchone()
        if row is None:
            raise WorkspaceError("File proposal not found.", 404)
        desired = "applied" if decision == "apply" else "dismissed"
        if row["status"] == "pending":
            if decision == "apply":
                session = conn.execute("SELECT project_name FROM sessions WHERE id=?", (session_id,)).fetchone()
                if not session or session["project_name"] != row["project_name"]:
                    raise WorkspaceError("Attach the original project before applying this change.", 409)
                _apply(row)
            conn.execute("UPDATE file_changes SET status=? WHERE id=?", (desired, change_id))
        elif row["status"] != desired:
            raise WorkspaceError("This proposal has already been reviewed.", 409)
        return dict(conn.execute(f"SELECT {_PUBLIC} FROM file_changes WHERE id=?", (change_id,)).fetchone())
