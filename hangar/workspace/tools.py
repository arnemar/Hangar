"""Read project files and prepare edits through a fixed set of Ollama tools."""

import json

from hangar.workspace.changes import propose_change
from hangar.workspace.files import LocalProject, WorkspaceError

LIST = "list_project_files"
READ = "read_project_file"
PROPOSE = "propose_file_change"

_PATH = {"type": "string", "description": "Path relative to the attached project."}
_LINE = {"type": "integer", "minimum": 1}
_DEFINITIONS = {
    LIST: (
        "List files and folders. An empty path lists the project root.",
        {"path": _PATH}, [],
    ),
    READ: (
        "Read a range of lines from a project file. Line numbers start at 1 and are inclusive.",
        {"path": _PATH, "start_line": _LINE, "end_line": _LINE}, ["path"],
    ),
    PROPOSE: (
        "Prepare an edit for the user to review.",
        {
            "path": _PATH,
            "operation": {"type": "string", "enum": ["replace", "create"]},
            "old_text": {"type": "string", "description": "Exact text that occurs once. Omit when creating."},
            "new_text": {"type": "string", "description": "Replacement text or the contents of the new file."},
        },
        ["path", "operation", "new_text"],
    ),
}
_ACTIVITY = {
    LIST: "Listing project files",
    READ: "Reading a project file",
    PROPOSE: "Preparing a change for review",
}


def activity(name: str) -> str:
    return _ACTIVITY.get(name, "Checking a project tool request")


def schemas(allow_proposals: bool) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }}
        for name, (description, properties, required) in _DEFINITIONS.items()
        if name != PROPOSE or allow_proposals
    ]


def read_lines(project: LocalProject, path: str, start: int, end: int | None) -> str:
    if end is not None and end < start:
        raise WorkspaceError("The last line must come after the first line.")
    rows = project.read(path).splitlines()
    count = len(rows)
    if start > count:
        return f"{path} has {count} lines."

    lines = []
    remaining = 12000
    for number in range(start, min(end or count, start + 159, count) + 1):
        value = f"{number}: {rows[number - 1]}"
        if len(value) >= remaining:
            if not lines:
                lines.append(value[:remaining - 18] + " [line truncated]")
            break
        lines.append(value)
        remaining -= len(value) + 1
    return f"{path} (lines {start}-{start + len(lines) - 1} of {count})\n" + "\n".join(lines)


def tool_arguments(name: str, args: object) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, RecursionError) as exc:
            raise WorkspaceError("Invalid tool arguments. Supply a JSON object.") from exc
    if not isinstance(args, dict):
        raise WorkspaceError("Invalid tool arguments. Supply an object.")
    _, properties, required = _DEFINITIONS[name]
    for field in required:
        if field not in args:
            raise WorkspaceError(f"Missing tool argument {field}.")
    values = {}
    for field, schema in properties.items():
        if field not in args or (field == "end_line" and args[field] is None):
            continue
        value = args[field]
        if schema["type"] == "integer":
            if type(value) is not int or value < 1:
                raise WorkspaceError(f"Tool argument {field} must be a positive whole number.")
        elif not isinstance(value, str):
            raise WorkspaceError(f"Tool argument {field} must be text.")
        else:
            value.encode("utf-8")
        values[field] = value
    return values


def run_tool(
    name: str, args: object, session_id: str, project: LocalProject, allow_proposals: bool,
) -> tuple[str, dict | None]:
    if name not in _DEFINITIONS or (name == PROPOSE and not allow_proposals):
        return "This tool is unavailable in this chat.", None
    try:
        values = tool_arguments(name, args)
        if name == LIST:
            return json.dumps(project.entries(values.get("path", "")), ensure_ascii=False), None
        if name == READ:
            return read_lines(project, values["path"], values.get("start_line", 1), values.get("end_line")), None
        proposal = propose_change(session_id, project, **values)
        return f"Proposed a change to {proposal['path']}. The user must review it before it is applied.", proposal
    except (WorkspaceError, UnicodeError) as exc:
        return str(exc), None
