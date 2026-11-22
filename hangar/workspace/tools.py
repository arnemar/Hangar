"""The complete set of model-accessible operations on an attached project."""

import json

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

from hangar.workspace.changes import propose_change
from hangar.workspace.files import LocalProject, WorkspaceError


class ListFiles(BaseModel):
    path: str = Field(default="", description="Folder relative to the project. Empty lists the root.")


class ReadFile(BaseModel):
    path: str = Field(description="File path relative to the project.")
    start_line: int = Field(default=1, ge=1, description="First line to read, starting at 1.")
    end_line: int | None = Field(default=None, ge=1, description="Last line to read, inclusive.")


class ProposeEdit(BaseModel):
    path: str = Field(description="File path relative to the project.")
    operation: Literal["replace", "create"]
    old_text: str = Field(default="", description="Exact text that appears once in an existing file. Omit for create.")
    new_text: str = Field(description="Replacement text or the contents of the new file.")


_TOOLS = {
    "list_project_files": (ListFiles, "List files and folders in the attached project."),
    "read_project_file": (ReadFile, "Read a range of lines from a text file in the attached project."),
    "propose_file_change": (ProposeEdit, "Draft a file change for the user to review. Does not write any file."),
}

_ACTIVITY = {
    "list_project_files": "Listing project files",
    "read_project_file": "Reading a project file",
    "propose_file_change": "Preparing a change for review",
}


def activity(name: str) -> str:
    return _ACTIVITY.get(name, "Checking a project tool request")


def schemas(allow_proposals: bool) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": name, "description": description, "parameters": model.model_json_schema(),
        }}
        for name, (model, description) in _TOOLS.items()
        if name != "propose_file_change" or allow_proposals
    ]


def _read(project: LocalProject, request: ReadFile) -> str:
    rows = project.read(request.path).splitlines()
    count = len(rows)
    start = request.start_line
    if start > count:
        return f"{request.path} has {count} lines."
    end = min(request.end_line or count, start + 159, count)
    if end < start:
        raise WorkspaceError("The last line must come after the first line.")
    lines = []
    remaining = 12000
    for number in range(start, end + 1):
        line = rows[number - 1]
        if len(line) + 16 > remaining:
            if number == start:
                lines.append(f"{number}: {line[:remaining-25]} [line truncated]")
            else:
                end = number - 1
            break
        value = f"{number}: {line}"
        lines.append(value)
        remaining -= len(value) + 1
    return f"{request.path} (lines {start}-{end} of {count})\n" + "\n".join(lines)


def run_tool(
    name: str, args: object, session_id: str, project: LocalProject, allow_proposals: bool,
) -> tuple[str, dict | None]:
    if name not in _TOOLS or (name == "propose_file_change" and not allow_proposals):
        return "This tool is unavailable in this chat.", None
    try:
        request = _TOOLS[name][0].model_validate(args)
        if name == "list_project_files":
            return json.dumps(project.entries(request.path), ensure_ascii=False), None
        if name == "read_project_file":
            return _read(project, request), None
        proposal = propose_change(session_id, project, **request.model_dump())
        return f"Proposed a change to {proposal['path']}. The user must review it before it is applied.", proposal
    except ValidationError as exc:
        issue = exc.errors(include_input=False)[0]
        field = ".".join(map(str, issue["loc"]))
        return f"Invalid tool arguments: {field} {issue['msg']}".strip(), None
    except WorkspaceError as exc:
        return str(exc), None
