"""Execute a model's requested tools within the attached project's scope."""

from hangar.workspace.files import LocalProject
from hangar.workspace.tools import activity, run_tool


def tool_events(calls: list[dict], session_id: str, project: LocalProject, allow_edits: bool):
    for call in calls:
        function = call["function"]
        name = function["name"]
        yield {"type": "activity", "content": activity(name)}, None
        result, proposal = run_tool(name, function.get("arguments"), session_id, project, allow_edits)
        reply = {"role": "tool", "tool_name": name, "content": result}
        yield {"type": "proposal", "change": proposal} if proposal else None, reply
