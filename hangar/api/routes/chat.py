"""Stream one chat turn through the attached project's explicit scope."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hangar.api.routes.projects import require_project
from hangar.api.routes.sessions import require_idle, require_session
from hangar.runtime.turns import stream_turn
from hangar.workspace.files import WorkspaceError

router = APIRouter()


class ChatTurn(BaseModel):
    message: str = Field(default="", max_length=1048576)
    attachments: list[str] = Field(default_factory=list, max_length=3)
    propose_edits: bool = False
    search_web: bool = False


@router.post("/api/sessions/{session_id}/chat")
async def chat(session_id: str, request: ChatTurn):
    require_idle(session_id)
    session = require_session(session_id)
    if not session["model"]:
        raise WorkspaceError("Choose an installed Ollama model first.")
    message = request.message.strip()
    attachments = list(dict.fromkeys(request.attachments))
    if not message and not attachments:
        raise WorkspaceError("Enter a message or attach a file.")
    if request.search_web and not message:
        raise WorkspaceError("Enter a question to search the web.")
    project_name = session["project_name"]
    project = require_project(project_name) if project_name else None
    if (attachments or request.propose_edits) and not project:
        raise WorkspaceError("Attach a project before using its files.", 403)
    excerpts = [project.read(path) for path in attachments]
    return StreamingResponse(
        stream_turn(session, message, attachments, excerpts, project, request.propose_edits, request.search_web),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
