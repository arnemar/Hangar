"""Validate one chat turn, then stream its events until completion or disconnect."""

import json
import logging
from contextlib import aclosing

from starlette.responses import StreamingResponse
from starlette.routing import Route

from hangar.api.input import read_object, read_switch, read_text
from hangar.api.routes.projects import require_project
from hangar.api.routes.sessions import require_idle, require_session
from hangar.integrations.ollama import OllamaError
from hangar.persistence.storage import StoreError
from hangar.runtime.turns import Turn, stream_turn
from hangar.workspace.files import WorkspaceError

logger = logging.getLogger(__name__)


def event_line(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


async def event_stream(turn: Turn):
    try:
        async with aclosing(stream_turn(turn)) as stream:
            async for event in stream:
                yield event_line(event)
    except (OllamaError, StoreError, WorkspaceError) as error:
        yield event_line({"type": "error", "content": str(error)})
    except Exception:
        logger.exception("Chat stream failed for %s", turn.session_id)
        yield event_line({"type": "error", "content": "Hangar could not finish the response. Check its server log."})


class ChatResponse(StreamingResponse):
    def __init__(self, turn: Turn, active_chats: set[str]):
        self.turn = turn
        self.active_chats = active_chats
        self.events = event_stream(turn)
        super().__init__(self.events, media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await self.events.aclose()
            finally:
                self.active_chats.discard(self.turn.session_id)


async def chat(request):
    data = await read_object(request)
    session_id = request.path_params["session_id"]
    require_idle(request, session_id)
    session = require_session(session_id)
    if not session["model"]:
        raise WorkspaceError("Choose an installed Ollama model first.")

    message = read_text(data.get("message", ""), "Message", 1048576).strip()
    paths = data.get("attachments", [])
    if not isinstance(paths, list) or len(paths) > 3:
        raise WorkspaceError("Attach up to three file excerpts per message.")
    paths = [read_text(path, "Attachment path", 4096) for path in paths]
    proposals = read_switch(data, "propose_edits")
    search = read_switch(data, "search_web")
    if not message and (not paths or search):
        raise WorkspaceError("Enter a question to search the web." if search else "Enter a message or attach a file.")

    project = None
    project_issue = ""
    if session["project_name"]:
        try:
            project = require_project(session["project_name"])
        except WorkspaceError as error:
            if paths or proposals:
                raise
            project_issue = str(error)
    if (paths or proposals) and project is None:
        raise WorkspaceError("Attach a local project before using its files.", 403)
    attachments = {path: project.read(path) for path in dict.fromkeys(paths)} if project else {}

    active = request.app.state.active_chats
    active.add(session_id)
    turn = Turn(session_id, session["model"], message, attachments, project, proposals, search, project_issue)
    return ChatResponse(turn, active)


routes = [Route("/api/sessions/{session_id}/chat", chat, methods=["POST"])]
