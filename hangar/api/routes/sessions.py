"""Saved chat history and per-chat configuration."""

from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from hangar.api.input import read_object, read_text
from hangar.api.routes.projects import require_project
from hangar.persistence import conversations
from hangar.workspace.files import WorkspaceError


def require_session(session_id: str) -> dict:
    session = conversations.get_session(session_id)
    if session is None:
        raise WorkspaceError("Chat not found.", 404)
    return session


def require_idle(request, session_id: str) -> None:
    if session_id in request.app.state.active_chats:
        raise WorkspaceError("Wait for the current response or stop it first.", 409)


def session_fields(data: dict) -> dict:
    fields = {}
    if "project_name" in data:
        name = data["project_name"]
        if name is not None:
            name = read_text(name, "Project name", 100).strip() or None
        if name:
            require_project(name)
        fields["project_name"] = name
    if "model" in data:
        fields["model"] = read_text(data["model"] if data["model"] is not None else "", "Model", 200).strip()
    return fields


async def list_sessions(request):
    return JSONResponse({"sessions": conversations.list_sessions()})


async def create_session(request):
    fields = session_fields(await read_object(request))
    session = conversations.create_session(fields.get("model", ""), fields.get("project_name"))
    return JSONResponse(session, status_code=201)


async def update_session(request):
    data = await read_object(request)
    session_id = request.path_params["session_id"]
    require_idle(request, session_id)
    session = conversations.update_session(session_id, session_fields(data))
    if session is None:
        raise WorkspaceError("Chat not found.", 404)
    return JSONResponse(session)


async def get_messages(request):
    session_id = request.path_params["session_id"]
    require_session(session_id)
    return JSONResponse({"messages": conversations.messages_for(session_id)})


async def delete_session(request):
    session_id = request.path_params["session_id"]
    require_idle(request, session_id)
    if not conversations.delete_session(session_id):
        raise WorkspaceError("Chat not found.", 404)
    return Response(status_code=204)


routes = [
    Route("/api/sessions", list_sessions),
    Route("/api/sessions", create_session, methods=["POST"]),
    Route("/api/sessions/{session_id}", update_session, methods=["PATCH"]),
    Route("/api/sessions/{session_id}", delete_session, methods=["DELETE"]),
    Route("/api/sessions/{session_id}/messages", get_messages),
]
