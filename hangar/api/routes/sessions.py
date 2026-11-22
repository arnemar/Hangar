"""Persistent conversations with per-chat model and project selection."""

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from hangar.api.routes.projects import require_project
from hangar.persistence import conversations
from hangar.persistence.db import get_db
from hangar.runtime.turns import is_running
from hangar.workspace.files import WorkspaceError

router = APIRouter()


class SessionCreate(BaseModel):
    model: str = Field(default="", max_length=200)
    project_name: str | None = None


class SessionUpdate(BaseModel):
    model: str | None = Field(default=None, max_length=200)
    project_name: str | None = None


def require_session(session_id: str) -> dict:
    with get_db() as conn:
        session = conversations.find_session(session_id, conn)
    if not session:
        raise WorkspaceError("Chat not found.", 404)
    return session


def require_idle(session_id: str) -> None:
    if is_running(session_id):
        raise WorkspaceError("Wait for the current response or stop it first.", 409)


@router.get("/api/sessions")
async def get_sessions():
    return {"sessions": conversations.list_sessions()}


@router.post("/api/sessions", status_code=201)
async def create_session(request: SessionCreate):
    if request.project_name:
        require_project(request.project_name)
    return conversations.create_session(request.model.strip(), request.project_name)


@router.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, request: SessionUpdate):
    require_idle(session_id)
    fields = request.model_dump(exclude_unset=True)
    if fields.get("project_name"):
        require_project(fields["project_name"])
    if "model" in fields:
        fields["model"] = (fields["model"] or "").strip()
    session = conversations.update_session(session_id, fields)
    if session is None:
        raise WorkspaceError("Chat not found.", 404)
    return session


@router.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    require_session(session_id)
    return {"messages": conversations.messages_for(session_id)}


@router.delete("/api/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    require_idle(session_id)
    if not conversations.delete_session(session_id):
        raise WorkspaceError("Chat not found.", 404)
    return Response(status_code=204)
