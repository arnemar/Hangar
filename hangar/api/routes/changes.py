"""Human review endpoints. Model tools only create pending proposals."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from hangar.api.routes.sessions import require_idle, require_session
from hangar.workspace.changes import changes_for, decide_change

router = APIRouter()


class ChangeDecision(BaseModel):
    decision: Literal["apply", "dismiss"]


@router.get("/api/sessions/{session_id}/changes")
async def list_changes(session_id: str):
    require_session(session_id)
    return {"changes": changes_for(session_id)}


@router.post("/api/sessions/{session_id}/changes/{change_id}")
async def review_change(session_id: str, change_id: str, request: ChangeDecision):
    require_idle(session_id)
    return decide_change(session_id, change_id, request.decision)
