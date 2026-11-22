"""Apply or dismiss a pending change after user review."""

from starlette.responses import JSONResponse
from starlette.routing import Route

from hangar.api.input import read_object
from hangar.api.routes.sessions import require_idle, require_session
from hangar.workspace.changes import changes_for, decide_change
from hangar.workspace.files import WorkspaceError


async def list_changes(request):
    session_id = request.path_params["session_id"]
    require_session(session_id)
    return JSONResponse({"changes": changes_for(session_id)})


async def review_change(request):
    data = await read_object(request)
    decision = data.get("decision")
    if decision not in ("apply", "dismiss"):
        raise WorkspaceError("Choose apply or dismiss.")
    session_id = request.path_params["session_id"]
    require_idle(request, session_id)
    return JSONResponse(decide_change(session_id, request.path_params["change_id"], decision))


routes = [
    Route("/api/sessions/{session_id}/changes", list_changes),
    Route("/api/sessions/{session_id}/changes/{change_id}", review_change, methods=["POST"]),
]
