"""Validate the small JSON payloads accepted by the local API."""

import json

from starlette.requests import Request

from hangar.workspace.files import WorkspaceError


async def read_object(request: Request) -> dict:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise WorkspaceError("Send the request as JSON.", 415)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 2 * 1024 * 1024:
            raise WorkspaceError("The request is too large.", 413)
        body.extend(chunk)
    try:
        value = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError) as exc:
        raise WorkspaceError("The request contains invalid JSON.") from exc
    if not isinstance(value, dict):
        raise WorkspaceError("The request must contain a JSON object.")
    return value


def read_text(value: object, label: str, limit: int) -> str:
    if not isinstance(value, str) or len(value) > limit:
        raise WorkspaceError(f"{label} must be text with at most {limit:,} characters.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceError(f"{label} must be valid UTF-8.") from exc
    return value


def read_switch(data: dict, key: str) -> bool:
    value = data.get(key, False)
    if not isinstance(value, bool):
        raise WorkspaceError(f"{key} must be true or false.")
    return value
