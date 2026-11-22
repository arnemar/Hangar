"""Settings for the local Ollama connection."""

from urllib.parse import urlsplit

from fastapi import APIRouter
from pydantic import BaseModel, Field

from hangar.config import OLLAMA_URL
from hangar.persistence.storage import load_memory, load_settings, save_settings
from hangar.runtime.context import positive_int
from hangar.workspace.files import WorkspaceError

router = APIRouter()


class Settings(BaseModel):
    ollama_url: str
    context_length: int = Field(ge=1024)
    system_prompt_chat: str = Field(default="", max_length=20000)
    use_saved_memory: bool = False


@router.get("/api/settings")
async def get_settings():
    settings = load_settings()
    return {
        "ollama_url": settings.get("ollama_url") or OLLAMA_URL,
        "context_length": max(1024, positive_int(settings.get("context_length"), 8192)),
        "system_prompt_chat": settings.get("system_prompt_chat") or "",
        "use_saved_memory": settings.get("use_saved_memory") is True,
        "saved_facts": [
            str(item["content"]) for item in (load_memory().get("facts") or [])
            if isinstance(item, dict) and item.get("content")
        ],
    }


@router.put("/api/settings")
async def update_settings(request: Settings):
    url = request.ollama_url.strip().rstrip("/")
    if not url:
        url = OLLAMA_URL
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise WorkspaceError("Use an HTTP or HTTPS Ollama address, such as http://localhost:11434.")
    data = request.model_dump()
    data["ollama_url"] = url
    save_settings({**load_settings(), **data})
    return data
