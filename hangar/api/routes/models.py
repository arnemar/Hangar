"""Report the configured Ollama connection and installed models."""

import httpx
from fastapi import APIRouter

from hangar.config import DEFAULT_MODEL
from hangar.integrations.ollama import get_models
from hangar.persistence.storage import load_settings

router = APIRouter()


@router.get("/api/models")
async def list_models():
    settings = load_settings()
    try:
        models = await get_models(settings.get("ollama_url", ""))
    except (httpx.HTTPError, ValueError):
        return {"models": [], "default": "", "connected": False}
    configured = settings.get("default_model", DEFAULT_MODEL)
    default = configured if configured in models else (models[0] if models else "")
    return {"models": models, "default": default, "connected": True}
