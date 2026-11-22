"""Report installed models and distinguish connection failures from an empty server."""

from starlette.responses import JSONResponse
from starlette.routing import Route

from hangar.config import DEFAULT_MODEL
from hangar.integrations.ollama import OllamaError, get_models
from hangar.persistence.storage import load_settings


async def list_models(request):
    settings = load_settings()
    try:
        models = await get_models(settings["ollama_url"])
    except OllamaError as error:
        return JSONResponse({"models": [], "default": "", "connected": False, "detail": str(error)})
    configured = settings.get("default_model", DEFAULT_MODEL)
    default = configured if configured in models else next(iter(models), "")
    return JSONResponse({"models": models, "default": default, "connected": True})


routes = [Route("/api/models", list_models)]
