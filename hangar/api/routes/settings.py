"""Configure the Ollama connection, context window, and system prompt."""

from urllib.parse import urlsplit

from starlette.responses import JSONResponse
from starlette.routing import Route

from hangar.api.input import read_object, read_switch, read_text
from hangar.config import OLLAMA_URL
from hangar.persistence.storage import load_settings, save_settings, saved_facts
from hangar.runtime.context import positive_int
from hangar.workspace.files import WorkspaceError


async def get_settings(request):
    settings = load_settings()
    result = {
        "ollama_url": settings["ollama_url"],
        "context_length": max(1024, positive_int(settings.get("context_length"), 8192)),
        "system_prompt_chat": settings.get("system_prompt_chat") or "",
        "use_saved_memory": settings.get("use_saved_memory") is True,
    }
    return JSONResponse({**result, "saved_facts": saved_facts()})


def ollama_url(value: object) -> str:
    url = read_text(value, "Ollama address", 2048).strip().rstrip("/") or OLLAMA_URL
    try:
        parsed = urlsplit(url)
        port = parsed.port
        if (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment and port != 0
            and not any(character.isspace() or ord(character) < 32 for character in url)
        ):
            return url
    except ValueError:
        pass
    raise WorkspaceError("Use an HTTP or HTTPS Ollama address, such as http://localhost:11434.")


async def update_settings(request):
    data = await read_object(request)
    context = data.get("context_length")
    if type(context) is not int or context < 1024:
        raise WorkspaceError("Context window must be a whole number of at least 1,024.")
    settings = {
        "ollama_url": ollama_url(data.get("ollama_url")),
        "context_length": context,
        "system_prompt_chat": read_text(data.get("system_prompt_chat", ""), "Custom instructions", 20000),
        "use_saved_memory": read_switch(data, "use_saved_memory"),
    }
    save_settings({**load_settings(), **settings})
    return JSONResponse(settings)


routes = [
    Route("/api/settings", get_settings),
    Route("/api/settings", update_settings, methods=["PUT"]),
]
