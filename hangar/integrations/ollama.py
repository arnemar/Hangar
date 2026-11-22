"""Ollama transport and model capability discovery."""

import json
from typing import AsyncIterator

import httpx

from hangar.config import OLLAMA_URL

_DEFAULT_TIMEOUT = 600
_tool_support_cache: dict[tuple[str, str], bool] = {}


def _url(ollama_url: str) -> str:
    return (ollama_url or OLLAMA_URL).rstrip("/")


async def get_models(ollama_url: str = "") -> list[str]:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(f"{_url(ollama_url)}/api/tags")
        response.raise_for_status()
        result = response.json()
        models = result.get("models") if isinstance(result, dict) else None
        if not isinstance(models, list):
            raise ValueError("Ollama model list is unavailable.")
        return [model["name"] for model in models if isinstance(model, dict) and isinstance(model.get("name"), str)]


async def model_supports_tools(model: str, ollama_url: str = "") -> bool:
    key = (_url(ollama_url), model)
    if key not in _tool_support_cache:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(f"{key[0]}/api/show", json={"model": model})
                response.raise_for_status()
                data = response.json()
                capabilities = data.get("capabilities", []) if isinstance(data, dict) else []
                if not isinstance(capabilities, list):
                    return False
        except (httpx.HTTPError, ValueError):
            return False
        _tool_support_cache[key] = "tools" in capabilities
    return _tool_support_cache[key]


async def stream_ollama(
    messages: list[dict],
    model: str,
    tools: list[dict] | None = None,
    num_ctx: int = 8192,
    ollama_url: str = "",
    timeout: float = _DEFAULT_TIMEOUT,
) -> AsyncIterator[dict]:
    """Send the selected context window and leave generation settings to Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    if tools:
        payload["tools"] = tools

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{_url(ollama_url)}/api/chat", json=payload) as response:
                if response.is_error:
                    await response.aread()
                    try:
                        detail = response.json().get("error")
                    except ValueError:
                        detail = None
                    if detail:
                        raise RuntimeError(f"Ollama: {detail}")
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Ollama returned an invalid streaming response.") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError("Ollama returned an invalid streaming response.")
                    if event.get("error"):
                        raise RuntimeError(f"Ollama: {event['error']}")
                    yield event
                    if event.get("done"):
                        return
                raise RuntimeError("Ollama's response ended before it was complete.")
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Timed out waiting for Ollama ({timeout:g} seconds).") from exc
    except httpx.ConnectError as exc:
        raise RuntimeError("Could not connect to the configured Ollama server.") from exc
